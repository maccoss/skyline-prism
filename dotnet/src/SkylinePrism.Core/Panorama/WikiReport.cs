using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using AngleSharp.Css.Parser;
using AngleSharp.Dom;
using AngleSharp.Html.Parser;

namespace SkylinePrism.Core.Panorama;

/// <summary>An image lifted out of a report, to be attached to the wiki page.</summary>
public sealed record WikiImage(string Name, byte[] Data);

/// <summary>
/// A report rewritten as a Panorama wiki body: <see cref="BodyTemplate"/> with each image's
/// <c>src</c> left as <see cref="WikiReport.ImagePlaceholder"/> + its name until the attachments'
/// addresses are known.
/// </summary>
public sealed record WikiDocument(string BodyTemplate, IReadOnlyList<WikiImage> Images)
{
    /// <summary>The body with every image pointed at its attachment.</summary>
    public string Render(Func<string, string> imageUrl) =>
        Regex.Replace(BodyTemplate, Regex.Escape(WikiReport.ImagePlaceholder) + @"([^""']+)", m => imageUrl(m.Groups[1].Value));
}

/// <summary>
/// Turns one of PRISM's self-contained HTML reports (<c>qc_report.html</c>, <c>quant_report.html</c>)
/// into a body Panorama's wiki will accept from an ordinary user.
/// </summary>
/// <remarks>
/// <para><b>What LabKey refuses.</b> Unless the account is a trusted browser developer, a wiki save
/// runs <c>PageFlowUtil.validateHtml</c> and refuses the page outright on any of: a
/// <c>&lt;style&gt;</c>, <c>&lt;link&gt;</c>, <c>&lt;script&gt;</c> or form element, an <c>on*</c>
/// attribute, a <c>script:</c> URL, or a <c>style</c> attribute containing <c>url</c>,
/// <c>expression</c> or <c>behavior</c> (read from LabKey's source, platform/api PageFlowUtil). The
/// reports keep their CSS in a <c>&lt;style&gt;</c> block, so it is applied to each element as a
/// <c>style</c> attribute here - selectors matched by a real CSS engine (AngleSharp), in specificity
/// order, an element's own inline style last - and the block removed.</para>
/// <para><b>Classes are dropped</b> once their styles are inline. Panorama's own stylesheet
/// (Bootstrap) also styles <c>.container</c>, <c>.box</c> and <c>.note</c>, and would otherwise
/// rearrange the page.</para>
/// <para><b>Images become attachments</b> rather than staying as data URIs. They are most of a
/// report's bytes (27 plots, ~3 MB, in a 96-sample QC report), and LabKey keeps every version of a
/// page body, so each republish would store them again.</para>
/// </remarks>
public static class WikiReport
{
    /// <summary>What an image's <c>src</c> holds until <see cref="WikiDocument.Render"/> fills it in.</summary>
    public const string ImagePlaceholder = "prism-wiki-image:";

    private static readonly string[] RemovedElements =
    {
        "style", "link", "script", "meta", "title", "object", "applet", "embed", "iframe", "frame", "frameset", "plaintext",
        "form", "input", "button",
    };

    // Substrings LabKey refuses inside a style attribute (after removing comments).
    private static readonly string[] RefusedInStyle = { "url", "expression", "behavior" };

    /// <summary>
    /// Rewrites a report. Each image's attachment name is <paramref name="imagePrefix"/>, its position,
    /// and a hash of its bytes - so an unchanged plot keeps its name from one publish to the next, and a
    /// changed one never takes over the name an earlier version of the page shows.
    /// </summary>
    public static WikiDocument FromReport(string html, string imagePrefix)
    {
        var document = new HtmlParser().ParseDocument(html);
        var css = string.Concat(document.QuerySelectorAll("style").Select(s => s.TextContent));
        var styles = ComputeStyles(document, css);

        foreach (var (element, declarations) in styles)
        {
            if (element == document.Body || element == document.DocumentElement)
                continue;
            // An element whose own style held nothing LabKey accepts loses the attribute altogether;
            // left as it was, it would carry the refused declarations through.
            if (declarations.Count == 0)
                element.RemoveAttribute("style");
            else
                element.SetAttribute("style", Format(declarations));
        }

        var images = new List<WikiImage>();
        foreach (var img in document.QuerySelectorAll("img").ToList())
        {
            var src = img.GetAttribute("src") ?? "";
            var data = Regex.Match(src, @"^data:image/(png|jpe?g|gif|svg\+xml);base64,(.*)$", RegexOptions.Singleline);
            if (!data.Success)
                continue;
            var extension = data.Groups[1].Value switch { "svg+xml" => ".svg", "jpeg" or "jpg" => ".jpg", var e => "." + e };
            var bytes = Convert.FromBase64String(data.Groups[2].Value.Trim());
            var name = $"{imagePrefix}-{images.Count + 1:00}-{Convert.ToHexStringLower(SHA256.HashData(bytes))[..8]}{extension}";
            images.Add(new WikiImage(name, bytes));
            img.SetAttribute("src", ImagePlaceholder + name);
        }

        foreach (var tag in RemovedElements)
            foreach (var element in document.QuerySelectorAll(tag).ToList())
                element.Remove();

        foreach (var element in document.All)
        {
            element.RemoveAttribute("class");
            foreach (var attribute in element.Attributes.Where(a => a.Name.StartsWith("on", StringComparison.OrdinalIgnoreCase)
                                                                     || a.Name.StartsWith("behavior", StringComparison.OrdinalIgnoreCase)
                                                                     || (a.Name is "href" or "src" && ScriptUrl(a.Value))).ToList())
                element.RemoveAttribute(attribute.Name);
        }

        // The page's own <body> styles (font, color) go on a wrapper, since a wiki body has no <body>.
        var bodyStyle = document.Body is { } body && styles.TryGetValue(body, out var b) ? Format(b) : "";
        var inner = document.Body?.InnerHtml ?? "";
        var wrapper = bodyStyle.Length > 0 ? $"<div style=\"{Escape(bodyStyle)}\">{inner}</div>" : $"<div>{inner}</div>";
        return new WikiDocument(wrapper, images);
    }

    /// <summary>
    /// Each element's resolved declarations: matching rules in ascending specificity (document order
    /// breaking ties, as the cascade does), then the element's own <c>style</c> attribute on top.
    /// </summary>
    private static Dictionary<IElement, Dictionary<string, string>> ComputeStyles(IDocument document, string css)
    {
        var matched = new Dictionary<IElement, List<(AngleSharp.Css.Priority Specificity, int Order, string Declarations)>>();
        var selectorParser = new CssSelectorParser();
        var order = 0;
        foreach (Match rule in Regex.Matches(StripComments(css), @"([^{}@]+)\{([^{}]*)\}"))
        {
            var declarations = rule.Groups[2].Value;
            foreach (var selectorText in rule.Groups[1].Value.Split(',').Select(s => s.Trim()).Where(s => s.Length > 0))
            {
                var selector = selectorParser.ParseSelector(selectorText);
                if (selector is null)
                    continue;
                IEnumerable<IElement> elements;
                try
                {
                    elements = document.QuerySelectorAll(selectorText);
                }
                catch (DomException)
                {
                    // A pseudo-class with no static meaning (:hover) cannot be inlined; it is dropped.
                    continue;
                }

                foreach (var element in elements)
                {
                    if (!matched.TryGetValue(element, out var list))
                        matched[element] = list = new();
                    list.Add((selector.Specificity, order++, declarations));
                }
            }
        }

        var resolved = new Dictionary<IElement, Dictionary<string, string>>();
        foreach (var element in document.All)
        {
            var merged = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            if (matched.TryGetValue(element, out var list))
                foreach (var (_, _, declarations) in list.OrderBy(m => m.Specificity).ThenBy(m => m.Order))
                    Apply(merged, declarations);
            var own = element.GetAttribute("style");
            if (own is not null)
                Apply(merged, own);
            // An element with a style attribute is always rewritten, even to nothing, so a declaration
            // Apply dropped does not survive in the original attribute.
            if (merged.Count > 0 || own is not null)
                resolved[element] = merged;
        }

        return resolved;
    }

    private static void Apply(Dictionary<string, string> merged, string declarations)
    {
        foreach (var part in declarations.Split(';'))
        {
            var colon = part.IndexOf(':');
            if (colon <= 0)
                continue;
            var property = part[..colon].Trim();
            var value = part[(colon + 1)..].Trim();
            if (property.Length == 0 || value.Length == 0)
                continue;
            // Anything LabKey would refuse is left out rather than allowed to fail the whole save.
            if (RefusedInStyle.Any(r => value.Contains(r, StringComparison.OrdinalIgnoreCase)
                                        || property.Contains(r, StringComparison.OrdinalIgnoreCase)))
                continue;
            // Interaction-only properties do nothing on a static wiki page.
            if (property.Equals("cursor", StringComparison.OrdinalIgnoreCase))
                continue;
            merged.Remove(property); // re-add so a later declaration also takes the later position
            merged[property] = value;
        }
    }

    private static string Format(Dictionary<string, string> declarations) =>
        string.Join("; ", declarations.Select(d => $"{d.Key}: {d.Value}"));

    private static string StripComments(string css) => Regex.Replace(css, @"/\*.*?\*/", "", RegexOptions.Singleline);

    private static string Escape(string attribute) => attribute.Replace("&", "&amp;").Replace("\"", "&quot;");

    /// <summary>
    /// Whether Panorama would refuse this HTML as an uploaded FILE: it applies the wiki's rules to an
    /// <c>.html</c> put into a file area (all of them - see <see cref="Refusals"/>), and refuses a
    /// <c>&lt;!doctype&gt;</c> as well (measured on panoramaweb.org, October 2026: the same bytes named
    /// <c>.txt</c> are accepted, and of doctype, <c>&lt;html&gt;</c>, <c>&lt;head&gt;</c>,
    /// <c>&lt;meta&gt;</c> and <c>&lt;title&gt;</c> only the doctype is refused).
    /// </summary>
    public static bool RefusedAsFile(string html) =>
        Regex.IsMatch(html, @"<!doctype|<style|<link|<script", RegexOptions.IgnoreCase) || Refusals(html).Count > 0;

    /// <summary>
    /// A report as a standalone file Panorama accepts: the same page with its CSS inlined, its images
    /// still embedded (so the file stands alone when downloaded), and no doctype.
    /// </summary>
    /// <remarks>
    /// Without a doctype a browser renders in quirks mode, where a table does not inherit the page's
    /// font. The tables are told to inherit it explicitly, so the downloaded file reads like the
    /// original.
    /// </remarks>
    public static string ToPanoramaFile(string html)
    {
        var title = Regex.Match(html, @"<title>(.*?)</title>", RegexOptions.Singleline | RegexOptions.IgnoreCase) is { Success: true } t
            ? t.Groups[1].Value
            : "PRISM report";
        var document = FromReport(html, "embedded");
        var images = document.Images.ToDictionary(i => i.Name, i => i);
        var body = document.Render(name => "data:" + MediaType(name) + ";base64," + Convert.ToBase64String(images[name].Data));
        body = Regex.Replace(body, @"<table style=""", "<table style=\"font: inherit; color: inherit; ");
        body = Regex.Replace(body, @"<table>", "<table style=\"font: inherit; color: inherit\">");
        return "<html lang=\"en\"><head><meta charset=\"utf-8\"><title>" + title + "</title></head>"
               + "<body style=\"margin: 0\">" + body + "</body></html>";

        static string MediaType(string name) => System.IO.Path.GetExtension(name) switch
        {
            ".svg" => "image/svg+xml",
            ".jpg" => "image/jpeg",
            ".gif" => "image/gif",
            _ => "image/png",
        };
    }

    /// <summary>
    /// Every reason Panorama would refuse this body from an ordinary user, by the same rules as
    /// LabKey's <c>validateHtml</c>; empty when it would be accepted. A test hook, and a last check
    /// before a save so a refusal names the element rather than arriving as a generic error.
    /// </summary>
    public static IReadOnlyList<string> Refusals(string body)
    {
        var problems = new List<string>();
        var document = new HtmlParser().ParseDocument("<!doctype html><html><body>" + body + "</body></html>");
        foreach (var tag in new[] { "link", "style", "script", "object", "applet", "form", "input", "button", "frame", "frameset", "iframe", "embed", "plaintext" })
            if (document.QuerySelector(tag) is not null)
                problems.Add($"Illegal element <{tag}>.");
        foreach (var element in document.All)
            foreach (var attribute in element.Attributes)
            {
                if (attribute.Name.StartsWith("on", StringComparison.OrdinalIgnoreCase)
                    || attribute.Name.StartsWith("behavior", StringComparison.OrdinalIgnoreCase))
                    problems.Add($"Illegal attribute '{attribute.Name}' on element <{element.LocalName}>.");
                if (attribute.Name is "href" or "src" && ScriptUrl(attribute.Value))
                    problems.Add($"Script is not allowed in '{attribute.Name}' attribute on element <{element.LocalName}>.");
                if (attribute.Name == "style"
                    && RefusedInStyle.Any(r => StripComments(attribute.Value).Contains(r, StringComparison.OrdinalIgnoreCase)))
                    problems.Add($"Style attribute cannot contain behaviors, expressions, or urls. Error on element <{element.LocalName}>.");
            }

        return problems;
    }

    private static bool ScriptUrl(string value) => Regex.IsMatch(value, @"^\s*[a-z]*script:", RegexOptions.IgnoreCase);

    /// <summary>The fingerprint the footer carries, so a later publish can tell whether the page was edited on Panorama.</summary>
    public static string Fingerprint(string text) =>
        Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text)))[..16];
}
