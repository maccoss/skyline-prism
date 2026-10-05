using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace SkylinePrism.Core.Panorama;

/// <summary>A wiki page as Panorama's editor holds it: what a save needs, and its current text.</summary>
/// <param name="EntityId">LabKey's ID for the page; attachments are filed under it.</param>
/// <param name="PageVersionId">The version being replaced; LabKey refuses a save when someone has edited it since.</param>
/// <param name="Attachments">
/// The files attached to the page now, as the editor lists them (<c>LABKEY._wiki.setAttachments</c>);
/// null when the editor did not say.
/// </param>
public sealed record WikiPageInfo(
    string EntityId, int? RowId, int? PageVersionId, string Name, string Title, string Body, int? Parent,
    bool ShowAttachments, bool ShouldIndex, IReadOnlyList<string>? Attachments = null);

/// <summary>A file attached to a wiki page, and where Panorama serves it.</summary>
public sealed record WikiAttachment(string Name, string? DownloadUrl);

public sealed partial class PanoramaClient
{
    /// <summary>The wiki page as Panorama's editor holds it, or null when the folder has no page by that name.</summary>
    public async Task<WikiPageInfo?> GetWikiPageAsync(string container, string name, CancellationToken cancellationToken = default)
    {
        var url = PanoramaPaths.Encode(PanoramaPaths.AsFolder(PanoramaPaths.Container(container)))
                  + "wiki-edit.view?name=" + Uri.EscapeDataString(name);
        var body = await SendAsync(() => new HttpRequestMessage(HttpMethod.Get, url),
            $"the wiki page {name} in {container}", allowMissing: true, MetadataTimeout, cancellationToken).ConfigureAwait(false);
        return body is null ? null : ParseWikiEditor(body);
    }

    /// <summary>
    /// Writes the wiki page as HTML: a new page, or a new version of <paramref name="existing"/>
    /// (Panorama keeps the earlier versions). Returns the page as saved.
    /// </summary>
    public async Task<WikiPageInfo> SaveWikiPageAsync(
        string container, string name, string title, string html, WikiPageInfo? existing,
        bool showAttachments, CancellationToken cancellationToken = default)
    {
        var payload = new Dictionary<string, object?>
        {
            ["entityId"] = existing?.EntityId,
            ["rowId"] = existing?.RowId,
            ["name"] = name,
            ["title"] = title,
            ["body"] = html,
            ["parent"] = existing?.Parent,
            ["pageVersionId"] = existing?.PageVersionId,
            ["rendererType"] = "HTML",
            ["showAttachments"] = showAttachments,
            ["shouldIndex"] = existing?.ShouldIndex ?? true,
        };
        var body = await PostWithCsrfAsync(
            WikiApiUrl(container, "wiki-saveWiki.api"),
            () => JsonContent.Create(payload),
            $"the wiki page {name} in {container}", cancellationToken).ConfigureAwait(false);
        return ParseSaved(body, name, title, html, existing);
    }

    /// <summary>
    /// Attaches files to a wiki page and/or removes attachments by name, through LabKey's
    /// <c>wiki-attachFiles.api</c> (multipart, filed under the page's entity ID).
    /// </summary>
    /// <returns>The page's attachments after the change, with their download addresses when Panorama gave them.</returns>
    public async Task<IReadOnlyList<WikiAttachment>> AttachFilesAsync(
        string container, string entityId, IReadOnlyList<(string Name, byte[] Data)> files,
        IReadOnlyList<string> toDelete, CancellationToken cancellationToken = default)
    {
        var body = await PostWithCsrfAsync(
            WikiApiUrl(container, "wiki-attachFiles.api"),
            () =>
            {
                var form = new MultipartFormDataContent { { new StringContent(entityId), "entityId" } };
                foreach (var name in toDelete)
                    form.Add(new StringContent(name), "toDelete");
                for (var i = 0; i < files.Count; i++)
                {
                    var part = new ByteArrayContent(files[i].Data);
                    part.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(MediaTypeFor(files[i].Name));
                    form.Add(part, $"formFiles[{i}]", files[i].Name);
                }
                return form;
            },
            $"the attachments of the wiki page in {container}", cancellationToken).ConfigureAwait(false);
        return ParseAttachments(body);
    }

    private static string WikiApiUrl(string container, string action) =>
        PanoramaPaths.Encode(PanoramaPaths.AsFolder(PanoramaPaths.Container(container))) + action;

    private static string MediaTypeFor(string name) => System.IO.Path.GetExtension(name).ToLowerInvariant() switch
    {
        ".png" => "image/png",
        ".jpg" or ".jpeg" => "image/jpeg",
        ".svg" => "image/svg+xml",
        ".csv" => "text/csv",
        ".json" => "application/json",
        ".yaml" or ".yml" => "text/yaml",
        _ => "application/octet-stream",
    };

    /// <summary>
    /// A POST to a LabKey API, which refuses one without a CSRF token even with an API key ("invalid
    /// security context"). As LabKey's own client libraries do, the token comes from
    /// <c>login-whoami.api</c>, and that answer's session cookies go with the POST, since the token
    /// belongs to that session. Both are used for one request and never kept or logged.
    /// </summary>
    private async Task<string> PostWithCsrfAsync(
        string url, Func<HttpContent> content, string display, CancellationToken cancellationToken)
    {
        var (token, cookie) = await SecurityContextAsync(cancellationToken).ConfigureAwait(false);
        var body = await SendAsync(() =>
        {
            var request = new HttpRequestMessage(HttpMethod.Post, url) { Content = content() };
            if (token is not null)
                request.Headers.TryAddWithoutValidation("X-LABKEY-CSRF", token);
            if (cookie is not null)
                request.Headers.TryAddWithoutValidation("Cookie", cookie);
            return request;
        }, display, allowMissing: false, MetadataTimeout, cancellationToken).ConfigureAwait(false);
        return body!;
    }

    private async Task<(string? Token, string? Cookie)> SecurityContextAsync(CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, "login-whoami.api");
        try
        {
            using var response = await SendRawAsync(request, "the sign-in", MetadataTimeout, cancellationToken).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
                return (null, null);

            var cookie = response.Headers.TryGetValues("Set-Cookie", out var values)
                ? string.Join("; ", values.Select(v => v.Split(';', 2)[0].Trim()).Where(v => v.Contains('=', StringComparison.Ordinal)))
                : null;
            using var document = JsonDocument.Parse(await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false));
            var token = Text(document.RootElement, "CSRF");
            return (token, string.IsNullOrEmpty(cookie) ? null : cookie);
        }
        catch (Exception ex) when (ex is PanoramaException or JsonException)
        {
            // Without a token the POST is refused, and that refusal says why.
            return (null, null);
        }
    }

    /// <summary>
    /// Reads the page's properties from Panorama's wiki editor (<c>LABKEY._wiki.setProps({...})</c>),
    /// which is where LabKey gives the version a save must name.
    /// </summary>
    public static WikiPageInfo ParseWikiEditor(string html)
    {
        // The block ends at the first "});" OUTSIDE a string: the page body is one of the strings, LabKey
        // escapes only what a string literal needs, and a report's text may well contain "});".
        var block = Regex.Match(html, @"LABKEY\._wiki\.setProps\(\{((?:'(?:[^'\\]|\\.)*'|""(?:[^""\\]|\\.)*""|[^'""])*?)\}\);",
            RegexOptions.Singleline);
        if (!block.Success)
            throw new PanoramaException("Panorama's wiki editor was not understood, so the page was left alone.");

        var props = block.Groups[1].Value;
        var entityId = JsString(props, "entityId");
        if (string.IsNullOrEmpty(entityId))
            throw new PanoramaException("Panorama's wiki editor did not say which page it is, so the page was left alone.");

        // The attachments come in a block of their own, one object per file, each with its name.
        var attachments = Regex.Match(html, @"LABKEY\._wiki\.setAttachments\(\[((?:'(?:[^'\\]|\\.)*'|""(?:[^""\\]|\\.)*""|[^'""])*?)\]\);",
            RegexOptions.Singleline) is { Success: true } list
            ? Regex.Matches(list.Groups[1].Value, @"(?m)^\s*name:\s*'((?:[^'\\]|\\.)*)'").Select(m => DecodeJsString(m.Groups[1].Value)).ToList()
            : null;

        return new WikiPageInfo(entityId, JsInt(props, "rowId"), JsInt(props, "pageVersionId"), JsString(props, "name") ?? "",
            JsString(props, "title") ?? "", JsString(props, "body") ?? "", JsInt(props, "parent"),
            JsBool(props, "showAttachments") ?? true, JsBool(props, "shouldIndex") ?? true, attachments);
    }

    /// <summary>A JavaScript string literal's text: <c>\x3C</c>, <c>é</c>, <c>\n</c>, <c>\'</c>, <c>\"</c> and <c>\\</c> undone.</summary>
    public static string DecodeJsString(string literal)
    {
        var text = new StringBuilder(literal.Length);
        for (var i = 0; i < literal.Length; i++)
        {
            var c = literal[i];
            if (c != '\\' || i + 1 >= literal.Length)
            {
                text.Append(c);
                continue;
            }

            var next = literal[++i];
            switch (next)
            {
                case 'n': text.Append('\n'); break;
                case 'r': text.Append('\r'); break;
                case 't': text.Append('\t'); break;
                case 'b': text.Append('\b'); break;
                case 'f': text.Append('\f'); break;
                case 'x' when i + 2 < literal.Length && IsHex(literal, i + 1, 2):
                    text.Append((char)int.Parse(literal.AsSpan(i + 1, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                    i += 2;
                    break;
                case 'u' when i + 4 < literal.Length && IsHex(literal, i + 1, 4):
                    text.Append((char)int.Parse(literal.AsSpan(i + 1, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                    i += 4;
                    break;
                default: text.Append(next); break;
            }
        }

        return text.ToString();

        static bool IsHex(string s, int start, int length) => s.AsSpan(start, length).ToString().All(Uri.IsHexDigit);
    }

    private static WikiPageInfo ParseSaved(string body, string name, string title, string html, WikiPageInfo? existing)
    {
        using var document = ParseJson(body, "the saved wiki page");
        var root = document.RootElement;
        if (root.TryGetProperty("success", out var ok) && ok.ValueKind == JsonValueKind.False)
            throw new PanoramaException("Panorama did not save the wiki page" + (ApiError(body) is { } why ? $": {why}" : "."));

        var props = root.TryGetProperty("wikiProps", out var p) ? p : root;
        return new WikiPageInfo(
            Text(props, "entityId") ?? existing?.EntityId ?? "", Int(props, "rowId") ?? existing?.RowId,
            Int(props, "pageVersionId"), name, title, html, existing?.Parent, existing?.ShowAttachments ?? true,
            existing?.ShouldIndex ?? true);
    }

    /// <summary>Reads <c>wiki-attachFiles.api</c>'s answer: <c>{"success", "warnings", "attachments": [{name, downloadUrl}]}</c>.</summary>
    public static IReadOnlyList<WikiAttachment> ParseAttachments(string body)
    {
        using var document = ParseJson(body, "the wiki attachments");
        var root = document.RootElement;
        if (root.TryGetProperty("success", out var ok) && ok.ValueKind == JsonValueKind.False)
            throw new PanoramaException("Panorama did not attach the files" + (ApiError(body) is { } why ? $": {why}" : "."));

        // A warning is how LabKey says a file was NOT attached (a name already there, for one), so
        // it is a failure here rather than something to show and carry on past.
        if (root.TryGetProperty("warnings", out var warnings) && warnings.ValueKind == JsonValueKind.Object)
        {
            var text = string.Join("; ", warnings.EnumerateObject()
                .Select(w => w.Value.ValueKind == JsonValueKind.String ? w.Value.GetString() : w.Value.ToString())
                .Where(t => !string.IsNullOrWhiteSpace(t)));
            if (text.Length > 0)
                throw new PanoramaException($"Panorama did not attach every file: {text}");
        }

        if (!root.TryGetProperty("attachments", out var list) || list.ValueKind != JsonValueKind.Array)
            return Array.Empty<WikiAttachment>();
        return list.EnumerateArray()
            .Select(a => new WikiAttachment(Text(a, "name") ?? "", Text(a, "downloadUrl")))
            .Where(a => a.Name.Length > 0)
            .ToList();
    }

    private static JsonDocument ParseJson(string body, string what)
    {
        try
        {
            return JsonDocument.Parse(body);
        }
        catch (JsonException ex)
        {
            throw new PanoramaException($"Panorama did not send {what}; the sign-in was probably not accepted.", ex)
            {
                IsSignInProblem = true,
            };
        }
    }

    private static string? Text(JsonElement row, string name) =>
        row.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static int? Int(JsonElement row, string name) =>
        row.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out var n) ? n : null;

    private static string? JsString(string props, string key) =>
        Regex.Match(props, $@"(?m)^\s*{key}:\s*'((?:[^'\\]|\\.)*)'", RegexOptions.Singleline) is { Success: true } m
            ? DecodeJsString(m.Groups[1].Value)
            : null;

    private static int? JsInt(string props, string key) =>
        Regex.Match(props, $@"(?m)^\s*{key}:\s*(-?\d+)") is { Success: true } m
            ? int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture)
            : null;

    private static bool? JsBool(string props, string key) =>
        Regex.Match(props, $@"(?m)^\s*{key}:\s*(true|false)") is { Success: true } m ? m.Groups[1].Value == "true" : null;
}
