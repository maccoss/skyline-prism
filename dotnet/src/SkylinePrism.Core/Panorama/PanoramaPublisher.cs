using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.Pipeline;

namespace SkylinePrism.Core.Panorama;

/// <summary>One line of the links page.</summary>
public sealed record PageLink(string Label, string Url, string? Note = null);

/// <summary>A wiki page as published.</summary>
public sealed record PublishedPage(string Folder, string PageName, string Url, int Images, bool Created);

/// <summary>An output directory as uploaded.</summary>
public sealed record UploadedDirectory(string Destination, string Url, int Uploaded, int Skipped, long BytesSent);

/// <summary>Where an upload has got to: the file in flight and the whole directory.</summary>
public sealed record UploadProgress(string File, int FileIndex, int FileCount, long FileSent, long FileBytes, long TotalSent, long TotalBytes);

/// <summary>
/// Publishes a PRISM output directory to Panorama: the QC and quant reports as wiki pages, and the
/// directory itself into a file area beside the raw files.
/// </summary>
/// <remarks>
/// <para><b>One page per report, updated in place.</b> Panorama keeps every version of a page, so
/// republishing replaces the page and loses nothing. A page carries a footer
/// (<see cref="FooterId"/>) with a fingerprint of what PRISM wrote. A page of that name that PRISM
/// did not write, or one edited on Panorama since PRISM published it, is refused unless the caller
/// says to replace it - an edit someone made by hand is never overwritten silently.</para>
/// <para><b>A page belongs to one output directory.</b> The footer also carries the publishing
/// directory's id (<see cref="PanoramaTargets.PublishId"/>), so a different output directory that
/// happens to have the same name - and so the same default page names - is refused rather than
/// allowed to replace another experiment's page.</para>
/// <para><b>An upload is verified, not assumed.</b> After a folder's files go up, Panorama's own MD5s
/// (computed over the bytes it stored) are compared with the local files. A mismatch is re-sent once,
/// then refused. The same MD5s let a republish skip a file already there.</para>
/// </remarks>
public sealed class PanoramaPublisher
{
    /// <summary>The footer's element id: what marks a page as PRISM's.</summary>
    public const string FooterId = "prism-wiki";

    /// <summary>Attachments per request: small enough that one failure costs little.</summary>
    private const int AttachmentBatch = 8;

    private readonly PanoramaClient _client;
    private readonly Action<string> _say;
    private readonly string? _sourceId;

    /// <param name="sourceId">
    /// The output directory's publish id, written into every page's footer. A page whose footer names
    /// another is refused unless replaced on purpose; null publishes without one and checks none.
    /// </param>
    public PanoramaPublisher(PanoramaClient client, Action<string>? say = null, string? sourceId = null)
    {
        _client = client;
        _say = say ?? (_ => { });
        _sourceId = sourceId;
    }

    // ---------------------------------------------------------------------------------------
    // Wiki pages
    // ---------------------------------------------------------------------------------------

    /// <summary>The QC report's default page name: <c>PRISM-QC-</c> and the output directory's name.</summary>
    public static string DefaultQcPage(string outputDir) => Bounded("PRISM-QC-" + Slug(DirectoryName(outputDir)));

    /// <summary>
    /// The quant report's default page name: the output directory and the contrast, since one output
    /// directory is analyzed with more than one contrast and each deserves its own page.
    /// </summary>
    public static string DefaultQuantPage(string outputDir, string? contrast) =>
        Bounded("PRISM-Quant-" + Slug(DirectoryName(outputDir)) + (string.IsNullOrWhiteSpace(contrast) ? "" : "-" + Slug(contrast!)));

    /// <summary>The contrast a quant report was produced for, from its <c>quant_parameters.json</c>; null when there is none.</summary>
    public static string? QuantContrast(string outputDir)
    {
        var path = Path.Combine(outputDir, "quant", "quant_parameters.json");
        if (!File.Exists(path))
            return null;
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(path));
            if (!document.RootElement.TryGetProperty("contrast", out var c))
                return null;
            string? Text(string name) => c.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
            IReadOnlyList<string>? List(string name) => c.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Array
                ? v.EnumerateArray().Select(x => x.GetString() ?? "").ToList()
                : null;
            return new QuantContrast(Text("group_by"), List("group_a"), List("group_b"), Text("trend_over")).Describe();
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>
    /// Publishes one report as a wiki page in <paramref name="folder"/>'s container.
    /// </summary>
    /// <param name="outputLink">The uploaded output directory, linked from the top of the page, when there is one.</param>
    /// <param name="replaceEdited">Replace a page edited on Panorama, or one PRISM did not write.</param>
    public async Task<PublishedPage> PublishReportAsync(
        string reportPath, string folder, string pageName, string title, string? outputLink,
        bool replaceEdited, CancellationToken cancellationToken = default)
    {
        if (!File.Exists(reportPath))
            throw new PanoramaException($"There is no report to publish at {reportPath}.");
        ValidatePageName(pageName);
        var container = PanoramaPaths.Container(folder);

        var document = WikiReport.FromReport(await File.ReadAllTextAsync(reportPath, cancellationToken).ConfigureAwait(false),
            pageName.ToLowerInvariant());
        if (WikiReport.Refusals(document.BodyTemplate) is { Count: > 0 } refusals)
            throw new PanoramaException("The report contains HTML Panorama would refuse: " + string.Join(" ", refusals));

        var existing = await _client.GetWikiPageAsync(container, pageName, cancellationToken).ConfigureAwait(false);
        var previousAttachments = existing is null ? Array.Empty<string>() : CheckOwnership(existing, container, pageName, replaceEdited);

        var created = existing is null;
        if (existing is null)
        {
            // A new page has no entity id to file attachments under until it is saved once. The stub
            // carries the footer, so a run that fails after this point leaves a page PRISM can retry.
            _say($"Creating the wiki page {pageName} in {container}");
            var stub = "<p>Publishing from Skyline-PRISM...</p>";
            await _client.SaveWikiPageAsync(container, pageName, title, stub + Footer.Html(stub, Array.Empty<string>(), _sourceId, ""),
                null, showAttachments: false, cancellationToken).ConfigureAwait(false);
            existing = await _client.GetWikiPageAsync(container, pageName, cancellationToken).ConfigureAwait(false)
                       ?? throw new PanoramaException($"Panorama did not keep the new wiki page {pageName} in {container}.");
        }

        // Attachments are named by their content (WikiReport.FromReport), so a plot that did not change
        // since the last publish is already attached under its name and is not sent again. Panorama does
        // not version attachments - they belong to the page, not to a version of it - so the previous
        // publish's other plots are removed only AFTER the new body is saved: the live page never points
        // at a plot that is gone, and an earlier version in the page's history never shows another run's
        // plot under its own name.
        var newNames = document.Images.Select(i => i.Name).ToList();
        var toSend = document.Images.Where(i => !previousAttachments.Contains(i.Name, StringComparer.Ordinal)).ToList();
        // A name on the page that the last publish did not record (one that failed before its save)
        // would be refused as taken, so any such leftover is cleared first.
        if (toSend.Count > 0)
            await _client.AttachFilesAsync(container, existing.EntityId, Array.Empty<(string, byte[])>(),
                toSend.Select(i => i.Name).ToList(), cancellationToken).ConfigureAwait(false);

        var urls = new Dictionary<string, string>(StringComparer.Ordinal);
        if (toSend.Count < document.Images.Count)
            _say($"{document.Images.Count - toSend.Count} of {document.Images.Count} plots are unchanged and already attached");
        for (var i = 0; i < toSend.Count; i += AttachmentBatch)
        {
            var batch = toSend.Skip(i).Take(AttachmentBatch).Select(img => (img.Name, img.Data)).ToList();
            _say(batch.Count == 1
                ? $"Attaching plot {i + 1} of {toSend.Count}"
                : $"Attaching plots {i + 1}-{i + batch.Count} of {toSend.Count}");
            var attached = await _client.AttachFilesAsync(container, existing.EntityId, batch, Array.Empty<string>(), cancellationToken)
                .ConfigureAwait(false);
            foreach (var a in attached.Where(a => a.DownloadUrl is not null))
                urls[a.Name] = a.DownloadUrl!;
        }

        string UrlFor(string name) => urls.TryGetValue(name, out var url)
            ? ServerRelative(url)
            : PanoramaPaths.Encode(PanoramaPaths.AsFolder(container)) + "wiki-download.view?entityId="
              + Uri.EscapeDataString(existing.EntityId) + "&name=" + Uri.EscapeDataString(name);

        var header = outputLink is null
            ? ""
            : $"<p style=\"margin: 0 0 12px\">Output directory on Panorama: <a href=\"{WebUtility.HtmlEncode(outputLink)}\">"
              + $"{WebUtility.HtmlEncode(PanoramaPaths.ToFolder(new Uri(outputLink).AbsolutePath))}</a></p>";
        var content = header + document.Render(n => WebUtility.HtmlEncode(UrlFor(n)));
        var body = content + Footer.Html(content, newNames, _sourceId, $"Published by Skyline-PRISM {PrismVersion.Current} on "
            + DateTime.Now.ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture)
            + ". Republishing from PRISM replaces this page; Panorama keeps the earlier versions of its text.");

        var current = await _client.GetWikiPageAsync(container, pageName, cancellationToken).ConfigureAwait(false) ?? existing;
        _say($"Saving the wiki page {pageName}");
        await _client.SaveWikiPageAsync(container, pageName, title, body, current, showAttachments: false, cancellationToken)
            .ConfigureAwait(false);

        var unused = previousAttachments.Except(newNames, StringComparer.Ordinal).ToList();
        if (unused.Count > 0)
        {
            try
            {
                await _client.AttachFilesAsync(container, existing.EntityId, Array.Empty<(string, byte[])>(), unused, cancellationToken)
                    .ConfigureAwait(false);
            }
            catch (PanoramaException ex)
            {
                // The page is published and complete; only the cleanup of plots it no longer shows failed.
                _say($"The page is published, but {unused.Count} earlier plots it no longer uses could not be removed ({ex.Message})");
            }
        }

        return new PublishedPage(container, pageName, PanoramaPaths.WikiPageUrl(container, pageName, _client.Server),
            document.Images.Count, created);
    }

    /// <summary>
    /// Refuses a page PRISM must not overwrite - one it did not write, one published from another output
    /// directory, or one edited on Panorama since - unless <paramref name="replaceEdited"/>; otherwise
    /// returns the attachments its last publish left.
    /// </summary>
    private string[] CheckOwnership(WikiPageInfo existing, string container, string pageName, bool replaceEdited)
    {
        var footer = Footer.Read(existing.Body);
        if (footer is null && !replaceEdited)
            throw new PanoramaException(
                $"{container} already has a wiki page named {pageName} that PRISM did not write. Choose another "
                + "page name, or replace it on purpose (--replace-edited in the CLI).");
        if (footer?.Source is { } source && _sourceId is not null && source != _sourceId && !replaceEdited)
            throw new PanoramaException(
                $"The wiki page {pageName} in {container} was published by PRISM from a different output directory - "
                + "one with the same name, from another experiment or analysis. Choose another page name, or replace it "
                + "on purpose (--replace-edited in the CLI).");
        if (footer is not null && footer.EditedSincePublished(existing.Body) && !replaceEdited)
            throw new PanoramaException(
                $"The wiki page {pageName} in {container} was edited on Panorama after PRISM published it. "
                + "Republishing would overwrite that edit; copy it somewhere first, or replace it on purpose "
                + "(--replace-edited in the CLI). Panorama keeps the edited version in the page's history either way.");
        return footer?.Attachments ?? Array.Empty<string>();
    }

    // ---------------------------------------------------------------------------------------
    // The links page, and the web part that shows it
    // ---------------------------------------------------------------------------------------

    /// <summary>The links page's default name: <c>PRISM-</c> and the output directory's name.</summary>
    public static string DefaultLinksPage(string outputDir) => Bounded("PRISM-" + Slug(DirectoryName(outputDir)));

    /// <summary>
    /// Publishes the small page that links an output directory's published results - its QC page, each
    /// quant page, the uploaded directory - so the folder's own page can show them all in one web part.
    /// </summary>
    public async Task<PublishedPage> PublishLinksPageAsync(
        string folder, string pageName, string title, string outputDirName, IReadOnlyList<PageLink> links,
        bool replaceEdited, CancellationToken cancellationToken = default)
    {
        ValidatePageName(pageName);
        var container = PanoramaPaths.Container(folder);
        var existing = await _client.GetWikiPageAsync(container, pageName, cancellationToken).ConfigureAwait(false);
        if (existing is not null)
            CheckOwnership(existing, container, pageName, replaceEdited);

        var items = string.Concat(links.Select(l =>
            $"<li style=\"margin: 2px 0\"><a href=\"{WebUtility.HtmlEncode(l.Url)}\">{WebUtility.HtmlEncode(l.Label)}</a>"
            + (l.Note is null ? "" : $" <span style=\"color: #666\">{WebUtility.HtmlEncode(l.Note)}</span>") + "</li>"));
        var content = $"<p style=\"margin: 0 0 6px\">Skyline-PRISM results for <strong>{WebUtility.HtmlEncode(outputDirName)}</strong>:</p>"
                      + $"<ul style=\"margin: 0; padding-left: 20px\">{items}</ul>";
        var body = content + Footer.Html(content, Array.Empty<string>(), _sourceId, $"Updated by Skyline-PRISM {PrismVersion.Current} on "
            + DateTime.Now.ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture) + ".");

        _say($"Saving the links page {pageName}");
        await _client.SaveWikiPageAsync(container, pageName, title, body, existing, showAttachments: false, cancellationToken)
            .ConfigureAwait(false);
        return new PublishedPage(container, pageName, PanoramaPaths.WikiPageUrl(container, pageName, _client.Server), 0, existing is null);
    }

    /// <summary>
    /// Makes the folder's own page show <paramref name="pageName"/> in a Wiki web part: the one PRISM
    /// added before (<paramref name="knownWebPartId"/>) if it is still there, otherwise a new one placed
    /// by <see cref="PositionFor"/>. Needs folder administrator permission.
    /// </summary>
    /// <remarks>
    /// LabKey's web part listing does not say which page a Wiki part shows, so PRISM recognizes its own
    /// part by the id it recorded, not by looking. A part it added before is left where someone may have
    /// moved it; only a new one is placed.
    /// </remarks>
    /// <param name="added">
    /// Told a new part's id as soon as Panorama has it, before it is set up and moved: a failure after
    /// that point must still leave the id recorded, or the next publish would add another part.
    /// </param>
    /// <returns>The web part's id, and whether it was added this time.</returns>
    public async Task<(int WebPartId, bool Added)> ShowOnFolderPageAsync(
        string folder, string pageName, int? knownWebPartId, Action<int>? added = null,
        CancellationToken cancellationToken = default)
    {
        var container = PanoramaPaths.Container(folder);
        var pageId = await _client.MainPageIdAsync(container, cancellationToken).ConfigureAwait(false);
        var parts = await _client.WebPartsAsync(container, pageId, cancellationToken).ConfigureAwait(false);
        var properties = new Dictionary<string, string>
        {
            ["name"] = pageName,
            ["webPartContainer"] = await _client.ContainerIdAsync(container, cancellationToken).ConfigureAwait(false),
        };

        if (knownWebPartId is { } known && parts.Any(p => p.WebPartId == known && p.Name == WikiWebPart))
        {
            await _client.CustomizeWebPartAsync(container, known, properties, cancellationToken).ConfigureAwait(false);
            return (known, false);
        }

        _say($"Adding the links to {container}'s page");
        var id = await _client.AddWebPartAsync(container, pageId, WikiWebPart, "body", cancellationToken).ConfigureAwait(false);
        added?.Invoke(id);
        await _client.CustomizeWebPartAsync(container, id, properties, cancellationToken).ConfigureAwait(false);

        // A new part arrives at the bottom of the body column; each move up passes one part.
        var body = (await _client.WebPartsAsync(container, pageId, cancellationToken).ConfigureAwait(false))
            .Where(p => p.Location == "body").OrderBy(p => p.Index).ToList();
        var at = body.FindIndex(p => p.WebPartId == id);
        for (var moves = at - PositionFor(body.Where(p => p.WebPartId != id).ToList()); moves > 0; moves--)
            await _client.MoveWebPartUpAsync(container, id, cancellationToken).ConfigureAwait(false);
        return (id, true);
    }

    /// <summary>The web part that lists a Panorama folder's Skyline documents.</summary>
    public const string RunsWebPart = "Targeted MS Runs";

    /// <summary>The web part that lists a folder's files.</summary>
    public const string FilesWebPart = "Files";

    /// <summary>
    /// Where in the body column, counted from the top of <paramref name="body"/>, the links go: right
    /// after the folder's <see cref="RunsWebPart"/>, so the results sit with the documents they came
    /// from (and so above a Files part that follows it); without one, right above
    /// <see cref="FilesWebPart"/>; without either, at the top.
    /// </summary>
    internal static int PositionFor(IReadOnlyList<WebPart> body)
    {
        int Find(string name) => body.Select((p, i) => (p, i))
            .FirstOrDefault(x => string.Equals(x.p.Name, name, StringComparison.OrdinalIgnoreCase), (null!, -1)).i;

        return Find(RunsWebPart) is var runs and >= 0 ? runs + 1
            : Find(FilesWebPart) is var files and >= 0 ? files
            : 0;
    }

    /// <summary>LabKey's name for the web part that shows one wiki page (<c>WikiModule.WEB_PART_NAME</c>).</summary>
    public const string WikiWebPart = "Wiki";

    /// <summary>Panorama gives absolute download addresses; a page links them server-relative, so a renamed host still works.</summary>
    private static string ServerRelative(string url) =>
        Uri.TryCreate(url, UriKind.Absolute, out var u) ? u.PathAndQuery : url;

    /// <summary>The longest page name PRISM writes.</summary>
    private const int MaxPageName = 200;

    /// <summary>
    /// A default name cut to <see cref="MaxPageName"/>, ending in a hash of the whole so two long names
    /// that share their first 191 characters still name different pages.
    /// </summary>
    private static string Bounded(string name) =>
        name.Length <= MaxPageName
            ? name
            : name[..(MaxPageName - 9)].TrimEnd('-') + "-"
              + Convert.ToHexStringLower(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(name)))[..8];

    private static void ValidatePageName(string name)
    {
        if (string.IsNullOrWhiteSpace(name) || !Regex.IsMatch(name, @"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,199}$"))
            throw new PanoramaException(
                $"'{name}' is not a usable wiki page name: use letters, digits, '-', '_' and '.', starting with a letter or digit.");
    }

    /// <summary>The footer PRISM writes at the bottom of each page it owns.</summary>
    /// <param name="Source">The publishing output directory's id; null on a page written before PRISM recorded one.</param>
    private sealed record Footer(string Fingerprint, string[] Attachments, string? Source)
    {
        private const string Open = "<div id=\"" + FooterId + "\"";

        public static string Html(string content, IReadOnlyList<string> attachments, string? source, string text) =>
            $"{Open} data-body=\"{WikiReport.Fingerprint(content)}\" data-attachments=\"{WebUtility.HtmlEncode(string.Join(",", attachments))}\" "
            + (source is null ? "" : $"data-source=\"{WebUtility.HtmlEncode(source)}\" ")
            + $"style=\"color: #888; font-size: 12px; margin-top: 24px\">{WebUtility.HtmlEncode(text)}</div>";

        public static Footer? Read(string body)
        {
            var m = Regex.Match(body, Regex.Escape(Open)
                                      + @"[^>]*data-body=""([0-9a-f]+)""[^>]*data-attachments=""([^""]*)""(?:[^>]*data-source=""([^""]*)"")?");
            return m.Success
                ? new Footer(m.Groups[1].Value, WebUtility.HtmlDecode(m.Groups[2].Value)
                        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries),
                    m.Groups[3].Success ? WebUtility.HtmlDecode(m.Groups[3].Value) : null)
                : null;
        }

        /// <summary>Whether the text above the footer is still what PRISM wrote.</summary>
        public bool EditedSincePublished(string body)
        {
            var at = body.IndexOf(Open, StringComparison.Ordinal);
            return at < 0 || WikiReport.Fingerprint(body[..at]) != Fingerprint;
        }
    }

    // ---------------------------------------------------------------------------------------
    // Output directory
    // ---------------------------------------------------------------------------------------

    /// <summary>
    /// Where an output directory goes when it is uploaded beside the raw files: a sibling of the raw
    /// files folder, named like the output directory. Raw files kept directly in a file root
    /// (<c>@files</c>) put it inside that root instead, since a file root has no sibling to share.
    /// </summary>
    public static string DestinationBeside(string rawFolder, string outputDir)
    {
        var raw = PanoramaPaths.ToFolder(rawFolder);
        if (!PanoramaPaths.IsFileArea(raw))
            throw new PanoramaException(
                $"{raw} is not a file area. Choose the folder that holds the raw files (inside @files).");
        var parent = PanoramaPaths.Name(raw).StartsWith('@') ? raw : PanoramaPaths.Parent(raw);
        return PanoramaPaths.Combine(parent, DirectoryName(outputDir));
    }

    /// <summary>Uploads every file under <paramref name="localDir"/> into <paramref name="destination"/>, keeping its layout.</summary>
    public async Task<UploadedDirectory> UploadDirectoryAsync(
        string localDir, string destination, IProgress<UploadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var root = Path.GetFullPath(localDir);
        if (!Directory.Exists(root))
            throw new PanoramaException($"There is no output directory at {root}.");
        if (!PanoramaPaths.IsFileArea(destination) || PanoramaPaths.Name(destination).StartsWith('@'))
            throw new PanoramaException($"{destination} is not a folder inside a file area (@files).");

        var files = Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
            .Select(f => new FileInfo(f))
            .OrderBy(f => f.FullName, StringComparer.Ordinal)
            .ToList();
        foreach (var f in files.Where(f => f.Name.Contains(';')))
            throw new PanoramaException($"{f.Name} contains a ';', at which Panorama silently truncates a name. Rename it first.");

        var total = files.Sum(f => f.Length);
        long done = 0;
        int uploaded = 0, skipped = 0, index = 0;
        long sent = 0;
        var temporary = new List<string>();

        try
        {
            foreach (var group in files.GroupBy(f => Path.GetRelativePath(root, f.DirectoryName!)))
            {
                var folder = group.Key == "." ? destination : PanoramaPaths.Combine(destination, group.Key.Replace('\\', '/'));
                var remote = await _client.Md5sAsync(folder, cancellationToken).ConfigureAwait(false);
                var pending = new List<(Source Source, string Md5, long DoneBefore, long LocalLength)>();
                foreach (var file in group)
                {
                    index++;
                    var source = await SourceForAsync(file, temporary, cancellationToken).ConfigureAwait(false);
                    var md5 = await LocalMd5Async(source.Path, cancellationToken).ConfigureAwait(false);
                    if (remote.TryGetValue(file.Name, out var there) && there == md5)
                    {
                        skipped++;
                        done += file.Length;
                        progress?.Report(new UploadProgress(file.Name, index, files.Count, file.Length, file.Length, done, total));
                        continue;
                    }

                    await UploadOneAsync(source, file.Length, folder, index, files.Count, done, total, progress, cancellationToken)
                        .ConfigureAwait(false);
                    pending.Add((source, md5, done, file.Length));
                    done += file.Length;
                    sent += source.Length;
                    uploaded++;
                }

                if (pending.Count == 0)
                    continue;

                // Verify against what the server stored, and re-send a mismatch once.
                var stored = await _client.Md5sAsync(folder, cancellationToken).ConfigureAwait(false);
                foreach (var (source, md5, doneBefore, localLength) in pending.Where(p => !stored.TryGetValue(p.Source.Name, out var s) || s != p.Md5).ToList())
                {
                    _say($"{source.Name} did not arrive intact; sending it again");
                    await UploadOneAsync(source, localLength, folder, index, files.Count, doneBefore, total, progress, cancellationToken)
                        .ConfigureAwait(false);
                    sent += source.Length;
                    var again = await _client.Md5sAsync(folder, cancellationToken).ConfigureAwait(false);
                    if (!again.TryGetValue(source.Name, out var s) || s != md5)
                        throw new PanoramaException(
                            $"{source.Name} was uploaded twice to {folder} and Panorama's copy still does not match the local file.");
                }
            }
        }
        finally
        {
            foreach (var path in temporary)
                TryDelete(path);
        }

        return new UploadedDirectory(destination, PanoramaPaths.BrowserUrl(destination, _client.Server), uploaded, skipped, sent);
    }

    /// <summary>What is actually sent for a local file: the file itself, or its Panorama-safe form.</summary>
    private sealed record Source(string Name, string Path, long Length);

    /// <summary>
    /// The bytes to send for a file. An HTML file Panorama would refuse (<see cref="WikiReport.RefusedAsFile"/>:
    /// PRISM's own reports, for their style block and doctype) is sent as
    /// <see cref="WikiReport.ToPanoramaFile"/> under the same name - the same page, styles inlined.
    /// Every other file goes as it is.
    /// </summary>
    private async Task<Source> SourceForAsync(FileInfo file, List<string> temporary, CancellationToken cancellationToken)
    {
        var html = file.Extension.Equals(".html", StringComparison.OrdinalIgnoreCase)
                   || file.Extension.Equals(".htm", StringComparison.OrdinalIgnoreCase);
        if (!html || file.Length > 256L << 20)
            return new Source(file.Name, file.FullName, file.Length);

        var text = await File.ReadAllTextAsync(file.FullName, cancellationToken).ConfigureAwait(false);
        if (!WikiReport.RefusedAsFile(text))
            return new Source(file.Name, file.FullName, file.Length);

        var path = Path.Combine(Path.GetTempPath(), $"prism-publish-{Guid.NewGuid():N}.html");
        await File.WriteAllTextAsync(path, WikiReport.ToPanoramaFile(text), cancellationToken).ConfigureAwait(false);
        temporary.Add(path);
        _say($"{file.Name}: sent with its styles inlined and no doctype, which Panorama requires of an HTML file");
        return new Source(file.Name, path, new FileInfo(path).Length);
    }

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (IOException)
        {
            // A temporary file; the system cleans its folder eventually.
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    /// <param name="localLength">
    /// The local file's size, which is what <paramref name="total"/> counts. A report sent in its
    /// Panorama form is a different size, so its bytes are scaled to the local file's share of the
    /// total; reported raw, the running total would step backwards after it.
    /// </param>
    private async Task UploadOneAsync(
        Source source, long localLength, string folder, int index, int count, long doneBefore, long total,
        IProgress<UploadProgress>? progress, CancellationToken cancellationToken)
    {
        _say($"Uploading {source.Name} ({FormatBytes(source.Length)}) to {folder}");
        // Relayed on the reporting thread: a Progress<T> here would post each report to the thread pool
        // (there is no synchronization context below the GUI's Task.Run, or in the CLI), so the caller's
        // reports would arrive out of order and concurrently.
        var perFile = progress is null
            ? null
            : new Relay(b => progress.Report(new UploadProgress(source.Name, index, count, b, source.Length,
                doneBefore + (source.Length == 0 ? 0 : (long)(b * (double)localLength / source.Length)), total)));
        await _client.UploadFileAsync(source.Path, folder, source.Name, perFile, cancellationToken).ConfigureAwait(false);
    }

    private sealed class Relay(Action<long> report) : IProgress<long>
    {
        public void Report(long value) => report(value);
    }

    private static async Task<string> LocalMd5Async(string path, CancellationToken cancellationToken)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete,
            1 << 20, FileOptions.Asynchronous | FileOptions.SequentialScan);
        var hash = await MD5.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);
        return Convert.ToHexStringLower(hash);
    }

    public static string FormatBytes(long bytes) => bytes switch
    {
        >= 1L << 30 => $"{bytes / (double)(1L << 30):0.0} GB",
        >= 1L << 20 => $"{bytes / (double)(1L << 20):0.0} MB",
        >= 1L << 10 => $"{bytes / 1024.0:0} KB",
        _ => $"{bytes} B",
    };

    private static string DirectoryName(string outputDir) =>
        new DirectoryInfo(Path.GetFullPath(outputDir).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)).Name;

    /// <summary>A page-name-safe form: letters, digits and '-', runs of anything else as one '-'.</summary>
    public static string Slug(string text)
    {
        var slug = Regex.Replace(text, "[^A-Za-z0-9]+", "-").Trim('-');
        return slug.Length > 120 ? slug[..120].TrimEnd('-') : slug;
    }
}
