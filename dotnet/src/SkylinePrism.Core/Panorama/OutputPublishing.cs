using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SkylinePrism.Core.Panorama;

/// <summary>What to publish from one output directory, and where.</summary>
/// <param name="QcFolder">Panorama folder for the QC report's wiki page; null to skip it.</param>
/// <param name="QuantFolder">Panorama folder for the quant report's wiki page; null to skip it.</param>
/// <param name="RawFolder">The folder holding the raw files; the output directory is uploaded beside it. Null to skip the upload.</param>
/// <param name="QcPage">Page name; null for <see cref="PanoramaPublisher.DefaultQcPage"/>.</param>
/// <param name="QuantPage">Page name; null for <see cref="PanoramaPublisher.DefaultQuantPage"/>.</param>
/// <param name="LinksFolder">
/// Panorama folder for the links page - links to this output directory's QC page, quant pages and
/// uploaded files - which that folder's own page then shows in a Wiki web part. Null to skip it.
/// </param>
public sealed record PublishRequest(
    string OutputDir,
    Uri Server,
    string? QcFolder,
    string? QuantFolder,
    string? RawFolder,
    string? QcPage = null,
    string? QuantPage = null,
    bool ReplaceEdited = false,
    string? LinksFolder = null,
    string? LinksPage = null)
{
    public string QcReport => Path.Combine(OutputDir, "qc_report.html");

    public string QuantReport => Path.Combine(OutputDir, "quant", "quant_report.html");

    public string ResolvedQcPage => QcPage ?? PanoramaPublisher.DefaultQcPage(OutputDir);

    public string ResolvedQuantPage => QuantPage ?? PanoramaPublisher.DefaultQuantPage(OutputDir, PanoramaPublisher.QuantContrast(OutputDir));

    public string? Destination => RawFolder is null ? null : PanoramaPublisher.DestinationBeside(RawFolder, OutputDir);

    public string ResolvedLinksPage => LinksPage ?? PanoramaPublisher.DefaultLinksPage(OutputDir);

    public bool HasWork => QcFolder is not null || QuantFolder is not null || RawFolder is not null || LinksFolder is not null;

    /// <summary>
    /// The targets to remember after this publish: this run's, with what was remembered kept for any step
    /// skipped this time - skipping a step for one publish does not forget where it goes.
    /// </summary>
    public PanoramaTargets ToTargets(PanoramaTargets remembered) => remembered with
    {
        Server = Server.GetLeftPart(UriPartial.Authority),
        QcWiki = QcFolder is null ? remembered.QcWiki : new WikiTarget(QcFolder, ResolvedQcPage),
        QuantWiki = QuantFolder is null ? remembered.QuantWiki : new WikiTarget(QuantFolder, ResolvedQuantPage),
        RawFolder = RawFolder ?? remembered.RawFolder,
        Destination = RawFolder is null ? remembered.Destination : Destination,
        LinksWiki = LinksFolder is null ? remembered.LinksWiki : new WikiTarget(LinksFolder, ResolvedLinksPage),
    };
}

/// <summary>What a publish produced.</summary>
/// <param name="LinksNote">Why the links could not be put on the folder's page, when they could not.</param>
public sealed record PublishOutcome(
    UploadedDirectory? Upload, PublishedPage? Qc, PublishedPage? Quant,
    PublishedPage? Links = null, bool LinksOnFolderPage = false, string? LinksNote = null);

/// <summary>
/// One publish of an output directory, shared by <c>prism publish</c> and the Skyline tool's
/// Publish to Panorama window so the two cannot drift.
/// </summary>
public static class OutputPublishing
{
    /// <summary>Why the request cannot run as given, as sentences; empty when it can.</summary>
    public static IReadOnlyList<string> Problems(PublishRequest request)
    {
        var problems = new List<string>();
        if (!Directory.Exists(request.OutputDir))
            problems.Add($"There is no output directory at {request.OutputDir}.");
        if (!request.HasWork)
            problems.Add("Nothing to publish: choose a wiki folder for the QC or quant report, the raw files folder to upload beside, or a folder for the links page.");
        if (request.QcFolder is not null && !File.Exists(request.QcReport))
            problems.Add($"There is no QC report to publish ({request.QcReport}); run 'prism qc' first.");
        if (request.QuantFolder is not null && !File.Exists(request.QuantReport))
            problems.Add($"There is no quant report to publish ({request.QuantReport}); make one with 'prism differential --report' "
                         + "or the Differential pane's Quant report button.");
        if (request.RawFolder is not null && !PanoramaPaths.IsFileArea(request.RawFolder))
            problems.Add($"{request.RawFolder} is not a file area. Choose the folder that holds the raw files (inside @files).");
        if (request.RawFolder is not null && PanoramaPaths.IsFileArea(request.RawFolder)
            && string.Equals(PanoramaPaths.ToFolder(request.RawFolder).TrimEnd('/'), request.Destination, StringComparison.OrdinalIgnoreCase))
            problems.Add("The output directory would be uploaded over the raw files folder; it has the same name.");
        // Checked here, before anything is sent: found inside the publish, a bad name would fail only
        // after the whole output directory had been uploaded.
        foreach (var (folder, page) in new[]
                 {
                     (request.QcFolder, request.ResolvedQcPage),
                     (request.QuantFolder, request.ResolvedQuantPage),
                     (request.LinksFolder, request.ResolvedLinksPage),
                 })
            if (folder is not null && PanoramaPublisher.PageNameProblem(page) is { } problem)
                problems.Add(problem);
        return problems;
    }

    /// <summary>The plan in words, for a dry run and for the window's confirmation.</summary>
    public static string Describe(PublishRequest request)
    {
        var sb = new StringBuilder();
        if (request.RawFolder is not null)
        {
            var files = Directory.Exists(request.OutputDir)
                ? Directory.EnumerateFiles(request.OutputDir, "*", SearchOption.AllDirectories).Select(f => new FileInfo(f)).ToList()
                : new List<FileInfo>();
            // panorama.json is written just before the upload, so it is one of the files sent.
            var count = files.Count + (File.Exists(Path.Combine(request.OutputDir, PanoramaTargets.FileName)) ? 0 : 1);
            sb.AppendLine($"Upload {count} files ({PanoramaPublisher.FormatBytes(files.Sum(f => f.Length))}) to {request.Destination}, "
                          + "skipping any already there unchanged");
        }
        if (request.QcFolder is not null)
            sb.AppendLine($"Publish the QC report as the wiki page {request.ResolvedQcPage} in {PanoramaPaths.Container(request.QcFolder)}");
        if (request.QuantFolder is not null)
            sb.AppendLine($"Publish the quant report as the wiki page {request.ResolvedQuantPage} in {PanoramaPaths.Container(request.QuantFolder)}");
        if (request.LinksFolder is not null)
            sb.AppendLine($"Publish the links page {request.ResolvedLinksPage} in {PanoramaPaths.Container(request.LinksFolder)}, "
                          + "and show it on that folder's page");
        return sb.ToString().TrimEnd();
    }

    /// <summary>
    /// A client signed in with the first saved sign-in Panorama accepts. Throws, naming each one tried
    /// and why it failed, when none works.
    /// </summary>
    public static async Task<PanoramaClient> SignInAsync(
        PanoramaSignIn signIn, CancellationToken cancellationToken = default, System.Net.Http.HttpMessageHandler? handler = null)
    {
        var tried = new List<string>();
        foreach (var candidate in signIn.Candidates())
        {
            var client = new PanoramaClient(signIn.Server, candidate, handler);
            var problem = await client.CheckSignInAsync(cancellationToken).ConfigureAwait(false);
            if (problem is null)
                return client;
            client.Dispose();
            if (!problem.IsSignInProblem)
                throw problem;
            tried.Add($"the sign-in from {candidate.Source} ({candidate}) was not accepted");
        }

        throw new PanoramaException(
            (tried.Count == 0
                ? "No Panorama sign-in is saved on this computer. "
                : "No saved Panorama sign-in works: " + string.Join("; ", tried) + ". ")
            + $"Sign in once from the Skyline tool's Publish to Panorama window, or in PanoramaBridge, or set {PanoramaSignIn.ApiKeyVariable} "
            + "to an API key (generate one on Panorama under your name, External Tool Access).")
        {
            IsSignInProblem = true,
        };
    }

    /// <summary>
    /// Runs a publish: the upload first, so both pages can link to the uploaded output directory,
    /// then the QC page, then the quant page.
    /// </summary>
    public static async Task<PublishOutcome> RunAsync(
        PanoramaClient client, PublishRequest request, Action<string>? say = null,
        IProgress<UploadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        if (Problems(request) is { Count: > 0 } problems)
            throw new PanoramaException(string.Join(" ", problems));
        // The request names the server the targets are recorded for; the client is the one the publish
        // goes to. Different, the pages would go to one server while panorama.json remembered the other.
        if (!string.Equals(client.Server.GetLeftPart(UriPartial.Authority), request.Server.GetLeftPart(UriPartial.Authority),
                StringComparison.OrdinalIgnoreCase))
            throw new PanoramaException(
                $"Signed in to {client.Server.GetLeftPart(UriPartial.Authority)}, but asked to publish to "
                + $"{request.Server.GetLeftPart(UriPartial.Authority)}. Sign in to that server first.");

        var targets = request.ToTargets(PanoramaTargets.Load(request.OutputDir));
        targets = targets with { PublishId = targets.PublishId ?? Guid.NewGuid().ToString("N") };
        // Saved before anything is sent: the id goes into every page's footer, and a page published under
        // an id that was then lost would be refused as another output directory's. It also means the
        // uploaded copy records where it was published.
        targets.Save(request.OutputDir);
        var publisher = new PanoramaPublisher(client, say, targets.PublishId);

        UploadedDirectory? upload = null;
        if (request.Destination is { } destination)
            upload = await publisher.UploadDirectoryAsync(request.OutputDir, destination, progress, cancellationToken).ConfigureAwait(false);

        var dirName = new DirectoryInfo(request.OutputDir).Name;
        PublishedPage? qc = null, quant = null;
        if (request.QcFolder is not null)
            qc = await publisher.PublishReportAsync(request.QcReport, request.QcFolder, request.ResolvedQcPage,
                $"PRISM QC: {dirName}", upload?.Url, request.ReplaceEdited, cancellationToken).ConfigureAwait(false);
        if (request.QuantFolder is not null)
        {
            var contrast = PanoramaPublisher.QuantContrast(request.OutputDir);
            quant = await publisher.PublishReportAsync(request.QuantReport, request.QuantFolder, request.ResolvedQuantPage,
                $"PRISM quant: {contrast ?? "quantification"} ({dirName})",
                upload is null ? null : upload.Url.TrimEnd('/') + "/quant/", request.ReplaceEdited, cancellationToken).ConfigureAwait(false);
        }

        if (quant is not null)
        {
            var contrast = PanoramaPublisher.QuantContrast(request.OutputDir);
            targets = targets with
            {
                QuantPages = (targets.QuantPages ?? Array.Empty<QuantPage>())
                    .Where(p => !(p.Folder == quant.Folder && p.Page == quant.PageName))
                    .Append(new QuantPage(quant.Folder, quant.PageName, contrast))
                    .ToList(),
            };
        }

        // What was published is recorded before the links step, which can fail on its own - a links
        // page edited on Panorama, a folder page this account may not change - and must not take the
        // record of a quant page that now exists down with it.
        targets.Save(request.OutputDir);

        PublishedPage? links = null;
        var onFolderPage = false;
        string? note = null;
        if (request.LinksFolder is not null)
        {
            try
            {
                var pageLinks = await LinksAsync(client, targets, qc, upload, cancellationToken).ConfigureAwait(false);
                links = await publisher.PublishLinksPageAsync(request.LinksFolder, request.ResolvedLinksPage,
                    $"PRISM: {dirName}", dirName, pageLinks, request.ReplaceEdited, cancellationToken).ConfigureAwait(false);
            }
            catch (PanoramaException ex)
            {
                note = $"The links page {request.ResolvedLinksPage} was not published ({ex.Message}); everything else was.";
                say?.Invoke(note);
            }

            if (links is not null)
            {
                try
                {
                    var (webPartId, _) = await publisher.ShowOnFolderPageAsync(request.LinksFolder, request.ResolvedLinksPage,
                        targets.LinksWebPartId,
                        added: id =>
                        {
                            // Recorded the moment the part exists, so a failure setting it up does not
                            // leave a part the next publish would not recognize and add again.
                            targets = targets with { LinksWebPartId = id };
                            targets.Save(request.OutputDir);
                        },
                        cancellationToken).ConfigureAwait(false);
                    targets = targets with { LinksWebPartId = webPartId };
                    onFolderPage = true;
                }
                catch (PanoramaException ex)
                {
                    // The pages are published either way; only their place on the folder's page is missing,
                    // and a folder administrator can add it by hand.
                    note = $"The links page was published, but could not be shown on {links.Folder}'s page ({ex.Message}). "
                           + $"A folder administrator can add a Wiki web part there showing the page {links.PageName}.";
                    say?.Invoke(note);
                }
            }
        }

        targets.Save(request.OutputDir);
        if (upload is not null)
        {
            // The publish changes panorama.json after uploading it - the quant pages, the web part's id - and
            // the uploaded copy is what a directory restored from Panorama remembers: left stale, a republish
            // from it would leave quant pages off the links page and add a second web part.
            try
            {
                await publisher.RefreshUploadedFileAsync(Path.Combine(request.OutputDir, PanoramaTargets.FileName), upload.Destination,
                    cancellationToken).ConfigureAwait(false);
            }
            catch (PanoramaException ex)
            {
                say?.Invoke($"Everything is published, but the uploaded {PanoramaTargets.FileName} could not be brought up to date ({ex.Message}).");
            }
        }

        return new PublishOutcome(upload, qc, quant, links, onFolderPage, note);
    }

    /// <summary>
    /// What the links page lists: the QC page, every quant page published from this output directory that
    /// is still on Panorama, and the uploaded output directory - this publish's, or the last one's.
    /// </summary>
    private static async Task<IReadOnlyList<PageLink>> LinksAsync(
        PanoramaClient client, PanoramaTargets targets, PublishedPage? qc, UploadedDirectory? upload,
        CancellationToken cancellationToken)
    {
        static string Relative(string url) => Uri.TryCreate(url, UriKind.Absolute, out var u) ? u.PathAndQuery : url;
        async Task<bool> Exists(string folder, string page) =>
            await client.GetWikiPageAsync(PanoramaPaths.Container(folder), page, cancellationToken).ConfigureAwait(false) is not null;

        var links = new List<PageLink>();
        if (qc is not null)
            links.Add(new PageLink("QC report", Relative(qc.Url)));
        else if (targets.QcWiki is { } remembered && await Exists(remembered.Folder, remembered.Page).ConfigureAwait(false))
            links.Add(new PageLink("QC report", Relative(PanoramaPaths.WikiPageUrl(remembered.Folder, remembered.Page, client.Server))));

        foreach (var page in targets.QuantPages ?? Array.Empty<QuantPage>())
            if (await Exists(page.Folder, page.Page).ConfigureAwait(false))
                links.Add(new PageLink("Quant report" + (page.Contrast is null ? "" : $": {page.Contrast}"),
                    Relative(PanoramaPaths.WikiPageUrl(page.Folder, page.Page, client.Server))));

        var output = upload?.Url ?? (targets.Destination is { } d ? PanoramaPaths.BrowserUrl(d, client.Server) : null);
        if (output is not null)
            links.Add(new PageLink("Output directory", Relative(output), "(every file this analysis produced)"));
        return links;
    }

    /// <summary>The <c>prism publish</c> line that does the same thing, for the window's Show Command Line.</summary>
    /// <remarks>
    /// A step the request skips is written as its <c>--no-</c> flag, never just left out: the CLI fills an
    /// absent folder from <c>panorama.json</c>, and the links page from the QC folder, so an omitted
    /// step would come back.
    /// </remarks>
    public static string CommandLine(PublishRequest request)
    {
        static string Q(string s) => s.Any(c => char.IsWhiteSpace(c) || c is '"' or '\'' or '&' or '|' or ';') ? "\"" + s.Replace("\"", "\\\"") + "\"" : s;
        var parts = new List<string> { "prism", "publish", "-d", Q(request.OutputDir) };
        if (!string.Equals(request.Server.GetLeftPart(UriPartial.Authority), PanoramaPaths.DefaultServer.GetLeftPart(UriPartial.Authority), StringComparison.OrdinalIgnoreCase))
            parts.AddRange(new[] { "--server", Q(request.Server.GetLeftPart(UriPartial.Authority)) });
        void Step(string? folder, string flag, string skip, string? page = null, string? pageFlag = null)
        {
            if (folder is null)
            {
                parts.Add(skip);
                return;
            }

            parts.AddRange(new[] { flag, Q(folder) });
            if (page is not null)
                parts.AddRange(new[] { pageFlag!, Q(page) });
        }

        Step(request.RawFolder, "--beside-raw", "--no-upload");
        Step(request.QcFolder, "--qc-wiki", "--no-qc", request.QcPage, "--qc-page");
        Step(request.QuantFolder, "--quant-wiki", "--no-quant", request.QuantPage, "--quant-page");
        Step(request.LinksFolder, "--links-wiki", "--no-links", request.LinksPage, "--links-page");
        if (request.ReplaceEdited)
            parts.Add("--replace-edited");
        return string.Join(' ', parts);
    }
}
