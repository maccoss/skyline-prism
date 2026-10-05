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
/// <param name="ReplaceEdited">Replace a page edited on Panorama since PRISM published it, or one PRISM did not write.</param>
/// <param name="TakeOver">Replace a page published from a different output directory (see <see cref="PanoramaTargets.PublishId"/>).</param>
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
    string? LinksPage = null,
    bool TakeOver = false)
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
            // panorama.json is written just before the upload and goes up with it - when it CAN be
            // written: a directory PRISM cannot write to is published without one, so it is named
            // rather than counted, and the run says which happened.
            var targetsPending = !File.Exists(Path.Combine(request.OutputDir, PanoramaTargets.FileName));
            sb.AppendLine($"Upload {files.Count} files ({PanoramaPublisher.FormatBytes(files.Sum(f => f.Length))})"
                          + (targetsPending ? $", and the {PanoramaTargets.FileName} this publish writes," : string.Empty)
                          + $" to {request.Destination}, skipping any already there unchanged");
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

        // Strictly: a file that is there but unreadable for a moment is not "nothing remembered", or the
        // directory's id would be replaced (see LoadForPublish).
        var remembered = PanoramaTargets.LoadForPublish(request.OutputDir, say);
        var targets = request.ToTargets(remembered);
        // The web part PRISM added before, and the folder whose page it is on - which older files do not
        // record, and then it is the links page's folder. Moving the links elsewhere adds a part there;
        // the old one stays recorded until the new one exists, so a move that fails (no administrator
        // permission in the new folder, say) still names the old part on the next publish.
        var partFolder = remembered.LinksWebPartFolder ?? remembered.LinksWiki?.Folder;
        var movedFrom = remembered.LinksWebPartId is { } oldPart && partFolder is not null && request.LinksFolder is not null
                        && !string.Equals(PanoramaPaths.Container(partFolder), PanoramaPaths.Container(request.LinksFolder),
                            StringComparison.OrdinalIgnoreCase)
            ? (Folder: PanoramaPaths.Container(partFolder), Page: remembered.LinksWebPartPage ?? remembered.LinksWiki?.Page ?? "",
                WebPartId: oldPart)
            : ((string Folder, string Page, int WebPartId)?)null;

        // Derived, not random: the same directory always gives the same id, so one PRISM cannot write to
        // (a read-only share, an archived analysis) is recognized from one publish to the next without
        // keeping it anywhere, and stays recognized if it later becomes writable.
        targets = targets with { PublishId = targets.PublishId ?? DerivedPublishId(request.OutputDir) };
        // Saved before anything is sent, so the uploaded copy records where it was published. A
        // directory PRISM cannot write to is published all the same.
        var remembering = Remember(targets, request.OutputDir);
        if (!remembering.Saved)
            say?.Invoke($"{PanoramaTargets.FileName} cannot be written in {request.OutputDir} ({remembering.Why}). Publishing anyway; "
                        + "where this directory was published to will not be remembered for next time.");

        var publisher = new PanoramaPublisher(client, say, targets.PublishId, request.TakeOver);

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
        if (remembering.Saved)
            Remember(targets, request.OutputDir);

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
                var superseded = false;
                try
                {
                    var (webPartId, _) = await publisher.ShowOnFolderPageAsync(request.LinksFolder, request.ResolvedLinksPage,
                        movedFrom is null ? targets.LinksWebPartId : null,
                        added: id =>
                        {
                            // Recorded the moment the part exists, so a failure setting it up does not
                            // leave a part the next publish would not recognize and add again. A file
                            // that cannot be written here must not fail a publish that worked.
                            targets = targets with
                            {
                                LinksWebPartId = id, LinksWebPartFolder = links.Folder, LinksWebPartPage = links.PageName,
                            };
                            superseded = true;
                            if (remembering.Saved)
                                Remember(targets, request.OutputDir);
                        },
                        cancellationToken).ConfigureAwait(false);
                    targets = targets with
                    {
                        LinksWebPartId = webPartId, LinksWebPartFolder = links.Folder, LinksWebPartPage = links.PageName,
                    };
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

                // Said once the new part exists - whether or not setting it up then failed - because from
                // then on the old one is no longer what this directory records.
                if (movedFrom is { } old && superseded)
                {
                    var moved = $"The links are now shown on {links.Folder}'s page. {old.Folder}'s page still shows the earlier "
                                + $"links page, {old.Page}, in the Wiki web part PRISM added there (id {old.WebPartId}); it is no "
                                + "longer updated. Remove that part on Panorama if it is not wanted there.";
                    say?.Invoke(moved);
                    note = note is null ? moved : note + " " + moved;
                }
            }
        }

        var final = remembering.Saved ? Remember(targets, request.OutputDir) : remembering;
        if (remembering.Saved && !final.Saved)
            say?.Invoke($"Everything is published, but {PanoramaTargets.FileName} could not be updated ({final.Why}); "
                        + "the next publish may not know about this one's quant page or web part.");
        if (upload is not null && final.Saved)
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
    /// Saves the targets, or says why they could not be saved. A publish never fails for want of
    /// writing <see cref="PanoramaTargets.FileName"/>: the file is a convenience, and the pages and the
    /// upload are what was asked for.
    /// </summary>
    private static (bool Saved, PanoramaTargets Targets, string? Why) Remember(PanoramaTargets targets, string outputDir)
    {
        try
        {
            targets.Save(outputDir);
            return (true, targets, null);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return (false, targets, ex.Message);
        }
    }

    /// <summary>
    /// The publish id of an output directory that has none recorded: a hash of its run record
    /// (<c>parameters.json</c>), else of its full path. The same however often it is derived - so a
    /// directory that cannot keep it is still recognized - and a copy of one analysis is that analysis,
    /// while another run, which writes its own <c>parameters.json</c>, is not.
    /// </summary>
    /// <remarks>
    /// A <c>parameters.json</c> that is there but cannot be read is refused rather than replaced by
    /// the path: the fallback would give this directory a second identity, and its pages would then
    /// be refused as another's.
    /// </remarks>
    internal static string DerivedPublishId(string outputDir)
    {
        var parameters = Path.Combine(outputDir, "parameters.json");
        byte[] basis;
        if (!File.Exists(parameters))
            basis = Encoding.UTF8.GetBytes(Path.GetFullPath(outputDir).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        else
        {
            try
            {
                using var stream = new FileStream(parameters, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
                using var copy = new MemoryStream();
                stream.CopyTo(copy);
                basis = copy.ToArray();
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                throw new PanoramaException(
                    $"parameters.json in {outputDir} could not be read ({ex.Message}); it is what this directory's publish "
                    + "id comes from. Close whatever has it open and publish again.");
            }
        }

        return Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(basis))[..32];
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
        if (request.TakeOver)
            parts.Add("--take-over");
        return string.Join(' ', parts);
    }
}
