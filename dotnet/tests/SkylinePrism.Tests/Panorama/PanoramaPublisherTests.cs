using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SkylinePrism.Core.Panorama;
using Xunit;

namespace SkylinePrism.Tests.Panorama;

/// <summary>
/// Publishing against <see cref="FakePanorama"/>: the wiki pages (created, updated in place, never
/// overwriting a page someone else wrote or edited) and the upload (skipping what is already there,
/// verified against the server's own MD5, re-sent once when damaged).
/// </summary>
public sealed class PanoramaPublisherTests : IDisposable
{
    private const string Container = "/MacCoss/maccoss/Test";
    private const string Png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

    private readonly string _outputDir;
    private readonly FakePanorama _server = new();
    private readonly PanoramaClient _client;

    public PanoramaPublisherTests()
    {
        _outputDir = Path.Combine(Path.GetTempPath(), "prism_pub_" + Guid.NewGuid().ToString("N"), "run-2026-10");
        Directory.CreateDirectory(Path.Combine(_outputDir, "quant"));
        File.WriteAllText(Path.Combine(_outputDir, "qc_report.html"), Report("QC"));
        File.WriteAllText(Path.Combine(_outputDir, "quant", "quant_report.html"), Report("Quant"));
        File.WriteAllText(Path.Combine(_outputDir, "quant", "quant_parameters.json"),
            "{\"contrast\":{\"group_by\":\"Arm\",\"group_a\":[\"Pre\"],\"group_b\":[\"Post\"]}}");
        File.WriteAllBytes(Path.Combine(_outputDir, "corrected_proteins.parquet"), Enumerable.Range(0, 5000).Select(i => (byte)i).ToArray());
        File.WriteAllText(Path.Combine(_outputDir, "sample_metadata.csv"), "sample_id,sample_type\na,qc\n");

        _server.AddContainer(Container);
        _server.Folders.Add(Container + "/@files/RawFiles");
        _client = new PanoramaClient(new Uri("https://panorama.test"), PanoramaCredential.ApiKey("k"), _server);
    }

    public void Dispose()
    {
        _client.Dispose();
        try { Directory.Delete(Path.GetDirectoryName(_outputDir)!, recursive: true); }
        catch (IOException) { /* best-effort */ }
    }

    private static string Report(string what) =>
        $"<!DOCTYPE html><html><head><title>PRISM {what} Report</title><style>h1 {{ color: #1a3c6e; }}</style></head>"
        + $"<body><h1>PRISM {what} Report</h1><img src=\"data:image/png;base64,{Png}\"><img src=\"data:image/png;base64,{Png}\"></body></html>";

    private PanoramaPublisher Publisher() => new(_client);

    private FakePanorama.Page PageNamed(string name) => _server.Pages[Container + "|" + name];

    // ---------------------------------------------------------------------------------------
    // Wiki pages
    // ---------------------------------------------------------------------------------------

    [Fact]
    public async Task ANewPage_IsCreated_WithItsPlotsAttached_AndAFooter()
    {
        var page = await Publisher().PublishReportAsync(Path.Combine(_outputDir, "qc_report.html"), Container,
            "PRISM-QC-run", "PRISM QC: run", outputLink: null, replaceEdited: false);

        Assert.True(page.Created);
        Assert.Equal(2, page.Images);
        var stored = PageNamed("PRISM-QC-run");
        Assert.Equal(new[] { "prism-qc-run-01-", "prism-qc-run-02-" }, stored.Attachments.Keys.OrderBy(k => k).Select(k => k[..16]));
        Assert.False(stored.ShowAttachments); // the plots are inline; the attachment list would repeat them
        Assert.Contains("<h1 style=\"color: #1a3c6e\">", stored.Body, StringComparison.Ordinal);
        Assert.Contains("src=\"/MacCoss/maccoss/Test/wiki-download.view?entityId=", stored.Body, StringComparison.Ordinal);
        Assert.Contains("id=\"prism-wiki\"", stored.Body, StringComparison.Ordinal);
        Assert.Empty(WikiReport.Refusals(stored.Body));
        Assert.Equal("https://panorama.test/MacCoss/maccoss/Test/wiki-page.view?name=PRISM-QC-run", page.Url);
    }

    [Fact]
    public async Task Republishing_UpdatesTheSamePage_AndSendsNoPlotThatIsAlreadyThere()
    {
        var report = Path.Combine(_outputDir, "qc_report.html");
        await Publisher().PublishReportAsync(report, Container, "PRISM-QC-run", "t", null, false);
        var version = PageNamed("PRISM-QC-run").Version;
        var attached = _server.FilesAttached;

        var again = await Publisher().PublishReportAsync(report, Container, "PRISM-QC-run", "t", null, false);

        Assert.False(again.Created);
        Assert.True(PageNamed("PRISM-QC-run").Version > version);
        Assert.Equal(2, PageNamed("PRISM-QC-run").Attachments.Count);
        Assert.Equal(attached, _server.FilesAttached); // the plots are unchanged, so none was sent again
    }

    [Fact]
    public async Task AChangedPlot_IsSentUnderANewName_AndTheOldOneRemovedOnlyAfterTheSave()
    {
        var report = Path.Combine(_outputDir, "qc_report.html");
        await Publisher().PublishReportAsync(report, Container, "PRISM-QC-run", "t", null, false);
        var before = PageNamed("PRISM-QC-run").Attachments.Keys.OrderBy(k => k).ToList();
        var attached = _server.FilesAttached;

        var changed = Convert.ToBase64String(Convert.FromBase64String(Png).Append((byte)0).ToArray());
        var html = File.ReadAllText(report);
        var last = html.LastIndexOf(Png, StringComparison.Ordinal);
        File.WriteAllText(report, html[..last] + changed + html[(last + Png.Length)..]);
        await Publisher().PublishReportAsync(report, Container, "PRISM-QC-run", "t", null, false);

        var after = PageNamed("PRISM-QC-run").Attachments.Keys.OrderBy(k => k).ToList();
        Assert.Equal(attached + 1, _server.FilesAttached);
        Assert.Equal(before[0], after[0]);
        Assert.NotEqual(before[1], after[1]);
        Assert.Equal(2, after.Count); // the replaced plot is gone, not kept forever
        // At no save did the page point at an attachment it did not have.
        Assert.Empty(_server.MissingAtSave);
    }

    [Fact]
    public async Task AnEditMadeOnPanoramaWhileThePlotsAttach_IsRefused_NotSavedOver()
    {
        // The save re-reads the page for its newest version; without a second ownership check it would
        // save over an edit made in the meantime, under that edit's own version token.
        _server.AfterAttach = page => page.Body = "<p>A note someone added just now.</p>" + page.Body;

        var ex = await Assert.ThrowsAsync<PanoramaException>(() => Publisher().PublishReportAsync(
            Path.Combine(_outputDir, "qc_report.html"), Container, "PRISM-QC-run", "t", null, replaceEdited: false));

        Assert.Contains("edited on Panorama", ex.Message, StringComparison.Ordinal);
        Assert.StartsWith("<p>A note someone added just now.</p>", PageNamed("PRISM-QC-run").Body, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("qc")]
    [InlineData("quant")]
    [InlineData("links")]
    public async Task AnUnusablePageName_IsRefusedBeforeAnythingIsSent_UploadIncluded(string which)
    {
        var request = Request() with
        {
            QcPage = which == "qc" ? "my page" : null,
            QuantPage = which == "quant" ? "my page" : null,
            LinksFolder = Container,
            LinksPage = which == "links" ? "my page" : null,
        };

        Assert.Contains(OutputPublishing.Problems(request), p => p.Contains("'my page' is not a usable wiki page name", StringComparison.Ordinal));
        await Assert.ThrowsAsync<PanoramaException>(() => OutputPublishing.RunAsync(_client, request));
        Assert.Empty(_server.Requests); // not a byte of the directory went up first
    }

    [Fact]
    public async Task AClientSignedInToAnotherServer_IsRefusedBeforeAnythingIsSent()
    {
        var ex = await Assert.ThrowsAsync<PanoramaException>(() => OutputPublishing.RunAsync(_client,
            Request() with { Server = new Uri("https://panorama.test:8443") }));

        Assert.Contains("Signed in to https://panorama.test, but asked to publish to https://panorama.test:8443", ex.Message,
            StringComparison.Ordinal);
        Assert.Empty(_server.Requests);
    }

    [Fact]
    public async Task TheUploadedPanoramaJson_EndsUpLikeTheLocalOne_WithWhatThePublishRecorded()
    {
        MakePanoramaFolder();
        await OutputPublishing.RunAsync(_client, WithLinks());

        // The quant page and the web part's id are recorded after the upload; the copy on Panorama is
        // what a directory restored from it remembers.
        var uploaded = Encoding.UTF8.GetString(_server.Files[Container + "/@files/run-2026-10/panorama.json"]);
        Assert.Equal(File.ReadAllText(Path.Combine(_outputDir, PanoramaTargets.FileName)), uploaded);
        Assert.Contains("links_web_part_id", uploaded, StringComparison.Ordinal);
        Assert.Contains("quant_pages", uploaded, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AnotherOutputDirectoryWithTheSameName_CannotReplaceItsPage()
    {
        var other = Path.Combine(Path.GetTempPath(), "prism_pub_" + Guid.NewGuid().ToString("N"), "run-2026-10");
        Directory.CreateDirectory(other);
        try
        {
            File.Copy(Path.Combine(_outputDir, "qc_report.html"), Path.Combine(other, "qc_report.html"));
            await OutputPublishing.RunAsync(_client, Request(quant: null, raw: null));

            // Same directory name, so the same default page name - but another experiment's results.
            var ex = await Assert.ThrowsAsync<PanoramaException>(() => OutputPublishing.RunAsync(_client,
                new PublishRequest(other, new Uri("https://panorama.test"), Container, null, null)));
            Assert.Contains("from a different output directory", ex.Message, StringComparison.Ordinal);

            // Replaced on purpose, it goes; and the first directory republishes its own page as before.
            await OutputPublishing.RunAsync(_client,
                new PublishRequest(other, new Uri("https://panorama.test"), Container, null, null, ReplaceEdited: true));
            await Assert.ThrowsAsync<PanoramaException>(() => OutputPublishing.RunAsync(_client, Request(quant: null, raw: null)));
        }
        finally
        {
            Directory.Delete(Path.GetDirectoryName(other)!, recursive: true);
        }
    }

    [Fact]
    public async Task APageWithTheSameName_ThatPrismDidNotWrite_IsRefused()
    {
        _server.Pages[Container + "|Notes"] = new FakePanorama.Page { EntityId = "e", Name = "Notes", Body = "<p>hand written</p>" };

        var ex = await Assert.ThrowsAsync<PanoramaException>(() => Publisher().PublishReportAsync(
            Path.Combine(_outputDir, "qc_report.html"), Container, "Notes", "t", null, replaceEdited: false));

        Assert.Contains("PRISM did not write", ex.Message, StringComparison.Ordinal);
        Assert.Equal("<p>hand written</p>", _server.Pages[Container + "|Notes"].Body);
    }

    [Fact]
    public async Task APageEditedOnPanorama_IsRefused_UnlessReplacedOnPurpose()
    {
        var report = Path.Combine(_outputDir, "qc_report.html");
        await Publisher().PublishReportAsync(report, Container, "PRISM-QC-run", "t", null, false);
        var page = PageNamed("PRISM-QC-run");
        page.Body = "<p>A note someone added.</p>" + page.Body;

        var ex = await Assert.ThrowsAsync<PanoramaException>(() =>
            Publisher().PublishReportAsync(report, Container, "PRISM-QC-run", "t", null, replaceEdited: false));
        Assert.Contains("edited on Panorama", ex.Message, StringComparison.Ordinal);
        Assert.StartsWith("<p>A note someone added.</p>", page.Body, StringComparison.Ordinal);

        await Publisher().PublishReportAsync(report, Container, "PRISM-QC-run", "t", null, replaceEdited: true);
        Assert.DoesNotContain("A note someone added", PageNamed("PRISM-QC-run").Body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ThePage_LinksTheUploadedOutputDirectory_WhenThereIsOne()
    {
        await Publisher().PublishReportAsync(Path.Combine(_outputDir, "qc_report.html"), Container, "PRISM-QC-run", "t",
            "https://panorama.test/_webdav/MacCoss/maccoss/Test/@files/run-2026-10/", false);
        Assert.Contains("href=\"https://panorama.test/_webdav/MacCoss/maccoss/Test/@files/run-2026-10/\"",
            PageNamed("PRISM-QC-run").Body, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("has space")]
    [InlineData("-leading-dash")]
    [InlineData("a/b")]
    public async Task AnUnusablePageName_IsRefusedBeforeAnythingIsSent(string name)
    {
        await Assert.ThrowsAsync<PanoramaException>(() => Publisher().PublishReportAsync(
            Path.Combine(_outputDir, "qc_report.html"), Container, name, "t", null, false));
        Assert.Empty(_server.Requests);
    }

    [Fact]
    public void DefaultPageNames_NameTheOutputDirectory_AndTheContrast()
    {
        Assert.Equal("PRISM-QC-run-2026-10", PanoramaPublisher.DefaultQcPage(_outputDir));
        Assert.Equal("Post vs Pre by Arm", PanoramaPublisher.QuantContrast(_outputDir));
        Assert.Equal("PRISM-Quant-run-2026-10-Post-vs-Pre-by-Arm",
            PanoramaPublisher.DefaultQuantPage(_outputDir, PanoramaPublisher.QuantContrast(_outputDir)));
    }

    [Fact]
    public void ARememberedQuantPage_FollowsANewContrast_UnlessItWasNamedByHand()
    {
        var generatedForOldContrast = new PanoramaTargets("https://panoramaweb.org", null,
            new WikiTarget(Container, "PRISM-Quant-run-2026-10-Week-12-vs-Week-0-by-Visit"), null, null);
        // A new contrast must not overwrite the old contrast's page.
        Assert.Equal("PRISM-Quant-run-2026-10-Post-vs-Pre-by-Arm", generatedForOldContrast.QuantPageFor(_outputDir));

        var chosen = generatedForOldContrast with { QuantWiki = new WikiTarget(Container, "Verapamil-trend") };
        Assert.Equal("Verapamil-trend", chosen.QuantPageFor(_outputDir));
    }

    // ---------------------------------------------------------------------------------------
    // Output directory
    // ---------------------------------------------------------------------------------------

    [Fact]
    public void TheOutputDirectory_GoesBesideTheRawFiles()
    {
        Assert.Equal("/P/@files/run-2026-10", PanoramaPublisher.DestinationBeside("/P/@files/RawFiles", _outputDir));
        // Raw files kept in the file root itself: a root has no sibling, so the outputs go inside it.
        Assert.Equal("/P/@files/run-2026-10", PanoramaPublisher.DestinationBeside("/P/@files", _outputDir));
        Assert.Throws<PanoramaException>(() => PanoramaPublisher.DestinationBeside("/P/Sub", _outputDir));
    }

    [Fact]
    public async Task Upload_SendsEveryFile_AndKeepsTheLayout()
    {
        var destination = PanoramaPublisher.DestinationBeside(Container + "/@files/RawFiles", _outputDir);

        var result = await Publisher().UploadDirectoryAsync(_outputDir, destination);

        Assert.Equal(5, result.Uploaded);
        Assert.Equal(0, result.Skipped);
        Assert.True(_server.Files.ContainsKey(destination + "/corrected_proteins.parquet"));
        Assert.True(_server.Files.ContainsKey(destination + "/quant/quant_parameters.json"));
        Assert.Equal(File.ReadAllBytes(Path.Combine(_outputDir, "corrected_proteins.parquet")),
            _server.Files[destination + "/corrected_proteins.parquet"]);
    }

    [Fact]
    public async Task Republishing_SkipsWhatIsAlreadyThere_AndSendsWhatChanged()
    {
        var destination = PanoramaPublisher.DestinationBeside(Container + "/@files/RawFiles", _outputDir);
        await Publisher().UploadDirectoryAsync(_outputDir, destination);
        File.AppendAllText(Path.Combine(_outputDir, "sample_metadata.csv"), "b,reference\n");

        var again = await Publisher().UploadDirectoryAsync(_outputDir, destination);

        Assert.Equal((1, 4), (again.Uploaded, again.Skipped));
        Assert.EndsWith("b,reference\n", Encoding.UTF8.GetString(_server.Files[destination + "/sample_metadata.csv"]), StringComparison.Ordinal);
    }

    [Fact]
    public async Task AReport_IsSentInTheFormPanoramaAccepts_UnderItsOwnName()
    {
        // The fake refuses an .html carrying a doctype or a <style>, as panoramaweb.org does (403).
        var destination = PanoramaPublisher.DestinationBeside(Container + "/@files/RawFiles", _outputDir);
        await Publisher().UploadDirectoryAsync(_outputDir, destination);

        var sent = Encoding.UTF8.GetString(_server.Files[destination + "/qc_report.html"]);
        Assert.False(WikiReport.RefusedAsFile(sent));
        Assert.Contains("<h1 style=\"color: #1a3c6e\">PRISM QC Report</h1>", sent, StringComparison.Ordinal);
        Assert.Contains("data:image/png;base64,", sent, StringComparison.Ordinal);

        // And it is recognized as unchanged next time, though its bytes differ from the local file.
        Assert.Equal(0, (await Publisher().UploadDirectoryAsync(_outputDir, destination)).Uploaded);
    }

    [Fact]
    public async Task ADamagedTransfer_IsSentAgain_AndRefusedIfItStaysDamaged()
    {
        var destination = PanoramaPublisher.DestinationBeside(Container + "/@files/RawFiles", _outputDir);

        var damagedOnce = false;
        _server.DamageOnPut = (path, bytes) =>
        {
            if (!path.EndsWith(".parquet", StringComparison.Ordinal) || damagedOnce)
                return bytes;
            damagedOnce = true;
            return bytes[..^1];
        };
        var recovered = await Publisher().UploadDirectoryAsync(_outputDir, destination);
        Assert.Equal(5, recovered.Uploaded);
        Assert.Equal(File.ReadAllBytes(Path.Combine(_outputDir, "corrected_proteins.parquet")),
            _server.Files[destination + "/corrected_proteins.parquet"]);

        _server.Files.Clear();
        _server.DamageOnPut = (path, bytes) => path.EndsWith(".parquet", StringComparison.Ordinal) ? bytes[..^1] : bytes;
        var ex = await Assert.ThrowsAsync<PanoramaException>(() => Publisher().UploadDirectoryAsync(_outputDir, destination));
        Assert.Contains("still does not match", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ANameWithASemicolon_IsRefused_SincePanoramaTruncatesAtIt()
    {
        File.WriteAllText(Path.Combine(_outputDir, "a;b.csv"), "x");
        var ex = await Assert.ThrowsAsync<PanoramaException>(() =>
            Publisher().UploadDirectoryAsync(_outputDir, Container + "/@files/run-2026-10"));
        Assert.Contains("';'", ex.Message, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------------------------------
    // The whole publish
    // ---------------------------------------------------------------------------------------

    private PublishRequest Request(string? qc = Container, string? quant = Container, string? raw = Container + "/@files/RawFiles") =>
        new(_outputDir, new Uri("https://panorama.test"), qc, quant, raw);

    [Fact]
    public async Task APublish_Uploads_ThenPublishesBothPages_AndRemembersTheTargets()
    {
        var outcome = await OutputPublishing.RunAsync(_client, Request());

        Assert.NotNull(outcome.Upload);
        Assert.True(_server.Pages.ContainsKey(Container + "|PRISM-QC-run-2026-10"));
        Assert.True(_server.Pages.ContainsKey(Container + "|PRISM-Quant-run-2026-10-Post-vs-Pre-by-Arm"));
        // The quant page links the uploaded quant folder, where its tables are.
        Assert.Contains("@files/run-2026-10/quant/", _server.Pages[Container + "|PRISM-Quant-run-2026-10-Post-vs-Pre-by-Arm"].Body,
            StringComparison.Ordinal);

        var remembered = PanoramaTargets.Load(_outputDir);
        Assert.Equal(Container, remembered.QcWiki!.Folder);
        Assert.Equal(Container + "/@files/RawFiles", remembered.RawFolder);
        // The targets went up with the outputs, and hold no credential.
        Assert.DoesNotContain("apikey", Encoding.UTF8.GetString(_server.Files[Container + "/@files/run-2026-10/panorama.json"]),
            StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Problems_AreSaidBeforeAnythingIsSent()
    {
        File.Delete(Path.Combine(_outputDir, "quant", "quant_report.html"));
        var problems = OutputPublishing.Problems(Request(raw: "/MacCoss/maccoss/Test/Sub"));

        Assert.Contains(problems, p => p.Contains("no quant report", StringComparison.Ordinal));
        Assert.Contains(problems, p => p.Contains("is not a file area", StringComparison.Ordinal));
        Assert.Contains(OutputPublishing.Problems(Request(null, null, null)), p => p.StartsWith("Nothing to publish", StringComparison.Ordinal));
    }

    [Fact]
    public void ThePlan_CountsTheTargetsFileThatGoesUpWithTheOutputs()
    {
        // Five files now, plus the panorama.json written just before the upload.
        Assert.StartsWith("Upload 6 files", OutputPublishing.Describe(Request()), StringComparison.Ordinal);
    }

    [Fact]
    public void TheCommandLine_ReproducesTheRequest()
    {
        var command = OutputPublishing.CommandLine(Request() with { QcPage = "my page", ReplaceEdited = true, LinksFolder = Container });
        Assert.Contains("--beside-raw /MacCoss/maccoss/Test/@files/RawFiles", command, StringComparison.Ordinal);
        Assert.Contains("--qc-wiki /MacCoss/maccoss/Test", command, StringComparison.Ordinal);
        Assert.Contains("--qc-page \"my page\"", command, StringComparison.Ordinal);
        Assert.Contains("--links-wiki /MacCoss/maccoss/Test", command, StringComparison.Ordinal);
        Assert.Contains("--server https://panorama.test", command, StringComparison.Ordinal);
        Assert.EndsWith("--replace-edited", command, StringComparison.Ordinal);
        Assert.DoesNotContain("--no-", command, StringComparison.Ordinal);
    }

    [Fact]
    public void TheCommandLine_SaysWhichStepsAreSkipped_SoTheCliDoesNotFillThemIn()
    {
        // The CLI fills an absent folder from panorama.json, and the links page from the QC folder;
        // leaving a skipped step out of the command would bring it back.
        var command = OutputPublishing.CommandLine(Request(quant: null, raw: null));
        Assert.Contains("--qc-wiki /MacCoss/maccoss/Test", command, StringComparison.Ordinal);
        Assert.Contains("--no-upload", command, StringComparison.Ordinal);
        Assert.Contains("--no-quant", command, StringComparison.Ordinal);
        Assert.Contains("--no-links", command, StringComparison.Ordinal);
        Assert.DoesNotContain("--no-qc", command, StringComparison.Ordinal);
    }

    [Fact]
    public async Task SkippingAStep_ForOnePublish_DoesNotForgetWhereItGoes()
    {
        await OutputPublishing.RunAsync(_client, Request() with { LinksFolder = Container });
        await OutputPublishing.RunAsync(_client, Request(qc: null, raw: null));

        var remembered = PanoramaTargets.Load(_outputDir);
        Assert.Equal(new WikiTarget(Container, "PRISM-QC-run-2026-10"), remembered.QcWiki);
        Assert.Equal(Container + "/@files/RawFiles", remembered.RawFolder);
        Assert.Equal(Container + "/@files/run-2026-10", remembered.Destination);
        Assert.Equal(new WikiTarget(Container, "PRISM-run-2026-10"), remembered.LinksWiki);
    }

    // ---------------------------------------------------------------------------------------
    // The links page, and the folder page that shows it
    // ---------------------------------------------------------------------------------------

    private PublishRequest WithLinks() => Request() with { LinksFolder = Container };

    // A Panorama folder's dashboard as panoramaweb.org creates it.
    private void MakePanoramaFolder()
    {
        _server.FolderTypes[Container] = "Targeted MS";
        _server.AddWebPart(Container, "DefaultDashboard", "!content", "Mass Spec Search (Tabbed)");
        _server.AddWebPart(Container, "DefaultDashboard", "!content", "Targeted MS Runs");
        _server.AddWebPart(Container, "DefaultDashboard", "!content", "Files");
        _server.AddWebPart(Container, "DefaultDashboard", "right", "Wiki Table of Contents");
    }

    [Fact]
    public async Task TheLinksPage_LinksTheQcPage_TheQuantPage_AndTheOutputDirectory()
    {
        MakePanoramaFolder();
        var outcome = await OutputPublishing.RunAsync(_client, WithLinks());

        Assert.NotNull(outcome.Links);
        var body = PageNamed("PRISM-run-2026-10").Body;
        Assert.Contains("href=\"/MacCoss/maccoss/Test/wiki-page.view?name=PRISM-QC-run-2026-10\"", body, StringComparison.Ordinal);
        Assert.Contains("href=\"/MacCoss/maccoss/Test/wiki-page.view?name=PRISM-Quant-run-2026-10-Post-vs-Pre-by-Arm\"", body, StringComparison.Ordinal);
        Assert.Contains("Quant report: Post vs Pre by Arm", body, StringComparison.Ordinal);
        Assert.Contains("@files/run-2026-10", body, StringComparison.Ordinal);
        Assert.Empty(WikiReport.Refusals(body));
    }

    [Fact]
    public async Task TheLinksPage_IsShownAfterTheRuns_AboveTheFiles()
    {
        MakePanoramaFolder();
        var outcome = await OutputPublishing.RunAsync(_client, WithLinks());

        Assert.True(outcome.LinksOnFolderPage);
        Assert.Null(outcome.LinksNote);
        var part = Assert.Single(_server.WebParts, p => p.Name == "Wiki");
        // Sent as LabKey's internal column name: "body" would be saved where no page shows it.
        Assert.Equal("!content", part.Location);
        Assert.Equal("PRISM-run-2026-10", part.Properties["name"]);
        Assert.Equal("id-of" + Container, part.Properties["webPartContainer"]);
        Assert.Equal(part.Id, PanoramaTargets.Load(_outputDir).LinksWebPartId);
        // The folder's own parts are all still there, in their order, with the links among them.
        Assert.Equal(new[] { "Mass Spec Search (Tabbed)", "Targeted MS Runs", "Wiki", "Files", "Wiki Table of Contents" },
            _server.RenderedParts(Container).Select(p => p.Name));
    }

    [Theory]
    [InlineData(new[] { "Mass Spec Search (Tabbed)", "Targeted MS Runs", "Files" }, 2)] // between the runs and the files
    [InlineData(new[] { "Files", "Targeted MS Runs", "Wiki" }, 2)]                       // after the runs, wherever the files are
    [InlineData(new[] { "Mass Spec Search (Tabbed)", "Files" }, 1)]                      // no runs: above the files
    [InlineData(new[] { "Mass Spec Search (Tabbed)" }, 0)]                               // neither: at the top
    [InlineData(new string[0], 0)]
    public void ANewLinksPart_GoesAfterTheRuns_ElseAboveTheFiles_ElseAtTheTop(string[] names, int position) =>
        Assert.Equal(position, PanoramaPublisher.PositionFor(names.Select((n, i) => new WebPart("body", n, i + 1, 1000 + i)).ToList()));

    [Fact]
    public async Task Republishing_ReusesItsWebPart_AndLeavesItWhereSomeoneMovedIt()
    {
        MakePanoramaFolder();
        await OutputPublishing.RunAsync(_client, WithLinks());
        var part = Assert.Single(_server.WebParts, p => p.Name == "Wiki");
        await _client.MoveWebPartUpAsync(Container, part.Id); // someone moves it above the runs

        await OutputPublishing.RunAsync(_client, WithLinks());

        Assert.Single(_server.WebParts, p => p.Name == "Wiki");
        Assert.Equal(part.Id, _server.RenderedParts(Container)[1].Id);
    }

    [Fact]
    public async Task AWebPartRemovedOnPanorama_IsAddedAgain()
    {
        MakePanoramaFolder();
        await OutputPublishing.RunAsync(_client, WithLinks());
        var first = PanoramaTargets.Load(_outputDir).LinksWebPartId;
        _server.WebParts.RemoveAll(p => p.Id == first);

        await OutputPublishing.RunAsync(_client, WithLinks());

        var again = PanoramaTargets.Load(_outputDir).LinksWebPartId;
        Assert.NotEqual(first, again);
        Assert.Equal(again, _server.RenderedParts(Container)[2].Id); // placed as a new one is, after the runs
    }

    [Fact]
    public async Task WithoutAdministratorPermission_EverythingElseIsPublished_AndTheOutcomeSaysWhat()
    {
        MakePanoramaFolder();
        _server.IsAdmin = false;

        var outcome = await OutputPublishing.RunAsync(_client, WithLinks());

        Assert.NotNull(outcome.Qc);
        Assert.NotNull(outcome.Quant);
        Assert.NotNull(outcome.Links);
        Assert.False(outcome.LinksOnFolderPage);
        Assert.Contains("A folder administrator can add a Wiki web part there showing the page PRISM-run-2026-10", outcome.LinksNote,
            StringComparison.Ordinal);
        Assert.DoesNotContain(_server.WebParts, p => p.Name == "Wiki");
    }

    [Fact]
    public async Task EveryQuantPagePublishedFromTheDirectory_IsListed_AndOneDeletedOnPanoramaIsNot()
    {
        MakePanoramaFolder();
        await OutputPublishing.RunAsync(_client, WithLinks());
        File.WriteAllText(Path.Combine(_outputDir, "quant", "quant_parameters.json"),
            "{\"contrast\":{\"group_by\":\"Visit\",\"group_a\":[\"Week 0\"],\"group_b\":[\"Week 12\"]}}");

        await OutputPublishing.RunAsync(_client, WithLinks());

        var body = PageNamed("PRISM-run-2026-10").Body;
        Assert.Contains("Quant report: Post vs Pre by Arm", body, StringComparison.Ordinal);
        Assert.Contains("Quant report: Week 12 vs Week 0 by Visit", body, StringComparison.Ordinal);

        _server.Pages.Remove(Container + "|PRISM-Quant-run-2026-10-Post-vs-Pre-by-Arm");
        await OutputPublishing.RunAsync(_client, WithLinks() with { QuantFolder = null });
        Assert.DoesNotContain("Post vs Pre by Arm", PageNamed("PRISM-run-2026-10").Body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheFolderPage_IsTheDashboardInAPanoramaFolder_AndThePortalPageOtherwise()
    {
        // An empty Panorama folder: nothing rendered to match, so the folder type decides.
        _server.FolderTypes[Container] = "Targeted MS";
        Assert.Equal(PanoramaClient.PanoramaDashboardPage, await _client.MainPageIdAsync(Container));

        _server.FolderTypes.Remove(Container);
        Assert.Equal(PanoramaClient.DefaultPortalPage, await _client.MainPageIdAsync(Container));

        // Parts left on the dashboard by an earlier folder type are not what the page shows: the page id
        // is the one whose parts the folder's page actually renders, though the dashboard is tried first.
        _server.AddWebPart(Container, "DefaultDashboard", "!content", "Targeted MS Runs");
        _server.AddWebPart(Container, "portal.default", "!content", "Wiki");
        Assert.Equal(PanoramaClient.DefaultPortalPage, await _client.MainPageIdAsync(Container));
    }

    [Fact]
    public async Task CustomizingAWebPartRemovedMeanwhile_IsRefused()
    {
        var ex = await Assert.ThrowsAsync<PanoramaException>(() =>
            _client.CustomizeWebPartAsync(Container, 99, new System.Collections.Generic.Dictionary<string, string> { ["name"] = "x" }));
        Assert.Contains("may have been removed", ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("body", "!content")]
    [InlineData("right", "right")]
    [InlineData("menu", "menubar")]
    public void APageColumn_IsSentUnderLabKeysInternalName(string listed, string sent) =>
        Assert.Equal(sent, PanoramaClient.InternalLocation(listed));

    [Fact]
    public void AnUnknownPageColumn_IsRefused_RatherThanSavedWhereNothingShowsIt() =>
        Assert.Throws<ArgumentException>(() => PanoramaClient.InternalLocation("left"));

    [Fact]
    public async Task AWebPartThatCouldNotBeSetUp_IsStillRecorded_AndSetUpByTheNextPublish()
    {
        MakePanoramaFolder();
        _server.FailCustomize = true;

        var failed = await OutputPublishing.RunAsync(_client, WithLinks());

        Assert.False(failed.LinksOnFolderPage);
        var part = Assert.Single(_server.WebParts, p => p.Name == "Wiki");
        Assert.Equal(part.Id, PanoramaTargets.Load(_outputDir).LinksWebPartId);

        _server.FailCustomize = false;
        var again = await OutputPublishing.RunAsync(_client, WithLinks());

        Assert.True(again.LinksOnFolderPage);
        Assert.Single(_server.WebParts, p => p.Name == "Wiki"); // not a second one
        Assert.Equal("PRISM-run-2026-10", part.Properties["name"]);
    }

    [Fact]
    public async Task ALinksPageThatCannotBePublished_FailsNothingElse_AndTheQuantPageIsStillRecorded()
    {
        MakePanoramaFolder();
        await OutputPublishing.RunAsync(_client, WithLinks());
        var links = PageNamed("PRISM-run-2026-10");
        links.Body = "<p>A note someone added.</p>" + links.Body;
        File.WriteAllText(Path.Combine(_outputDir, "quant", "quant_parameters.json"),
            "{\"contrast\":{\"group_by\":\"Visit\",\"group_a\":[\"Week 0\"],\"group_b\":[\"Week 12\"]}}");

        var outcome = await OutputPublishing.RunAsync(_client, WithLinks());

        Assert.NotNull(outcome.Quant);
        Assert.Null(outcome.Links);
        Assert.Contains("The links page PRISM-run-2026-10 was not published", outcome.LinksNote, StringComparison.Ordinal);
        Assert.Contains(PanoramaTargets.Load(_outputDir).QuantPages!, p => p.Contrast == "Week 12 vs Week 0 by Visit");
    }

    [Fact]
    public async Task AQuantPageNamedByEditingTheDefault_IsKept_AndAGeneratedOneFollowsTheContrast()
    {
        var edited = PanoramaPublisher.DefaultQuantPage(_outputDir, PanoramaPublisher.QuantContrast(_outputDir)) + "-v2";
        await OutputPublishing.RunAsync(_client, Request(qc: null, raw: null) with { QuantPage = edited });
        File.WriteAllText(Path.Combine(_outputDir, "quant", "quant_parameters.json"),
            "{\"contrast\":{\"group_by\":\"Visit\",\"group_a\":[\"Week 0\"],\"group_b\":[\"Week 12\"]}}");

        // It starts like a generated name, but it is recorded as the page published for a contrast whose
        // default it is not, so it was chosen.
        Assert.Equal(edited, PanoramaTargets.Load(_outputDir).QuantPageFor(_outputDir));

        await OutputPublishing.RunAsync(_client, Request(qc: null, raw: null));
        Assert.Equal("PRISM-Quant-run-2026-10-Week-12-vs-Week-0-by-Visit", PanoramaTargets.Load(_outputDir).QuantWiki!.Page);
        File.WriteAllText(Path.Combine(_outputDir, "quant", "quant_parameters.json"),
            "{\"contrast\":{\"group_by\":\"Arm\",\"group_a\":[\"Pre\"],\"group_b\":[\"Post\"]}}");
        Assert.Equal("PRISM-Quant-run-2026-10-Post-vs-Pre-by-Arm", PanoramaTargets.Load(_outputDir).QuantPageFor(_outputDir));
    }

    [Fact]
    public void LongDefaultPageNames_StayUsable_AndDistinct()
    {
        var dir = Path.Combine(Path.GetTempPath(), new string('d', 150));
        var a = PanoramaPublisher.DefaultQuantPage(dir, new string('c', 100) + " A");
        var b = PanoramaPublisher.DefaultQuantPage(dir, new string('c', 100) + " B");

        Assert.True(a.Length <= 200);
        Assert.Matches("^[A-Za-z0-9][A-Za-z0-9_.-]*$", a);
        Assert.NotEqual(a, b); // cut to the same 191 characters, told apart by the hash of the whole
    }

    [Fact]
    public async Task UploadProgress_ArrivesInOrder_AndAllOfItBeforeTheUploadReturns()
    {
        var reports = new System.Collections.Generic.List<UploadProgress>();
        var destination = PanoramaPublisher.DestinationBeside(Container + "/@files/RawFiles", _outputDir);

        await Publisher().UploadDirectoryAsync(_outputDir, destination, new Recorder(reports));

        // A Progress<T> inside the publisher would post these to the thread pool: late, and out of order.
        Assert.NotEmpty(reports);
        Assert.Equal(reports.Select(r => r.TotalSent).OrderBy(t => t), reports.Select(r => r.TotalSent));
        Assert.Equal(reports[^1].TotalBytes, reports[^1].TotalSent);
    }

    private sealed class Recorder(System.Collections.Generic.List<UploadProgress> into) : IProgress<UploadProgress>
    {
        public void Report(UploadProgress value) => into.Add(value);
    }

    [Fact]
    public async Task WhenNoSavedSignInWorks_TheMessageSaysWhatWasTriedAndWhatToDo()
    {
        _server.RejectSignIn = true;
        var signIn = new PanoramaSignIn(null, new Uri("https://panorama.test"), n => n == PanoramaSignIn.ApiKeyVariable ? "bad" : null);

        var ex = await Assert.ThrowsAsync<PanoramaException>(() => OutputPublishing.SignInAsync(signIn, default, _server));

        Assert.True(ex.IsSignInProblem);
        Assert.Contains($"the sign-in from {PanoramaSignIn.ApiKeyVariable} (an API key) was not accepted", ex.Message, StringComparison.Ordinal);
        Assert.Contains("External Tool Access", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheFirstSignInPanoramaAccepts_IsTheOneUsed()
    {
        var signIn = new PanoramaSignIn(null, new Uri("https://panorama.test"), n => n == PanoramaSignIn.ApiKeyVariable ? "good" : null);
        using var client = await OutputPublishing.SignInAsync(signIn, default, _server);
        Assert.Equal(PanoramaSignIn.ApiKeyVariable, client.Credential.Source);
    }
}
