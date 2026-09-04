using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading;
using SkylinePrism.App;
using SkylinePrism.Skyline;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Skyline shared document archives (<c>.sky.zip</c>), which is how a document arrives from
/// PanoramaWeb: the <c>.sky</c>, its <c>.skyd</c>, any library and the audit log in one file.
///
/// <para>Skyline's own command line cannot open one - <c>--in</c> XML-parses the path, and the
/// pre-flight check waves a <c>.sky.zip</c> through on its extension, so the failure is a generic
/// parse error. Verified against SkylineCmd on a real 13.7 GB Panorama archive:
/// <i>"There was an error opening the file ... does not appear to be a Skyline document. Skyline
/// documents normally have a ".sky" or ".sky.zip" filename extension"</i> - the message contradicts
/// itself. So PRISM extracts, and these tests pin the parts of that which can go quietly wrong:
/// which entry is the document, what the batch label becomes, that an extraction is reused rather
/// than repeated (17.4 GB a plate), and that an archive is not trusted with a path.</para>
/// </summary>
public class SharedDocumentArchiveTests
{
    private readonly ITestOutputHelper _out;

    public SharedDocumentArchiveTests(ITestOutputHelper output) => _out = output;

    private const string Document = """
<?xml version="1.0" encoding="utf-8"?>
<srm_settings format_version="25.1" software_version="Skyline (64-bit) 25.1.0.237">
  <settings_summary name="Default">
    <peptide_settings>
      <enzyme name="Trypsin" cut="KR" no_cut="P" sense="C" />
    </peptide_settings>
    <data_settings document_guid="3f1a0893-9070-4bdc-9910-d8005ed2e76f">
      <annotation name="Plate" targets="replicate" type="text" />
    </data_settings>
    <transition_settings>
      <transition_full_scan product_mass_analyzer="centroided" product_res="10" />
    </transition_settings>
    <measured_results>
      <replicate name="Ref_01" sample_type="standard">
        <sample_file id="f0" file_path="C:\data\Ref_01.raw" />
        <annotation name="Plate">P1</annotation>
      </replicate>
      <replicate name="Study_07" sample_type="unknown">
        <sample_file id="f1" file_path="C:\data\Study_07.raw" />
      </replicate>
    </measured_results>
  </settings_summary>
  <protein name="sp|P12345|TEST_HUMAN">
    <peptide sequence="PEPTIDER" />
  </protein>
</srm_settings>
""";

    private static string TempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_skyzip_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>A shared archive shaped like Panorama's: the document, a cache, a library, an audit log.</summary>
    private static string WriteArchive(
        string dir, string stem = "Plate1", IEnumerable<(string Name, string Body)>? entries = null)
    {
        var path = Path.Combine(dir, stem + SharedDocumentArchive.Extension);
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var zip = new ZipArchive(stream, ZipArchiveMode.Create);
        foreach (var (name, body) in entries ?? new[]
                 {
                     (stem + ".sky", Document),
                     (stem + ".skyd", "not really a cache"),
                     ("library.blib", "not really a library"),
                     (stem + ".skyl", "audit log"),
                 })
        {
            var entry = zip.CreateEntry(name);
            using var writer = new StreamWriter(entry.Open(), Encoding.UTF8);
            writer.Write(body);
        }
        return path;
    }

    [Fact]
    public void AnArchiveIsRecognizedByItsDoubleExtension()
    {
        Assert.True(SharedDocumentArchive.IsArchive(@"C:\p\Plate1.sky.zip"));
        Assert.True(SharedDocumentArchive.IsArchive(@"C:\p\Plate1.SKY.ZIP"));   // Panorama casing varies
        Assert.False(SharedDocumentArchive.IsArchive(@"C:\p\Plate1.sky"));
        Assert.False(SharedDocumentArchive.IsArchive(@"C:\p\raw-files.zip"));
        Assert.False(SharedDocumentArchive.IsArchive(null));
    }

    /// <summary>
    /// The label drops BOTH extensions. Path.GetFileNameWithoutExtension strips only ".zip", and the
    /// leftover ".sky" would ride into every merged sample ID as part of the batch - the merge derives
    /// Batch from the file stem.
    /// </summary>
    [Fact]
    public void TheStemDropsBothExtensions()
    {
        Assert.Equal(
            "EXP25033_UWash_Floyd_P1a_annotated",
            SharedDocumentArchive.StemOf(@"V:\dl\EXP25033_UWash_Floyd_P1a_annotated.sky.zip"));
        Assert.Equal("Plate1", SharedDocumentArchive.StemOf(@"C:\p\Plate1.sky"));

        // ...and the input built from an archive carries that label, not the file name.
        var input = PrismInput.FromClosedDocument(
            Path.Combine(TempDir(), "EXP_P1a_annotated.sky.zip"));
        Assert.Equal("EXP_P1a_annotated", input.BatchLabel);
        Assert.True(input.IsSharedArchive);
        Assert.False(PrismInput.FromClosedDocument(Path.Combine(TempDir(), "doc.sky")).IsSharedArchive);
    }

    /// <summary>
    /// The document header is read out of the archive with NOTHING extracted, so the Inputs tab can
    /// show replicates, annotations and the enzyme the moment an archive is added. Worth having when
    /// the entry inside is 4.4 GB, as it was on the measured Panorama plate.
    /// </summary>
    [Fact]
    public void TheHeaderIsReadFromTheArchiveWithoutExtracting()
    {
        var dir = TempDir();
        var archive = WriteArchive(dir);

        var info = SkyDocumentInfo.TryRead(archive, _out.WriteLine);

        Assert.NotNull(info);
        Assert.Equal(new[] { "Plate" }, info!.ReplicateAnnotationNames);
        Assert.Equal(new[] { "Ref_01", "Study_07" }, info.Replicates.Select(r => r.Name).ToArray());
        Assert.Equal("trypsin", info.PrismEnzyme);
        Assert.Equal("+/-10 ppm (centroided)", info.ProductTolerance!.Describe());

        // Nothing was written anywhere: the only file in the directory is still the archive.
        Assert.Equal(new[] { archive }, Directory.GetFileSystemEntries(dir));
    }

    /// <summary>An archive and the loose document inside it must read identically.</summary>
    [Fact]
    public void TheArchiveAndTheLooseDocumentAgree()
    {
        var dir = TempDir();
        var archive = WriteArchive(dir);
        var loose = Path.Combine(dir, "loose.sky");
        File.WriteAllText(loose, Document);

        var fromZip = SkyDocumentInfo.TryRead(archive)!;
        var fromFile = SkyDocumentInfo.TryRead(loose)!;

        Assert.Equal(fromFile.ReplicateAnnotationNames, fromZip.ReplicateAnnotationNames);
        Assert.Equal(
            fromFile.Replicates.Select(r => r.Name + "/" + r.SampleType),
            fromZip.Replicates.Select(r => r.Name + "/" + r.SampleType));
        Assert.Equal(fromFile.EnzymeXml, fromZip.EnzymeXml);
        Assert.Equal(fromFile.SampleFilePaths, fromZip.SampleFilePaths);
    }

    [Fact]
    public void ExtractingProducesTheWholeArchiveAndReturnsTheDocument()
    {
        var dir = TempDir();
        var archive = WriteArchive(dir);

        var document = SharedDocumentArchive.Extract(archive, dir, _out.WriteLine);

        Assert.True(File.Exists(document));
        Assert.Equal("Plate1.sky", Path.GetFileName(document));
        Assert.Equal(Document, File.ReadAllText(document));
        // The .skyd matters as much as the .sky: without the chromatogram cache the report has no
        // areas to export.
        var extractDir = Path.GetDirectoryName(document)!;
        Assert.True(File.Exists(Path.Combine(extractDir, "Plate1.skyd")));
        Assert.True(File.Exists(Path.Combine(extractDir, "library.blib")));
        // Beside the archive, under one folder the user can find and delete.
        Assert.Equal(
            Path.Combine(dir, SharedDocumentArchive.ExtractRootName, "Plate1"),
            extractDir);
    }

    /// <summary>
    /// The whole point of the stamp: 17.4 GB a plate is not something to repeat because a downstream
    /// setting changed. A second run of the same archive reuses what the first extracted.
    /// </summary>
    [Fact]
    public void AnUnchangedArchiveIsNotExtractedTwice()
    {
        var dir = TempDir();
        var archive = WriteArchive(dir);

        var first = new List<string>();
        var a = SharedDocumentArchive.Extract(archive, dir, first.Add);
        var second = new List<string>();
        var b = SharedDocumentArchive.Extract(archive, dir, second.Add);

        Assert.Equal(a, b);
        Assert.Contains(first, m => m.Contains("Extracting", StringComparison.Ordinal));
        Assert.Contains(second, m => m.Contains("Reusing", StringComparison.Ordinal));
        Assert.DoesNotContain(second, m => m.Contains("Extracting", StringComparison.Ordinal));
    }

    /// <summary>
    /// A re-downloaded archive is a different archive. Reusing the old extraction there would analyze
    /// data the user replaced - the same stale-input trap the export cache guards against.
    /// </summary>
    [Fact]
    public void AChangedArchiveIsExtractedAgain()
    {
        var dir = TempDir();
        var archive = WriteArchive(dir);
        SharedDocumentArchive.Extract(archive, dir, _out.WriteLine);

        // Re-download: same path, different content and timestamp.
        File.Delete(archive);
        WriteArchive(dir, entries: new[]
        {
            ("Plate1.sky", Document.Replace("Study_07", "Study_08", StringComparison.Ordinal)),
            ("Plate1.skyd", "a different cache"),
        });
        File.SetLastWriteTimeUtc(archive, DateTime.UtcNow.AddMinutes(5));

        var log = new List<string>();
        var document = SharedDocumentArchive.Extract(archive, dir, log.Add);

        Assert.Contains(log, m => m.Contains("Extracting", StringComparison.Ordinal));
        Assert.Contains("Study_08", File.ReadAllText(document), StringComparison.Ordinal);
    }

    /// <summary>
    /// A killed extraction leaves no stamp, so the next run redoes it rather than trusting a partial
    /// document - and the overwrite means it does not have to clean up first.
    /// </summary>
    [Fact]
    public void APartialExtractionIsNotReused()
    {
        var dir = TempDir();
        var archive = WriteArchive(dir);
        var document = SharedDocumentArchive.Extract(archive, dir, _out.WriteLine);
        var extractDir = Path.GetDirectoryName(document)!;

        // What a stopped run leaves: files, no stamp. Also truncate one, so a reuse would be visible.
        File.Delete(Path.Combine(extractDir, SharedDocumentArchive.StampFileName));
        File.WriteAllText(document, "truncated");

        var log = new List<string>();
        SharedDocumentArchive.Extract(archive, dir, log.Add);

        Assert.Contains(log, m => m.Contains("Extracting", StringComparison.Ordinal));
        Assert.Equal(Document, File.ReadAllText(document));
    }

    [Fact]
    public void AnArchiveWithNoDocumentIsRefusedWithAnExplanation()
    {
        var dir = TempDir();
        var archive = WriteArchive(dir, entries: new[] { ("data.skyd", "x"), ("notes.txt", "y") });

        var ex = Assert.Throws<InvalidDataException>(
            () => SharedDocumentArchive.Extract(archive, dir, _out.WriteLine));

        Assert.Contains("contains no Skyline document", ex.Message, StringComparison.Ordinal);
        // TryRead turns it into a message rather than an exception, so adding one in the GUI explains
        // itself instead of throwing.
        Assert.Null(SkyDocumentInfo.TryRead(archive, _out.WriteLine));
    }

    [Fact]
    public void AnArchiveWithSeveralDocumentsIsRefused()
    {
        var dir = TempDir();
        var archive = WriteArchive(dir, entries: new[] { ("a.sky", Document), ("b.sky", Document) });

        var ex = Assert.Throws<InvalidDataException>(
            () => SharedDocumentArchive.Extract(archive, dir, _out.WriteLine));

        Assert.Contains("contains 2 Skyline documents", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// A zip is user data that arrives over the network, and an entry name is not a trusted path.
    /// "zip slip" - a name climbing out of the extraction folder - is refused rather than followed.
    /// </summary>
    [Fact]
    public void AnEntryThatEscapesTheExtractionFolderIsRefused()
    {
        var dir = TempDir();
        var archive = WriteArchive(dir, entries: new[]
        {
            ("Plate1.sky", Document),
            ("../../escaped.txt", "should never be written"),
        });

        var ex = Assert.Throws<InvalidDataException>(
            () => SharedDocumentArchive.Extract(archive, dir, _out.WriteLine));

        Assert.Contains("outside", ex.Message, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(dir, "escaped.txt")));
        Assert.False(File.Exists(Path.Combine(Path.GetDirectoryName(dir)!, "escaped.txt")));
    }

    /// <summary>The document a closed-document input exports from: the archive's extraction, or itself.</summary>
    [Fact]
    public void AnArchiveInputExportsTheExtractedDocumentNotTheZip()
    {
        var dir = TempDir();
        var archive = WriteArchive(dir);
        var reports = Path.Combine(dir, "out", "skyline-reports");
        Directory.CreateDirectory(reports);

        var resolved = PrismInput.FromClosedDocument(archive)
            .ResolveDocumentForExport(reports, _out.WriteLine, CancellationToken.None);

        Assert.EndsWith(".sky", resolved, StringComparison.OrdinalIgnoreCase);
        Assert.NotEqual(archive, resolved);
        Assert.True(File.Exists(resolved));

        // A loose document is used in place - nothing to extract, nothing copied.
        var loose = Path.Combine(dir, "loose.sky");
        File.WriteAllText(loose, Document);
        Assert.Equal(
            Path.GetFullPath(loose),
            PrismInput.FromClosedDocument(loose)
                .ResolveDocumentForExport(reports, _out.WriteLine, CancellationToken.None));
    }

    /// <summary>
    /// Extraction goes where PRISM_EXTRACT_DIR says, for a Panorama folder on a slow share (or one the
    /// user would rather keep clean).
    /// </summary>
    [Fact]
    public void TheExtractionDirectoryCanBeOverridden()
    {
        var dir = TempDir();
        var elsewhere = TempDir();
        var archive = WriteArchive(dir);

        Environment.SetEnvironmentVariable(SharedDocumentArchive.ExtractDirEnvVar, elsewhere);
        try
        {
            var document = SharedDocumentArchive.Extract(archive, dir, _out.WriteLine);
            Assert.StartsWith(elsewhere, document, StringComparison.OrdinalIgnoreCase);
            Assert.False(Directory.Exists(Path.Combine(dir, SharedDocumentArchive.ExtractRootName)));
        }
        finally
        {
            Environment.SetEnvironmentVariable(SharedDocumentArchive.ExtractDirEnvVar, null);
        }
    }

    /// <summary>
    /// The export memory budget has to size off the DOCUMENT, not the archive. A headless Skyline is
    /// budgeted at roughly twice the <c>.sky</c>; the archive is compressed and holds the chromatogram
    /// cache too, so on the measured Panorama plate (13.7 GB archive, 4.4 GB document) sizing off the
    /// file would over-estimate ~3x - safe, but enough to drop a nine-plate cohort to one export at a
    /// time for nothing.
    /// </summary>
    [Fact]
    public void TheExportBudgetSizesOffTheDocumentNotTheArchive()
    {
        var dir = TempDir();
        // A document that compresses hard, which is what a .sky does - it is XML.
        var body = Document + new string(' ', 200_000);
        var archive = WriteArchive(dir, entries: new[] { ("Plate1.sky", body), ("Plate1.skyd", "x") });

        var input = PrismInput.FromClosedDocument(archive);

        // The entry's own uncompressed length, read independently - not the string's length, which
        // differs by the byte-order mark the writer adds.
        using (var zip = ZipFile.OpenRead(archive))
        {
            Assert.Equal(
                zip.Entries.Single(e => e.Name.EndsWith(".sky", StringComparison.Ordinal)).Length,
                input.DocumentBytes());
        }
        Assert.True(
            new FileInfo(archive).Length < input.DocumentBytes(),
            "the fixture has to compress, or it is not testing anything");

        // A loose document reports its own length, as it always did.
        var loose = Path.Combine(dir, "loose.sky");
        File.WriteAllText(loose, body);
        Assert.Equal(
            new FileInfo(loose).Length, PrismInput.FromClosedDocument(loose).DocumentBytes());

        // A pre-exported report is never loaded into Skyline, so it stays out of the budget.
        var report = Path.Combine(dir, "already.parquet");
        File.WriteAllBytes(report, new byte[4096]);
        Assert.Equal(0, PrismInput.FromReportFile(report).DocumentBytes());
    }

    /// <summary>
    /// Two archives that share a NAME must not share an extraction. Beside the archive the stem is
    /// unique by construction, but a redirected target gathers archives from everywhere - two Panorama
    /// folders both holding "Plate1.sky.zip" is ordinary - and a shared folder there is not a
    /// survivable collision: one input extracts, hands the path to Skyline, and the other re-extracts
    /// over the files that Skyline is reading. The stamp cannot help; it is checked before extracting.
    /// </summary>
    [Fact]
    public void SameNamedArchivesFromDifferentFoldersDoNotShareAnExtraction()
    {
        var one = TempDir();
        var two = TempDir();
        var elsewhere = TempDir();
        var a = WriteArchive(one, entries: new[] { ("Plate1.sky", Document), ("Plate1.skyd", "cache A") });
        var b = WriteArchive(two, entries: new[] { ("Plate1.sky", Document), ("Plate1.skyd", "cache B") });
        Assert.Equal(Path.GetFileName(a), Path.GetFileName(b));   // the premise

        Environment.SetEnvironmentVariable(SharedDocumentArchive.ExtractDirEnvVar, elsewhere);
        try
        {
            var docA = SharedDocumentArchive.Extract(a, null, _out.WriteLine);
            var docB = SharedDocumentArchive.Extract(b, null, _out.WriteLine);

            Assert.NotEqual(Path.GetDirectoryName(docA), Path.GetDirectoryName(docB));
            // ...and neither extraction was overwritten by the other.
            Assert.Equal("cache A", File.ReadAllText(Path.Combine(Path.GetDirectoryName(docA)!, "Plate1.skyd")));
            Assert.Equal("cache B", File.ReadAllText(Path.Combine(Path.GetDirectoryName(docB)!, "Plate1.skyd")));

            // Both are still reused on a second ask, so distinguishing them did not cost the cache.
            var log = new List<string>();
            SharedDocumentArchive.Extract(a, null, log.Add);
            Assert.Contains(log, m => m.Contains("Reusing", StringComparison.Ordinal));
        }
        finally
        {
            Environment.SetEnvironmentVariable(SharedDocumentArchive.ExtractDirEnvVar, null);
        }
    }

    /// <summary>
    /// The document size is remembered per (length, last-write-time), never as a bare answer. A zero
    /// does not read as "unknown" downstream, it reads as "small": the export budget falls back to its
    /// floor and may start four concurrent exports of a document needing ~9 GB each - the memory
    /// exhaustion the budget exists to prevent, from which a starved Skyline does not recover. An
    /// archive added while it is still downloading must not answer 0 for the rest of the session.
    /// </summary>
    [Fact]
    public void TheDocumentSizeIsNotRememberedFromAFailedRead()
    {
        var dir = TempDir();
        var path = Path.Combine(dir, "Plate1.sky.zip");
        File.WriteAllBytes(path, new byte[] { 0x50, 0x4B, 0x03, 0x04 });   // a truncated download

        var input = PrismInput.FromClosedDocument(path);
        Assert.Equal(0, input.DocumentBytes());

        // The download finishes at the same path.
        File.Delete(path);
        var body = Document + new string(' ', 100_000);
        WriteArchive(dir, entries: new[] { ("Plate1.sky", body), ("Plate1.skyd", "x") });

        Assert.True(
            input.DocumentBytes() > 100_000,
            "a completed download must be measured, not answered from the failed read");
    }

    /// <summary>
    /// Where an extraction goes turns on this one question, so it is worth pinning: a network share is
    /// not a local disk.
    ///
    /// <para>Extracting onto the share the archive came from was measured at 3.7 MB/s per archive with
    /// four running - ~15 MB/s together - against 227 MB/s for the same archive extracted to a local
    /// disk. It reads ~12 GB and writes ~17 GB back over one link, so a 12-plate cohort is the
    /// difference between minutes and most of a day, and it first showed up as "it starts 4 files and
    /// never seems to fully uncompress".</para>
    /// </summary>
    [Fact]
    public void ANetworkPathIsNotALocalDisk()
    {
        // The temp directory is where an extraction lands when nothing better is offered, so it had
        // better read as local.
        Assert.True(SharedDocumentArchive.IsOnLocalDisk(Path.GetTempPath()));
        Assert.True(SharedDocumentArchive.IsOnLocalDisk(TempDir()));

        // A UNC path is answered without touching the network - DriveInfo cannot speak for one.
        Assert.False(SharedDocumentArchive.IsOnLocalDisk(@"\\panorama\share\Plate1.sky.zip"));
        Assert.False(SharedDocumentArchive.IsOnLocalDisk(@"\\127.0.0.1\c$\x"));

        // A letter with nothing mounted on it is not somewhere to put 17 GB either.
        Assert.False(SharedDocumentArchive.IsOnLocalDisk(@"Q:\prism"));
        Assert.False(SharedDocumentArchive.IsOnLocalDisk(null));
        Assert.False(SharedDocumentArchive.IsOnLocalDisk("   "));
    }

    /// <summary>
    /// A real Panorama archive, opt-in via <c>PRISM_SKY_ZIP</c>: how long the extraction actually
    /// takes, and that the document inside reads. Skipped in CI - these are 13.7 GB files on a share.
    /// </summary>
    [Fact]
    public void RealArchiveExtractionTiming()
    {
        var archive = Environment.GetEnvironmentVariable("PRISM_SKY_ZIP");
        if (string.IsNullOrWhiteSpace(archive))
        {
            _out.WriteLine("skipped: set PRISM_SKY_ZIP to a .sky.zip (and PRISM_EXTRACT_DIR to a fast disk).");
            return;
        }
        Assert.True(File.Exists(archive), $"no archive at {archive}");

        var info = SkyDocumentInfo.TryRead(archive, _out.WriteLine);
        Assert.NotNull(info);
        _out.WriteLine(
            $"header read from the archive: {info!.Replicates.Count} replicate(s), "
            + $"enzyme {info.PrismEnzyme ?? "(none)"}, "
            + $"tolerance {info.ProductTolerance?.Describe() ?? "(none)"}, "
            + $"annotations {string.Join(", ", info.ReplicateAnnotationNames)}");

        var timer = System.Diagnostics.Stopwatch.StartNew();
        var document = SharedDocumentArchive.Extract(archive, null, _out.WriteLine);
        timer.Stop();

        var bytes = new DirectoryInfo(Path.GetDirectoryName(document)!)
            .EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length);
        _out.WriteLine(
            $"extracted {bytes / (1024.0 * 1024 * 1024):N1} GB in {timer.Elapsed.TotalMinutes:N1} min "
            + $"({bytes / (1024.0 * 1024) / Math.Max(1, timer.Elapsed.TotalSeconds):N0} MB/s)");
        Assert.True(File.Exists(document));
        Assert.NotNull(SkyDocumentInfo.TryRead(document, _out.WriteLine));
    }
}
