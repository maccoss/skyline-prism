using System;
using System.IO;
using System.Threading.Tasks;
using SkylinePrism.App;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Whether a report already exported from a closed document has to be exported again.
/// </summary>
/// <remarks>
/// <para>Exporting is minutes of Skyline per document, and it used to happen on every run - which also
/// meant every run got a report with a new write time, so the merge stamp moved, so the merge and
/// every stage under it recomputed. Nothing in the pipeline could ever be reused from the tool, however
/// little had changed.</para>
///
/// <para>The stamp is deliberately the document's NAME, size and write time rather than its full path.
/// A cohort on a share is <c>Z:\...</c> from one machine and <c>Y:\...</c> from another, and that
/// difference alone used to force a full re-export and a full recompute of results that would have come
/// out identical.</para>
/// </remarks>
public class ExportReuseTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "prism-export-" + Guid.NewGuid().ToString("N"));

    public ExportReuseTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    /// <summary>The same document reached down two different drive letters is the same document.</summary>
    [Fact]
    public void TheSameDocumentUnderTwoPathsHasOneFingerprint()
    {
        var onZ = Document(Path.Combine(_dir, "as-Z"), "Plate1.sky");
        var onY = Document(Path.Combine(_dir, "as-Y"), "Plate1.sky");

        Assert.NotEqual(onZ, onY); // different paths ...
        Assert.Equal(                // ... same document
            PrismInput.ExportFingerprint(onZ, "Plate1", null, null),
            PrismInput.ExportFingerprint(onY, "Plate1", null, null));
    }

    /// <summary>And a document that has actually changed is not.</summary>
    [Fact]
    public void AnEditedDocumentHasANewFingerprint()
    {
        var doc = Document(Path.Combine(_dir, "edited"), "Plate1.sky");
        var before = PrismInput.ExportFingerprint(doc, "Plate1", null, null);

        File.WriteAllText(doc, "edited, and longer than it was");
        File.SetLastWriteTimeUtc(doc, new DateTime(2026, 2, 2, 0, 0, 0, DateTimeKind.Utc));

        Assert.NotEqual(before, PrismInput.ExportFingerprint(doc, "Plate1", null, null));
    }

    /// <summary>
    /// Everything that decides the exported file's content is in the stamp - the batch label names it,
    /// the metadata report and the batch annotation change what is in it.
    /// </summary>
    [Fact]
    public void TheExportSettingsAreInTheFingerprint()
    {
        var doc = Document(Path.Combine(_dir, "settings"), "Plate1.sky");
        var baseline = PrismInput.ExportFingerprint(doc, "Plate1", null, null);

        Assert.NotEqual(baseline, PrismInput.ExportFingerprint(doc, "Plate2", null, null));
        Assert.NotEqual(baseline, PrismInput.ExportFingerprint(doc, "Plate1", "Replicates", null));
        Assert.NotEqual(baseline, PrismInput.ExportFingerprint(doc, "Plate1", null, "Plate"));
    }

    /// <summary>A document that is not there cannot be stamped, which means export rather than reuse.</summary>
    [Fact]
    public void AMissingDocumentIsNotReusable()
    {
        Assert.Null(PrismInput.ExportFingerprint(Path.Combine(_dir, "gone.sky"), "Plate1", null, null));
        Assert.Null(PrismInput.ExportFingerprint(null, "Plate1", null, null));
        Assert.Null(PrismInput.ExportFingerprint("   ", "Plate1", null, null));
    }

    /// <summary>
    /// The round trip through the run's own stage_cache.json: an export recorded there is offered back
    /// with both files, and stops being offered when one of them is deleted.
    /// </summary>
    [Fact]
    public void ARecordedExportIsOfferedBackUntilItsFilesGo()
    {
        var doc = Document(Path.Combine(_dir, "round-trip"), "Plate1.sky");
        var outputDir = Path.Combine(_dir, "PRISM-Output");
        var reportsDir = Path.Combine(outputDir, "skyline-reports");
        Directory.CreateDirectory(reportsDir);
        var report = Path.Combine(reportsDir, "Plate1.parquet");
        var metadata = Path.Combine(reportsDir, "Plate1.metadata.csv");
        File.WriteAllText(report, "report");
        File.WriteAllText(metadata, "Replicate\r\nA\r\n");

        var stage = PrismInput.ExportStageId("Plate1");
        var fingerprint = PrismInput.ExportFingerprint(doc, "Plate1", null, null)!;
        StageCache.Load(outputDir).Record(stage, fingerprint, report, metadata);

        // A second machine reads the same directory: the entry is there and both files are present.
        Assert.True(StageCache.Load(outputDir).CanReuse(stage, fingerprint));
        var outputs = StageCache.Load(outputDir).OutputsOf(stage);
        Assert.Equal(2, outputs.Count);
        Assert.Equal("Plate1.parquet", Path.GetFileName(outputs[0]));
        Assert.Equal("Plate1.metadata.csv", Path.GetFileName(outputs[1]));

        // Delete the report and the export is no longer something to stand on.
        File.Delete(report);
        Assert.False(StageCache.Load(outputDir).CanReuse(stage, fingerprint));
    }

    /// <summary>
    /// Inputs are exported concurrently, and no worker's entry may be lost.
    /// </summary>
    /// <remarks>
    /// <c>stage_cache.json</c> is read whole and written whole. Each worker holding its own instance
    /// meant each wrote back a snapshot taken before the others recorded, so the last write erased the
    /// rest - and the next run re-exported whichever documents lost, silently, which is the failure
    /// this whole change exists to remove.
    /// </remarks>
    [Fact]
    public void ConcurrentExportsDoNotEraseEachOther()
    {
        var outputDir = Path.Combine(_dir, "concurrent", "PRISM-Output");
        var reportsDir = Path.Combine(outputDir, "skyline-reports");
        Directory.CreateDirectory(reportsDir);

        const int plates = 24;
        var fingerprints = new string[plates];
        Parallel.For(0, plates, i =>
        {
            var label = "Plate" + i;
            var report = Path.Combine(reportsDir, label + ".parquet");
            File.WriteAllText(report, "report " + i);
            fingerprints[i] = "stamp-" + i;
            PrismInput.RecordExport(
                outputDir, PrismInput.ExportStageId(label), fingerprints[i],
                new ExportedReports(report, true, null, "doc.sky", label));
        });

        var cache = StageCache.Load(outputDir);
        for (var i = 0; i < plates; i++)
        {
            Assert.True(
                cache.CanReuse(PrismInput.ExportStageId("Plate" + i), fingerprints[i]),
                $"Plate{i}'s export entry was lost");
        }
    }

    /// <summary>A document file with fixed content and write time, under its own directory.</summary>
    private static string Document(string dir, string name)
    {
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, name);
        File.WriteAllText(path, "<srm_settings/>");
        File.SetLastWriteTimeUtc(path, new DateTime(2026, 1, 1, 12, 0, 0, DateTimeKind.Utc));
        return path;
    }
}
