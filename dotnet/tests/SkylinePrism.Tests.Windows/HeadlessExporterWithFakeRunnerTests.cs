using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Xml.Linq;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Drives <see cref="HeadlessSkylineExporter"/> against a FAKE <see cref="ISkylineCommandRunner"/>, so the
/// export decisions are tested without Skyline installed: the parquet-first / CSV-fallback choice, the
/// per-document file naming that carries batch labels into the merge, and the generated Replicates report.
/// This is the closed-document counterpart to <see cref="SkylineReportDriverTests"/>, which does the same
/// for the live RPC path.
/// </summary>
public class HeadlessExporterWithFakeRunnerTests
{
    /// <summary>
    /// Stands in for Skyline: records each invocation and writes whatever <see cref="OnRun"/> decides,
    /// so a test can simulate a working parquet writer, a broken one, or an outright failure.
    /// </summary>
    private sealed class FakeRunner : ISkylineCommandRunner
    {
        public FakeRunner(bool supportsParquet = true) => SupportsParquet = supportsParquet;

        public string Description => "fake runner";
        public bool SupportsParquet { get; }

        public readonly List<string[]> Invocations = new();

        /// <summary>(args, reportFile) -> write the output, or throw to simulate a Skyline error.</summary>
        public Action<string[], string>? OnRun;

        public void Run(string[] args, Action<string> log, CancellationToken cancellationToken,
            TimeSpan? timeout = null)
        {
            Invocations.Add(args);
            var file = args.First(a => a.StartsWith("--report-file=", StringComparison.Ordinal))
                ["--report-file=".Length..];
            OnRun?.Invoke(args, file);
        }

        /// <summary>The report file of invocation <paramref name="i"/>.</summary>
        public string ReportFile(int i) => Invocations[i]
            .First(a => a.StartsWith("--report-file=", StringComparison.Ordinal))["--report-file=".Length..];

        /// <summary>The .skyr installed by invocation <paramref name="i"/>, if any.</summary>
        public string? ReportAdd(int i) => Invocations[i]
            .FirstOrDefault(a => a.StartsWith("--report-add=", StringComparison.Ordinal))?["--report-add=".Length..];
    }

    private static void WriteParquet(string path)
    {
        var b = new byte[16];
        "PAR1"u8.CopyTo(b);
        "PAR1"u8.CopyTo(b.AsSpan(12));
        File.WriteAllBytes(path, b);
    }

    private static void WriteCsv(string path) => File.WriteAllText(path, "Replicate,SampleType\nR1,Standard\n");

    private static string TempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "prism_headless_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    /// <summary>A minimal .sky with one replicate annotation, for the metadata-report generation.</summary>
    private static string WriteSky(string dir, string name = "PlateA")
    {
        var path = Path.Combine(dir, name + ".sky");
        File.WriteAllText(path, """
<?xml version="1.0" encoding="utf-8"?>
<srm_settings format_version="25.1">
  <settings_summary name="Default">
    <peptide_settings><enzyme name="Trypsin" cut="KR" no_cut="P" sense="C" /></peptide_settings>
    <data_settings document_guid="g">
      <annotation name="Plate" targets="replicate" type="text" />
    </data_settings>
    <measured_results>
      <replicate name="Ref_01" sample_type="standard"><annotation name="Plate">P1</annotation></replicate>
    </measured_results>
  </settings_summary>
</srm_settings>
""");
        return path;
    }

    /// <summary>
    /// Re-exporting a closed document that has not changed is the most expensive wasted work in a
    /// re-run - a large cohort's transition report is tens of GB - so an unchanged document skips it.
    /// Safe for a CLOSED document specifically: the file cannot change without its size or
    /// last-write-time changing.
    /// </summary>
    [Fact]
    public void Export_ReusesThePreviousExport_WhenTheDocumentIsUnchanged()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        var runner = new FakeRunner(supportsParquet: false) { OnRun = (_, file) => WriteCsv(file) };

        var first = new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");
        var callsAfterFirst = runner.Invocations.Count;
        Assert.True(callsAfterFirst > 0);

        var second = new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");

        Assert.Equal(callsAfterFirst, runner.Invocations.Count); // Skyline was not run again
        Assert.Equal(first.InputPath, second.InputPath);
        Assert.Equal(first.ReplicatesCsv, second.ReplicatesCsv);
    }

    [Fact]
    public void Export_DoesNotReuse_WhenTheDocumentChanged()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        var runner = new FakeRunner(supportsParquet: false) { OnRun = (_, file) => WriteCsv(file) };

        new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");
        var callsAfterFirst = runner.Invocations.Count;

        // Re-integrated, re-saved: same path, different content and timestamp.
        File.AppendAllText(sky, "\n<!-- edited -->\n");
        File.SetLastWriteTimeUtc(sky, DateTime.UtcNow.AddMinutes(1));

        new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");

        Assert.True(runner.Invocations.Count > callsAfterFirst, "a changed document must be re-exported");
    }

    [Fact]
    public void Export_DoesNotReuse_WhenTheBatchAnnotationChanged()
    {
        // The annotation decides a column of the metadata report, so the previous export is the wrong
        // shape even though the document is untouched.
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        var runner = new FakeRunner(supportsParquet: false) { OnRun = (_, file) => WriteCsv(file) };

        new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA", "Plate");
        var callsAfterFirst = runner.Invocations.Count;

        new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA", "Condition");

        Assert.True(runner.Invocations.Count > callsAfterFirst);
    }

    [Fact]
    public void Export_DoesNotReuse_WhenTheExportedFileIsGone()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        var runner = new FakeRunner(supportsParquet: false) { OnRun = (_, file) => WriteCsv(file) };

        var first = new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");
        var callsAfterFirst = runner.Invocations.Count;
        File.Delete(first.InputPath);

        new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");

        Assert.True(runner.Invocations.Count > callsAfterFirst);
        Assert.True(File.Exists(first.InputPath));
    }

    [Fact]
    public void Export_UsesParquet_WhenTheRunnerSupportsItAndItIsValid()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var runner = new FakeRunner(supportsParquet: true)
        {
            OnRun = (_, file) =>
            {
                if (file.EndsWith(".parquet", StringComparison.Ordinal)) WriteParquet(file);
                else WriteCsv(file);
            },
        };

        var result = new HeadlessSkylineExporter(runner, reportsDir: dir)
            .Export(sky, Path.Combine(dir, "out"), "PlateA");

        Assert.True(result.InputIsParquet);
        Assert.EndsWith("PlateA.parquet", result.InputPath);
        // The parquet attempt must NOT pass --report-format, or Skyline writes text into the .parquet.
        Assert.DoesNotContain(runner.Invocations[0], a => a.StartsWith("--report-format", StringComparison.Ordinal));
    }

    /// <summary>
    /// A CSV left by a previous run that fell back is superseded once a parquet export succeeds. They
    /// are alternatives for the same input and only one is ever read, so keeping the loser wastes real
    /// space - on a real cohort, 14.8 GB of CSV beside the 695 MB parquet that replaced it - and leaves
    /// two files a user can pick between by hand, one of them stale.
    /// </summary>
    [Fact]
    public void Export_RemovesACsvSupersededByASuccessfulParquet()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        Directory.CreateDirectory(work);

        // Left behind by an earlier run whose parquet attempt failed.
        var stale = Path.Combine(work, "PlateA.csv");
        File.WriteAllText(stale, "Replicate,SampleType" + Environment.NewLine + "R1,Standard");

        var runner = new FakeRunner(supportsParquet: true)
        {
            OnRun = (_, file) =>
            {
                if (file.EndsWith(".parquet", StringComparison.Ordinal)) WriteParquet(file);
                else WriteCsv(file);
            },
        };

        var result = new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");

        Assert.True(result.InputIsParquet);
        Assert.False(File.Exists(stale), "the superseded CSV should have been removed");
    }

    [Fact]
    public void Export_FallsBackToCsv_WhenTheParquetAttemptProducesGarbage()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        // Simulates the broken SkylineCmd parquet writer that "succeeds" but leaves a non-parquet stub.
        var runner = new FakeRunner(supportsParquet: true) { OnRun = (_, file) => WriteCsv(file) };

        var result = new HeadlessSkylineExporter(runner, reportsDir: dir)
            .Export(sky, Path.Combine(dir, "out"), "PlateA");

        Assert.False(result.InputIsParquet);
        Assert.EndsWith("PlateA.csv", result.InputPath);
        Assert.False(File.Exists(Path.Combine(dir, "out", "PlateA.parquet"))); // stub cleaned up
        Assert.Contains("--report-format=csv", runner.Invocations[1]);
    }

    [Fact]
    public void Export_FallsBackToCsv_WhenTheParquetAttemptThrows()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var runner = new FakeRunner(supportsParquet: true)
        {
            OnRun = (_, file) =>
            {
                if (file.EndsWith(".parquet", StringComparison.Ordinal))
                    throw new InvalidOperationException("Could not load file or assembly 'Parquet'");
                WriteCsv(file);
            },
        };

        var result = new HeadlessSkylineExporter(runner, reportsDir: dir)
            .Export(sky, Path.Combine(dir, "out"), "PlateA");

        Assert.False(result.InputIsParquet);
        Assert.True(File.Exists(result.InputPath));
    }

    [Fact]
    public void Export_SkipsTheParquetAttemptEntirely_WhenTheRunnerCannotDoIt()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var runner = new FakeRunner(supportsParquet: false) { OnRun = (_, file) => WriteCsv(file) };

        var result = new HeadlessSkylineExporter(runner, reportsDir: dir)
            .Export(sky, Path.Combine(dir, "out"), "PlateA");

        Assert.False(result.InputIsParquet);
        // No wasted document load: straight to CSV (transition report + metadata = 2 invocations).
        Assert.Equal(2, runner.Invocations.Count);
        Assert.DoesNotContain(runner.Invocations, inv => inv.Any(a => a.EndsWith(".parquet", StringComparison.Ordinal)));
    }

    [Fact]
    public void Export_ThrowsWhenNoReportIsProducedAtAll()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var runner = new FakeRunner(supportsParquet: false) { OnRun = (_, _) => { /* write nothing */ } };

        var ex = Assert.Throws<InvalidOperationException>(
            () => new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, Path.Combine(dir, "out"), "PlateA"));
        Assert.Contains("did not produce a PRISM report", ex.Message);
    }

    /// <summary>
    /// The worst failure this class has produced: an export that never reached Skyline reported the file
    /// a PREVIOUS export had left at the same path as its own output. It is valid parquet, so the PAR1
    /// check passed it, and a 2-plate cohort was analyzed against a report exported 25 days earlier from a
    /// since-re-integrated document - logged as "Exported ... bytes, parquet", with no warning anywhere.
    /// A stale export may only be adopted by TryReuseExport, which checks the document first.
    /// </summary>
    [Fact]
    public void Export_DoesNotAdoptAStaleParquet_WhenTheExportFails()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        Directory.CreateDirectory(work);

        // Left by an earlier, successful export of an older version of the document.
        var stale = Path.Combine(work, "PlateA.parquet");
        WriteParquet(stale);

        // Skyline never starts, so neither the parquet attempt nor the CSV fallback writes anything.
        var runner = new FakeRunner(supportsParquet: true)
        {
            OnRun = (_, _) => throw new IOException("Pipe is broken."),
        };

        var staleBytes = File.ReadAllBytes(stale);

        // Whatever the failure is, it must surface. The CSV fallback's own exception is what escapes here
        // (it is fatal by design); what matters is that nothing is returned - that throw IS the
        // non-adoption, because there is no path handed back for the merge to read.
        Assert.ThrowsAny<Exception>(
            () => new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA"));

        // The stale file is deliberately still there. Not adopting it and destroying it are different
        // things, and this test used to demand the second: the export deleted the destination up front,
        // so a re-export that then failed threw away a perfectly usable cached export - the single most
        // expensive step of a re-run - over a transient error. Skyline now writes to a sidecar, so a
        // failure cannot reach the destination at all. It can still only be picked up again through
        // TryReuseExport, which checks the document and the tool version first.
        Assert.True(File.Exists(stale), "a failed export must not destroy the previous one");
        Assert.Equal(staleBytes, File.ReadAllBytes(stale));

        // ...and the failed attempt leaves nothing behind for the next run to trip over.
        Assert.Empty(Directory.GetFiles(work, HeadlessSkylineExporter.SidecarPrefix + "*"));
    }

    /// <summary>
    /// A sidecar from a run that was KILLED - the one case the cleanup paths above cannot cover, because
    /// none of them got to run. Nothing else would ever remove it: it is named after a GUID no later run
    /// knows, and it is a full report, so a stopped export of a real cohort strands gigabytes. Worse, it
    /// is named after its destination, so ".prism-partial-....metadata.csv" matches the "*.metadata.csv"
    /// glob the GUI uses to find every document's replicate report - a partial file would be read as a
    /// document's metadata under a bogus label.
    /// </summary>
    [Fact]
    public void Export_SweepsSidecarsLeftByAKilledRun()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        Directory.CreateDirectory(work);

        // What a killed run leaves: named for this label, never promoted, never deleted.
        var orphanReport = Path.Combine(work, HeadlessSkylineExporter.SidecarPrefix + "deadbeef.PlateA.parquet");
        var orphanMetadata = Path.Combine(work, HeadlessSkylineExporter.SidecarPrefix + "deadbeef.PlateA.metadata.csv");
        WriteParquet(orphanReport);
        WriteCsv(orphanMetadata);
        // Another document's orphan must survive - it may belong to an export running right now.
        var otherDocument = Path.Combine(work, HeadlessSkylineExporter.SidecarPrefix + "cafebabe.PlateB.parquet");
        WriteParquet(otherDocument);

        var runner = new FakeRunner(supportsParquet: true)
        {
            OnRun = (_, file) =>
            {
                if (file.EndsWith(".csv", StringComparison.Ordinal)) WriteCsv(file);
                else WriteParquet(file);
            },
        };

        new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");

        Assert.False(File.Exists(orphanReport), "a killed run's report sidecar should be swept");
        Assert.False(File.Exists(orphanMetadata), "a killed run's metadata sidecar should be swept");
        Assert.True(File.Exists(otherDocument), "another document's sidecar is not ours to delete");
    }

    /// <summary>
    /// The sidecar naming is what makes the sweep and the GUI's glob exclusion possible, so pin it:
    /// a sidecar must be recognizable as one from its name alone.
    /// </summary>
    [Fact]
    public void IsSidecar_RecognizesAnUnfinishedExportButNotAFinishedOne()
    {
        var side = @"C:\x\" + HeadlessSkylineExporter.SidecarPrefix;
        Assert.True(HeadlessSkylineExporter.IsSidecar(side + "deadbeef.PlateA.metadata.csv"));
        Assert.True(HeadlessSkylineExporter.IsSidecar(side + "deadbeef.PlateA.parquet"));
        Assert.False(HeadlessSkylineExporter.IsSidecar(@"C:\x\PlateA.metadata.csv"));
        Assert.False(HeadlessSkylineExporter.IsSidecar(@"C:\x\PlateA.parquet"));

        // NO LEADING DOT. An SMB server maps the Unix dot-file convention onto the DOS hidden attribute,
        // and the promoted export inherits it: measured on a real share, a 1.4 GB parquet and its
        // metadata arrived at their final names as "Hidden, Archive" while the files PRISM writes
        // directly were plain "Archive". The pipeline still finds them; the user opening the folder does
        // not, and concludes the export failed.
        Assert.False(HeadlessSkylineExporter.SidecarPrefix.StartsWith('.'));
    }

    /// <summary>
    /// The point of the sidecar, stated positively: a re-export that fails leaves the previous export
    /// not merely present but REUSABLE. Deleting up front made a transient Skyline failure cost a full
    /// re-export of a multi-GB document on the next attempt as well.
    /// </summary>
    [Fact]
    public void Export_LeavesTheCachedExportReusable_WhenAReExportFails()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");

        var fail = false;
        var runner = new FakeRunner(supportsParquet: true)
        {
            OnRun = (_, file) =>
            {
                if (fail)
                    throw new IOException("Pipe is broken.");
                if (file.EndsWith(".metadata.csv", StringComparison.Ordinal))
                    WriteCsv(file);
                else
                    WriteParquet(file);
            },
        };

        // A good export, cached.
        var first = new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");
        Assert.NotNull(first.ReplicatesCsv);
        var cached = File.ReadAllBytes(first.InputPath);

        // Something makes the document look changed, so the cache is bypassed - and the re-export fails.
        File.SetLastWriteTimeUtc(sky, File.GetLastWriteTimeUtc(sky).AddMinutes(5));
        fail = true;
        Assert.ThrowsAny<Exception>(
            () => new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA"));

        // The previous export survived the failed attempt intact.
        Assert.True(File.Exists(first.InputPath));
        Assert.Equal(cached, File.ReadAllBytes(first.InputPath));
        Assert.Empty(Directory.GetFiles(work, HeadlessSkylineExporter.SidecarPrefix + "*"));
    }

    /// <summary>
    /// The same trap without an exception to notice: a runner that "succeeds" but writes nothing, over a
    /// stale file. Freshness is what decides, not the runner's silence.
    /// </summary>
    [Fact]
    public void Export_DoesNotAdoptAStaleCsv_WhenTheRunnerWritesNothing()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        Directory.CreateDirectory(work);
        WriteCsv(Path.Combine(work, "PlateA.csv"));

        var runner = new FakeRunner(supportsParquet: false) { OnRun = (_, _) => { /* write nothing */ } };

        var ex = Assert.Throws<InvalidOperationException>(
            () => new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA"));
        Assert.Contains("did not produce a PRISM report", ex.Message);
    }

    /// <summary>
    /// Metadata has its own copy of the trap, and adopting a stale one is quieter than it looks: sample
    /// types and batches would come from an older document instead of being inferred from names, which is
    /// a different reference/QC split and therefore different ComBat - with the log saying "Exported".
    /// </summary>
    [Fact]
    public void Export_DoesNotAdoptStaleMetadata_WhenOnlyTheMetadataExportFails()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        Directory.CreateDirectory(work);
        WriteCsv(Path.Combine(work, "PlateA.metadata.csv"));

        var runner = new FakeRunner(supportsParquet: true)
        {
            OnRun = (_, file) =>
            {
                if (file.EndsWith(".metadata.csv", StringComparison.Ordinal))
                    throw new IOException("Pipe is broken.");
                WriteParquet(file);
            },
        };

        var result = new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");

        Assert.True(result.InputIsParquet); // the transition report itself was fine
        Assert.Null(result.ReplicatesCsv);
    }

    /// <summary>
    /// An export missing its metadata must not be cached, or one transient failure becomes permanent: the
    /// reuse check only re-exports when a STAMPED metadata file has gone missing, so a stamp recording no
    /// metadata at all reads as "correctly has none" and is honoured for every later run of the unchanged
    /// document - inferring sample types from replicate names forever, which is a different reference/QC
    /// split and therefore different ComBat, with only "Reusing the previous export" in the log.
    /// </summary>
    [Fact]
    public void Export_DoesNotCacheAnExportWhoseMetadataFailed()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        var failMetadata = true;
        var runner = new FakeRunner(supportsParquet: true)
        {
            OnRun = (_, file) =>
            {
                if (file.EndsWith(".metadata.csv", StringComparison.Ordinal))
                {
                    if (failMetadata)
                        throw new IOException("Pipe is broken.");
                    WriteCsv(file);
                    return;
                }
                WriteParquet(file);
            },
        };

        var first = new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");
        Assert.Null(first.ReplicatesCsv);
        var callsAfterFirst = runner.Invocations.Count;

        // Same unchanged document, and metadata now works. The run must export again rather than reuse.
        failMetadata = false;
        var second = new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");

        Assert.True(runner.Invocations.Count > callsAfterFirst, "the failed export should not be reused");
        Assert.NotNull(second.ReplicatesCsv);
    }

    /// <summary>
    /// A THROWN failure is transient - Skyline slow to start, stalled, killed - so the parquet export is
    /// retried rather than abandoned. Falling back on the first throw is what turned a startup timeout on
    /// an 11 GB document into a 21.39 GB CSV that took ~87 minutes, when the very next launch (the CSV
    /// attempt itself) succeeded and would have produced a 1.28 GB parquet in ~21.
    /// </summary>
    [Fact]
    public void Export_RetriesParquet_WhenAnAttemptThrows()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        var parquetAttempts = 0;

        var runner = new FakeRunner(supportsParquet: true)
        {
            OnRun = (_, file) =>
            {
                if (file.EndsWith(".parquet", StringComparison.Ordinal))
                {
                    // Fails the way a startup timeout does, then succeeds - the observed pattern.
                    if (++parquetAttempts == 1)
                        throw new TimeoutException("Skyline-daily did not start within 180s");
                    WriteParquet(file);
                    return;
                }
                WriteCsv(file);
            },
        };

        var result = new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");

        Assert.True(result.InputIsParquet, "a transient failure must not downgrade the run to CSV");
        Assert.Equal(2, parquetAttempts);
        Assert.False(File.Exists(Path.Combine(work, "PlateA.csv")), "no CSV should have been written");
    }

    /// <summary>
    /// The OTHER failure, which must NOT be retried: the run completes cleanly but the file is not
    /// parquet. That is SkylineCmd, whose config lacks the Parquet.Net bindings - the reason the CSV
    /// fallback exists at all - and no number of retries changes it.
    /// </summary>
    [Fact]
    public void Export_FallsBackImmediately_WhenTheRunSucceedsButIsNotParquet()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        var parquetAttempts = 0;

        var runner = new FakeRunner(supportsParquet: true)
        {
            OnRun = (_, file) =>
            {
                if (file.EndsWith(".parquet", StringComparison.Ordinal))
                    parquetAttempts++;
                WriteCsv(file);   // never parquet, however often it is asked
            },
        };

        var result = new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");

        Assert.False(result.InputIsParquet);
        Assert.Equal(1, parquetAttempts);   // no retry: retrying cannot help this failure
    }

    /// <summary>
    /// The metadata export is retried for a sharper reason than the transition report: metadata pairs 1:1
    /// with the inputs, so one document failing here downgrades the WHOLE run to name-based sample typing -
    /// which on replicates named "001-A_F4_064" resolves to 0 reference, 0 QC and no control-CV table in
    /// the QC report.
    /// </summary>
    [Fact]
    public void Export_RetriesTheMetadataExport_WhenAnAttemptThrows()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var work = Path.Combine(dir, "out");
        var metadataAttempts = 0;

        var runner = new FakeRunner(supportsParquet: true)
        {
            OnRun = (_, file) =>
            {
                if (file.EndsWith(".metadata.csv", StringComparison.Ordinal))
                {
                    if (++metadataAttempts == 1)
                        throw new TimeoutException("Skyline-daily did not start within 180s");
                    WriteCsv(file);
                    return;
                }
                WriteParquet(file);
            },
        };

        var result = new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, work, "PlateA");

        Assert.Equal(2, metadataAttempts);
        Assert.NotNull(result.ReplicatesCsv);
    }

    [Fact]
    public void Export_NamesFilesAfterTheDocumentLabel_SoTheMergeDerivesTheRightBatch()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var runner = new FakeRunner(supportsParquet: false) { OnRun = (_, file) => WriteCsv(file) };

        var result = new HeadlessSkylineExporter(runner, reportsDir: dir)
            .Export(sky, Path.Combine(dir, "out"), "Plate_07");

        // DuckDbMerge takes each input's Batch / Source Document from the FILE STEM, so the stem must be
        // the label - otherwise several documents would collide on one batch.
        Assert.Equal("Plate_07", Path.GetFileNameWithoutExtension(result.InputPath));
        Assert.Equal("Plate_07", result.DocumentLabel);
        Assert.EndsWith("Plate_07.metadata.csv", result.ReplicatesCsv);
    }

    [Fact]
    public void Export_DefaultsTheLabelToTheDocumentName()
    {
        var dir = TempDir();
        var sky = WriteSky(dir, "MyStudy");
        var runner = new FakeRunner(supportsParquet: false) { OnRun = (_, file) => WriteCsv(file) };

        var result = new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, Path.Combine(dir, "out"));

        Assert.Equal("MyStudy", result.DocumentLabel);
    }

    [Fact]
    public void Export_GeneratesAReplicatesReportCarryingTheDocumentsAnnotations()
    {
        var dir = TempDir();
        var sky = WriteSky(dir); // declares a replicate annotation "Plate"
        var runner = new FakeRunner(supportsParquet: false) { OnRun = (_, file) => WriteCsv(file) };

        new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, Path.Combine(dir, "out"), "PlateA");

        // The second invocation exports the metadata report from a .skyr we generated for THIS document.
        var skyr = runner.ReportAdd(1);
        Assert.NotNull(skyr);
        var columns = XDocument.Load(skyr!).Root!.Element("view")!
            .Elements("column").Select(c => (string)c.Attribute("name")!).ToList();
        // Quoted - the bare annotation_Plate form is rejected by Skyline's property-path parser.
        Assert.Contains("\"annotation_Plate\"", columns);
        Assert.Contains("SampleType", columns);
        Assert.Contains("--report-name=PRISM-Replicates", runner.Invocations[1]);
    }

    [Fact]
    public void Export_ForcesTheRequestedBatchAnnotationIntoTheReport()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var runner = new FakeRunner(supportsParquet: false) { OnRun = (_, file) => WriteCsv(file) };

        new HeadlessSkylineExporter(runner, reportsDir: dir)
            .Export(sky, Path.Combine(dir, "out"), "PlateA", batchAnnotation: "RunBlock");

        var columns = XDocument.Load(runner.ReportAdd(1)!).Root!.Element("view")!
            .Elements("column").Select(c => (string)c.Attribute("name")!).ToList();
        // Not declared in the .sky, but the user named it as the batch column - it must still be exported.
        Assert.Contains("\"annotation_RunBlock\"", columns);
    }

    [Fact]
    public void Export_SucceedsWithoutMetadata_WhenTheReplicatesReportFails()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var runner = new FakeRunner(supportsParquet: false)
        {
            OnRun = (args, file) =>
            {
                if (args.Any(a => a.Contains("PRISM-Replicates", StringComparison.Ordinal)))
                    throw new InvalidOperationException("report rejected");
                WriteCsv(file);
            },
        };

        // Metadata is optional: sample types can be inferred from replicate names, so a metadata failure
        // must not lose the transition report the user waited for.
        var result = new HeadlessSkylineExporter(runner, reportsDir: dir)
            .Export(sky, Path.Combine(dir, "out"), "PlateA");

        Assert.True(File.Exists(result.InputPath));
        Assert.Null(result.ReplicatesCsv);
    }

    [Fact]
    public void Export_NeverPassesSave_OnAnyInvocation()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var runner = new FakeRunner(supportsParquet: true)
        {
            OnRun = (_, file) =>
            {
                if (file.EndsWith(".parquet", StringComparison.Ordinal)) WriteParquet(file);
                else WriteCsv(file);
            },
        };

        new HeadlessSkylineExporter(runner, reportsDir: dir).Export(sky, Path.Combine(dir, "out"), "PlateA");

        // The user's document is read-only here; a stray --save would rewrite it.
        Assert.All(runner.Invocations,
            inv => Assert.DoesNotContain(inv, a => a.StartsWith("--save", StringComparison.OrdinalIgnoreCase)));
        Assert.Equal(File.ReadAllText(sky), File.ReadAllText(sky)); // sanity: untouched
    }

    [Fact]
    public void Export_OmitsReportAdd_WhenTheBundledSkyrIsMissing()
    {
        var dir = TempDir();
        var sky = WriteSky(dir);
        var runner = new FakeRunner(supportsParquet: false) { OnRun = (_, file) => WriteCsv(file) };

        // reportsDir points somewhere with no Skyline-PRISM.skyr.
        new HeadlessSkylineExporter(runner, reportsDir: Path.Combine(dir, "no-reports"))
            .Export(sky, Path.Combine(dir, "out"), "PlateA");

        // Falls back to the "PRISM" report already in Skyline's settings; an empty --report-add would
        // abort the whole invocation.
        Assert.Null(runner.ReportAdd(0));
        Assert.Contains("--report-name=PRISM", runner.Invocations[0]);
    }

    [Fact]
    public void Export_ThrowsForAMissingDocument()
    {
        var dir = TempDir();
        var runner = new FakeRunner();
        Assert.Throws<FileNotFoundException>(() => new HeadlessSkylineExporter(runner, reportsDir: dir)
            .Export(Path.Combine(dir, "nope.sky"), Path.Combine(dir, "out"), "PlateA"));
    }
}
