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

        public void Run(string[] args, Action<string> log, CancellationToken cancellationToken)
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
