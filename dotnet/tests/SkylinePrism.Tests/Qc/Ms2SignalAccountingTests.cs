using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.Visualization;
using SkylinePrism.Tests.TestSupport;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// A real pipeline run of the committed cohort fixture, produced ONCE and shared. The accounting reads
/// peptides_rollup / corrected_peptides for peptide identity, so it needs an output directory the
/// pipeline actually wrote - hand-assembled files would test the reader against a shape the pipeline
/// does not produce.
///
/// <para>Seeded exactly as <c>CohortRegressionTests</c> does: the committed merged_data is put where
/// Stage 1 would have written it, with a matching cache entry, so the merge is reused and the empty
/// placeholder inputs are never opened.</para>
/// </summary>
public sealed class CohortRunFixture : IDisposable
{
    private const long FixtureRows = 1_129_728;
    private const string FixturePartitionKey = "Peptide";

    public string Dir { get; }

    public CohortRunFixture()
    {
        Dir = Path.Combine(Path.GetTempPath(), "prism_ms2acct_" + Guid.NewGuid().ToString("N"));
        var cohort = Fixtures.Path2("cohort");

        var config = new PrismConfig();
        config.QcReport.Enabled = false;    // the report is generated per test, with its own settings

        Directory.CreateDirectory(Dir);
        var mergedDir = Path.Combine(Dir, "merged_data");
        foreach (var bucket in Directory.GetDirectories(cohort, "_pep_bucket=*"))
        {
            var target = Path.Combine(mergedDir, Path.GetFileName(bucket));
            Directory.CreateDirectory(target);
            foreach (var f in Directory.GetFiles(bucket))
                File.Copy(f, Path.Combine(target, Path.GetFileName(f)), overwrite: true);
        }

        var inputs = new[] { "Batch1", "Batch2" }
            .Select(b => Path.Combine(Dir, b + ".parquet"))
            .ToArray();
        foreach (var p in inputs)
            File.WriteAllBytes(p, Array.Empty<byte>());

        var fingerprint = SourceFingerprint.Compute(inputs)
            + "|" + StageDependencies.Values(StageDependencies.Merge, config);
        File.WriteAllText(
            Path.Combine(Dir, "merged_data.cache.json"),
            JsonSerializer.Serialize(
                new SourceFingerprint.CacheEntry(fingerprint, FixtureRows, FixturePartitionKey)));

        PrismPipeline.Run(
            inputs, Dir, config,
            new[]
            {
                Path.Combine(cohort, "Batch1.metadata.csv"),
                Path.Combine(cohort, "Batch2.metadata.csv"),
            });
    }

    /// <summary>A private copy, so a test that writes results cannot disturb another.</summary>
    public string CopyForWriting()
    {
        var copy = Dir + "_" + Guid.NewGuid().ToString("N")[..8];
        Directory.CreateDirectory(copy);
        foreach (var dir in Directory.EnumerateDirectories(Dir, "*", SearchOption.AllDirectories))
            Directory.CreateDirectory(dir.Replace(Dir, copy, StringComparison.Ordinal));
        foreach (var file in Directory.EnumerateFiles(Dir, "*", SearchOption.AllDirectories))
            File.Copy(file, file.Replace(Dir, copy, StringComparison.Ordinal), overwrite: true);
        return copy;
    }

    public void Dispose() => TryDelete(Dir);

    internal static void TryDelete(string dir)
    {
        try
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
        catch (IOException)
        {
            // A parquet handle still held by the reader must not fail a test that already passed.
        }
    }
}

/// <summary>
/// The whole-cohort accounting: one pass over a real run of the committed fixture, persisted, read
/// back, and plotted. The properties asserted are the ones a reader of the plot depends on - totals
/// nest, the union never exceeds the sum, and a cached read reproduces what was computed.
/// </summary>
public class Ms2SignalAccountingTests : IClassFixture<CohortRunFixture>, IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly CohortRunFixture _cohort;
    private readonly List<string> _scratch = new();

    public Ms2SignalAccountingTests(ITestOutputHelper output, CohortRunFixture cohort)
    {
        _out = output;
        _cohort = cohort;
    }

    public void Dispose()
    {
        foreach (var dir in _scratch)
            CohortRunFixture.TryDelete(dir);
    }

    private static readonly ProductMassTolerance Ppm10 = ProductMassTolerance.Parse("centroided", "10")!;

    /// <summary>A private copy of the completed run, cleaned up with the test.</summary>
    private string SeedCohort()
    {
        var dir = _cohort.CopyForWriting();
        _scratch.Add(dir);
        return dir;
    }

    /// <summary>
    /// One pass covers every replicate. Sums are per replicate and each union is at most its own sum -
    /// the inequality that is the whole correction, asserted on real data rather than a synthetic case.
    /// </summary>
    [Fact]
    public void ComputesEveryReplicateInOnePass()
    {
        var dir = SeedCohort();

        var result = Ms2SignalAccounting.Compute(
            dir, IsolationScheme.AstralDefault(), Ppm10,
            Array.Empty<ProteinList>(), null, _out.WriteLine);

        Assert.NotNull(result);
        Assert.True(result!.Rows.Count > 1, "the fixture cohort has more than one replicate");

        foreach (var row in result.Rows)
        {
            Assert.True(row.AssignedArea <= row.SummedArea + 1e-9,
                $"{row.Sample}: union {row.AssignedArea} exceeded the sum {row.SummedArea}");
            Assert.True(row.MergedGroups <= row.Regions);
        }

        // Rows come back in sample order, which is what makes the plot's bar order reproducible.
        Assert.Equal(
            result.Rows.Select(r => r.Sample).OrderBy(s => s, StringComparer.Ordinal).ToList(),
            result.Rows.Select(r => r.Sample).ToList());

        _out.WriteLine($"{result.Rows.Count} replicates, {result.AssignedPeptides:N0} assigned peptides");
    }

    /// <summary>
    /// The property the plot depends on: a list bar can never be taller than the assigned bar, on every
    /// replicate. Two lists claiming the same peptide BOTH get it, so they can sum past the assigned
    /// total - which is why they are drawn overlaid rather than stacked.
    /// </summary>
    [Fact]
    public void ListTotalsNestInsideTheAssignedTotal()
    {
        var dir = SeedCohort();
        var lists = TwoOverlappingLists(dir);

        var result = Ms2SignalAccounting.Compute(
            dir, IsolationScheme.AstralDefault(), Ppm10, lists, null, _out.WriteLine);

        Assert.NotNull(result);
        Assert.Equal(2, result!.ListNames.Count);

        var anyListSignal = false;
        foreach (var row in result.Rows)
        {
            for (var l = 0; l < result.ListNames.Count; l++)
            {
                Assert.True(row.ListArea[l] <= row.AssignedArea + 1e-9,
                    $"{row.Sample}/{result.ListNames[l]}: list total exceeded the assigned total");
                anyListSignal |= row.ListArea[l] > 0;
            }
        }
        Assert.True(anyListSignal, "the lists should account for some signal");
    }

    /// <summary>
    /// Persist and read back. This is the path <c>prism qc -d</c> takes, so a drift here would show as a
    /// report that silently disagrees with the run that produced it.
    /// </summary>
    [Fact]
    public void RoundTripsThroughParquet()
    {
        var dir = SeedCohort();
        var lists = TwoOverlappingLists(dir);

        var computed = Ms2SignalAccounting.Compute(
            dir, IsolationScheme.AstralDefault(), Ppm10, lists,
            new Dictionary<string, string> { }, _out.WriteLine);
        Assert.NotNull(computed);

        Ms2SignalAccounting.Write(dir, computed!);
        Assert.True(File.Exists(Path.Combine(dir, Ms2SignalAccounting.AccountingFile)));
        Assert.True(File.Exists(Path.Combine(dir, Ms2SignalAccounting.ListsFile)));

        var cached = Ms2SignalAccounting.ReadCached(dir);
        Assert.NotNull(cached);
        Assert.Equal(computed!.Rows.Count, cached!.Rows.Count);
        Assert.Equal(computed.ListNames, cached.ListNames);
        Assert.Equal(computed.Tolerance, cached.Tolerance);
        Assert.Equal(computed.IsolationScheme, cached.IsolationScheme);
        Assert.Equal(computed.AssignedPeptides, cached.AssignedPeptides);

        for (var i = 0; i < computed.Rows.Count; i++)
        {
            var a = computed.Rows[i];
            var b = cached.Rows[i];
            Assert.Equal(a.Sample, b.Sample);
            Assert.Equal(a.AssignedArea, b.AssignedArea, 6);
            Assert.Equal(a.SummedArea, b.SummedArea, 6);
            Assert.Equal(a.Regions, b.Regions);
            Assert.Equal(a.DuplicateRows, b.DuplicateRows);
            Assert.Equal(a.SharedAcrossPeptides, b.SharedAcrossPeptides);
            for (var l = 0; l < computed.ListNames.Count; l++)
                Assert.Equal(a.ListArea[l], b.ListArea[l], 6);
        }
    }

    /// <summary>
    /// A re-run with the lists removed must not leave the previous run's list bars on the plot. The
    /// stale file would otherwise read as current, which is worse than having no bars at all.
    /// </summary>
    [Fact]
    public void RewritingWithoutListsClearsTheStaleListFile()
    {
        var dir = SeedCohort();
        var withLists = Ms2SignalAccounting.Compute(
            dir, IsolationScheme.AstralDefault(), Ppm10, TwoOverlappingLists(dir));
        Ms2SignalAccounting.Write(dir, withLists!);
        Assert.True(File.Exists(Path.Combine(dir, Ms2SignalAccounting.ListsFile)));

        var without = Ms2SignalAccounting.Compute(
            dir, IsolationScheme.AstralDefault(), Ppm10, Array.Empty<ProteinList>());
        Ms2SignalAccounting.Write(dir, without!);

        Assert.False(File.Exists(Path.Combine(dir, Ms2SignalAccounting.ListsFile)));
        Assert.Empty(Ms2SignalAccounting.ReadCached(dir)!.ListNames);
    }

    /// <summary>An output directory with no merged_data/ yields null, not an exception.</summary>
    [Fact]
    public void WithoutMergedDataThereIsNothingToCompute()
    {
        var empty = Path.Combine(Path.GetTempPath(), "prism_ms2empty_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(empty);
        _scratch.Add(empty);

        var messages = new List<string>();
        var result = Ms2SignalAccounting.Compute(
            empty, IsolationScheme.AstralDefault(), Ppm10, Array.Empty<ProteinList>(),
            null, messages.Add);

        Assert.Null(result);
        Assert.Contains(messages, m => m.Contains("merged_data", StringComparison.Ordinal));
        Assert.Null(Ms2SignalAccounting.ReadCached(empty));
    }

    /// <summary>
    /// The plot renders on real results. Not a pixel comparison - the point is that the drawing code
    /// survives a cohort with more replicates than tick labels and lists that overlap each other.
    /// </summary>
    [Fact]
    public void RendersAPlot()
    {
        var dir = SeedCohort();
        var result = Ms2SignalAccounting.Compute(
            dir, IsolationScheme.AstralDefault(), Ppm10, TwoOverlappingLists(dir));
        Assert.NotNull(result);

        var png = PlotRenderer.Ms2AccountingPng(result!, "MS2 Signal Assigned to Peptides");

        // Set PRISM_MS2_PLOT_OUT to a directory to eyeball the result; a plot whose numbers are right
        // can still be unreadable, and that is not something an assertion catches.
        if (Environment.GetEnvironmentVariable("PRISM_MS2_PLOT_OUT") is { Length: > 0 } dump)
        {
            Directory.CreateDirectory(dump);
            File.WriteAllBytes(Path.Combine(dump, "ms2_signal_accounting.png"), png);
        }

        Assert.True(png.Length > 1000, "the PNG should not be empty");
        // PNG magic, so a truncated or non-image buffer fails here rather than in a browser.
        Assert.Equal(new byte[] { 0x89, 0x50, 0x4E, 0x47 }, png.Take(4).ToArray());
    }

    /// <summary>
    /// The report section appears once the results are cached, and names what the bars actually are.
    /// The wording is asserted because it is load-bearing: this is integrated signal for the document's
    /// targets, and calling it total MS2 would turn unknown coverage into apparent full coverage.
    /// </summary>
    [Fact]
    public void TheReportShowsTheSectionOnceResultsAreCached()
    {
        var dir = SeedCohort();
        var result = Ms2SignalAccounting.Compute(
            dir, IsolationScheme.AstralDefault(), Ppm10, Array.Empty<ProteinList>());
        Ms2SignalAccounting.Write(dir, result!);
        AskForMs2Signal(dir);

        var html = File.ReadAllText(QcReport.Generate(dir, new PrismConfig(), savePlots: false));

        Assert.Contains("MS2 Signal Accounting", html, StringComparison.Ordinal);
        // Stops before the apostrophe so this pins the WORDING, not the HTML escaping scheme.
        Assert.Contains("Signal Skyline Integrated for This Document", html, StringComparison.Ordinal);
        Assert.DoesNotContain("Total MS2 Signal", html, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Put the cohort's isolation windows where the report looks for them. The report resolves the
    /// scheme from <c>isolation_schemes.xml</c> (the Skyline tool writes it during a run) rather than
    /// being handed one, so a test that exercises the REPORT has to seed it; the tests that call
    /// <c>Compute</c> directly pass the scheme instead.
    /// </summary>
    private static void SeedIsolationScheme(string dir)
    {
        var catalog = new IsolationSchemeCatalog();
        catalog.AddDocumentScheme("Batch1", IsolationScheme.AstralDefault());
        catalog.Save(Path.Combine(dir, IsolationSchemeCatalog.FileName));
    }

    /// <summary>
    /// Record in the directory's provenance that the run asked for the accounting, which is what makes
    /// the report replot a cached result: the report takes what it SAYS about a run from that run's
    /// own parameters.json, never from the config handed to this invocation.
    /// </summary>
    private static void AskForMs2Signal(string dir, Action<PrismConfig>? tune = null)
    {
        var path = Path.Combine(dir, "parameters.json");
        var config = Provenance.LoadConfig(path);
        config.QcReport.Ms2Signal.Enabled = true;
        tune?.Invoke(config);
        Provenance.Write(
            path, config, Array.Empty<string>(), new Provenance.Stats(0, 0, 0, 0),
            DateTime.UtcNow.ToString("O"));
    }

    /// <summary>
    /// Without cached results and without the flag, the report has no such section and does not go
    /// looking at merged_data/ - a report regeneration must not silently start a minutes-long pass.
    /// </summary>
    [Fact]
    public void TheReportOmitsTheSectionWhenNothingWasComputed()
    {
        var dir = SeedCohort();

        var messages = new List<string>();
        var html = File.ReadAllText(
            QcReport.Generate(dir, new PrismConfig(), savePlots: false, log: messages.Add));

        Assert.DoesNotContain("MS2 Signal Accounting", html, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(dir, Ms2SignalAccounting.AccountingFile)));
    }

    /// <summary>
    /// Two lists that deliberately share peptides, built from the cohort's own protein groups so they
    /// actually match something. Overlap is the interesting case: both lists get the shared signal in
    /// full, which is what makes the totals nest rather than partition.
    /// </summary>
    private static IReadOnlyList<ProteinList> TwoOverlappingLists(string dir)
    {
        var accessions = LeadingProteins(Path.Combine(dir, "corrected_peptides.parquet"));
        Assert.True(accessions.Count >= 4, "fixture should have several protein groups");

        var first = accessions.Take(Math.Max(2, accessions.Count / 2)).ToList();
        var second = accessions.Skip(Math.Max(1, accessions.Count / 3)).ToList();

        return new[]
        {
            new ProteinList { Name = "Panel A", ColorHex = "#2ca02c", Members = first },
            new ProteinList { Name = "Panel B", ColorHex = "#9467bd", Members = second },
        };
    }

    private static List<string> LeadingProteins(string correctedPeptides)
    {
        using var reader = ParquetColumnReader.Open(correctedPeptides);
        if (!reader.HasColumn("leading_protein"))
            return new List<string>();
        return reader.ReadStrings("leading_protein")
            .Where(v => !string.IsNullOrWhiteSpace(v))
            .SelectMany(v => ProteinListMatcher.SplitGroups(v))
            .Distinct(StringComparer.Ordinal)
            .OrderBy(v => v, StringComparer.Ordinal)
            .ToList();
    }
    /// <summary>
    /// A cached section belongs to the run that asked for it. Turning the accounting off and
    /// regenerating must not resurrect the previous run's plot - which the report did, because it
    /// preferred the cache before it even looked at whether the section had been asked for.
    /// </summary>
    [Fact]
    public void TheReportDropsACachedSectionTheRunDidNotAskFor()
    {
        var dir = SeedCohort();
        Ms2SignalAccounting.Write(
            dir,
            Ms2SignalAccounting.Compute(
                dir, IsolationScheme.AstralDefault(), Ppm10, Array.Empty<ProteinList>())!);
        // Provenance left as the fixture wrote it: qc_report.ms2_signal.enabled is false.

        var html = File.ReadAllText(QcReport.Generate(dir, new PrismConfig(), savePlots: false));

        Assert.DoesNotContain("MS2 Signal Accounting", html, StringComparison.Ordinal);
        // ...and the cache is left alone, so asking for the section again replots it for free.
        Assert.True(File.Exists(Path.Combine(dir, Ms2SignalAccounting.AccountingFile)));
    }

    /// <summary>
    /// The cache is keyed on the SETTINGS, not just on its file name.
    ///
    /// <para>This is the failure the keying exists for: a re-run that switched to ion counts - after a
    /// four-hour Skyline export to get them - found a cached parquet and replotted the previous run's
    /// peak areas under the new run's caption, silently. Both plots look right; nothing in the log
    /// mentioned it.</para>
    ///
    /// <para>The second half matters as much. This cohort's export has no ion-count column, so the
    /// recomputed result falls back to signal - and if the cache were keyed on what was COMPUTED
    /// rather than on what was ASKED for, every report from then on would recompute forever.</para>
    /// </summary>
    [Fact]
    public void ChangingTheMeasureRecomputesRatherThanReplottingTheCache()
    {
        var dir = SeedCohort();
        SeedIsolationScheme(dir);
        Ms2SignalAccounting.Write(
            dir,
            Ms2SignalAccounting.Compute(
                dir, IsolationScheme.AstralDefault(), Ppm10, Array.Empty<ProteinList>())!);
        AskForMs2Signal(dir, c => c.QcReport.Ms2Signal.Measure = "ions");

        var first = new List<string>();
        QcReport.Generate(dir, new PrismConfig(), savePlots: false, log: first.Add);
        Assert.Contains(first, m => m.Contains("Recomputing", StringComparison.Ordinal));

        // Second pass: the key now records that ions were asked for, so the same request is served
        // from the cache. Without that, the fallback to signal would miss the key every time.
        var second = new List<string>();
        QcReport.Generate(dir, new PrismConfig(), savePlots: false, log: second.Add);
        Assert.DoesNotContain(second, m => m.Contains("Recomputing", StringComparison.Ordinal));
    }

    /// <summary>Same rule for the extraction tolerance, which changes the numbers just as much.</summary>
    [Fact]
    public void ChangingTheToleranceRecomputes()
    {
        var dir = SeedCohort();
        SeedIsolationScheme(dir);
        Ms2SignalAccounting.Write(
            dir,
            Ms2SignalAccounting.Compute(
                dir, IsolationScheme.AstralDefault(), Ppm10, Array.Empty<ProteinList>())!);
        AskForMs2Signal(dir, c => c.QcReport.Ms2Signal.ExtractionTolerance = "100 ppm");

        var log = new List<string>();
        QcReport.Generate(dir, new PrismConfig(), savePlots: false, log: log.Add);

        Assert.Contains(log, m => m.Contains("Recomputing", StringComparison.Ordinal));
        Assert.Equal(
            "+/-100 ppm (centroided)", Ms2SignalAccounting.ReadCached(dir)!.Tolerance);
    }

    /// <summary>
    /// Which quantity the cached numbers ARE has to survive the round trip: ion counts and gross peak
    /// areas are both large positive doubles, so a replot that assumes signal captions ion counts as
    /// "Integrated MS2 signal" in the title, the axis and the caption at once, with nothing to give it
    /// away. (<c>ListsMatchable</c> was hardcoded true on the way back for the same reason.)
    /// </summary>
    [Fact]
    public void TheCacheRemembersWhichMeasureProducedIt()
    {
        var dir = SeedCohort();
        var computed = Ms2SignalAccounting.Compute(
            dir, IsolationScheme.AstralDefault(), Ppm10, Array.Empty<ProteinList>())!;

        Ms2SignalAccounting.Write(
            dir, computed with { Measure = Ms2SignalMeasure.Ions, ListsMatchable = false });
        var read = Ms2SignalAccounting.ReadCached(dir)!;

        Assert.Equal(Ms2SignalMeasure.Ions, read.Measure);
        Assert.False(read.ListsMatchable);
        Assert.Equal(computed.SettingsKey, read.SettingsKey);
    }

    /// <summary>
    /// A protein list this machine has no list for must cost the plot's lines, not the whole run. The
    /// resolver threw, and nothing between it and the caller caught: a provenance file from a
    /// colleague's machine aborted the run at Stage 5b - after the export, the merge, the rollups and
    /// the correction had all completed - and left no qc_report.html at all. Every other failure in
    /// that section logs and carries on.
    /// </summary>
    [Fact]
    public void AnUnresolvableProteinListDoesNotThrowAwayTheReport()
    {
        var dir = SeedCohort();
        SeedIsolationScheme(dir);
        AskForMs2Signal(dir, c => c.QcReport.Ms2Signal.ProteinLists =
            new List<string> { "a list saved on somebody else's machine" });

        var log = new List<string>();
        var html = File.ReadAllText(
            QcReport.Generate(dir, new PrismConfig(), savePlots: false, log: log.Add));

        Assert.Contains("MS2 Signal Accounting", html, StringComparison.Ordinal);
        Assert.Contains(log, m => m.Contains("somebody else's machine", StringComparison.Ordinal));
    }
}
