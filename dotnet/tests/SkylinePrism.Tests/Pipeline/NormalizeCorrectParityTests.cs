using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// C#-vs-C# regression harness for Stage 2b/2c (and 4b/4c): every implementation reachable through
/// <see cref="NormalizeCorrectStage.Run"/> must reproduce <see cref="NormalizeCorrectStage.RunInMemory"/>
/// bit-for-bit-ish (1e-12 relative) on the same input.
/// <para>
/// This exists because the cross-language goldens cannot protect a refactor of this stage:
/// <c>PipelineParityTests</c> compares corrected_peptides / corrected_proteins at <b>3e-2 relative</b>
/// (C# and Python ComBat legitimately diverge), which would wave through a serious bug. Here both
/// implementations run in one process over identical synthetic input, so the tolerance can be tight
/// enough that only a genuine numerical change shows up.
/// </para>
/// <para>
/// While a single implementation exists this passes trivially - that is the point: it proves the
/// fixtures, the case matrix and the comparer work before a second implementation lands. The
/// comparer's own sensitivity is pinned by <see cref="Comparer_DetectsADifference"/>, so "green"
/// never means "compared nothing".
/// </para>
/// </summary>
public class NormalizeCorrectParityTests
{
    private const string KeyCol = SyntheticCohort.KeyColumn;
    private const string InternalFile = "peptides_log2_internal.parquet";
    private const string CorrectedFile = "corrected.parquet";

    /// <summary>
    /// Every code path that changes what the stage computes: each normalization method (quantile
    /// included - it is the one method that is not streamable and must stay on the in-memory path),
    /// ComBat on/off, reference-anchored on/off, auto-revert on/off, and both the "no rows dropped"
    /// (aliasing) and "rows dropped" branches of the all-NaN filter.
    /// </summary>
    public static IEnumerable<object[]> Cases()
    {
        foreach (var norm in new[] { "median", "quantile", "vsn", "none", "rt_lowess" })
        foreach (var dropRows in new[] { false, true })
        {
            yield return new object[] { norm, false, false, false, dropRows };
            foreach (var referenceAnchored in new[] { false, true })
            foreach (var autoRevert in new[] { false, true })
                yield return new object[] { norm, true, referenceAnchored, autoRevert, dropRows };
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void DispatchedPath_MatchesInMemory(
        string norm, bool combat, bool referenceAnchored, bool autoRevert, bool dropRows)
    {
        var root = NewTempDir();
        try
        {
            var cohort = SyntheticCohort.Write(Path.Combine(root, "in"), allNanEvery: dropRows ? 17 : 0);
            // The two branches of the all-NaN filter really are both exercised: n == nAll aliases the
            // loaded matrix instead of copying it, so a bug there only shows on one of them.
            Assert.Equal(dropRows, cohort.KeptRows < cohort.NRows);

            var (rowsA, reportA) = RunStage(
                NormalizeCorrectStage.RunInMemory, cohort, Path.Combine(root, "in-memory"),
                norm, combat, referenceAnchored, autoRevert);

            var reportB = new List<string>();
            var requestB = BuildRequest(
                cohort, Path.Combine(root, "dispatched"), norm, combat, referenceAnchored,
                autoRevert, reportB);

            // Which implementation this case is meant to exercise. Asserted, not assumed: without it
            // a streaming path that quietly stopped being eligible would leave every case below
            // comparing the in-memory implementation to itself, and still be green.
            // Reference-anchored is streamed now, so the only exclusion left here is quantile.
            var expectStreaming = norm != "quantile";
            Assert.Equal(expectStreaming, StreamingNormalizeCorrect.CanHandle(requestB));

            var rowsB = NormalizeCorrectStage.Run(requestB);

            Assert.Equal(cohort.KeptRows, rowsA);
            Assert.Equal(rowsA, rowsB);
            Assert.Equal(reportA, reportB); // same CV metrics, same revert / overfitting decisions

            var compared = AssertOutputsMatch(
                Path.Combine(root, "in-memory"), Path.Combine(root, "dispatched"), cohort);

            // The cohort is dense, so every written cell must be finite and must have been compared
            // numerically. Without this a path that quietly emitted NaN everywhere would still pass.
            Assert.Equal(cohort.KeptRows * cohort.Samples.Length * 2, compared);
        }
        finally
        {
            Cleanup(root);
        }
    }

    /// <summary>
    /// Real <c>peptides_rollup.parquet</c> files arrive in 2,000-row groups, and the streaming path
    /// carries per-feature state (the kept-row ordinal that indexes ComBat's per-feature terms)
    /// ACROSS those boundaries. A single-row-group fixture cannot catch a mistake there, so these
    /// cases use tiny groups: 7 rows against 80 features is a dozen boundaries, with dropped rows
    /// landing on the first row of a group, the last row of a group, and - in
    /// <see cref="EntirelyDroppedRowGroup_MatchesInMemory"/> - all of them.
    /// </summary>
    [Theory]
    [InlineData("median", true, false, false)]
    [InlineData("median", true, true, false)]
    [InlineData("median", false, true, false)]
    [InlineData("rt_lowess", true, true, false)]
    [InlineData("vsn", true, true, false)]
    [InlineData("none", true, true, false)]
    // Reference-anchored carries per-feature state across row-group boundaries too, and its fit sets
    // are a subset of each batch - so a boundary bug there would not show in the standard cases.
    [InlineData("median", true, false, true)]
    [InlineData("median", true, true, true)]
    [InlineData("rt_lowess", true, true, true)]
    public void MultipleRowGroups_MatchesInMemory(
        string norm, bool combat, bool dropRows, bool referenceAnchored)
    {
        var root = NewTempDir();
        try
        {
            var cohort = SyntheticCohort.Write(
                Path.Combine(root, "in"), allNanEvery: dropRows ? 7 : 0, rowGroupRows: 7);
            AssertBothPathsAgree(root, cohort, norm, combat, autoRevert: false, expectDense: true,
                referenceAnchored: referenceAnchored);
        }
        finally
        {
            Cleanup(root);
        }
    }

    /// <summary>
    /// A row group in which every feature is dropped produces an empty output row group, which the
    /// writer must skip rather than emit - and the kept-row ordinal must not advance past it.
    /// </summary>
    [Fact]
    public void EntirelyDroppedRowGroup_MatchesInMemory()
    {
        var root = NewTempDir();
        try
        {
            // Groups are rows [0,5) [5,10) [10,15) ...; blank the whole second one, plus stragglers
            // on either side of the third boundary.
            var cohort = SyntheticCohort.Write(
                Path.Combine(root, "in"), nFeatures: 60, rowGroupRows: 5,
                allNanRows: new[] { 5, 6, 7, 8, 9, 14, 15, 59 });
            Assert.Equal(60 - 8, cohort.KeptRows);
            AssertBothPathsAgree(root, cohort, "median", combat: true, autoRevert: false, expectDense: true);
        }
        finally
        {
            Cleanup(root);
        }
    }

    /// <summary>
    /// Missing values (a peptide absent from some samples) are the normal case in real cohorts, and
    /// they change what every stat sees. Pinned separately from the dense matrix because the outputs
    /// are then partly NaN and the "everything is finite" assertion above cannot apply.
    /// </summary>
    [Theory]
    [InlineData("median", false)]
    [InlineData("median", true)]
    [InlineData("quantile", false)]
    [InlineData("rt_lowess", false)]
    public void MissingValues_DispatchedPathMatchesInMemory(string norm, bool combat)
    {
        var root = NewTempDir();
        try
        {
            var cohort = SyntheticCohort.Write(Path.Combine(root, "in"), missingFraction: 0.08, allNanEvery: 23);
            Assert.True(cohort.KeptRows < cohort.NRows);

            var (rowsA, reportA) = RunStage(
                NormalizeCorrectStage.RunInMemory, cohort, Path.Combine(root, "in-memory"),
                norm, combat, referenceAnchored: false, autoRevert: false);

            var reportB = new List<string>();
            var requestB = BuildRequest(
                cohort, Path.Combine(root, "dispatched"), norm, combat,
                referenceAnchored: false, autoRevert: false, report: reportB);
            Assert.Equal(norm != "quantile", StreamingNormalizeCorrect.CanHandle(requestB));
            var rowsB = NormalizeCorrectStage.Run(requestB);

            Assert.Equal(rowsA, rowsB);
            Assert.Equal(reportA, reportB);
            var compared = AssertOutputsMatch(
                Path.Combine(root, "in-memory"), Path.Combine(root, "dispatched"), cohort);
            if (!combat)
                Assert.True(compared > cohort.KeptRows * cohort.Samples.Length, // > half the cells
                    $"only {compared} finite cells compared - the comparison was nearly vacuous.");
        }
        finally
        {
            Cleanup(root);
        }
    }

    /// <summary>
    /// The auto-revert branch: ComBat rescales a low-spread batch up, inflating the CV of the QC
    /// replicates that live in it, so the evaluator throws the correction away. Both paths must reach
    /// the same decision - a streaming implementation computes the before/after CVs differently and
    /// could silently flip it.
    /// </summary>
    [Fact]
    public void AutoRevert_TriggersAndMatches()
    {
        var root = NewTempDir();
        try
        {
            // Batch spreads differ by ~100x; the QC replicates sit in the tightest batch.
            var cohort = SyntheticCohort.Write(
                Path.Combine(root, "in"), batchSpread: new[] { 8.0, 0.05, 4.0 }, qcOnlyInBatch: 1);

            var (_, reportA) = RunStage(
                NormalizeCorrectStage.RunInMemory, cohort, Path.Combine(root, "in-memory"),
                "median", combat: true, referenceAnchored: false, autoRevert: true);

            var reportB = new List<string>();
            var requestB = BuildRequest(
                cohort, Path.Combine(root, "dispatched"), "median", combat: true,
                referenceAnchored: false, autoRevert: true, report: reportB);
            Assert.True(StreamingNormalizeCorrect.CanHandle(requestB));
            NormalizeCorrectStage.Run(requestB);

            Assert.Contains(reportA, line => line.Contains("ComBat REVERTED", StringComparison.Ordinal));
            Assert.Equal(reportA, reportB);
            AssertOutputsMatch(
                Path.Combine(root, "in-memory"), Path.Combine(root, "dispatched"), cohort);
        }
        finally
        {
            Cleanup(root);
        }
    }

    /// <summary>
    /// A non-parquet corrected output (config output.format = csv/tsv) is not streamed - the
    /// delimited writer builds the whole file in memory - so the dispatcher must fall back rather
    /// than write a truncated file.
    /// </summary>
    [Fact]
    public void DelimitedOutput_FallsBackAndStillMatches()
    {
        var root = NewTempDir();
        try
        {
            var cohort = SyntheticCohort.Write(Path.Combine(root, "in"), nFeatures: 40);
            RunStage(NormalizeCorrectStage.RunInMemory, cohort, Path.Combine(root, "in-memory"),
                "median", combat: true, referenceAnchored: false, autoRevert: false, correctedExt: "csv");

            var request = BuildRequest(
                cohort, Path.Combine(root, "dispatched"), "median", combat: true,
                referenceAnchored: false, autoRevert: false, report: new List<string>(),
                correctedExt: "csv");
            Assert.False(StreamingNormalizeCorrect.CanHandle(request));
            NormalizeCorrectStage.Run(request);

            var a = File.ReadAllText(Path.Combine(root, "in-memory", "corrected.csv"));
            var b = File.ReadAllText(Path.Combine(root, "dispatched", "corrected.csv"));
            Assert.Equal(a, b);
            Assert.Contains("protein_group", a); // derived columns land on the corrected output
        }
        finally
        {
            Cleanup(root);
        }
    }

    /// <summary>
    /// The derived (parsimony) columns belong on the corrected output ONLY. The internal log2 file
    /// feeds ProteinRollup and QcReport, which treat any undeclared column as a sample and try to
    /// parse "PG12" as an abundance - a regression that has happened before.
    /// </summary>
    [Fact]
    public void DerivedColumns_OnCorrectedOutputOnly()
    {
        var root = NewTempDir();
        try
        {
            var cohort = SyntheticCohort.Write(Path.Combine(root, "in"), nFeatures: 40);
            var outDir = Path.Combine(root, "out");
            RunStage(NormalizeCorrectStage.Run, cohort, outDir,
                "median", combat: false, referenceAnchored: false, autoRevert: false);

            var corrected = ParquetTable.Load(Path.Combine(outDir, CorrectedFile));
            var internalLog2 = ParquetTable.Load(Path.Combine(outDir, InternalFile));
            Assert.Contains("protein_group", corrected.ColumnNames);
            Assert.Contains("leading_protein", corrected.ColumnNames);
            Assert.DoesNotContain("protein_group", internalLog2.ColumnNames);
            Assert.DoesNotContain("leading_protein", internalLog2.ColumnNames);
        }
        finally
        {
            Cleanup(root);
        }
    }

    /// <summary>
    /// Negative control for the harness itself: the comparer must fail on a difference far smaller
    /// than the 3e-2 the cross-language goldens tolerate. Without this, "all parity tests green"
    /// could mean the comparer never looked at anything.
    /// </summary>
    [Fact]
    public void Comparer_DetectsADifference()
    {
        var root = NewTempDir();
        try
        {
            var samples = new[] { "s1", "s2" };
            var meta = new[] { ParquetWideWriter.Strings(KeyCol, new[] { "A", "B" }) };
            var baseline = new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 } };
            var perturbed = new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 * (1 + 1e-9) } };

            var pathA = Path.Combine(root, "a.parquet");
            var pathB = Path.Combine(root, "b.parquet");
            ParquetWideWriter.Write(pathA, meta, samples, baseline, 2);
            ParquetWideWriter.Write(pathB, meta, samples, perturbed, 2);

            var sampleSet = new HashSet<string>(samples, StringComparer.Ordinal);
            Assert.ThrowsAny<Xunit.Sdk.XunitException>(
                () => AssertParquetEqual(pathA, pathB, sampleSet));

            // ... and passes when the files really are identical.
            ParquetWideWriter.Write(pathB, meta, samples, baseline, 2);
            Assert.Equal(4, AssertParquetEqual(pathA, pathB, sampleSet));
        }
        finally
        {
            Cleanup(root);
        }
    }

    /// <summary>
    /// Run both implementations over <paramref name="cohort"/> and assert they agree on everything
    /// observable: row count, report lines (so the CV metrics and any revert decision match), and
    /// every cell of both outputs.
    /// </summary>
    private static void AssertBothPathsAgree(
        string root, SyntheticCohort cohort, string norm, bool combat, bool autoRevert, bool expectDense,
        bool referenceAnchored = false)
    {
        var (rowsA, reportA) = RunStage(
            NormalizeCorrectStage.RunInMemory, cohort, Path.Combine(root, "in-memory"),
            norm, combat, referenceAnchored, autoRevert);

        var reportB = new List<string>();
        var requestB = BuildRequest(
            cohort, Path.Combine(root, "dispatched"), norm, combat,
            referenceAnchored, autoRevert: autoRevert, report: reportB);
        Assert.True(StreamingNormalizeCorrect.CanHandle(requestB));
        var rowsB = NormalizeCorrectStage.Run(requestB);

        Assert.Equal(cohort.KeptRows, rowsA);
        Assert.Equal(rowsA, rowsB);
        Assert.Equal(reportA, reportB);

        var compared = AssertOutputsMatch(
            Path.Combine(root, "in-memory"), Path.Combine(root, "dispatched"), cohort);
        if (expectDense)
            Assert.Equal(cohort.KeptRows * cohort.Samples.Length * 2, compared);
    }

    private static (int Rows, List<string> Report) RunStage(
        Func<NormalizeCorrectRequest, int> implementation,
        SyntheticCohort cohort, string outDir, string norm,
        bool combat, bool referenceAnchored, bool autoRevert, string correctedExt = "parquet")
    {
        var report = new List<string>();
        var request = BuildRequest(
            cohort, outDir, norm, combat, referenceAnchored, autoRevert, report, correctedExt);
        return (implementation(request), report);
    }

    private static NormalizeCorrectRequest BuildRequest(
        SyntheticCohort cohort, string outDir, string norm,
        bool combat, bool referenceAnchored, bool autoRevert, List<string> report,
        string correctedExt = "parquet")
    {
        Directory.CreateDirectory(outDir);
        return new NormalizeCorrectRequest
        {
            WideParquet = cohort.InputPath,
            MetaSpec = new[]
            {
                (KeyCol, MetaType.Str), ("n_transitions", MetaType.Long),
                ("mean_rt", MetaType.Double), ("low_confidence", MetaType.Bool),
            },
            Samples = cohort.Samples,
            BatchLabels = cohort.BatchLabels,
            CombatEnabled = combat,
            NormMethod = norm,
            InternalLog2Path = Path.Combine(outDir, InternalFile),
            CorrectedLinearPath = Path.Combine(outDir, "corrected." + correctedExt),
            Report = report.Add,
            RefIdx = cohort.RefIdx,
            QcIdx = cohort.QcIdx,
            ReferenceAnchored = referenceAnchored,
            ReferenceMask = cohort.ReferenceMask,
            RtColumn = "mean_rt",
            AutoRevert = autoRevert,
            DerivedMeta = new (string, Func<string, string>)[]
            {
                ("protein_group", p => "PG" + (int.Parse(p[3..]) / 3)),
                ("leading_protein", p => $"sp|P{p[3..]}|X_HUMAN"),
            },
            DerivedKeyColumn = KeyCol,
        };
    }

    // ---------------------------------------------------------------- comparison

    /// <summary>Compare both output files of two runs. Returns the number of numeric cells compared.</summary>
    private static int AssertOutputsMatch(string expectedDir, string actualDir, SyntheticCohort cohort)
    {
        var sampleSet = new HashSet<string>(cohort.Samples, StringComparer.Ordinal);
        var compared = 0;
        foreach (var file in new[] { InternalFile, CorrectedFile })
        {
            var expected = Path.Combine(expectedDir, file);
            var actual = Path.Combine(actualDir, file);
            Assert.True(File.Exists(expected), $"missing baseline output {file}");
            Assert.True(File.Exists(actual), $"missing output {file}");
            compared += AssertParquetEqual(expected, actual, sampleSet);
        }
        return compared;
    }

    /// <summary>
    /// Cell-by-cell comparison of two wide parquets. Sample columns compare numerically at
    /// <paramref name="relTol"/> (NaN matches only NaN); every other column - the meta and derived
    /// columns, whose row filtering must line up exactly - compares as formatted text. Returns the
    /// number of finite numeric cells actually compared, so a caller can reject a vacuous pass.
    /// </summary>
    private static int AssertParquetEqual(
        string expectedPath, string actualPath, ISet<string> sampleNames,
        double relTol = 1e-12, double absTol = 1e-12)
    {
        var expected = ParquetTable.Load(expectedPath);
        var actual = ParquetTable.Load(actualPath);
        var file = Path.GetFileName(expectedPath);

        Assert.Equal(expected.ColumnNames, actual.ColumnNames);
        Assert.Equal(expected.RowCount, actual.RowCount);

        var compared = 0;
        foreach (var name in expected.ColumnNames)
        {
            if (sampleNames.Contains(name))
            {
                var e = expected.GetDouble(name);
                var a = actual.GetDouble(name);
                for (var i = 0; i < e.Length; i++)
                {
                    var x = e[i] ?? double.NaN;
                    var y = a[i] ?? double.NaN;
                    if (double.IsNaN(x) || double.IsNaN(y))
                    {
                        Assert.True(double.IsNaN(x) && double.IsNaN(y),
                            $"{file} [{name}][{i}]: {x} vs {y} (one is NaN)");
                        continue;
                    }
                    Assert.True(Math.Abs(x - y) <= absTol + relTol * Math.Abs(x),
                        $"{file} [{name}][{i}]: {x:R} vs {y:R} (rel {(x == 0 ? 0 : Math.Abs(x - y) / Math.Abs(x)):E2})");
                    compared++;
                }
            }
            else
            {
                var e = expected.Column(name);
                var a = actual.Column(name);
                for (var i = 0; i < expected.RowCount; i++)
                    Assert.Equal(Fixtures.FormatCell(e.GetValue(i)), Fixtures.FormatCell(a.GetValue(i)));
            }
        }
        return compared;
    }

    // ---------------------------------------------------------------- plumbing

    private static string NewTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_nc_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void Cleanup(string dir)
    {
        if (Directory.Exists(dir))
            Directory.Delete(dir, recursive: true);
    }
}
