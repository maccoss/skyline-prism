using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.DifferentialAnalysis.Detection;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Tests for <see cref="DetectionAnalysis"/>: that it picks the test the design calls for, in the
/// Detection pane's order, and runs it on the right samples. Each case is checked against a direct
/// call of the estimator it should have chosen - those estimators are pinned to goldens elsewhere,
/// so what is tested here is the choice and the column bookkeeping, not the statistics.
/// </summary>
public class DetectionAnalysisTests
{
    private static string MiniOutput => Fixtures.Path2("mini", "e2e-sum", "output");

    private const int Peptides = 6;

    /// <summary>
    /// A deterministic binary detection matrix over the dataset's samples, in REVERSED order, so a
    /// bug that indexed detection columns by dataset position would test the wrong samples.
    /// </summary>
    internal static (DifferentialDataset Ds, DetectionMatrixData Det, List<int> A, List<int> B) Setup()
    {
        var ds = DifferentialDataset.Load(MiniOutput, FeatureLevel.Protein);
        var detSamples = ds.SampleIds.Reverse().ToArray();
        var peptides = Enumerable.Range(0, Peptides).Select(p => $"PEP{p}").ToArray();
        var m = new double[Peptides, detSamples.Length];
        for (var p = 0; p < Peptides; p++)
        for (var s = 0; s < detSamples.Length; s++)
            m[p, s] = (p * 7 + s * 3 + (s % 4 == 0 ? p : 0)) % 5 < 2 + p % 3 ? 1.0 : 0.0;
        var det = new DetectionMatrixData(peptides, detSamples, m);

        // Interleaved experimental samples, so both arms span both plates (batch is then not
        // confounded with the arm, and the adjusted model stays identifiable).
        var types = ds.MetadataValues("sample_type");
        var experimental = Enumerable.Range(0, ds.SampleIds.Length).Where(j => types[j] == "experimental").ToList();
        var a = experimental.Where((_, i) => i % 2 == 0).Take(20).ToList();
        var b = experimental.Where((_, i) => i % 2 == 1).Take(20).ToList();
        return (ds, det, a, b);
    }

    private static List<int> InDet(DifferentialDataset ds, DetectionMatrixData det, IEnumerable<int> cols) =>
        cols.Select(j => Array.IndexOf(det.SampleIds, ds.SampleIds[j])).ToList();

    /// <summary>Subject k owns A[k] and B[k]; everyone else has no subject.</summary>
    internal static string?[] Subjects(DifferentialDataset ds, List<int> a, List<int> b)
    {
        var s = new string?[ds.SampleIds.Length];
        for (var k = 0; k < Math.Min(a.Count, b.Count); k++)
        {
            s[a[k]] = $"subj{k}";
            s[b[k]] = $"subj{k}";
        }

        return s;
    }

    /// <summary>
    /// Under the blocked design detection tests the samples that HAVE a subject - the ones the
    /// differential fits and the plots draw - and says it does not account for repeated subjects.
    /// </summary>
    [Fact]
    public void Blocked_TestsTheSamplesWithASubject_AndSaysRepeatsAreNotAccountedFor()
    {
        var (ds, det, a, b) = Setup();
        var subjects = Subjects(ds, a, b);
        subjects[a[0]] = null;
        var keptA = a.Skip(1).ToList();

        var r = DetectionAnalysis.Run(det, ds, a, b, DifferentialDesign.BlockedBySubject, subjects, null,
            MultipleTesting.BenjaminiHochberg);

        Assert.Equal(DetectionMethod.FisherExact, r.Method);
        Assert.Equal(UnpairedReason.RepeatedSubjects, r.UnpairedReason);
        Assert.Equal((keptA.Count, b.Count), (r.NA, r.NB));
        Assert.Equal(1, r.SamplesWithoutSubject);
        Assert.Equal(0, r.DroppedSamples);
        var expected = DetectionTest.Run(det.Matrix, det.PeptideIds, InDet(ds, det, keptA), InDet(ds, det, b));
        Assert.Equal(expected, r.Rows);
        var (usedA, usedB) = PairedSamples.ColumnsUsed(DifferentialDesign.BlockedBySubject, subjects, a, b);
        Assert.Equal(keptA, usedA);
        Assert.Equal(b, usedB);
    }

    [Fact]
    public void Blocked_WithNoRepeatedSubject_HasNoNote()
    {
        var (ds, det, a, b) = Setup();
        var subjects = new string?[ds.SampleIds.Length];
        foreach (var c in a.Concat(b))
            subjects[c] = $"own{c}";

        var r = DetectionAnalysis.Run(det, ds, a, b, DifferentialDesign.BlockedBySubject, subjects, null,
            MultipleTesting.BenjaminiHochberg);

        Assert.Equal(UnpairedReason.None, r.UnpairedReason);
    }

    [Fact]
    public void Unpaired_NoCovariates_RunsFisherOnTheRemappedColumns()
    {
        var (ds, det, a, b) = Setup();

        var r = DetectionAnalysis.Run(det, ds, a, b, DifferentialDesign.Unpaired, null, null,
            MultipleTesting.BenjaminiHochberg);

        Assert.Equal(DetectionMethod.FisherExact, r.Method);
        Assert.Equal(UnpairedReason.None, r.UnpairedReason);
        Assert.Equal(a.Count, r.NA);
        Assert.Equal(b.Count, r.NB);
        var expected = DetectionTest.Run(det.Matrix, det.PeptideIds, InDet(ds, det, a), InDet(ds, det, b));
        Assert.Equal(expected, r.Rows);
    }

    [Fact]
    public void Paired_NoCovariates_RunsMcNemarOverTheMatchedSubjects()
    {
        var (ds, det, a, b) = Setup();

        var r = DetectionAnalysis.Run(det, ds, a, b, DifferentialDesign.Paired, Subjects(ds, a, b), null,
            MultipleTesting.Holm);

        Assert.Equal(DetectionMethod.McNemarPaired, r.Method);
        Assert.Equal(20, r.NA);
        var pairs = Enumerable.Range(0, 20)
            .Select(k => new SamplePair($"subj{k}", InDet(ds, det, new[] { a[k] })[0], InDet(ds, det, new[] { b[k] })[0]))
            .ToList();
        // The correction is passed through: McNemar is the one detection test that takes it.
        var expected = DetectionPairedTest.Run(det.Matrix, det.PeptideIds, pairs, MultipleTesting.Holm);
        Assert.Equal(expected, r.PairedRows);
        // The uniform rows keep each peptide's rates and p, with the pair count standing in for n.
        Assert.All(r.Rows, row => Assert.Equal(20, row.NA));
        Assert.Equal(expected.Select(e => e.P), r.Rows.Select(row => row.P));
    }

    [Fact]
    public void Covariates_RunTheFirthGlm_AndWinOverPairing()
    {
        var (ds, det, a, b) = Setup();
        var batch = new[] { "batch" };

        var r = DetectionAnalysis.Run(det, ds, a, b, DifferentialDesign.Paired, Subjects(ds, a, b), batch,
            MultipleTesting.BenjaminiHochberg);

        Assert.Equal(DetectionMethod.FirthGlm, r.Method);
        Assert.Equal(UnpairedReason.Covariates, r.UnpairedReason);
        Assert.True(r.Identifiable);
        var covariates = DetectionAnalysis.CovariatesFor(ds, batch, det.SampleIds);
        var expected = DetectionGlm.Run(det.Matrix, det.PeptideIds, InDet(ds, det, a), InDet(ds, det, b), covariates);
        Assert.Equal(expected.Rows.Select(e => e.P), r.Glm!.Rows.Select(g => g.P));
        Assert.Equal(expected.Rows.Select(e => e.P), r.Rows.Select(row => row.P));
    }

    [Fact]
    public void Paired_WithoutSubjects_FallsBackToFisherAndSaysWhy()
    {
        var (ds, det, a, b) = Setup();

        var r = DetectionAnalysis.Run(det, ds, a, b, DifferentialDesign.Paired, null, null,
            MultipleTesting.BenjaminiHochberg);

        Assert.Equal(DetectionMethod.FisherExact, r.Method);
        Assert.Equal(UnpairedReason.NoMatchedSubjects, r.UnpairedReason);
    }

    /// <summary>
    /// Subjects DID match, but no matched pair has both halves in merged_data. That is not "no subject
    /// could be matched" - saying so would send someone to fix the wrong column - so it has its own reason.
    /// </summary>
    [Fact]
    public void Paired_MatchedButNoPairInMergedData_FallsBackWithItsOwnReason()
    {
        var (ds, det, a, b) = Setup();
        // Arm B gains five unpaired samples, then every PAIRED B sample is removed from merged_data:
        // pairs exist in the metadata, none survives, and both arms still have samples to test.
        var types = ds.MetadataValues("sample_type");
        var extras = Enumerable.Range(0, ds.SampleIds.Length)
            .Where(j => types[j] == "experimental" && !a.Contains(j) && !b.Contains(j)).Take(5).ToList();
        var armB = b.Concat(extras).ToList();
        var gone = new HashSet<string>(b.Select(j => ds.SampleIds[j]), StringComparer.Ordinal);
        var keep = Enumerable.Range(0, det.SampleIds.Length).Where(s => !gone.Contains(det.SampleIds[s])).ToArray();
        var m = new double[Peptides, keep.Length];
        for (var p = 0; p < Peptides; p++)
        for (var k = 0; k < keep.Length; k++)
            m[p, k] = det.Matrix[p, keep[k]];
        var partial = new DetectionMatrixData(det.PeptideIds, keep.Select(s => det.SampleIds[s]).ToArray(), m);

        var r = DetectionAnalysis.Run(partial, ds, a, armB, DifferentialDesign.Paired, Subjects(ds, a, b), null,
            MultipleTesting.BenjaminiHochberg);

        Assert.Equal(DetectionMethod.FisherExact, r.Method);
        Assert.Equal(UnpairedReason.NoPairInMergedData, r.UnpairedReason);
        Assert.Equal(20, r.PairsNotInMerged);
        Assert.Contains("no matched pair has both its samples in merged_data",
            DetectionAnalysis.UnpairedNote(r.UnpairedReason));
    }

    [Fact]
    public void CovariatesFor_AlignsToTheTargetOrder()
    {
        var (ds, det, _, _) = Setup();
        var batch = ds.MetadataValues("batch");

        var cov = DetectionAnalysis.CovariatesFor(ds, new[] { "batch" }, det.SampleIds)!;
        var direct = Covariate.FromMetadata("batch", det.SampleIds
            .Select(id => batch[Array.IndexOf(ds.SampleIds, id)]).ToArray());

        Assert.Single(cov);
        Assert.Equal("batch", cov[0].Name);
        Assert.Null(DetectionAnalysis.CovariatesFor(ds, Array.Empty<string>(), det.SampleIds));
        // Same values in the same (detection) order as building the covariate by hand.
        var built = DetectionGlm.Run(det.Matrix, det.PeptideIds, new[] { 0, 2, 4, 6 }, new[] { 1, 3, 5, 7 }, cov);
        var hand = DetectionGlm.Run(det.Matrix, det.PeptideIds, new[] { 0, 2, 4, 6 }, new[] { 1, 3, 5, 7 }, new[] { direct });
        Assert.Equal(hand.Rows.Select(h => h.P), built.Rows.Select(x => x.P));
    }

    [Fact]
    public void Trend_IsRefused()
    {
        var (ds, det, a, b) = Setup();
        Assert.Throws<ArgumentException>(() => DetectionAnalysis.Run(det, ds, a, b,
            DifferentialDesign.LinearTrend, null, null, MultipleTesting.BenjaminiHochberg));
    }

    [Fact]
    public void SamplesMissingFromMergedData_AreCounted_AndAllMissingIsRefused()
    {
        var (ds, det, a, b) = Setup();
        // Drop arm A's first two samples from the detection matrix.
        var gone = new HashSet<string>(new[] { ds.SampleIds[a[0]], ds.SampleIds[a[1]] }, StringComparer.Ordinal);
        var keep = Enumerable.Range(0, det.SampleIds.Length).Where(s => !gone.Contains(det.SampleIds[s])).ToArray();
        var m = new double[Peptides, keep.Length];
        for (var p = 0; p < Peptides; p++)
        for (var k = 0; k < keep.Length; k++)
            m[p, k] = det.Matrix[p, keep[k]];
        var partial = new DetectionMatrixData(det.PeptideIds, keep.Select(s => det.SampleIds[s]).ToArray(), m);

        var r = DetectionAnalysis.Run(partial, ds, a, b, DifferentialDesign.Unpaired, null, null,
            MultipleTesting.BenjaminiHochberg);
        Assert.Equal(2, r.DroppedSamples);
        Assert.Equal(a.Count - 2, r.NA);

        var none = new DetectionMatrixData(det.PeptideIds, new[] { "not-a-sample" }, new double[Peptides, 1]);
        Assert.Throws<DetectionSamplesNotFoundException>(() => DetectionAnalysis.Run(none, ds, a, b,
            DifferentialDesign.Unpaired, null, null, MultipleTesting.BenjaminiHochberg));
    }
}
