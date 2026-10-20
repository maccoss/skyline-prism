using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Library;
using Xunit;

namespace SkylinePrism.Tests.Library;

/// <summary>
/// Which injections the fragment noise is measured within, and the two files the noise-gated library
/// fit leaves beside the outputs. The arithmetic is pinned by <see cref="NoiseGatedRatioGoldenTests"/>.
/// </summary>
public class FragmentNoisePlanTests
{
    private static Dictionary<string, string> Types(params (string Sample, string Type)[] pairs) =>
        pairs.ToDictionary(p => p.Sample, p => p.Type, StringComparer.Ordinal);

    [Fact]
    public void Plan_UsesEachPoolWithThreeInjections_KeptApart()
    {
        var samples = new[] { "r1", "e1", "r2", "q1", "r3", "q2", "e2" };
        var types = Types(("r1", "reference"), ("r2", "reference"), ("r3", "reference"),
            ("q1", "qc"), ("q2", "qc"), ("e1", "experimental"), ("e2", "experimental"));

        var plan = FragmentNoise.Plan(samples, types);

        // Two QC injections are too few for a group; the reference pool alone is used, in run order.
        Assert.Equal(FragmentNoiseCurve.PoolReplicates, plan.Source);
        var group = Assert.Single(plan.Groups);
        Assert.Equal("reference", group.Name);
        Assert.Equal(new[] { "r1", "r2", "r3" }, group.Samples);
    }

    [Fact]
    public void Plan_FallsBackToAllSamples_WithoutAPool_AndLeavesBlanksOut()
    {
        var samples = new[] { "a", "b", "c", "blank1" };
        var types = Types(("a", "experimental"), ("b", "experimental"), ("c", "qc"), ("blank1", "blank"));

        var plan = FragmentNoise.Plan(samples, types);

        Assert.Equal(FragmentNoiseCurve.AllSamples, plan.Source);
        Assert.Equal(new[] { "a", "b", "c" }, Assert.Single(plan.Groups).Samples);
    }

    [Fact]
    public void Plan_IsNone_BelowThreeUsableSamples()
    {
        var plan = FragmentNoise.Plan(new[] { "a", "b" }, Types(("a", "experimental"), ("b", "experimental")));
        Assert.Equal(FragmentNoiseCurve.None, plan.Source);
        Assert.Empty(plan.Groups);
    }

    [Fact]
    public void PlanKey_ChangesWithPoolMembership_NotWithOtherSamples()
    {
        var samples = new[] { "r1", "r2", "r3", "e1" };
        var types = Types(("r1", "reference"), ("r2", "reference"), ("r3", "reference"), ("e1", "experimental"));
        var key = FragmentNoise.Plan(samples, types).Key;

        var moreExperimental = FragmentNoise.Plan(samples.Append("e2").ToArray(),
            new Dictionary<string, string>(types) { ["e2"] = "experimental" }).Key;
        Assert.Equal(key, moreExperimental);

        var retyped = new Dictionary<string, string>(types) { ["e1"] = "reference" };
        Assert.NotEqual(key, FragmentNoise.Plan(samples, retyped).Key);
    }

    [Fact]
    public void NoiseCurve_SurvivesSaveAndLoad()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_noise_" + Guid.NewGuid().ToString("N") + ".json");
        try
        {
            var curve = new FragmentNoiseCurve
            {
                Source = FragmentNoiseCurve.PoolReplicates,
                Groups = new[] { new FragmentNoiseCurve.GroupSummary("reference", 7) },
                Cells = 1234,
                NoiseLevel = new[] { 10.5, 1000.25 },
                NoiseSd = new[] { 30.1, 400.9 },
                SignalLevel = new[] { 100.0 },
                SignalCv = new[] { 42.0 },
            };
            curve.Save(path);
            var back = FragmentNoiseCurve.Load(path)!;

            Assert.Equal(curve.Source, back.Source);
            Assert.Equal(curve.Groups, back.Groups);
            Assert.Equal(curve.Cells, back.Cells);
            Assert.Equal(curve.NoiseLevel, back.NoiseLevel);
            Assert.Equal(curve.NoiseSd, back.NoiseSd);
            Assert.Equal(curve.SignalLevel, back.SignalLevel);
            Assert.Equal(curve.SignalCv, back.SignalCv);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void NoiseCurve_LoadIsNull_ForAMissingOrBrokenFile()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_noise_" + Guid.NewGuid().ToString("N") + ".json");
        Assert.Null(FragmentNoiseCurve.Load(path));
        try
        {
            File.WriteAllText(path, "{ not json");
            Assert.Null(FragmentNoiseCurve.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void PrecisionFile_RoundTripsNaNAndQuotedPeptides()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_precision_" + Guid.NewGuid().ToString("N") + ".csv");
        try
        {
            var rows = new List<(string, int, PrecisionRow)>
            {
                ("PEPTIDE,K", 6, new PrecisionRow("reference", 7, 1000.5, 800.25, 5.5, 21.5, 20.0)),
                ("\"Q\"ELK", 3, new PrecisionRow("qc", 7, 10, 0, 3, double.NaN, 95.0)),
            };
            LibraryAssistPrecision.Write(path, rows);
            var back = LibraryAssistPrecision.Read(path);

            Assert.Equal(2, back.Count);
            Assert.Equal(rows[0], back[0]);
            Assert.Equal("\"Q\"ELK", back[1].Peptide);
            Assert.True(double.IsNaN(back[1].Row.ObservedCv));
            Assert.True(double.IsNaN(back[1].Row.Ratio));
            Assert.Equal(21.5 / 20.0, back[0].Row.Ratio);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void NoiseGatedRatio_WithNoRemovals_IsTheSumOfMatchedFragments()
    {
        // The identity the method is built around: nothing dropped -> exactly the matched fragments'
        // sum, so on clean data it behaves like sum, which is where its precision comes from.
        var observed = new double[,] { { 900, 1200 }, { 410, 700 }, { 260, 300 }, { 5, 9 } };
        var lib = new[] { 1.0, 0.5, 0.3, double.NaN };
        var fit = LibraryRollup.NoiseGatedRatio(observed, lib, 3, 1.0, 3.0, _ => 1e9);

        Assert.Equal(900 + 410 + 260, fit.Abundance[0], 9);
        Assert.Equal(1200 + 700 + 300, fit.Abundance[1], 9);
        Assert.Equal(new[] { 3, 3 }, fit.Kept);
    }
}
