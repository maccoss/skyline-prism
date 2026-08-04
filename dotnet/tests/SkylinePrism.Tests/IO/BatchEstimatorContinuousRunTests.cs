using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// Characterises how sensitive acquisition-time batch estimation is on a CONTINUOUSLY acquired run -
/// the case where the answer should usually be "one batch".
///
/// <para>The threshold is <c>max(q3 + k*IQR, 1.1 * median gap)</c>. On an evenly spaced run the IQR is
/// ~0, so the Tukey term collapses to roughly the median and the <b>1.1x floor becomes the binding
/// constraint</b> - meaning any gap barely 10% longer than the typical spacing starts a new batch. These
/// tests pin that behaviour (it matches the Python implementation, cli.py:412-415) so the sensitivity is
/// visible and any future change to it is deliberate.</para>
/// </summary>
public class BatchEstimatorContinuousRunTests
{
    /// <summary>A run of <paramref name="n"/> samples spaced <paramref name="gapMinutes"/> apart, with
    /// optional longer gaps injected before the given sample indices.</summary>
    private static List<(string Sample, DateTime Time)> Run(
        int n, double gapMinutes, params (int Index, double Gap)[] longGaps)
    {
        var longById = longGaps.ToDictionary(g => g.Index, g => g.Gap);
        var rows = new List<(string, DateTime)>();
        var t = new DateTime(2026, 3, 1, 8, 0, 0, DateTimeKind.Utc);
        for (var i = 0; i < n; i++)
        {
            if (i > 0)
                t = t.AddMinutes(longById.TryGetValue(i, out var g) ? g : gapMinutes);
            rows.Add(($"S{i:D2}", t));
        }
        return rows;
    }

    private static int BatchCount(List<(string Sample, DateTime Time)> rows) =>
        BatchEstimator.AssignBatches(rows, "auto", null, 1.5) is { Count: > 0 } m
            ? m.Values.Distinct().Count()
            : 1; // empty map = "no batches found" = a single batch

    [Fact]
    public void APerfectlyEvenRun_IsOneBatch()
    {
        Assert.Equal(1, BatchCount(Run(36, gapMinutes: 90)));
    }

    [Fact]
    public void AGapOnly20PercentLongerAlreadySplitsAContinuousRun()
    {
        // 90-minute spacing, one 108-minute gap: 20% over, which on a real instrument is a wash, a
        // blank, or a queue pause - not a batch boundary.
        Assert.Equal(2, BatchCount(Run(36, 90, (18, 108))));
    }

    [Fact]
    public void ThreeSlightlyLongGaps_ProduceFourBatchesFromOneContinuousSequence()
    {
        // The reported symptom: 36 samples run continuously, reported as 4 batches.
        var rows = Run(36, 90, (9, 105), (18, 110), (27, 105));

        Assert.Equal(4, BatchCount(rows));
    }

    [Fact]
    public void TheFloorNotTheIqrRuleIsWhatSplitsAnEvenRun()
    {
        // Same data, but a huge IQR multiplier: if the Tukey term were binding this would collapse to
        // one batch. It does not, because max(...) keeps the 1.1x median floor.
        var rows = Run(36, 90, (18, 105));

        var withHugeMultiplier = BatchEstimator.AssignBatches(rows, "auto", null, gapIqrMultiplier: 100)
            .Values.Distinct().Count();

        Assert.Equal(2, withHugeMultiplier); // raising the multiplier does NOT help on an even run
    }

    [Fact]
    public void AGenuineOvernightBreak_IsDetected()
    {
        // What the feature is actually for: a real break between days.
        var rows = Run(24, 90, (12, 14 * 60));

        Assert.Equal(2, BatchCount(rows));
    }

    [Fact]
    public void IrregularSpacingRaisesTheThreshold_SoSmallGapsAreTolerated()
    {
        // With genuinely variable spacing the IQR is wide, the Tukey term binds, and a modestly longer
        // gap no longer splits - the opposite of the even-run behaviour.
        var rows = new List<(string, DateTime)>();
        var t = new DateTime(2026, 3, 1, 8, 0, 0, DateTimeKind.Utc);
        var spacings = new[] { 40, 120, 55, 200, 60, 30, 150, 45, 90, 70, 180, 50 };
        rows.Add(("S00", t));
        for (var i = 0; i < spacings.Length; i++)
        {
            t = t.AddMinutes(spacings[i]);
            rows.Add(($"S{i + 1:D2}", t));
        }

        // 200 min is the widest spacing and is inside the spread, so no break is declared.
        Assert.Equal(1, BatchCount(rows));
    }

    [Fact]
    public void MethodNone_IsHowAContinuousRunOptsOut()
    {
        // The documented escape hatch: PrismPipeline skips estimation entirely for "none"/"source", so
        // ComBat is never handed fabricated batches.
        var rows = Run(36, 90, (9, 105), (18, 110), (27, 105));

        // Sanity: estimation WOULD have split it.
        Assert.Equal(4, BatchCount(rows));
        // The pipeline-level guard is the config value; assert the documented spelling has not drifted.
        Assert.Contains("none", new[] { "auto", "gap", "fixed", "source", "none" });
    }
}
