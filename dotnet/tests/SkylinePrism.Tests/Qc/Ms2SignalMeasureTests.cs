using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Choosing between totalling signal and totalling ions.
///
/// <para>The column selection is the whole feature, so it is tested directly rather than through a
/// cohort: what matters is that "ions" reaches Skyline's ion-count column, that "signal" sums gross
/// area, and above all that asking for ions on an export that has none does not quietly answer the
/// other question - the two numbers look alike and nothing downstream could tell them apart.</para>
/// </summary>
public class Ms2SignalMeasureTests
{
    /// <summary>Column names as Skyline's own export spells them.</summary>
    private static readonly string[] WithIons =
    {
        "Sample ID", "Peptide Modified Sequence", "Fragment Ion", "Area", "Background",
        "Precursor Mz", "Product Mz", "Start Time", "End Time",
        "LC Peak Transition Ion Count",
    };

    private static readonly string[] AreaOnly =
    {
        "Sample ID", "Peptide Modified Sequence", "Fragment Ion", "Area",
        "Precursor Mz", "Product Mz", "Start Time", "End Time",
    };

    [Fact]
    public void IonsUseSkylinesIonCountColumn()
    {
        var cols = Ms2SignalRegions.Resolve(WithIons);
        Assert.NotNull(cols);
        Assert.True(cols!.HasIonCounts);

        var sql = Ms2SignalRegions.GrossSignalSql(cols, Ms2SignalMeasure.Ions);
        Assert.Contains("LC Peak Transition Ion Count", sql, StringComparison.Ordinal);
        // Ions are already gross and already a count - nothing may be added to them.
        Assert.DoesNotContain("Background", sql, StringComparison.Ordinal);
        Assert.DoesNotContain("\"Area\"", sql, StringComparison.Ordinal);
    }

    [Fact]
    public void SignalSumsAreaPlusBackground()
    {
        var cols = Ms2SignalRegions.Resolve(WithIons)!;
        var sql = Ms2SignalRegions.GrossSignalSql(cols, Ms2SignalMeasure.Signal);

        Assert.Contains("Area", sql, StringComparison.Ordinal);
        Assert.Contains("Background", sql, StringComparison.Ordinal);
        // Even when ion counts are present, asking for signal must not reach for them.
        Assert.DoesNotContain("Ion Count", sql, StringComparison.Ordinal);
    }

    /// <summary>An export without the Background column falls back to net area, still not to ions.</summary>
    [Fact]
    public void SignalWithoutBackgroundIsNetArea()
    {
        var cols = Ms2SignalRegions.Resolve(AreaOnly)!;
        Assert.False(cols.HasIonCounts);

        var sql = Ms2SignalRegions.GrossSignalSql(cols, Ms2SignalMeasure.Signal);
        Assert.Contains("Area", sql, StringComparison.Ordinal);
        Assert.DoesNotContain("Background", sql, StringComparison.Ordinal);
    }

    /// <summary>
    /// Ions requested on an export that has none falls back to area rather than producing nothing -
    /// but the caller is told, because the fallback answers a different question.
    /// </summary>
    [Fact]
    public void IonsOnAnOldExportFallBackToArea()
    {
        var cols = Ms2SignalRegions.Resolve(AreaOnly)!;
        var sql = Ms2SignalRegions.GrossSignalSql(cols, Ms2SignalMeasure.Ions);

        Assert.Contains("Area", sql, StringComparison.Ordinal);
        Assert.DoesNotContain("Ion Count", sql, StringComparison.Ordinal);
    }

    /// <summary>
    /// The APEX columns must never be picked up. An apex value is the single spectrum holding the
    /// highest transition intensity - a peak-shape statistic, not a total - so accepting it would
    /// silently answer a different question with a plausible-looking number.
    /// </summary>
    [Fact]
    public void ApexColumnsAreNotMistakenForTotals()
    {
        var apexOnly = AreaOnly.Concat(new[]
        {
            "Apex Transition Ion Count", "Apex Analyte Ion Count Fragment", "Apex Total Ion Count",
        }).ToArray();

        var cols = Ms2SignalRegions.Resolve(apexOnly)!;
        Assert.False(cols.HasIonCounts);
        Assert.DoesNotContain(
            "Apex", Ms2SignalRegions.GrossSignalSql(cols, Ms2SignalMeasure.Ions),
            StringComparison.Ordinal);
    }

    /// <summary>The config accepts both measures and rejects anything else by name.</summary>
    [Fact]
    public void ConfigValidatesTheMeasure()
    {
        foreach (var value in new[] { "signal", "ions", "SIGNAL", "Ions" })
        {
            var config = new PrismConfig();
            config.QcReport.Ms2Signal.Measure = value;
            config.Validate();   // must not throw
        }

        var bad = new PrismConfig();
        bad.QcReport.Ms2Signal.Measure = "ion";   // singular - a plausible typo
        var ex = Assert.Throws<NotSupportedException>(() => bad.Validate());
        Assert.Contains("ions", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>The measure travels on the result, because it is not recoverable from the numbers.</summary>
    [Fact]
    public void TheResultRemembersWhichMeasureProducedIt()
    {
        var rows = new[]
        {
            new Ms2SignalAccounting.Row(
                "S1", "experimental", 100, 120, Array.Empty<double>(),
                10, 9, 2, 1, 0, 0, 0, 0, 20, 0),
        };

        var signal = new Ms2SignalAccounting.Result(
            rows, Array.Empty<string>(), Array.Empty<string>(), Array.Empty<int>(),
            1, "+/-10 ppm", "scheme", true);
        var ions = signal with { Measure = Ms2SignalMeasure.Ions };

        Assert.Equal(Ms2SignalMeasure.Signal, signal.Measure);   // the default
        Assert.Equal(Ms2SignalMeasure.Ions, ions.Measure);
    }
    /// <summary>
    /// The same question the Skyline tool asks of a pre-exported report before it offers "ions": does
    /// this file carry the column? Spelled as Skyline exports it, as the merge might rewrite it, and
    /// never satisfied by an Apex column.
    /// </summary>
    [Fact]
    public void FindIonCountColumn_AcceptsTheLcPeakSpellingsAndNotApex()
    {
        Assert.Equal("LC Peak Transition Ion Count",
            Ms2SignalRegions.FindIonCountColumn(new[] { "Area", "LC Peak Transition Ion Count" }));
        Assert.Equal("LCPeakTransitionIonCount",
            Ms2SignalRegions.FindIonCountColumn(new[] { "Area", "LCPeakTransitionIonCount" }));
        Assert.Null(Ms2SignalRegions.FindIonCountColumn(new[] { "Area", "Background" }));
        Assert.Null(Ms2SignalRegions.FindIonCountColumn(new[] { "Area", "Apex Transition Ion Count" }));
    }
}
