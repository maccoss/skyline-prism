using System;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// A trend pools every sample that has a value on its axis, which silently merges studies sharing a
/// timepoint scale - on the cohort this was built for, a Verapamil arm and a Liraglutide arm both
/// measured "at week 4" became one slope through both. Restricting the samples is how the meant one
/// is fitted, and these pin what a restriction keeps, what it refuses, and that the command a report
/// records carries it.
/// </summary>
public class TrendRestrictionTests
{
    private static DifferentialDataset Mini() =>
        DifferentialDataset.Load(Fixtures.Path2("mini", "e2e-sum", "output"), FeatureLevel.Protein);

    private static QuantRequest TrendRequest(DifferentialDataset ds, params QuantRestriction[] restrictions) => new()
    {
        OutputDir = "out",
        Dataset = ds,
        Options = new DifferentialOptions { Design = DifferentialDesign.LinearTrend, TrendColumn = "x" },
        Rule = SignificanceRule.Default,
        Restrictions = restrictions,
    };

    [Fact]
    public void NoRestriction_KeepsEverySample()
    {
        var ds = Mini();
        Assert.Equal(ds.SampleIds.Length, QuantAnalysis.TrendColumnsFor(TrendRequest(ds), ds).Length);
    }

    [Fact]
    public void ARestriction_KeepsOnlyTheMatchingSamples()
    {
        var ds = Mini();
        var types = ds.MetadataValues("sample_type");
        var expected = Enumerable.Range(0, types.Length).Where(i => types[i] == "experimental").ToArray();

        var kept = QuantAnalysis.TrendColumnsFor(
            TrendRequest(ds, new QuantRestriction("sample_type", new[] { "experimental" })), ds);

        Assert.Equal(expected, kept);
    }

    [Fact]
    public void SeveralRestrictions_NarrowTogether()
    {
        // Intersected, not last-one-wins: "experimental" AND "reference" is no sample at all, which
        // is refused rather than fitted as either.
        var ds = Mini();
        var request = TrendRequest(ds,
            new QuantRestriction("sample_type", new[] { "experimental" }),
            new QuantRestriction("sample_type", new[] { "reference" }));

        Assert.Throws<ArgumentException>(() => QuantAnalysis.TrendColumnsFor(request, ds));
    }

    [Fact]
    public void ARestrictionToAValueTheColumnNeverHolds_IsRefused_NamingWhatIsThere()
    {
        // Silently fitting zero samples would report "nothing to fit" and send the reader to their
        // data, when the fault is in the restriction.
        var ds = Mini();
        var request = TrendRequest(ds, new QuantRestriction("sample_type", new[] { "no-such-type" }));
        var ex = Assert.Throws<ArgumentException>(() => QuantAnalysis.TrendColumnsFor(request, ds));
        Assert.Contains("'sample_type' has no value 'no-such-type'. Present:", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARestrictionOnAnUnknownColumn_IsRefused_NotIgnored()
    {
        // Ignored, the trend would be fitted over every sample while the report's recorded command
        // claimed the restriction. The CLI checked this; the shared path every request runs through
        // did not.
        var ds = Mini();
        var request = TrendRequest(ds, new QuantRestriction("no_such_column", new[] { "x" }));
        var ex = Assert.Throws<ArgumentException>(() => QuantAnalysis.TrendColumnsFor(request, ds));
        Assert.Contains("no metadata column 'no_such_column'", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void KeptColumns_IsThePanesRestriction_AndKeepsEverySampleWithoutOne()
    {
        var ds = Mini();
        Assert.Equal(Enumerable.Range(0, ds.SampleIds.Length), QuantAnalysis.KeptColumns(Array.Empty<QuantRestriction>(), ds));
        Assert.Equal(
            QuantAnalysis.TrendColumnsFor(TrendRequest(ds, new QuantRestriction("sample_type", new[] { "experimental" })), ds),
            QuantAnalysis.KeptColumns(new[] { new QuantRestriction("sample_type", new[] { "experimental" }) }, ds));
    }

    [Fact]
    public void ARestrictedValueStartingWithADash_StillGetsACommand()
    {
        // It travels as COLUMN=VALUE, so the argument never starts with '-' and cannot be read as a
        // flag; refusing it withheld a perfectly reproducible command (a visit "-1", say).
        var ds = Mini();
        var request = TrendRequest(ds, new QuantRestriction("sample_type", new[] { "-1" }));

        Assert.True(QuantCommand.TryArguments(request, out var args, out var reason), reason);
        Assert.Equal("sample_type=-1", args[args.ToList().IndexOf("--restrict-to") + 1]);
    }

    [Fact]
    public void TheRecordedCommand_CarriesTheRestriction()
    {
        // Left out, the command reproduces the analysis over a DIFFERENT set of samples - every one
        // with a value on the axis - and nothing in its output says so.
        var ds = Mini();
        var request = TrendRequest(ds, new QuantRestriction("sample_type", new[] { "experimental", "qc" }));

        Assert.True(QuantCommand.TryArguments(request, out var args, out var reason), reason);
        var at = args.ToList().IndexOf("--restrict-to");
        Assert.True(at >= 0, "the command has no --restrict-to");
        Assert.Equal("sample_type=experimental,qc", args[at + 1]);
    }

    [Fact]
    public void EachRestriction_IsItsOwnFlag()
    {
        // One flag per restriction - the form the CLI intersects. A single flag carrying two columns
        // would need a separator the parser does not take.
        var ds = Mini();
        var request = TrendRequest(ds,
            new QuantRestriction("sample_type", new[] { "experimental" }),
            new QuantRestriction("batch", new[] { "b1" }));

        Assert.True(QuantCommand.TryArguments(request, out var args, out _));
        Assert.Equal(2, args.Count(a => a == "--restrict-to"));
    }

    [Fact]
    public void ARestrictedValueTheCommandLineCannotCarry_GivesNoCommand()
    {
        // A comma inside a value would be split into two values by the CLI - and select something else.
        var ds = Mini();
        var request = TrendRequest(ds, new QuantRestriction("sample_type", new[] { "AD, early" }));

        Assert.False(QuantCommand.TryArguments(request, out _, out var reason));
        Assert.Contains("comma", reason, StringComparison.Ordinal);
    }
}
