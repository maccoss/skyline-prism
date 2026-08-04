using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Which samples reach a QC plot. Getting this wrong renders a perfectly plausible plot of the wrong
/// samples with nothing to indicate a problem, so the selection rules are pinned rather than left to the
/// window.
/// </summary>
public class QcGroupFilterTests
{
    // Sample Type as the Replicates report spells it: 2 standards, 2 QC, 4 unknowns.
    private static readonly string[] Types =
        { "Standard", "Standard", "Quality Control", "Quality Control", "Unknown", "Unknown", "Unknown", "Unknown" };

    private static List<int> Match(params string[] selected) =>
        QcGroupFilter.Matching(Types.Length, i => Types[i],
            new HashSet<string>(selected, QcGroupFilter.Comparer));

    [Fact]
    public void NothingTicked_ShowsEverySample()
    {
        // An untouched dropdown must not silently filter; empty means "no filter", not "nothing matched".
        Assert.Equal(Enumerable.Range(0, Types.Length), Match());
    }

    [Fact]
    public void TwoValuesTicked_CombineRatherThanReplace()
    {
        // The case this feature exists for: both control types, without the unknowns.
        var idx = Match("Standard", "Quality Control");

        Assert.Equal(new[] { 0, 1, 2, 3 }, idx);
        Assert.DoesNotContain(idx, i => Types[i] == "Unknown");
    }

    [Fact]
    public void OneValueTicked_StillWorksAsBefore()
        => Assert.Equal(new[] { 2, 3 }, Match("Quality Control"));

    [Fact]
    public void MatchingIsCaseInsensitive()
    {
        // PRISM's synthetic column says "reference"/"qc"; the Replicates report says "Standard"/"Quality
        // Control". Neither spelling should depend on case.
        Assert.Equal(new[] { 0, 1 }, Match("STANDARD"));
    }

    [Fact]
    public void AValueThatMatchesNothing_YieldsAnEmptySelectionNotEverything()
    {
        // Must not fall back to "all" - that would quietly plot samples the user excluded.
        Assert.Empty(Match("Blank"));
    }

    [Theory]
    [InlineData("Standard")]
    [InlineData("Quality Control")]
    [InlineData("QC")]
    [InlineData("reference")]
    [InlineData("qc")]
    [InlineData("STANDARD")]
    public void ControlValuesAreRecognisedInBothVocabularies(string value)
        => Assert.True(QcGroupFilter.IsControlValue(value));

    [Theory]
    [InlineData("Unknown")]
    [InlineData("experimental")]
    [InlineData("Blank")]
    [InlineData("Plate1")]
    [InlineData(null)]
    public void NonControlValuesAreNot(string? value)
        => Assert.False(QcGroupFilter.IsControlValue(value));

    [Fact]
    public void ControlsAmong_PicksOnlyTheControlTypesPresent()
    {
        var controls = QcGroupFilter.ControlsAmong(new[] { "Standard", "Quality Control", "Unknown" });

        Assert.Equal(new[] { "Standard", "Quality Control" }, controls);
    }

    [Fact]
    public void ControlsAmong_IsEmptyForANonSampleTypeColumn()
    {
        // Grouping by a Condition annotation has no controls; the caller uses this to leave the selection
        // alone rather than tick nothing and render an empty plot.
        Assert.Empty(QcGroupFilter.ControlsAmong(new[] { "Treated", "Untreated", "Plate1" }));
    }

    [Fact]
    public void Describe_ReadsAsAPlotSubtitle()
    {
        Assert.Equal("all samples", QcGroupFilter.Describe(Array.Empty<string>()));
        Assert.Equal("Quality Control + Standard",
            QcGroupFilter.Describe(new[] { "Standard", "Quality Control" }));
    }

    [Fact]
    public void Summarize_CollapsesEverythingTickedBackToAll()
    {
        // All ticked and none ticked show the same samples, so they should read the same.
        Assert.Equal("All", QcGroupFilter.Summarize(Array.Empty<string>(), 3));
        Assert.Equal("All", QcGroupFilter.Summarize(new[] { "a", "b", "c" }, 3));
        Assert.Equal("a, b", QcGroupFilter.Summarize(new[] { "a", "b" }, 3));
        Assert.Equal("Standard", QcGroupFilter.Summarize(new[] { "Standard" }, 3));
    }

    [Fact]
    public void Summarize_HandlesAnEmptyValueList()
        => Assert.Equal("All", QcGroupFilter.Summarize(Array.Empty<string>(), 0));

    // ---- Multi-select is not specific to Sample Type ----
    // The Group-by list is every column in the Replicates report (Condition, Subject, Plate, Timepoint,
    // ...), and the value list is built from whichever column is chosen. Nothing in the filter knows or
    // cares which column it is; only the control-correlation DEFAULT looks for sample types, and it
    // no-ops elsewhere.

    private static readonly string[] Conditions =
        { "Treated", "Treated", "Untreated", "Untreated", "Vehicle", "Vehicle" };

    [Fact]
    public void AnArbitraryAnnotationColumnSupportsTheSameMultiSelect()
    {
        var selected = new HashSet<string>(new[] { "Treated", "Vehicle" }, QcGroupFilter.Comparer);

        var idx = QcGroupFilter.Matching(Conditions.Length, i => Conditions[i], selected);

        Assert.Equal(new[] { 0, 1, 4, 5 }, idx);
        Assert.DoesNotContain(idx, i => Conditions[i] == "Untreated");
    }

    [Fact]
    public void AnArbitraryAnnotationColumnStillTreatsEmptyAsNoFilter()
        => Assert.Equal(
            Enumerable.Range(0, Conditions.Length),
            QcGroupFilter.Matching(Conditions.Length, i => Conditions[i],
                new HashSet<string>(QcGroupFilter.Comparer)));

    [Fact]
    public void DescribeAndSummarizeAreColumnAgnostic()
    {
        Assert.Equal("Treated + Vehicle", QcGroupFilter.Describe(new[] { "Vehicle", "Treated" }));
        Assert.Equal("Treated, Vehicle", QcGroupFilter.Summarize(new[] { "Treated", "Vehicle" }, 3));
    }

    [Fact]
    public void ControlDefaultingLeavesANonSampleTypeColumnAlone()
    {
        // Selecting Control correlation while grouped by Condition must not tick anything: there are no
        // control values to tick, and ticking none would render an empty plot instead of all samples.
        Assert.Empty(QcGroupFilter.ControlsAmong(Conditions));
    }
}
