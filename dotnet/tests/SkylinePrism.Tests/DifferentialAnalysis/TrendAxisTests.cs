using System;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// A study's timepoint is usually written for a human - <c>V2_Week 8</c> - so a trend could not be
/// fitted against it at all: the picker listed only wholly numeric columns, and the column simply did
/// not appear.
///
/// <para>The hazard is picking the WRONG number rather than finding none. These values hold two, and
/// they are different quantities: on the cohort this was built from, <c>V1_Week 2</c> and
/// <c>V1_Week 4</c> share a visit while <c>V2_Week 4</c> and <c>V1_Week 4</c> share a week, so a rule
/// like "take the first number" yields a plausible axis that is not time. Every readable position is
/// therefore offered, with a preview, and nothing guesses.</para>
/// </summary>
public class TrendAxisTests
{
    // The Verapamil arm of the cohort this was built for, verbatim.
    private static readonly string?[] Verapamil =
    {
        "V0_Week 0", "V1_Week 2", "V2_Week 4", "V3_Week 6", "V4_Week 12",
    };

    // Both arms plus the unassigned rows, as the full run carries them.
    private static readonly string?[] Cohort =
    {
        "V0_Week 0", "V1_Week 2", "V2_Week 4", "V3_Week 6", "V4_Week 12",
        "V-1_Week -2", "V1_Week 4", "V2_Week 8", "na", null, "",
    };

    [Fact]
    public void TheWeekIsReadable_AndIsTheValuesTheStudyUses()
    {
        var week = Assert.Single(
            TrendAxis.OptionsFor("Longitudinal Draw Description", Verapamil),
            o => o.Label.Contains("Week", StringComparison.Ordinal));

        var x = TrendAxis.Read(Verapamil, week);
        Assert.Equal(new[] { 0.0, 2.0, 4.0, 6.0, 12.0 }, x);
    }

    [Fact]
    public void TheVisitIsOfferedToo_AndIsADifferentAxis()
    {
        // Both parse, and they disagree - which is exactly why neither is chosen for the user.
        var options = TrendAxis.OptionsFor("Longitudinal Draw Description", Verapamil);
        Assert.Equal(2, options.Count);

        var visit = options.Single(o => o.Label.Contains("(V)", StringComparison.Ordinal));
        var week = options.Single(o => o.Label.Contains("(Week)", StringComparison.Ordinal));

        Assert.Equal(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 }, TrendAxis.Read(Verapamil, visit));
        Assert.Equal(new[] { 0.0, 2.0, 4.0, 6.0, 12.0 }, TrendAxis.Read(Verapamil, week));
    }

    [Fact]
    public void OptionsAreNamedByTheWordInFrontOfTheNumber()
    {
        var labels = TrendAxis.OptionsFor("Longitudinal Draw Description", Cohort)
            .Select(o => o.Label).ToList();
        Assert.Contains("Longitudinal Draw Description (Week)", labels);
        Assert.Contains("Longitudinal Draw Description (V)", labels);
    }

    [Fact]
    public void EachAxisHasAShortNameForAPlotAxis()
    {
        // The full label tells two readings of one column apart; as an axis title it ran past both
        // ends of the plot. The short name is the word the number was read after.
        var options = TrendAxis.OptionsFor("Longitudinal Draw Description", Verapamil);
        Assert.Equal("Week", options.Single(o => o.Label.EndsWith("(Week)", StringComparison.Ordinal)).Short);
        Assert.Equal("V", options.Single(o => o.Label.EndsWith("(V)", StringComparison.Ordinal)).Short);

        // A plain numeric column is already its own short name.
        Assert.Equal("Week", Assert.Single(TrendAxis.OptionsFor("Week", new string?[] { "0", "2", "4" })).Short);
    }

    [Fact]
    public void AValueWithNoNumberReadsAsNaN_AndDoesNotDisqualifyTheColumn()
    {
        // "na" carries no timepoint; TrendSamples drops a non-finite x and says how many.
        var week = TrendAxis.OptionsFor("Longitudinal Draw Description", Cohort)
            .Single(o => o.Label.Contains("(Week)", StringComparison.Ordinal));
        var x = TrendAxis.Read(Cohort, week);

        Assert.Equal(-2.0, x[5]);
        Assert.True(double.IsNaN(x[8]));  // "na"
        Assert.True(double.IsNaN(x[9]));  // null
        Assert.True(double.IsNaN(x[10])); // empty
    }

    [Fact]
    public void NegativeNumbersSurvive()
    {
        // V-1_Week -2 is two weeks BEFORE the drug; a trend that read it as +2 would run backwards.
        var week = TrendAxis.OptionsFor("t", new string?[] { "V-1_Week -2", "V0_Week 0", "V1_Week 2" })
            .Single(o => o.Label.Contains("(Week)", StringComparison.Ordinal));
        Assert.Equal(new[] { -2.0, 0.0, 2.0 }, TrendAxis.Read(
            new string?[] { "V-1_Week -2", "V0_Week 0", "V1_Week 2" }, week));
    }

    [Fact]
    public void APositionSomeValuesCannotReach_IsNotOffered()
    {
        // "Day 3" has one number where the others have two. Reading position 2 for some samples and
        // nothing for the rest would be two readings of one column, not one axis.
        var options = TrendAxis.OptionsFor("t", new string?[] { "V1_Week 2", "V2_Week 4", "Day 3" });
        Assert.All(options, o => Assert.Equal(1, o.Position));
    }

    [Fact]
    public void AWhollyNumericColumn_StaysOneOptionWithNothingToExplain()
    {
        var options = TrendAxis.OptionsFor("Week", new string?[] { "0", "2", "4", "12" });
        var only = Assert.Single(options);
        Assert.Equal(0, only.Position);
        Assert.Equal("Week", only.Label);
        Assert.Empty(only.Preview);
        Assert.Equal(new[] { 0.0, 2.0, 4.0, 12.0 },
            TrendAxis.Read(new string?[] { "0", "2", "4", "12" }, only));
    }

    [Theory]
    [InlineData("one value")]        // a constant cannot carry a slope
    [InlineData("no numbers here")]  // nothing to read
    public void AColumnThatCannotCarryASlope_IsNotOffered(string value)
        => Assert.Empty(TrendAxis.OptionsFor("t", new string?[] { value, value, value }));

    [Fact]
    public void EverySampleAtATimepointSharesItsValue()
    {
        // The shape of a real cohort, and the one the first draft crashed on: ten subjects all read
        // "V1_Week 2", so a map keyed by sample rather than by distinct value throws on load.
        var repeated = Enumerable.Range(0, 10)
            .SelectMany(_ => new string?[] { "V0_Week 0", "V1_Week 2", "V4_Week 12" })
            .ToArray();

        var week = TrendAxis.OptionsFor("t", repeated)
            .Single(o => o.Label.Contains("(Week)", StringComparison.Ordinal));

        Assert.Equal(3, week.Distinct);
        Assert.Equal(30, week.Covered);
        Assert.Equal(new[] { 0.0, 2.0, 12.0 }, TrendAxis.Read(repeated, week).Take(3));
    }

    [Fact]
    public void AnUnambiguousAxisLeads_AndInterpretedOnesFollowInColumnOrder()
    {
        // A column that IS a number needed no interpretation, so it leads; the interpreted ones
        // follow in the metadata's own order. No attempt is made to rank how timepoint-like a column
        // looks: two such rules were tried on this cohort and both led with an identifier - a patient
        // id outranked the timepoint on distinct values, then a near-constant column outranked it on
        // repetition. Callers default to an unambiguous axis and make the user pick an interpreted one.
        var subjects = Enumerable.Range(0, 19)
            .SelectMany(s => Enumerable.Repeat<string?>($"SpeakeLab{100000 + s}", 4)).ToArray();
        var weeks = Enumerable.Range(0, 76)
            .Select(i => (string?)new[] { "V0_Week 0", "V1_Week 2", "V2_Week 4", "V3_Week 6",
                "V4_Week 12", "V-1_Week -2", "V2_Week 8" }[i % 7]).ToArray();
        var order = Enumerable.Range(0, 76).Select(i => (string?)i.ToString()).ToArray();

        var axes = TrendAxis.AllFor(
            new[] { "BRI Subject ID", "Longitudinal Draw Description", "MS_Run_Order" },
            c => c switch
            {
                "BRI Subject ID" => subjects,
                "Longitudinal Draw Description" => weeks,
                _ => order,
            });

        Assert.Equal(0, axes[0].Position);                 // the plain number first
        Assert.Equal("MS_Run_Order", axes[0].Column);
        Assert.All(axes.Skip(1), a => Assert.NotEqual(0, a.Position));
    }

    [Fact]
    public void ThePreviewShowsWhatTheParseDid()
    {
        var week = TrendAxis.OptionsFor("t", Verapamil)
            .Single(o => o.Label.Contains("(Week)", StringComparison.Ordinal));
        var line = TrendAxis.DescribePreview(week);

        // Ordered by the NUMBER, so a reader checks the axis runs the right way and starts where
        // they expect - not by the text, where "V1_Week 2" sorts before "V0_Week 0" would not.
        Assert.StartsWith("V0_Week 0 -> 0", line, StringComparison.Ordinal);
        Assert.Contains("V4_Week 12 -> 12", line, StringComparison.Ordinal);
    }

    [Fact]
    public void AnIdentifierLikeNumberIsNotMistakenForAnAxis()
    {
        // Exponent notation is deliberately not read: "1e5" in a label is an identifier far more
        // often than a number, and an axis read out of a sample name is the wrong kind of surprise.
        var options = TrendAxis.OptionsFor("t", new string?[] { "S1e5", "S2e5", "S3e5" });
        Assert.All(options, o => Assert.Equal(1, o.Position));
        Assert.Equal(new[] { 1.0, 2.0, 3.0 },
            TrendAxis.Read(new string?[] { "S1e5", "S2e5", "S3e5" }, options[0]));
    }
}
