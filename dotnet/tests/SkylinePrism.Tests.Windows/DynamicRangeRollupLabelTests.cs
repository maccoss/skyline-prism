using System;
using System.IO;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The Dynamic Range tab has to say which rollup produced its values. Comparing it against Skyline's
/// relative-abundance plot is the natural thing to do, and the two are different quantities: Skyline
/// sums peak areas, while median polish estimates the level of a typical peptide. On a real cohort
/// that reorders the top of the plot (C4A leads the summed view with 121 peptides; ITIH2 leads here
/// with 44), and an unlabeled axis makes a legitimate difference look like a defect.
/// </summary>
public class DynamicRangeRollupLabelTests
{
    private static string AppDir => Path.Combine(
        Path.GetDirectoryName(typeof(DynamicRangeRollupLabelTests).Assembly.Location)!,
        "..", "..", "..", "..", "..", "src", "SkylinePrism.App");

    [Theory]
    [InlineData("median_polish", "not a sum")]
    [InlineData("sum", "summed")]
    [InlineData("topn", "partial total")]
    [InlineData("maxlfq", "not a sum")]
    [InlineData("ibaq", "comparing proteins")]
    public void EveryRollupMethod_SaysWhatItsNumbersAre(string method, string expected)
    {
        var meaning = MainWindow.RollupMeaning(method);
        Assert.Contains(expected, meaning, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// The axis says MEAN, because a mean across replicates is what it plots.
    /// </summary>
    /// <remarks>
    /// It read "Log10 abundance (sum)", where the parenthesis named the rollup - how peptides were
    /// combined into a protein - and a reader took it for the quantity on the axis. That is the
    /// difference that sends someone to Skyline's relative-abundance view wondering why the numbers
    /// sit a replicate count apart: Skyline sums across replicates, this averages.
    /// </remarks>
    [Fact]
    public void TheAxisNamesTheMeanAndTheRollupSeparately()
    {
        Assert.Equal("Log10 mean abundance (sum rollup)", MainWindow.RangeYLabel("sum", 82));
        Assert.Equal(
            "Log10 mean abundance (median_polish rollup)",
            MainWindow.RangeYLabel("median_polish", 2));
    }

    /// <summary>A mean of one thing is just the thing, so it is not called a mean.</summary>
    [Fact]
    public void OneReplicateIsNotAveraged()
    {
        Assert.Equal("Log10 abundance (sum rollup)", MainWindow.RangeYLabel("sum", 1));
        Assert.DoesNotContain("mean", MainWindow.RangeYLabel("sum", 1), StringComparison.Ordinal);
    }

    /// <summary>With no rollup known, the axis still says what the quantity is.</summary>
    [Fact]
    public void AnUnknownRollupStillNamesTheMean()
    {
        Assert.Equal("Log10 mean abundance", MainWindow.RangeYLabel("", 12));
        Assert.Equal("Log10 abundance", MainWindow.RangeYLabel("", 1));
    }

    [Fact]
    public void TheMethodNameIsCaseInsensitive()
    {
        Assert.Equal(MainWindow.RollupMeaning("median_polish"), MainWindow.RollupMeaning("Median_Polish"));
    }

    [Fact]
    public void AnUnknownMethod_SaysNothingRatherThanGuessing()
    {
        // A method added later must not be described wrongly; the name alone still reaches the label.
        Assert.Equal("", MainWindow.RollupMeaning("something_new"));
    }

    [Fact]
    public void TheAxisLabelCarriesTheMethod_NotJustTheStatusLine()
    {
        // The axis travels with the image when the plot is copied into a slide; the status line does
        // not. Asserted on the wiring rather than the text, which RangeYLabel now owns and the tests
        // above pin - a source assertion that spelled the label out was one more place to update, and
        // it was the place that noticed last.
        var src = File.ReadAllText(Path.Combine(AppDir, "MainWindow.DynamicRange.cs"));
        Assert.Contains("yLabel: RangeYLabel(rollup, SelectedReplicateCount())", src);
    }
}
