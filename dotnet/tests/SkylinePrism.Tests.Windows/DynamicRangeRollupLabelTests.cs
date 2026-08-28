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
        // The axis travels with the image when the plot is copied into a slide; the status line does not.
        var src = File.ReadAllText(Path.Combine(AppDir, "MainWindow.DynamicRange.cs"));
        Assert.Contains("yLabel: rollup.Length > 0 ? $\"Log10 abundance ({rollup})\"", src);
    }
}
