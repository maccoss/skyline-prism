using System;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The Spectrum density pane's opening RT bin, which lives in the XAML and in a Core constant and
/// drifted once: the constant moved from 0.1 to 0.01 min - ten acquisition cycles to about one - and
/// the box kept 0.1 for two releases. Every user of the tool saw the map at a bin the documentation
/// said was not the default, and the widened-bin caveat never showed, because the bin was the one
/// they had asked for. Nothing fails when the two disagree, so a source check is what notices.
/// </summary>
public class DensityPaneDefaultsTests
{
    private static string AppDir =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));

    /// <summary>
    /// About one acquisition cycle, which is what makes a cell one spectrum - see
    /// <see cref="PrecursorDensity.DefaultRtBinMin"/>. Equal as numbers, not as text, so 0.01 and .01
    /// are not a failure.
    /// </summary>
    [Fact]
    public void TheRtBinBoxOpensAtTheDocumentedDefault()
    {
        var xaml = File.ReadAllText(Path.Combine(AppDir, "MainWindow.xaml"));

        var box = Regex.Match(xaml, @"x:Name=""DensityRtBinBox""[^>]*?Text=""([^""]+)""");
        Assert.True(box.Success, "DensityRtBinBox has no Text in MainWindow.xaml.");

        Assert.Equal(
            PrecursorDensity.DefaultRtBinMin,
            double.Parse(box.Groups[1].Value, CultureInfo.InvariantCulture),
            6);
    }
}
