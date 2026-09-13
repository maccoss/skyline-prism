using System;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The Ion Accounting pane's opening state, which lives in two files and can drift silently.
/// </summary>
/// <remarks>
/// Both defaults here were asked for by name, and both are split between the XAML that renders the
/// control and the code that reads or resets it. Nothing fails when they disagree - the pane simply
/// opens on one value and resets to another - so a source check is the only thing that notices.
/// </remarks>
public class IonPaneDefaultsTests
{
    private static string AppDir =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));

    /// <summary>
    /// 0.01 min is 0.6 s, shorter than one acquisition cycle on the instruments this was built for -
    /// so the default bins essentially nothing and the trace is drawn at the rate it was acquired
    /// at. A minute-wide bin averaged sixty cycles together.
    /// </summary>
    [Fact]
    public void TheGradientBinOpensAtTheAcquisitionsOwnRate()
    {
        var xaml = File.ReadAllText(Path.Combine(AppDir, "MainWindow.xaml"));
        var code = File.ReadAllText(Path.Combine(AppDir, "MainWindow.IonAccounting.cs"));

        var box = Regex.Match(xaml, @"x:Name=""IonBinBox""[^>]*?Text=""([^""]+)""");
        Assert.True(box.Success, "IonBinBox has no Text in MainWindow.xaml.");
        Assert.Equal("0.01", box.Groups[1].Value);

        var constant = Regex.Match(code, @"DefaultIonBinMinutes\s*=\s*([0-9.]+)");
        Assert.True(constant.Success, "DefaultIonBinMinutes is gone.");

        // The box is what the pane opens with and the constant is what Reset puts back. Equal as
        // numbers, not as text, so 0.01 and .01 are not a failure.
        Assert.Equal(
            double.Parse(box.Groups[1].Value, CultureInfo.InvariantCulture),
            double.Parse(constant.Groups[1].Value, CultureInfo.InvariantCulture),
            6);
    }

    /// <summary>
    /// Signal is what the instrument reports and what a mass spectrometrist reads a run in, so it is
    /// what the pane opens on. Its position in the list IS the default - the first item is selected
    /// when nothing else has been - so reordering the two silently changes it.
    /// </summary>
    [Fact]
    public void QuantityOpensOnSignal()
    {
        var xaml = File.ReadAllText(Path.Combine(AppDir, "MainWindow.xaml"));
        var combo = Regex.Match(
            xaml, @"x:Name=""IonQuantityCombo"".*?</ComboBox>", RegexOptions.Singleline);
        Assert.True(combo.Success, "IonQuantityCombo is gone from MainWindow.xaml.");

        var tags = Regex.Matches(combo.Value, @"<ComboBoxItem Tag=""([^""]+)""");
        Assert.Equal(2, tags.Count);
        Assert.Equal("Signal", tags[0].Groups[1].Value);
        Assert.Equal("Ions", tags[1].Groups[1].Value);
    }
}
