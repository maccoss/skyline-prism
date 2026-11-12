using System;
using System.Linq;
using System.Xml.Linq;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Every statistical option in the Differential pane says what it does when hovered.
/// </summary>
/// <remarks>
/// Each picker had one paragraph of help on the control as a whole, and the individual choices -
/// Paired, Blocked by subject, Welch t, the variance priors, the corrections - had none. On a
/// segmented control the choices cover the control, so the paragraph was hard to reach at all, and
/// the design paragraph told the reader to "hover each" when only the two Trend entries answered.
/// </remarks>
public class DifferentialOptionHelpTests
{
    private static readonly XNamespace Wpf = "http://schemas.microsoft.com/winfx/2006/xaml/presentation";
    private static readonly XNamespace X = "http://schemas.microsoft.com/winfx/2006/xaml";

    [Theory]
    [InlineData("DiffDesignCombo")]
    [InlineData("DiffLevelCombo")]
    [InlineData("DiffPKindCombo")]
    [InlineData("DiffTestCombo")]
    [InlineData("DiffPriorCombo")]
    [InlineData("DiffCorrectionCombo")]
    public void EveryOptionExplainsItself(string picker)
    {
        var doc = XDocument.Parse(XamlMarkup.MainWindow);
        var control = doc.Descendants().Single(e => (string?)e.Attribute(X + "Name") == picker);
        var options = control.Elements()
            .Where(e => e.Name == Wpf + "ListBoxItem" || e.Name == Wpf + "ComboBoxItem")
            .ToList();
        Assert.NotEmpty(options);

        foreach (var option in options)
        {
            var name = (string?)option.Attribute(X + "Name") ?? option.Value.Trim();
            // The Trend entries write theirs when it opens, from the state at that moment
            // (TrendDesignItemsHelpTests); a static one would silently win over it.
            if (option.Attribute("ToolTipOpening") is not null)
                continue;
            var help = (string?)option.Attribute("ToolTip");
            Assert.False(string.IsNullOrWhiteSpace(help), $"{picker}: '{name}' has no tooltip of its own");
            // A sentence, not a restatement of the label.
            Assert.True(help!.Length >= 40, $"{picker}: '{name}' tooltip is only '{help}'");
        }
    }
}
