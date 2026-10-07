using System;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The two Trend design entries are GRAYED, not hidden, when there is no axis to fit a trend against,
/// and their tooltip is what says why. The same two rules as <see cref="DisabledControlHelpTests"/> -
/// the tooltip must survive being disabled, and it must not be fixed in the markup - plus the hook
/// that writes it when it opens, from the state at that moment.
/// </summary>
public class TrendDesignItemsHelpTests
{
    [Theory]
    [InlineData("DiffDesignTrendItem")]
    [InlineData("DiffDesignTrendSubjectItem")]
    public void ATrendEntry_ShowsItsTooltipWhileDisabled_AndWritesItWhenItOpens(string name)
    {
        var element = XamlMarkup.Element(name);

        Assert.Contains("ToolTipService.ShowOnDisabled=\"True\"", element, StringComparison.Ordinal);
        Assert.Contains("ToolTipOpening=\"OnTrendItemToolTipOpening\"", element, StringComparison.Ordinal);
        Assert.DoesNotContain(" ToolTip=\"", element, StringComparison.Ordinal);
    }
}
