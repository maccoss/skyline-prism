using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Which visualization pane is on screen is now TWO conditions - the outer tab and the nav rail - and
/// both ways of getting it wrong are silent. Too eager and PRISM polls Skyline for its selection while
/// the user is over on Settings; too lazy and a pane never loads its data and just sits empty.
/// </summary>
public class VizNavigationTests
{
    // VizPane is internal, so it can appear in a test BODY (the App grants InternalsVisibleTo) but
    // never in a public test signature - hence one Fact rather than a Theory over the three rows.
    [Fact]
    public void TheRailPicksThePaneWhenVisualizationIsSelected()
    {
        Assert.Equal(VizPane.Qc, VizNavigation.Current(visualizationTabSelected: true, navIndex: 0));
        Assert.Equal(VizPane.Density, VizNavigation.Current(visualizationTabSelected: true, navIndex: 1));
        Assert.Equal(VizPane.DynamicRange, VizNavigation.Current(visualizationTabSelected: true, navIndex: 2));
        Assert.Equal(VizPane.Ms2Signal, VizNavigation.Current(visualizationTabSelected: true, navIndex: 3));
    }

    /// <summary>
    /// The rail keeps its selection while the user is on Analysis, so the rail alone cannot answer
    /// this. Reading it alone is what would leave the dynamic-range plot polling Skyline from behind
    /// another tab.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void NoPaneIsOnScreenWhileTheAnalysisTabIsSelected(int navIndex)
    {
        Assert.Null(VizNavigation.Current(visualizationTabSelected: false, navIndex));
    }

    /// <summary>
    /// -1 is a real state: WPF reports it for a ListBox before anything is selected, which is how the
    /// rail sits between InitializeComponent and the constructor setting it.
    /// </summary>
    [Theory]
    [InlineData(-1)]
    [InlineData(4)]
    [InlineData(99)]
    public void AnIndexThatNamesNoRowIsNoPane(int navIndex)
    {
        Assert.Null(VizNavigation.Current(visualizationTabSelected: true, navIndex));
    }

    /// <summary>
    /// The poll is a live RPC to Skyline on a timer. It must run for the dynamic-range pane and for
    /// nothing else - including "no pane", which is the case a null check is easiest to forget.
    /// </summary>
    [Fact]
    public void OnlyTheDynamicRangePaneFollowsSkylinesSelection()
    {
        Assert.True(VizNavigation.ShouldFollowSkylineSelection(VizPane.DynamicRange));
        Assert.False(VizNavigation.ShouldFollowSkylineSelection(VizPane.Qc));
        Assert.False(VizNavigation.ShouldFollowSkylineSelection(VizPane.Density));
        Assert.False(VizNavigation.ShouldFollowSkylineSelection(VizPane.Ms2Signal));
        Assert.False(VizNavigation.ShouldFollowSkylineSelection(null));
    }

    /// <summary>
    /// The enum doubles as the rail's row order - ShowVisualization assigns it straight to
    /// SelectedIndex - so a reordering of the rows without a matching reordering here would silently
    /// send "land on the plots when the run finishes" to the wrong pane.
    /// </summary>
    [Fact]
    public void ThePaneValuesAreTheRailRowIndices()
    {
        Assert.Equal(0, (int)VizPane.Qc);
        Assert.Equal(1, (int)VizPane.Density);
        Assert.Equal(2, (int)VizPane.DynamicRange);
        Assert.Equal(3, (int)VizPane.Ms2Signal);
    }
}
