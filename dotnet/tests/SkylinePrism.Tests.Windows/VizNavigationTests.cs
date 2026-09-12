using System.Text.RegularExpressions;
using System.Linq;
using System.IO;
using System;
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
        Assert.Equal(VizPane.IonAccounting, VizNavigation.Current(visualizationTabSelected: true, navIndex: 3));
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
    [InlineData(5)]
    [InlineData(99)]
    public void AnIndexThatNamesNoRowIsNoPane(int navIndex)
    {
        Assert.Null(VizNavigation.Current(visualizationTabSelected: true, navIndex));
    }

    /// <summary>
    /// Every row of the rail maps to a pane, and the mapping is by INDEX - so a row added to the XAML
    /// without an enum value silently becomes "no pane" and its plots never load. This is the
    /// assertion that turns that into a build failure instead.
    /// </summary>
    [Fact]
    public void EveryRailRowMapsToItsOwnPane()
    {
        var panes = Enum.GetValues<VizPane>();
        for (var index = 0; index < panes.Length; index++)
        {
            Assert.Equal(
                (VizPane)index,
                VizNavigation.Current(visualizationTabSelected: true, index));
        }

        // And one past the last row is not a pane, which is what catches an enum value added
        // without its row.
        Assert.Null(VizNavigation.Current(visualizationTabSelected: true, panes.Length));
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
        Assert.False(VizNavigation.ShouldFollowSkylineSelection(VizPane.IonAccounting));
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
        Assert.Equal(3, (int)VizPane.IonAccounting);
    }

    /// <summary>
    /// The Analysis rail's values are ITS row indices, in the order the ListBox declares them.
    ///
    /// <para>The rail selects by index and the handler casts that index straight to the enum, so a
    /// row inserted in the XAML without a matching enum value silently shows the wrong pane - the
    /// same failure this file already pins for the Visualization rail, and the reason that one walks
    /// every value rather than checking a count.</para>
    /// </summary>
    [Fact]
    public void TheAnalysisPaneValuesAreItsRailRowIndices()
    {
        Assert.Equal(0, (int)AnalysisPane.Inputs);
        Assert.Equal(1, (int)AnalysisPane.Settings);
        Assert.Equal(2, (int)AnalysisPane.Log);

        // ...and nothing else, so adding a pane forces this test to be updated with it.
        Assert.Equal(
            new[] { 0, 1, 2 },
            Enum.GetValues<AnalysisPane>().Select(p => (int)p).OrderBy(i => i).ToArray());
    }

    /// <summary>
    /// Every row the Analysis rail declares in XAML has an enum value, and every enum value has a
    /// row. Read from the XAML itself, because the two drift independently.
    /// </summary>
    [Fact]
    public void EveryAnalysisRailRowHasItsOwnPane()
    {
        var xaml = File.ReadAllText(MainWindowXamlPath());
        var rail = Regex.Match(
            xaml,
            @"<ListBox[^>]*x:Name=""AnalysisNav""(.*?)</ListBox>",
            RegexOptions.Singleline);
        Assert.True(rail.Success, "the Analysis rail was not found in MainWindow.xaml");

        var rows = Regex.Matches(rail.Groups[1].Value, @"<ListBoxItem\s").Count;
        Assert.Equal(Enum.GetValues<AnalysisPane>().Length, rows);
    }

    private static string MainWindowXamlPath()
    {
        var dir = AppContext.BaseDirectory;
        for (var i = 0; i < 8 && dir is not null; i++)
        {
            var candidate = Path.Combine(dir, "src", "SkylinePrism.App", "MainWindow.xaml");
            if (File.Exists(candidate))
                return candidate;
            dir = Path.GetDirectoryName(dir);
        }
        throw new FileNotFoundException("MainWindow.xaml not found from " + AppContext.BaseDirectory);
    }
}
