using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The Spectrum density tab's view picker is wired by name: each <c>ComboBoxItem</c> carries a
/// <c>Tag</c> that <c>SelectedDensityView</c> parses back into the <c>DensityView</c> enum.
///
/// <para>That indirection is what lets the visible label be reworded freely, but it also means a
/// mistyped or renamed Tag does not fail the build - <c>Enum.TryParse</c> just returns false and the
/// tab silently keeps drawing the heatmap whichever view is chosen. A source check is the only place
/// this is visible without a running window.</para>
/// </summary>
public class DensityViewComboTests
{
    private static string AppDir =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));

    [Fact]
    public void EveryViewComboItem_NamesADensityViewValue()
    {
        var tags = ComboTags();
        var values = EnumValues();

        Assert.NotEmpty(values); // the enum must actually have been found
        Assert.Equal(values, tags);
    }

    /// <summary>The Tag of every item inside the DensityViewCombo, in the order the user sees them.</summary>
    private static List<string> ComboTags()
    {
        var xaml = File.ReadAllText(Path.Combine(AppDir, "MainWindow.xaml"));
        var start = xaml.IndexOf("x:Name=\"DensityViewCombo\"", StringComparison.Ordinal);
        Assert.True(start >= 0, "DensityViewCombo not found in MainWindow.xaml - was it renamed?");
        var end = xaml.IndexOf("</ComboBox>", start, StringComparison.Ordinal);
        Assert.True(end > start, "DensityViewCombo has no items");

        var items = Regex.Matches(xaml[start..end], @"<ComboBoxItem\s+Tag=""(\w+)""")
            .Select(m => m.Groups[1].Value)
            .ToList();
        Assert.True(items.Count == CountItems(xaml[start..end]),
            "every item in the view picker needs a Tag, or it falls back to the heatmap");
        return items;
    }

    private static int CountItems(string block) => Regex.Matches(block, "<ComboBoxItem\\b").Count;

    /// <summary>The DensityView members, in declaration order.</summary>
    private static List<string> EnumValues()
    {
        var src = File.ReadAllText(Path.Combine(AppDir, "MainWindow.Density.cs"));
        var at = src.IndexOf("enum DensityView", StringComparison.Ordinal);
        Assert.True(at >= 0, "the DensityView enum was not found - was it renamed or moved?");

        var open = src.IndexOf('{', at);
        var close = src.IndexOf('}', open);
        var body = src[(open + 1)..close];
        // Drop the doc comments, then take what is left of each comma-separated member.
        return Regex.Replace(body, @"///[^\r\n]*", "")
            .Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(v => v.Trim())
            .Where(v => v.Length > 0)
            .ToList();
    }
}
