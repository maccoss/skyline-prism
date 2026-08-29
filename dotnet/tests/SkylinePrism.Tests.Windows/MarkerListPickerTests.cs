using System;
using System.IO;
using System.Text.RegularExpressions;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The marker-normalization list picker has to be populated when the window opens, not only when a
/// provenance file is opened.
///
/// <para>It shipped filled from <c>ApplyConfigToUi</c> alone, which runs only on "Open provenance...".
/// On a machine with no saved protein lists the picker was therefore empty on a fresh start, and the
/// panels PRISM ships looked as though they had not been installed - the feature appeared broken while
/// being perfectly configured. Opening a provenance file filled it, which is exactly why the defect was
/// easy to miss: the person testing usually has a previous run to load.</para>
///
/// <para>A source check rather than a behavioral one, in the same style and for the same reason as
/// <see cref="UiThreadSafetyTests"/>: the code lives in <c>MainWindow</c> and needs a real window and
/// dispatcher to exercise, but the defect that shipped is visible in the source.</para>
/// </summary>
public class MarkerListPickerTests
{
    private static string AppDir =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));

    private static string CodeBehind => File.ReadAllText(Path.Combine(AppDir, "MainWindow.xaml.cs"));

    /// <summary>
    /// The constructor must fill the picker. Without this the shipped panels are invisible until the
    /// user happens to open a previous run's parameters.json.
    /// </summary>
    [Fact]
    public void TheConstructorPopulatesThePicker()
    {
        var ctor = Between(CodeBehind, "public MainWindow()", "\n    private ");

        Assert.Contains("RefreshMarkerListCombo(", ctor);
    }

    /// <summary>
    /// Exactly one place fills it. Two would drift, and the one that ran last would decide what the run
    /// normalizes against.
    /// </summary>
    [Fact]
    public void OnlyTheHelperFillsThePicker()
    {
        var adds = Regex.Matches(CodeBehind, @"MarkerNormListCombo\.Items\.(Add|Clear)\b").Count;
        var inHelper = Regex.Matches(
            Between(CodeBehind, "private void RefreshMarkerListCombo", "\n    /// <summary>\n    /// Populate"),
            @"MarkerNormListCombo\.Items\.(Add|Clear)\b").Count;

        Assert.Equal(inHelper, adds);
    }

    /// <summary>
    /// Editing the protein lists refreshes the picker, so a list created in the editor is selectable as
    /// a marker set without restarting the tool.
    /// </summary>
    [Fact]
    public void EditingTheListsRefreshesThePicker()
    {
        var source = File.ReadAllText(Path.Combine(AppDir, "MainWindow.DynamicRange.cs"));
        var handler = Between(source, "private void OnManageProteinLists", "\n    }");

        Assert.Contains("RefreshMarkerListCombo(", handler);
    }

    /// <summary>
    /// The picker is built from the in-session lists, not a fresh read from disk - otherwise an edit
    /// made in the Protein lists editor would not appear until the tool was restarted.
    /// </summary>
    [Fact]
    public void ThePickerReadsTheInSessionLists()
    {
        var helper = Between(
            CodeBehind, "private void RefreshMarkerListCombo",
            "\n    /// <summary>\n    /// Populate");

        Assert.Contains("_proteinLists.WithBuiltIns()", helper);
        Assert.DoesNotContain("ProteinListSet.Load()", helper);
    }

    private static string Between(string source, string start, string end)
    {
        var from = source.IndexOf(start, StringComparison.Ordinal);
        Assert.True(from >= 0, $"'{start}' not found - the test needs updating with the code.");
        var to = source.IndexOf(end, from, StringComparison.Ordinal);
        return to < 0 ? source[from..] : source[from..to];
    }
}
