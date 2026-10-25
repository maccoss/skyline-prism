using System;
using System.IO;
using System.Text.RegularExpressions;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The main window's Output directory box: what a change to it reaches, and that a run holds it still.
///
/// <para>Source checks, in the same style and for the same reason as <see cref="UiThreadSafetyTests"/>:
/// the code lives in <c>MainWindow</c> and needs a real window and dispatcher to exercise, but the
/// defect is visible in the source.</para>
/// </summary>
public class OutputDirectoryTests
{
    private static string AppDir =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));

    private static string Read(string file) =>
        File.ReadAllText(Path.Combine(AppDir, file)).Replace("\r\n", "\n");

    private static string Body(string source, string signature)
    {
        var from = source.IndexOf(signature, StringComparison.Ordinal);
        Assert.True(from >= 0, $"'{signature}' not found - the test needs updating with the code.");
        var to = source.IndexOf("\n    }\n", from, StringComparison.Ordinal);
        Assert.True(to > from, $"The end of '{signature}' was not found.");
        return source[from..to];
    }

    [Fact]
    public void AChangedDirectory_ReloadsThePaneOnScreen()
    {
        // Invalidating alone left the plot being looked at showing the previous directory until the
        // user switched panes; the QC pane was not even invalidated.
        var changed = Body(Read("MainWindow.xaml.cs"), "private void OnOutputDirChanged(");
        Assert.Contains("InvalidateQc();", changed, StringComparison.Ordinal);
        Assert.Contains("_ = ReloadShownPaneAsync();", changed, StringComparison.Ordinal);
    }

    [Fact]
    public void ARun_HoldsTheOutputDirectoryStill()
    {
        var source = Read("MainWindow.xaml.cs");
        var update = Body(source, "private void UpdateRunEnabled()");
        Assert.Contains("OutputDirBox.IsReadOnly = _isRunning;", update, StringComparison.Ordinal);
        Assert.Contains("OutputBrowseButton.IsEnabled = !_isRunning;", update, StringComparison.Ordinal);
        Assert.Contains("x:Name=\"OutputBrowseButton\"", Read("MainWindow.xaml"), StringComparison.Ordinal);
    }

    [Fact]
    public void QcMatrices_AreOnlyLoadedOneAtATime()
    {
        // LoadQcMatrices fills one dictionary from a worker thread. Called directly, a run finishing
        // while the pane reloads for a changed directory fills it from two threads at once.
        var source = Read("MainWindow.xaml.cs") + Read("MainWindow.Density.cs");
        var calls = Regex.Matches(source, @"(?<!void )\bLoadQcMatrices\(");
        Assert.Single(calls);
        var gate = Body(Read("MainWindow.xaml.cs"), "private async Task LoadQcAsync(");
        Assert.Contains("LoadQcMatrices(outputDir)", gate, StringComparison.Ordinal);
    }
}
