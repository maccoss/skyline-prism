using System;
using System.IO;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The main window's Open QC Report and Publish to Panorama buttons, which follow the Output directory box.
///
/// <para>Source checks, in the same style and for the same reason as <see cref="UiThreadSafetyTests"/>:
/// the code lives in <c>MainWindow</c> and needs a real window and dispatcher to exercise, but the
/// defect is visible in the source.</para>
/// </summary>
public class ReportButtonStateTests
{
    private static string AppDir =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));

    private static string Source =>
        File.ReadAllText(Path.Combine(AppDir, "MainWindow.xaml.cs")).Replace("\r\n", "\n");

    private static string Probe
    {
        get
        {
            var source = Source;
            const string start = "private async Task UpdateReportButtonsAsync()";
            var from = source.IndexOf(start, StringComparison.Ordinal);
            Assert.True(from >= 0, $"'{start}' not found - the test needs updating with the code.");
            var to = source.IndexOf("\n    }\n", from, StringComparison.Ordinal);
            Assert.True(to > from, "The end of UpdateReportButtonsAsync was not found.");
            return source[from..to];
        }
    }

    [Theory]
    [InlineData("PublishButton")]
    [InlineData("OpenReportButton")]
    public void AnotherDirectory_TurnsTheButtonOff_BeforeTheWait(string button)
    {
        // The look waits for the box to be still. Left as it was meanwhile, the button stayed enabled
        // for the previous directory, and a click in the wait acted on whatever the box held now.
        var probe = Probe;
        var off = probe.IndexOf("if (!string.Equals(dir, _reportProbedDir", StringComparison.Ordinal);
        var wait = probe.IndexOf("await Task.Delay(ReportProbeDelay)", StringComparison.Ordinal);
        Assert.True(off >= 0, "The buttons are not turned off for a directory they were not decided for.");
        Assert.True(wait > off, "The buttons are turned off only after the wait.");
        Assert.Contains($"{button}.IsEnabled = false;", probe[off..wait], StringComparison.Ordinal);
    }

    [Fact]
    public void TheDirectoryDecidedFor_IsRecordedOnlyByTheLatestLook()
    {
        var probe = Probe;
        var latest = probe.LastIndexOf("if (probe == _reportProbe)", StringComparison.Ordinal);
        Assert.True(latest >= 0);
        Assert.Contains("_reportProbedDir = dir;", probe[latest..], StringComparison.Ordinal);
    }

    [Fact]
    public void OpenQcReport_FollowsTheDirectory_NotTheRun()
    {
        // Open QC Report was enabled only when a run finished in this window, so pointing the box at an
        // output directory produced earlier loaded every plot but left the report unopenable. The look
        // at the directory is now the one place that enables it and says which report it opens.
        var probe = Probe;
        var latest = probe.LastIndexOf("if (probe == _reportProbe)", StringComparison.Ordinal);
        Assert.True(latest >= 0);
        Assert.Contains("OpenReportButton.IsEnabled = hasQc && !_isRunning;", probe[latest..], StringComparison.Ordinal);
        Assert.Contains("_lastReportPath = hasQc ? qcReport : null;", probe[latest..], StringComparison.Ordinal);

        var outside = Source.Replace(probe, "", StringComparison.Ordinal);
        Assert.DoesNotContain("OpenReportButton.IsEnabled = reportExists", outside, StringComparison.Ordinal);
        Assert.DoesNotContain("_lastReportPath = reportPath", outside, StringComparison.Ordinal);
    }
}
