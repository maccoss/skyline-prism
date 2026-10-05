using System;
using System.IO;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The main window's Publish to Panorama button, which follows the Output directory box.
///
/// <para>Source checks, in the same style and for the same reason as <see cref="UiThreadSafetyTests"/>:
/// the code lives in <c>MainWindow</c> and needs a real window and dispatcher to exercise, but the
/// defect is visible in the source.</para>
/// </summary>
public class PublishButtonStateTests
{
    private static string AppDir =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));

    private static string Probe
    {
        get
        {
            var source = File.ReadAllText(Path.Combine(AppDir, "MainWindow.xaml.cs")).Replace("\r\n", "\n");
            const string start = "private async Task UpdatePublishEnabledAsync()";
            var from = source.IndexOf(start, StringComparison.Ordinal);
            Assert.True(from >= 0, $"'{start}' not found - the test needs updating with the code.");
            var to = source.IndexOf("\n    }\n", from, StringComparison.Ordinal);
            Assert.True(to > from, "The end of UpdatePublishEnabledAsync was not found.");
            return source[from..to];
        }
    }

    [Fact]
    public void AnotherDirectory_TurnsTheButtonOff_BeforeTheWait()
    {
        // The look waits for the box to be still. Left as it was meanwhile, the button stayed enabled
        // for the previous directory, and a click in the wait published whatever the box held now.
        var probe = Probe;
        var off = probe.IndexOf("if (!string.Equals(dir, _publishProbedDir", StringComparison.Ordinal);
        var wait = probe.IndexOf("await Task.Delay(PublishProbeDelay)", StringComparison.Ordinal);
        Assert.True(off >= 0, "The button is not turned off for a directory it was not decided for.");
        Assert.True(wait > off, "The button is turned off only after the wait.");
        Assert.Contains("PublishButton.IsEnabled = false;", probe[off..wait], StringComparison.Ordinal);
    }

    [Fact]
    public void TheDirectoryDecidedFor_IsRecordedOnlyByTheLatestLook()
    {
        var probe = Probe;
        var latest = probe.LastIndexOf("if (probe == _publishProbe)", StringComparison.Ordinal);
        Assert.True(latest >= 0);
        Assert.Contains("_publishProbedDir = dir;", probe[latest..], StringComparison.Ordinal);
    }
}
