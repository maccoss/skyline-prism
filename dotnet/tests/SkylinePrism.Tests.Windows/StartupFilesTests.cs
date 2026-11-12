using System;
using System.IO;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// A PRISM process whose runtime folder was removed under it must ask for a restart, once, instead of
/// showing each fault it raises as a stack trace.
///
/// <para>The regression: a window left open across a .NET 10.0.14 to 10.0.15 update raised
/// <c>DllNotFoundException</c> from <c>MS.Win32.Penimc.UnsafeNativeMethods.CreateResetEvent</c> seven
/// times in six seconds when Windows reported a tablet - WPF loading its pen support from a folder the
/// update had deleted. See <see cref="StartupFiles"/>.</para>
///
/// <para>The detection is tested on real files. How <c>App.Report</c> uses it is checked against the
/// source, for the reason <see cref="ShutdownReportingTests"/> gives: Report needs a live
/// Application and message loop, and the fault needs a runtime folder deleted from under a running
/// process, which a test cannot stage.</para>
/// </summary>
public class StartupFilesTests
{
    [Fact]
    public void FirstMissing_NamesTheFileThatWasRemoved()
    {
        var dir = Directory.CreateTempSubdirectory("prism-startupfiles-");
        try
        {
            var kept = Path.Combine(dir.FullName, "kept.dll");
            var removed = Path.Combine(dir.FullName, "removed.dll");
            File.WriteAllText(kept, "");
            File.WriteAllText(removed, "");
            Assert.Null(StartupFiles.FirstMissing(new[] { kept, removed }));

            File.Delete(removed);

            Assert.Equal(removed, StartupFiles.FirstMissing(new[] { kept, removed }));
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public void FirstMissing_SkipsAnEmptyLocation()
    {
        // A single-file publish reports Location as "" for every bundled assembly. Treating that as
        // missing would replace every error the tool ever shows with a restart prompt.
        Assert.Null(StartupFiles.FirstMissing(new[] { "", typeof(object).Assembly.Location }));
    }

    [Fact]
    public void AHealthyProcessHasEveryStartupFile()
    {
        // The false positive is the expensive direction: a probe that reads as missing in a normal
        // process would hide every real error behind "restart PRISM". This process loaded CoreLib,
        // PresentationCore and the App assembly from disk, so all three must be found.
        var probes = StartupFiles.Probes();
        Assert.All(probes, p => Assert.False(string.IsNullOrEmpty(p), "A probe resolved to no path."));
        Assert.Null(StartupFiles.FirstMissing(probes));
    }

    [Fact]
    public void TheProbesCoverTheWpfRuntimeFolder()
    {
        // The folder that disappeared in the reported case. The pen DLL is resolved from it, so a
        // probe set without a file there would have missed exactly the case this exists for.
        var wpfFolder = Path.GetDirectoryName(typeof(System.Windows.UIElement).Assembly.Location);
        Assert.Contains(StartupFiles.Probes(),
            p => string.Equals(Path.GetDirectoryName(p), wpfFolder, StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void TheRestartMessageSaysWhatToDo()
    {
        var text = StartupFiles.RestartMessage(@"C:\runtime\10.0.14\PresentationCore.dll", @"C:\logs\prism-tool.log");

        Assert.Contains(@"C:\runtime\10.0.14\PresentationCore.dll", text);
        Assert.Contains(@"C:\logs\prism-tool.log", text);
        Assert.Contains("Tools menu", text);
    }

    [Fact]
    public void ReportChecksForMissingFilesBeforeTryingTheCopyableWindow()
    {
        var text = ReadAppSource();

        var guard = text.IndexOf("if (_shutdownStarted)", StringComparison.Ordinal);
        var check = text.IndexOf("StartupFiles.FirstMissing(StartupFiles.Probes())", StringComparison.Ordinal);
        // The call, with its parenthesis: Report's comment names the method too, ahead of the check.
        var window = text.IndexOf("PrismMessageWindow.ShowError(", StringComparison.Ordinal);
        var dialog = text.IndexOf("MessageBox.Show", StringComparison.Ordinal);

        Assert.True(check >= 0, "App.Report must check for missing startup files.");
        // After the shutdown guard: a fault while exiting stays log-only even in a stale process.
        Assert.True(guard < check, "The shutdown guard must still come first.");
        // Before the copyable window: in the reported case that window was what could not be built.
        Assert.True(check < window, "The missing-files check must come before PrismMessageWindow.");
        Assert.True(check < dialog, "The missing-files check must come before the message box.");
    }

    [Fact]
    public void TheRestartMessageIsShownOnce()
    {
        var text = ReadAppSource();

        // Claimed atomically, and a second claim returns before the restart message is even built.
        // Bounded by that assignment, not by the dialog: the copyable window's branch between the two
        // has a return of its own, which would satisfy a looser check with this one deleted.
        var claim = text.IndexOf("Interlocked.Exchange(ref _restartMessageShown, 1) != 0", StringComparison.Ordinal);
        var built = text.IndexOf("StartupFiles.RestartMessage(", StringComparison.Ordinal);
        Assert.True(claim >= 0, "The restart message must be claimed with Interlocked.Exchange.");
        Assert.True(built > claim, "The claim must come before the restart message is built.");
        Assert.Contains("return;", text[claim..built]);
    }

    private static string ReadAppSource()
    {
        var path = Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..",
            "src", "SkylinePrism.App", "App.xaml.cs"));
        Assert.True(File.Exists(path), $"App.xaml.cs not found at {path} - has the output layout moved?");
        return File.ReadAllText(path);
    }
}
