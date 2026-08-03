using System.IO;
using System.Linq;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The pure parts of the SkylineRunner mechanism - launching the installed Skyline headlessly and reading
/// its result. Because the launcher process exits immediately, the ONLY signal that a batch failed is an
/// "Error:" line in the piped output, so <see cref="SkylineAppRunner.IsErrorLine"/> is the whole error
/// path; a false negative would report a failed export as a success.
/// </summary>
public class SkylineAppRunnerTests
{
    [Theory]
    [InlineData("Error: The report PRISM does not exist.")]
    [InlineData("\tError: Failure attempting to save PRISM report to out.parquet.")] // timestamped output
    [InlineData("エラー：レポートが存在しません")]
    [InlineData("错误：报告不存在")]
    public void IsErrorLine_DetectsFailures(string line)
        => Assert.True(SkylineAppRunner.IsErrorLine(line));

    [Theory]
    [InlineData("Report PRISM exported successfully to out.parquet.")]
    [InlineData("Opening file...")]
    [InlineData("Success! Imported Reports from Skyline-PRISM.skyr")]
    [InlineData("")]
    public void IsErrorLine_AcceptsNormalProgress(string line)
        => Assert.False(SkylineAppRunner.IsErrorLine(line));

    [Fact]
    public void IsErrorLine_IgnoresTheWordErrorMidLine()
    {
        // A path or report name containing "Error:" must not be mistaken for a failure - the prefix only
        // counts at the start of a line or straight after a tab.
        Assert.False(SkylineAppRunner.IsErrorLine(
            @"Report exported successfully to C:\data\Error: notes\out.csv."));
    }

    [Fact]
    public void EscapeForCmd_LeavesOrdinaryPathsUntouched()
    {
        const string path = @"C:\Users\jo\AppData\Roaming\...\Skyline-daily.appref-ms";
        Assert.Equal(path, SkylineAppRunner.EscapeForCmd(path));
    }

    [Fact]
    public void EscapeForCmd_EscapesAmpersandsAndSpaces()
    {
        // cmd.exe mangles & even inside quotes; once escaping starts, spaces need it too. A user account
        // like "V&V" put this exact bug in SkylineRunner.
        var escaped = SkylineAppRunner.EscapeForCmd(@"C:\Users\A&B\Start Menu\Skyline.appref-ms");

        Assert.Contains("^&", escaped);
        Assert.Contains("^ ", escaped); // spaces escaped once escaping is in play
        Assert.DoesNotContain("&B", escaped.Replace("^&", ""));
    }

    [Fact]
    public void EscapeForCmd_EscapesTheCaretFirstSoItIsNotDoubled()
    {
        var escaped = SkylineAppRunner.EscapeForCmd(@"C:\a^b&c");

        // "^" -> "^^" then "&" -> "^&"; escaping & first would corrupt the caret's own escape.
        Assert.Equal(@"C:\a^^b^&c", escaped);
    }

    [Fact]
    public void FindInstallations_OnlyReturnsShortcutsThatExist()
    {
        // Machine-dependent (0 entries on a build agent with no Skyline), so assert the invariant rather
        // than a count: whatever comes back must be a real .appref-ms.
        foreach (var installation in SkylineAppRunner.FindInstallations())
        {
            Assert.True(File.Exists(installation.ShortcutPath), installation.ShortcutPath);
            Assert.EndsWith(".appref-ms", installation.ShortcutPath);
            Assert.Contains(installation.AppName, new[] { "Skyline", "Skyline-daily" });
        }
    }

    [Fact]
    public void FindInstallations_PrefersDaily()
    {
        var apps = SkylineAppRunner.FindInstallations().Select(i => i.AppName).ToList();
        if (apps.Count < 2)
            return; // only one channel installed here; nothing to order

        Assert.Equal("Skyline-daily", apps[0]);
    }

    [Fact]
    public void Find_ReportsParquetSupport()
    {
        var runner = SkylineAppRunner.Find();
        if (runner is null)
            return; // no Skyline installed on this machine

        // The whole point of preferring this over SkylineCmd: it runs the full Skyline application, which
        // has the Parquet.Net assembly bindings that SkylineCmd.exe.config lacks.
        Assert.True(runner.SupportsParquet);
        Assert.Contains("Skyline", runner.Description);
    }
}
