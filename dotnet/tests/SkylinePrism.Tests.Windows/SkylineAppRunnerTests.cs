using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
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

    /// <summary>
    /// A Skyline that never connects must be reported as a startup timeout - the one thing the caller can
    /// act on. It used to surface as <c>IOException: Pipe is broken.</c> instead: giving up on the
    /// synchronous WaitForConnection meant connecting a dummy client to our own pipe, the waiter counted
    /// that as Skyline, and Run then wrote the arguments to an already-disposed client. That message sent
    /// a real investigation after Skyline and the network share, when the cause was this timeout - and it
    /// was nondeterministic, so the very next call reported the same failure honestly.
    /// </summary>
    [Fact]
    public void Run_ReportsAStartupTimeout_NotABrokenPipe()
    {
        // A shortcut that does not exist: cmd.exe fails immediately, so nothing ever connects.
        var installation = new SkylineAppRunner.Installation(
            "Skyline", Path.Combine(Path.GetTempPath(), "prism_no_such_" + Guid.NewGuid().ToString("N") + ".appref-ms"));
        // The cap matters even though nothing should start: the extended wait triggers on ANY Skyline
        // appearing during the window - another test, a real export, a developer opening Skyline - and
        // with the 10-minute default that would hang the run rather than fail it.
        var runner = new SkylineAppRunner(
            installation, TimeSpan.FromSeconds(1), maxStartupTimeout: TimeSpan.FromSeconds(3));

        // TimeoutException, not InvalidOperationException: PRISM giving up is a different thing from
        // Skyline refusing, and HeadlessSkylineExporter now retries the first and not the second.
        var ex = Assert.Throws<TimeoutException>(
            () => runner.Run(new[] { "--in=x.sky" }, _ => { }, CancellationToken.None));

        // "did not start" is the expected verdict, but an unrelated Skyline appearing mid-wait legitimately
        // produces "started but never connected". Both are honest reports of the same startup timeout;
        // what must never come back is the broken pipe, which blamed Skyline for our own give-up.
        Assert.DoesNotContain("Pipe is broken", ex.Message);
        Assert.True(
            ex.Message.Contains("did not start") || ex.Message.Contains("never connected"),
            $"expected a startup-timeout message, got: {ex.Message}");
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

    /// <summary>
    /// A shortcut that cmd.exe cannot run, so nothing ever connects. The liveness check is what the test
    /// varies; everything else is held still.
    /// </summary>
    private static SkylineAppRunner.Installation MissingShortcut() =>
        new("Skyline", Path.Combine(
            Path.GetTempPath(), "prism_no_such_" + Guid.NewGuid().ToString("N") + ".appref-ms"));

    /// <summary>
    /// The failure this exists for: Skyline crashes about a second after launch, and PRISM must notice
    /// then - not when its startup window expires.
    ///
    /// <para>Every failed headless launch in the field log had a Skyline crash recorded in the Windows
    /// Application event log in the SAME SECOND as the launch (an UnauthorizedAccessException in
    /// ClickOnce's IsolationInterop.CreateActContext, before any Skyline code runs). PRISM's first
    /// liveness check used to be after the base deadline, so a one-second crash bought three minutes of
    /// waiting, and the retry budget - counted in attempts, not seconds - was spent on timeouts until the
    /// export fell back to CSV. The base deadline here is a full minute and the test finishes in a few
    /// seconds: that gap IS the fix.</para>
    /// </summary>
    [Fact]
    public void Run_NoticesSkylineCrashingOnStartup_WithoutWaitingOutTheWindow()
    {
        var calls = 0;
        // Up for the first two polls, then gone - a process that started and died, which is what a
        // startup crash looks like from the outside.
        var runner = new SkylineAppRunner(
            MissingShortcut(),
            startupTimeout: TimeSpan.FromSeconds(60),
            maxStartupTimeout: TimeSpan.FromSeconds(90),
            startedSkylineIsRunning: _ => ++calls <= 2);

        var sw = Stopwatch.StartNew();
        var ex = Assert.Throws<TimeoutException>(
            () => runner.Run(new[] { "--in=x.sky" }, _ => { }, CancellationToken.None));
        sw.Stop();

        // The message is the proof of WHICH path ran: waiting out the window produces "did not start
        // within", and only the crash path says the process exited.
        Assert.Contains("started and then exited", ex.Message);
        Assert.DoesNotContain("did not start within", ex.Message);
        // And it names the actual fault, so the next person does not go looking at the machine's load
        // the way the old message told them to.
        Assert.Contains("CreateActContext", ex.Message);
        // Generous by 4x even against the ~7x slowdown a loaded parallel suite has produced here
        // before; the point is only that it did not sit out the 60 s base deadline.
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(40),
            $"expected the crash to be noticed in seconds, took {sw.Elapsed.TotalSeconds:F0}s");
    }

    /// <summary>
    /// One missed enumeration must NOT be read as a crash. SkylineProcessIds swallows transient failures,
    /// so a single miss is expected noise - and treating it as death would abandon a Skyline that is
    /// legitimately minutes into starting a large document.
    /// </summary>
    [Fact]
    public void Run_DoesNotCallSkylineDead_OnOneTransientEnumerationMiss()
    {
        var calls = 0;
        var runner = new SkylineAppRunner(
            MissingShortcut(),
            startupTimeout: TimeSpan.FromSeconds(1),
            maxStartupTimeout: TimeSpan.FromSeconds(6),
            // Alive, alive, one miss, then alive for good: never connects, so this must end as the
            // "running but silent" timeout rather than as a crash.
            startedSkylineIsRunning: _ => ++calls != 3);

        var ex = Assert.Throws<TimeoutException>(
            () => runner.Run(new[] { "--in=x.sky" }, _ => { }, CancellationToken.None));

        Assert.Contains("never connected", ex.Message);
        Assert.DoesNotContain("started and then exited", ex.Message);
    }

    /// <summary>
    /// The pre-existing verdict still has to work: a launch where no Skyline EVER appears is a different
    /// fault from one that appeared and died, and keeps its own message.
    /// </summary>
    [Fact]
    public void Run_StillReportsNeverStarted_WhenNoProcessAppearsAtAll()
    {
        var runner = new SkylineAppRunner(
            MissingShortcut(),
            startupTimeout: TimeSpan.FromSeconds(1),
            maxStartupTimeout: TimeSpan.FromSeconds(3),
            startedSkylineIsRunning: _ => false);

        var ex = Assert.Throws<TimeoutException>(
            () => runner.Run(new[] { "--in=x.sky" }, _ => { }, CancellationToken.None));

        Assert.Contains("did not start within", ex.Message);
        Assert.DoesNotContain("started and then exited", ex.Message);
    }
}
