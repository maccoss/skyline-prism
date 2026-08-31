using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The SkylineCmd fallback's silence bound, driven against a REAL child process.
///
/// <para>This is the wiring the shared watchdog was written for, and it is three independent pieces:
/// SawOutput in the stdout handler, SawOutput in the stderr handler, and the CheckStalled call in the
/// poll loop. Drop any one and the failure is silent in OPPOSITE directions - a missing SawOutput
/// abandons healthy multi-hour exports, a missing CheckStalled restores the unbounded hang that cost
/// 7 h 35 min on the run that prompted all of this. Testing the watchdog in isolation catches neither -
/// and it was a rewrite of these that exposed the handlers resetting nothing at all.</para>
///
/// <para><b>Only ONE of these tests may depend on wall-clock timing, and it is the kill.</b> That one is
/// safe in the flaky direction: a slower machine pushes the child's marker further out, so the kill has
/// more room, not less. The reset assertion is NOT safe that way - it needs the child talking faster than
/// the bound while running longer than it, which leaves no margin on a loaded machine. Four attempts at
/// tuning sleeps proved that, the last passing alone and failing inside the parallel suite, so it is
/// asked by counting resets instead and has no timing dependence at all.</para>
/// </summary>
public class SkylineCmdRunnerIdleTests
{
    /// <summary>
    /// `ping -n N` waits about N-1 seconds; `-n 1` returns immediately, which is what made an earlier
    /// version of the talking test pass without ever approaching the bound.
    /// </summary>
    private static string SleepSeconds(int seconds) => $"ping -n {seconds + 1} 127.0.0.1 >nul";


    private static (string Script, string Dir) WriteScript(string body)
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_cmdrunner_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, "fake-skylinecmd.cmd");
        File.WriteAllText(path, "@echo off\r\n" + body + "\r\n");
        return (path, dir);
    }

    /// <summary>
    /// The output handlers must reset the watchdog, or a healthy multi-hour export is abandoned after
    /// twenty minutes of real work. Asked by COUNTING resets rather than by racing a deadline: a child
    /// that must talk faster than the bound while running longer than it leaves no margin on a loaded
    /// machine, which four attempts at tuning sleeps demonstrated - the last one passing alone and
    /// failing inside the parallel suite. This version has no timing dependence at all.
    /// </summary>
    [Fact]
    public void TheOutputHandlersResetTheWatchdog()
    {
        // Prints and exits immediately; the bound is never approached, so nothing here can race.
        var (script, _) = WriteScript("echo progress 1& echo progress 2& echo progress 3");
        var log = new List<string>();

        var runner = new SkylineCmdRunner(script, idleTimeout: TimeSpan.FromMinutes(10));
        runner.Run(Array.Empty<string>(), log.Add, CancellationToken.None);

        Assert.Contains(log, l => l.Contains("progress 3"));
        Assert.NotNull(runner.LastWatchdog);
        Assert.True(runner.LastWatchdog!.SawOutputCount >= 3,
            $"expected the handlers to reset the watchdog per line, saw {runner.LastWatchdog.SawOutputCount}");
    }

    /// <summary>
    /// A child that says NOTHING is stopped - and KILLED, not merely abandoned, because a Skyline left
    /// running holds its document open with nothing left to stop it. The kill is proved by a marker the
    /// child writes only if it survives to the end of its sleep.
    ///
    /// <para>The child is deliberately SILENT rather than speaking once first. An earlier version had it
    /// echo a line and asserted that line was logged, to show the handlers ran - and that assertion raced
    /// the bound in the unsafe direction: the kill is safer the slower the machine gets, but "did the
    /// first line arrive before the clock ran out" is LESS likely to hold. It failed CI twice, once at a
    /// 1 s bound and again at 4 s, which is a window being widened rather than a race being removed.
    /// <see cref="TheOutputHandlersResetTheWatchdog"/> proves that half with no clock at all; no test
    /// here needs to prove both.</para>
    /// </summary>
    [Fact]
    public void ASilentChildIsStoppedAndKilled()
    {
        var marker = Path.Combine(Path.GetTempPath(), "prism_marker_" + Guid.NewGuid().ToString("N"));

        // Silent for 8 s, then writes the marker. The bound is 4 s, measured from Process.Start, so the
        // kill must land with 4 s to spare - and a slower machine only pushes the marker further out.
        var (script, _) = WriteScript($"{SleepSeconds(8)}& echo done > \"{marker}\"");
        var log = new List<string>();

        var runner = new SkylineCmdRunner(script, idleTimeout: TimeSpan.FromSeconds(4));
        var ex = Assert.Throws<TimeoutException>(
            () => runner.Run(Array.Empty<string>(), log.Add, CancellationToken.None));

        Assert.Contains("stopped reporting progress", ex.Message);
        // The message must name the cause, because the user's next action depends on it.
        Assert.Contains("runs out of memory", ex.Message);

        // Well past when the child would have written its marker had it survived.
        Thread.Sleep(TimeSpan.FromSeconds(6));
        Assert.False(File.Exists(marker), "the stalled child should have been killed, not abandoned");
    }
}
