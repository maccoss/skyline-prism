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
/// 7 h 35 min on the run that prompted all of this. Testing the watchdog in isolation catches neither.</para>
///
/// <para><b>Both timings are chosen so the flaky direction is the SAFE one.</b> A first attempt at these
/// got both wrong: a 1 s bound raced the async output plumbing and failed on CI before the child's first
/// line arrived, and the talking child used <c>ping -n 1</c>, which returns instantly - so it finished in
/// milliseconds, never approached the bound, and would have passed with SawOutput deleted. A slower
/// machine now makes each test MORE likely to pass, never less.</para>
/// </summary>
public class SkylineCmdRunnerIdleTests
{
    /// <summary>`ping -n N` waits about N-1 seconds; `-n 1` returns immediately.</summary>
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
    /// 1 s bound and again at 4 s, which is a window being widened rather than a race being removed. The
    /// SawOutput path is proved by AChildThatKeepsTalkingIsNotStopped instead, which is mutation-verified;
    /// no test here needs to prove both halves.</para>
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
