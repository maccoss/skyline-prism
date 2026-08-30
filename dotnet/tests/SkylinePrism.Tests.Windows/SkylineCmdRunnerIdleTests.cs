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
    /// A child that keeps talking must be left alone, however long it runs. This is the direction that
    /// matters most - a broken SawOutput would abandon a perfectly healthy multi-hour export - so the
    /// child deliberately runs FOUR TIMES the idle bound while speaking every second.
    /// </summary>
    [Fact]
    public void AChildThatKeepsTalkingIsNotStopped()
    {
        // ~6 s of work, a line every second, against a 4 s bound: without SawOutput resetting the clock
        // this trips at 4 s and the test fails, which is exactly the regression being guarded.
        var (script, _) = WriteScript(
            $"for /L %%i in (1,1,6) do (echo progress %%i& {SleepSeconds(1)})");
        var log = new List<string>();

        new SkylineCmdRunner(script, idleTimeout: TimeSpan.FromSeconds(4))
            .Run(Array.Empty<string>(), log.Add, CancellationToken.None);

        Assert.Contains(log, l => l.Contains("progress 6"));
        Assert.DoesNotContain(log, l => l.Contains("no output from"));
    }

    /// <summary>
    /// A child that goes silent is stopped - and KILLED, not merely abandoned, because a Skyline left
    /// running holds its document open with nothing left to stop it. The kill is proved by a marker the
    /// child writes only if it survives to the end of its sleep.
    /// </summary>
    [Fact]
    public void ASilentChildIsStoppedAndKilled()
    {
        var marker = Path.Combine(Path.GetTempPath(), "prism_marker_" + Guid.NewGuid().ToString("N"));

        // Speaks once (so the handlers are exercised), sleeps 8 s, then writes the marker. The bound is
        // 4 s, so the kill must land with 4 s to spare - and a slow machine only widens that gap.
        var (script, _) = WriteScript(
            $"echo starting& {SleepSeconds(8)}& echo done > \"{marker}\"");
        var log = new List<string>();

        var runner = new SkylineCmdRunner(script, idleTimeout: TimeSpan.FromSeconds(4));
        var ex = Assert.Throws<TimeoutException>(
            () => runner.Run(Array.Empty<string>(), log.Add, CancellationToken.None));

        Assert.Contains("stopped reporting progress", ex.Message);
        // The message must name the cause, because the user's next action depends on it.
        Assert.Contains("runs out of memory", ex.Message);
        // The first line had a full 4 s to arrive, so this also pins that the handlers ran at all.
        Assert.Contains(log, l => l.Contains("starting"));

        // Well past when the child would have written its marker had it survived.
        Thread.Sleep(TimeSpan.FromSeconds(6));
        Assert.False(File.Exists(marker), "the stalled child should have been killed, not abandoned");
    }
}
