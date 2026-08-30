using System;
using System.Collections.Generic;
using System.Diagnostics;
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
/// poll loop. Drop any one and the failure is silent in opposite directions - a missing SawOutput kills
/// healthy exports after twenty minutes of real work, a missing CheckStalled restores the unbounded hang
/// that cost 7 h 35 min on the run that prompted all of this. Testing the watchdog in isolation, which is
/// what was done first, catches neither.</para>
///
/// <para>A batch file stands in for SkylineCmd: the runner only cares that its child writes to stdout and
/// eventually exits, so nothing here needs Skyline installed.</para>
/// </summary>
public class SkylineCmdRunnerIdleTests
{
    private static string WriteScript(string body)
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_cmdrunner_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, "fake-skylinecmd.cmd");
        File.WriteAllText(path, "@echo off\r\n" + body + "\r\n");
        return path;
    }

    /// <summary>
    /// A child that talks and then exits must be left alone. This is the direction that matters most: a
    /// broken SawOutput would abandon a perfectly healthy multi-hour export.
    /// </summary>
    [Fact]
    public void AChildThatKeepsTalkingIsNotStopped()
    {
        // Prints steadily for ~2 s, well past several poll intervals.
        var script = WriteScript("for /L %%i in (1,1,8) do (echo progress %%i& ping -n 1 127.0.0.1 >nul)");
        var log = new List<string>();

        new SkylineCmdRunner(script).Run(Array.Empty<string>(), log.Add, CancellationToken.None);

        Assert.Contains(log, l => l.Contains("progress 8"));
        Assert.DoesNotContain(log, l => l.Contains("no output from"));
    }

    /// <summary>
    /// A child that goes silent is stopped, with the message that names the cause - and it is KILLED, not
    /// merely abandoned, because a Skyline left running holds its document open with nothing to stop it.
    /// </summary>
    [Fact]
    public void ASilentChildIsStoppedAndKilled()
    {
        var before = Process.GetProcessesByName("ping").Length;

        // Speaks once so the handlers are exercised, then sleeps far beyond the bound.
        var script = WriteScript("echo starting& ping -n 60 127.0.0.1 >nul");
        var log = new List<string>();

        // A one-second bound: the runner's own poll interval is 250 ms, so this resolves in about a second
        // of real time rather than the shipped twenty minutes.
        var runner = new SkylineCmdRunner(script, idleTimeout: TimeSpan.FromSeconds(1));
        var ex = Assert.Throws<TimeoutException>(() => runner.Run(
            Array.Empty<string>(), log.Add, CancellationToken.None));

        Assert.Contains("stopped reporting progress", ex.Message);
        Assert.Contains("runs out of memory", ex.Message);
        Assert.Contains(log, l => l.Contains("starting"));

        // The child is gone rather than left running: give the kill a moment to land.
        Thread.Sleep(500);
        Assert.True(Process.GetProcessesByName("ping").Length <= before,
            "the stalled child should have been killed, not abandoned");
    }
}
