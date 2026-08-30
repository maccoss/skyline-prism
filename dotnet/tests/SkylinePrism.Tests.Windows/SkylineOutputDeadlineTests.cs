using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The bound on a headless export. The SkylineRunner protocol has no exit code and reports only through
/// its output pipe, so a stall is indistinguishable from slow work until something gives up - and for a
/// long time nothing did. Two runs of the same cohort waited on a Skyline that was never coming back:
/// 7 h 35 min on one, an hour on the next, both ending with a process that had stopped accumulating CPU
/// entirely after the machine ran out of memory.
///
/// <para>The bound is on SILENCE rather than total duration, because total duration cannot be set without
/// knowing how long the document should take. A healthy export of an 11 GB document runs 20-45 minutes
/// and narrates continuously - "Opening file...", "4%", "31% - Writing row 22,210,657" - with gaps under
/// a minute. A stalled one goes quiet and stays quiet.</para>
/// </summary>
public class SkylineOutputDeadlineTests
{
    /// <summary>A clock the test moves by hand, so twenty minutes of silence costs no wall time.</summary>
    private sealed class FakeClock
    {
        public DateTime UtcNow { get; private set; } = new(2026, 8, 30, 12, 0, 0, DateTimeKind.Utc);
        public void Advance(TimeSpan by) => UtcNow += by;
    }

    private static readonly TimeSpan Idle = TimeSpan.FromMinutes(20);

    [Fact]
    public void SilenceBeyondTheIdleBoundStopsTheExport()
    {
        var clock = new FakeClock();
        var lines = new BlockingCollection<string>();
        lines.Add("Opening file...");   // it started fine, then went quiet - the observed failure

        // The consumer drains the queued line, then finds nothing; the clock jumps past the bound.
        var log = new List<string>();
        var ex = Assert.Throws<TimeoutException>(() => SkylineAppRunner.ConsumeOutput(
            lines, log.Add, "Skyline-daily", Idle, deadline: null, CancellationToken.None,
            utcNow: () => { clock.Advance(TimeSpan.FromMinutes(5)); return clock.UtcNow; }));

        Assert.Contains("stopped reporting progress", ex.Message);
        // The message has to point at the cause, because the user's next action depends on it.
        Assert.Contains("runs out of memory", ex.Message);
        Assert.Contains("fewer documents at a time", ex.Message);
    }

    /// <summary>
    /// The bound must never fire on a healthy export, however long it runs. Output resets it, so an
    /// export that keeps talking can take hours - which a real 11 GB cohort legitimately does.
    /// </summary>
    [Fact]
    public void ContinuedOutputKeepsALongExportAlive()
    {
        var clock = new FakeClock();
        var lines = new BlockingCollection<string>();

        // Two hours of progress, in steps well under the idle bound.
        for (var i = 0; i < 24; i++)
            lines.Add($"{i * 4}% - Writing row {i * 1_000_000:N0}");
        lines.CompleteAdding();

        var log = new List<string>();
        var errors = SkylineAppRunner.ConsumeOutput(
            lines, log.Add, "Skyline-daily", Idle, deadline: null, CancellationToken.None,
            utcNow: () => { clock.Advance(TimeSpan.FromMinutes(5)); return clock.UtcNow; });

        Assert.Empty(errors.ToString());
        Assert.Equal(24, log.Count);
    }

    /// <summary>
    /// An "Error:" line is the protocol's ONLY failure signal, so it has to survive the new structure -
    /// a false negative here reports a failed export as a success.
    /// </summary>
    [Fact]
    public void ErrorLinesAreStillCollected()
    {
        var lines = new BlockingCollection<string>
        {
            "Opening file...",
            "Error: Failure attempting to save PRISM report to out.parquet.",
            "Done.",
        };
        lines.CompleteAdding();

        var errors = SkylineAppRunner.ConsumeOutput(
            lines, _ => { }, "Skyline-daily", Idle, deadline: null, CancellationToken.None);

        Assert.Contains("Failure attempting to save", errors.ToString());
    }

    /// <summary>A caller's overall bound still applies, and is reported as its own thing.</summary>
    [Fact]
    public void TheCallersOverallDeadlineStillApplies()
    {
        var clock = new FakeClock();
        var lines = new BlockingCollection<string>();
        var deadline = clock.UtcNow + TimeSpan.FromMinutes(2);

        var ex = Assert.Throws<TimeoutException>(() => SkylineAppRunner.ConsumeOutput(
            lines, _ => { }, "Skyline-daily", Idle, deadline, CancellationToken.None,
            utcNow: () => { clock.Advance(TimeSpan.FromMinutes(1)); return clock.UtcNow; },
            totalBound: TimeSpan.FromMinutes(2)));

        // The overall bound expires first here (2 min vs the 20 min idle bound), so that is what is said -
        // and it must name the number. "Within the time allowed" leaves a reader unable to tell a
        // 5-minute bound from a 60-minute one, or which caller set it.
        Assert.Contains("did not finish within 2 min", ex.Message);
    }

    /// <summary>
    /// Silence is warned about before it is fatal, so a user watching the log sees the stall developing
    /// rather than only learning about it when the export is abandoned.
    /// </summary>
    [Fact]
    public void ApproachingTheIdleBoundIsWarnedAboutOnce()
    {
        var clock = new FakeClock();
        var lines = new BlockingCollection<string>();
        var log = new List<string>();

        Assert.Throws<TimeoutException>(() => SkylineAppRunner.ConsumeOutput(
            lines, log.Add, "Skyline-daily", Idle, deadline: null, CancellationToken.None,
            utcNow: () => { clock.Advance(TimeSpan.FromMinutes(2)); return clock.UtcNow; }));

        Assert.Single(log, l => l.Contains("no output from Skyline-daily"));
    }
}
