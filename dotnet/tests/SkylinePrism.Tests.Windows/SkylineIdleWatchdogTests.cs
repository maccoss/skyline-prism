using System;
using System.Collections.Generic;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The silence bound itself, tested once for both runners. It lives in a shared class precisely because
/// the first version lived inside <see cref="SkylineAppRunner"/>, which left <see cref="SkylineCmdRunner"/>
/// - the fallback taken whenever no ClickOnce shortcut is found - with no bound at all, so the hang it was
/// written to prevent stayed reachable.
/// </summary>
public class SkylineIdleWatchdogTests
{
    private sealed class FakeClock
    {
        public DateTime UtcNow = new(2026, 8, 30, 12, 0, 0, DateTimeKind.Utc);
        public DateTime Now() => UtcNow;
        public void Advance(TimeSpan by) => UtcNow += by;
    }

    [Fact]
    public void SilenceBeyondTheLimitThrows()
    {
        var clock = new FakeClock();
        var watch = new SkylineIdleWatchdog("Skyline-daily", TimeSpan.FromMinutes(20), clock.Now);
        var log = new List<string>();

        clock.Advance(TimeSpan.FromMinutes(19));
        Assert.Null(watch.CheckStalled(log.Add));   // still inside the bound

        clock.Advance(TimeSpan.FromMinutes(2));
        var ex = watch.CheckStalled(log.Add);

        Assert.NotNull(ex);
        Assert.Contains("stopped reporting progress", ex.Message);
        // The message must name the cause, because the user's next action depends on it.
        Assert.Contains("runs out of memory", ex.Message);
        Assert.Contains("fewer documents at a time", ex.Message);
    }

    /// <summary>
    /// Output resets the clock, so a healthy export may run for as long as it likes - which an 11 GB
    /// document legitimately does. This is the property that makes bounding silence safe where bounding
    /// total duration is not.
    /// </summary>
    [Fact]
    public void OutputResetsTheClockSoALongExportSurvives()
    {
        var clock = new FakeClock();
        var watch = new SkylineIdleWatchdog("Skyline-daily", TimeSpan.FromMinutes(20), clock.Now);
        var log = new List<string>();

        // Three hours of work, narrated every fifteen minutes.
        for (var i = 0; i < 12; i++)
        {
            clock.Advance(TimeSpan.FromMinutes(15));
            Assert.Null(watch.CheckStalled(log.Add));
            watch.SawOutput();
        }

        clock.Advance(TimeSpan.FromMinutes(19));
        Assert.Null(watch.CheckStalled(log.Add)); // still fine: the last line was 19 minutes ago
    }

    [Fact]
    public void TheHalfWayWarningIsLoggedOnceAndResetsOnOutput()
    {
        var clock = new FakeClock();
        var watch = new SkylineIdleWatchdog("Skyline-daily", TimeSpan.FromMinutes(20), clock.Now);
        var log = new List<string>();

        clock.Advance(TimeSpan.FromMinutes(11));
        watch.CheckStalled(log.Add);
        watch.CheckStalled(log.Add);
        watch.CheckStalled(log.Add);
        Assert.Single(log);
        Assert.Contains("no output from Skyline-daily", log[0]);

        // Skyline speaks again, so the next quiet spell warns afresh rather than staying silent.
        watch.SawOutput();
        clock.Advance(TimeSpan.FromMinutes(11));
        watch.CheckStalled(log.Add);
        Assert.Equal(2, log.Count);
    }

    /// <summary>
    /// The default has to be far above any healthy gap and far below either observed stall - the runs
    /// that prompted this went quiet for an hour and for 7 h 35 min, while the longest gap between lines
    /// on a healthy export was under a minute.
    /// </summary>
    [Fact]
    public void TheDefaultLimitSitsBetweenAHealthyGapAndAnObservedStall()
    {
        Assert.True(SkylineIdleWatchdog.DefaultLimit > TimeSpan.FromMinutes(5),
            "must not fire on a slow but healthy phase");
        Assert.True(SkylineIdleWatchdog.DefaultLimit < TimeSpan.FromHours(1),
            "must fire well inside the shortest stall actually observed");
    }
}
