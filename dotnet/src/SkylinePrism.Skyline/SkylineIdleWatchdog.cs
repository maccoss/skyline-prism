#nullable enable

using System;
using System.Threading;

namespace SkylinePrism.Skyline;

/// <summary>
/// Gives up on a headless Skyline that has stopped saying anything.
///
/// <para><b>Why silence rather than total duration.</b> A wall-clock bound on the whole command cannot be
/// set without knowing how long the document should take, and that is the unknowable part: a legitimate
/// export of an 11 GB document runs 20-45 minutes, so a limit generous enough for the largest cohort
/// catches nothing in a small one. Silence has no such problem. Skyline narrates continuously -
/// "Opening file...", "4%", "31% - Writing row 22,210,657" - and across the exports measured here the
/// longest gap between lines was under a minute. A stalled one goes quiet and stays quiet: 7 h 35 min on
/// the run that prompted this, and an hour on the next, both ending with a Skyline that had stopped
/// accumulating CPU entirely after the machine ran out of memory.</para>
///
/// <para><b>Shared by both runners on purpose.</b> The first version of this lived inside
/// <see cref="SkylineAppRunner"/>, which left <see cref="SkylineCmdRunner"/> - the fallback taken whenever
/// no ClickOnce shortcut is found - with no bound at all, so the hang it was written to prevent was still
/// reachable. The bound belongs to the contract in <see cref="ISkylineCommandRunner"/>, not to one
/// implementation of it.</para>
///
/// <para>Deliberately NOT combined with a CPU-liveness check on the Skyline process, even though a
/// stalled one sits at exactly zero CPU: with several exports in flight there is no way to tell which
/// process is ours - the same reason <c>SkylineAppRunner.KillSpawned</c> declines to reap under
/// concurrency - so the signal would be unavailable exactly when it is needed, or would read another
/// export's work as our own.</para>
/// </summary>
internal sealed class SkylineIdleWatchdog
{
    /// <summary>
    /// Far above any healthy gap between output lines, far below either observed stall.
    /// </summary>
    public static readonly TimeSpan DefaultLimit = TimeSpan.FromMinutes(20);

    private readonly TimeSpan _limit;
    private readonly Func<DateTime> _now;
    private readonly string _appName;

    /// <summary>
    /// Last output, in UTC ticks. Written from the process's stdout/stderr callbacks (a threadpool
    /// thread for <see cref="SkylineCmdRunner"/>) and read from the polling loop, so it is exchanged
    /// atomically rather than assigned.
    /// </summary>
    private long _lastOutputTicks;

    private bool _warned;

    public SkylineIdleWatchdog(string appName, TimeSpan? limit = null, Func<DateTime>? now = null)
    {
        _appName = appName;
        _limit = limit ?? DefaultLimit;
        _now = now ?? (() => DateTime.UtcNow);
        _lastOutputTicks = _now().Ticks;
    }

    /// <summary>Skyline said something, so it is alive; the clock restarts.</summary>
    public void SawOutput()
    {
        Interlocked.Exchange(ref _lastOutputTicks, _now().Ticks);
        _warned = false;
    }

    /// <summary>
    /// Throws <see cref="TimeoutException"/> once Skyline has been silent for the limit, and logs a
    /// single warning at the half-way mark so a stall is visible while it develops rather than only when
    /// the command is abandoned.
    /// </summary>
    public void ThrowIfStalled(Action<string> log)
    {
        var idle = _now() - new DateTime(Interlocked.Read(ref _lastOutputTicks), DateTimeKind.Utc);
        if (idle >= _limit)
            throw new TimeoutException(
                $"{_appName} stopped reporting progress {idle.TotalMinutes:F0} min ago and was stopped. "
                + "A Skyline that runs out of memory stalls exactly like this and does not recover - "
                + "export fewer documents at a time, or close other Skyline windows, and try again.");

        if (_warned || idle < _limit / 2)
            return;
        log($"    (no output from {_appName} for {idle.TotalMinutes:F0} min; "
            + $"giving up at {_limit.TotalMinutes:F0})");
        _warned = true;
    }
}
