#nullable enable

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Text;
using System.Threading;

namespace SkylinePrism.Skyline;

/// <summary>
/// Runs Skyline command-line arguments by driving the INSTALLED Skyline application headlessly - the
/// mechanism <c>SkylineRunner.exe</c> uses, reimplemented here (it is ~40 lines of protocol, and the
/// official shim is a separate download that is built per channel: one binary looks only for
/// <c>Skyline</c>, another only for <c>Skyline-daily</c>).
///
/// <para><b>Why this is preferred over SkylineCmd:</b> it starts the real <c>Skyline.exe</c>, so it runs
/// under <c>Skyline.exe.config</c> - which carries the Parquet.Net assembly bindings that
/// <c>SkylineCmd.exe.config</c> lacks. Report export to <c>.parquet</c> therefore works here and fails
/// there. The trade-off is startup: a full (UI-less) Skyline process takes longer to come up.</para>
///
/// <para><b>The protocol</b> (ported from <c>pwiz_tools/Skyline/Executables/SkylineRunner/Program.cs</c>):
/// launch the ClickOnce <c>.appref-ms</c> shortcut with a single argument <c>CMD-&lt;guid&gt;</c>; Skyline
/// then connects back on the named pipe <c>SkylineInputPipe-&lt;guid&gt;</c> to read its arguments (one per
/// line) and writes its console output to <c>SkylineOutputPipe-&lt;guid&gt;</c>. Because the launcher process
/// returns immediately, there is no exit code to inspect - success is determined from the output lines,
/// the same way SkylineRunner does it.</para>
/// </summary>
public sealed class SkylineAppRunner : ISkylineCommandRunner
{
    /// <summary>An installed Skyline that can be driven headlessly.</summary>
    public sealed record Installation(string AppName, string ShortcutPath)
    {
        public override string ToString() => $"{AppName} ({ShortcutPath})";
    }

    /// <summary>ClickOnce application names to look for, newest channel first.</summary>
    private static readonly string[] AppNames = { "Skyline-daily", "Skyline" };

    private readonly Installation _installation;
    private readonly TimeSpan _startupTimeout;

    /// <summary>
    /// How long to keep waiting once <see cref="_startupTimeout"/> has passed but the Skyline we launched
    /// is demonstrably alive. Only a backstop: the normal exit from that wait is Skyline connecting.
    /// </summary>
    private readonly TimeSpan _maxStartupTimeout;

    /// <summary>How long Skyline may print nothing before the command is abandoned.</summary>
    private readonly TimeSpan _idleTimeout;

    /// <param name="maxStartupTimeout">
    /// The backstop for the extended wait; never shorter than <paramref name="startupTimeout"/>, since a
    /// cap below the base deadline could never be reached and would silently do nothing. Injectable so a
    /// test that deliberately times out cannot sit on the 10-minute default: the extension triggers
    /// whenever ANY Skyline appears during the wait, which on a developer machine or a shared agent is
    /// not something a test can rule out.
    /// </param>
    /// <param name="idleTimeout">
    /// How long Skyline may say NOTHING before the command is abandoned; defaults to
    /// <see cref="SkylineIdleWatchdog.DefaultLimit"/>. See that class for why the bound is on silence
    /// rather than total duration. Injectable so a test need not wait twenty minutes for nothing.
    /// </param>
    public SkylineAppRunner(
        Installation installation, TimeSpan? startupTimeout = null, TimeSpan? maxStartupTimeout = null,
        TimeSpan? idleTimeout = null)
    {
        _installation = installation;
        _idleTimeout = idleTimeout ?? SkylineIdleWatchdog.DefaultLimit;
        // A cold ClickOnce start (plus the update check Skyline does on launch) can take a while; the
        // official runner waits 15 s, which is too tight on a first run of the day.
        //
        // 90 s was too tight as well, and failed in the way that costs the most: PRISM exports several
        // documents at once, so a headless start competes with another Skyline already streaming a
        // multi-GB report to a network share. Measured case - a 2-document cohort where the open
        // document's own parquet export was in flight - both headless starts blew past 90 s and the run
        // silently fell back to month-old report files. The base wait is now 3 min, and WaitForConnection
        // extends it while our Skyline is actually running, so a slow start is waited out and a Skyline
        // that never appears still fails promptly.
        _startupTimeout = startupTimeout ?? TimeSpan.FromMinutes(3);
        var cap = maxStartupTimeout ?? TimeSpan.FromMinutes(10);
        _maxStartupTimeout = cap < _startupTimeout ? _startupTimeout : cap;
    }

    public string Description => $"{_installation.AppName} (headless application)";

    /// <summary>The full Skyline application honours <c>Skyline.exe.config</c>, so parquet export works.</summary>
    public bool SupportsParquet => true;

    /// <summary>
    /// Every installed Skyline this can drive. Empty when Skyline was installed some way that leaves no
    /// ClickOnce shortcut - the caller then falls back to <see cref="SkylineCmdRunner"/>.
    /// </summary>
    public static IReadOnlyList<Installation> FindInstallations()
    {
        var programs = Environment.GetFolderPath(Environment.SpecialFolder.Programs);
        var found = new List<Installation>();
        if (string.IsNullOrEmpty(programs))
            return found;

        foreach (var app in AppNames)
        {
            // The two layouts SkylineRunner probes: the publisher folder, and a per-app folder.
            var candidates = new[]
            {
                Path.Combine(programs, "MacCoss Lab, UW", app + ".appref-ms"),
                Path.Combine(programs, app, app + ".appref-ms"),
            };
            var path = candidates.FirstOrDefault(File.Exists);
            if (path is not null)
                found.Add(new Installation(app, path));
        }
        return found;
    }

    /// <summary>The preferred installation, or null when none is present.</summary>
    public static SkylineAppRunner? Find(Action<string>? log = null)
    {
        var installation = FindInstallations().FirstOrDefault();
        if (installation is null)
        {
            log?.Invoke("No installed Skyline shortcut found; headless export will use SkylineCmd instead.");
            return null;
        }
        return new SkylineAppRunner(installation);
    }

    /// <summary>
    /// How many <see cref="Run"/> calls are in flight in this process. PRISM exports several documents at
    /// once, so this is routinely greater than one - and <see cref="KillSpawned"/> then has no way to tell
    /// the headless Skyline it should kill from one another export is still using, because neither is our
    /// child and the connection carries no PID.
    /// </summary>
    private static int _inFlight;

    public void Run(string[] args, Action<string> log, CancellationToken cancellationToken,
        TimeSpan? timeout = null)
    {
        Interlocked.Increment(ref _inFlight);
        try
        {
            RunCore(args, log, cancellationToken, timeout);
        }
        finally
        {
            Interlocked.Decrement(ref _inFlight);
        }
    }

    private void RunCore(string[] args, Action<string> log, CancellationToken cancellationToken,
        TimeSpan? timeout)
    {
        var suffix = "-" + Guid.NewGuid();
        var inPipeName = "SkylineInputPipe" + suffix;
        var outPipeName = "SkylineOutputPipe" + suffix;

        log($"  > {_installation.AppName} {string.Join(" ", args)}");

        // Which Skyline processes existed BEFORE we launched, so the one we start can be identified
        // and killed if we give up. It is not a child of ours - the cmd.exe that launches the
        // ClickOnce shortcut exits immediately - so there is no handle to wait on or kill.
        var preexisting = SkylineProcessIds();

        // cmd.exe /c is how the shortcut gets launched with an argument; .appref-ms is not directly
        // executable. Paths containing ^ or & must be escaped for cmd even though they are quoted.
        var psi = new ProcessStartInfo("cmd.exe")
        {
            CreateNoWindow = true,
            UseShellExecute = false,
            Arguments = $"/c \"{EscapeForCmd(_installation.ShortcutPath)}\" CMD{suffix}",
        };

        // PipeOptions.Asynchronous so WaitForConnection can wait with a deadline that actually cancels.
        // See WaitForConnection for why the synchronous overload cannot be given up on safely.
        using var serverStream = new NamedPipeServerStream(
            inPipeName, PipeDirection.InOut, 1, PipeTransmissionMode.Byte, PipeOptions.Asynchronous);
        using (var launcher = Process.Start(psi))
        {
            if (launcher is null)
                throw new InvalidOperationException($"Could not launch {_installation.AppName}.");
        }

        // ONE absolute deadline for the whole call, not a fresh copy of `timeout` per phase. Each phase
        // gets what is LEFT of it: startup, the output-pipe connect and the read loop are sequential
        // steps of one call, so giving each the full bound made a caller's stated limit a per-step limit.
        // SkylineIsolationImporter passes 5 min "precisely so it can never hold a run up" - and a Skyline
        // slow at every step could take three times that before anything gave up.
        var callDeadline = timeout.HasValue ? DateTime.UtcNow + timeout.Value : (DateTime?)null;

        switch (WaitForConnection(serverStream, preexisting, log, cancellationToken, Remaining(callDeadline)))
        {
            case StartupResult.Connected:
                break;

            case StartupResult.StartedButNeverConnected:
                // It is up but useless to us - we are about to abandon its pipe, so it would sit there
                // holding the document open forever.
                KillSpawned(preexisting, log);
                throw new InvalidOperationException(
                    $"{_installation.AppName} started but never connected on the command pipe within "
                    + $"{Min(_maxStartupTimeout, Remaining(callDeadline)).TotalMinutes:F1} min.");

            default:
                throw new InvalidOperationException(
                    $"{_installation.AppName} did not start within "
                    + $"{Min(_startupTimeout, Remaining(callDeadline)).TotalSeconds:F0}s (no connection on the command "
                    + "pipe). Another export may be saturating the machine - try exporting one document "
                    + "at a time.");
        }

        using (var writer = new StreamWriter(serverStream))
        {
            // Skyline formats its console output to this width. There is no console in a WPF host, so
            // pick a wide fixed value rather than reading Console.BufferWidth (which throws there).
            writer.WriteLine("--sw=" + 512);
            writer.WriteLine("--dir=" + Directory.GetCurrentDirectory());
            foreach (var arg in args)
                writer.WriteLine(arg);
        }

        // Skyline streams its progress here and closes the pipe when the batch finishes. The launching
        // cmd.exe has already exited, so THIS is how we know the work is done - and the only place errors
        // surface, since there is no exit code to read.
        StringBuilder errors;
        using var outPipe = new NamedPipeClientStream(outPipeName);

        // Distinguish "Skyline is unresponsive" from "the caller's budget ran out on the way here".
        // Without this, an exhausted budget makes Remaining() zero, Connect(0) fails instantly, and the
        // blame lands on Skyline for never opening a pipe it was given no time to open - the same
        // misdiagnosis as the "Pipe is broken." message this class was fixed for.
        // A MINIMUM workable budget, not equality with zero. Remaining() only clamps to zero once the
        // deadline has already passed, so testing `== TimeSpan.Zero` let a remainder of a few
        // milliseconds through - Connect(3) then fails instantly and blames Skyline anyway, which is the
        // same misdiagnosis narrowed to a race with the clock rather than removed.
        var outPipeBudget = Remaining(callDeadline);
        if (outPipeBudget is not null && outPipeBudget.Value < MinimumOutPipeBudget)
        {
            KillSpawned(preexisting, log);
            throw new TimeoutException(
                $"{_installation.AppName} connected, but the {timeout!.Value.TotalMinutes:F0} min allowed "
                + "for this command was already spent getting it started, so it was stopped.");
        }
        try
        {
            outPipe.Connect((int)Min(_startupTimeout, outPipeBudget).TotalMilliseconds);
        }
        catch (TimeoutException)
        {
            throw new InvalidOperationException(
                $"{_installation.AppName} started but never opened its output pipe.");
        }

        // Read on a worker and consume with a deadline. Reading inline would block in ReadLine with
        // no way out: a Skyline that produces no output - stuck, or just slow on a huge file over a
        // network share - hangs the caller forever, and the cancellation check after ReadLine never
        // runs, so Stop cannot break it either.
        var lines = new BlockingCollection<string>();
        var readerThread = new Thread(() =>
        {
            try
            {
                using var reader = new StreamReader(outPipe);
                string? line;
                while ((line = reader.ReadLine()) is not null)
                    lines.Add(line);
            }
            catch (Exception)
            {
                // Pipe closed or disposed under us (which is how we unblock it); nothing to add.
            }
            finally
            {
                lines.CompleteAdding();
            }
        }) { IsBackground = true, Name = "SkylineAppRunner output" };
        readerThread.Start();

        try
        {
            errors = ConsumeOutput(
                lines, log, _installation.AppName, _idleTimeout, callDeadline, cancellationToken,
                totalBound: timeout);
        }
        catch (TimeoutException)
        {
            // Either bound expired. Whatever Skyline is doing, it is not doing it for us any more, and
            // leaving it running holds the document open with nothing left to stop it.
            KillSpawned(preexisting, log);
            throw;
        }
        catch (OperationCanceledException)
        {
            // The whole run is stopping, so reap even with other exports in flight - they are cancelling
            // on the same token, and each is doing this for its own Skyline.
            KillSpawned(preexisting, log, allStopping: true);
            throw;
        }

        if (errors.Length > 0)
            throw new InvalidOperationException(errors.ToString().Trim());
    }

    /// <summary>How often the consumer wakes while Skyline is producing no output.</summary>
    private static readonly TimeSpan PollInterval = TimeSpan.FromMilliseconds(250);

    private static HashSet<int> SkylineProcessIds()
    {
        var ids = new HashSet<int>();
        foreach (var name in AppNames)
        {
            try
            {
                foreach (var p in Process.GetProcessesByName(name))
                {
                    ids.Add(p.Id);
                    p.Dispose();
                }
            }
            catch (Exception)
            {
                // Enumeration can fail transiently; a missed PID only means we cannot kill that one.
            }
        }
        return ids;
    }

    /// <summary>
    /// Kill the Skyline this run started, and only that one. Two guards, because getting this wrong
    /// means killing the user's open document: the process must not have existed before we launched,
    /// AND it must have no main window - the headless instance has none, an interactive one always
    /// does. A Skyline the user opens by hand mid-run fails the second test.
    /// </summary>
    /// <param name="allStopping">
    /// True when EVERY in-flight export is being abandoned, not just this one - which is the case on
    /// cancellation, because the token is the run's and they all observe it together. The concurrency
    /// guard below must not apply then: there is no export left for an unrecognized Skyline to belong
    /// to, and declining to reap leaves one holding its document open with nothing to stop it.
    /// </param>
    private static void KillSpawned(HashSet<int> preexisting, Action<string> log, bool allStopping = false)
    {
        // A third guard, for the case the two below cannot cover: with another export in flight, a
        // headless Skyline that appeared after WE launched may well be ITS Skyline, quietly exporting a
        // different document. "Not preexisting" is only meaningful when we are the only one launching -
        // or when nothing is left running, which is what allStopping says.
        if (!allStopping && Volatile.Read(ref _inFlight) > 1)
        {
            log("    Leaving any headless Skyline running: another export is in flight and the one to "
                + "stop cannot be told from the one still working.");
            return;
        }

        foreach (var name in AppNames)
        {
            Process[] running;
            try
            {
                running = Process.GetProcessesByName(name);
            }
            catch (Exception)
            {
                continue;
            }

            foreach (var p in running)
            {
                try
                {
                    if (preexisting.Contains(p.Id) || p.MainWindowHandle != IntPtr.Zero)
                        continue;
                    log($"    Stopping the headless {name} started for this command (pid {p.Id}).");
                    p.Kill(entireProcessTree: true);
                }
                catch (Exception)
                {
                    // Already gone, or not ours to kill.
                }
                finally
                {
                    p.Dispose();
                }
            }
        }
    }

    /// <summary>
    /// Skyline marks failures with an "Error:" prefix at the start of a line (or right after a tab, when
    /// timestamps are enabled). Ported from SkylineRunner's ErrorChecker, including the localized
    /// prefixes - a translated Skyline still emits the English one for new messages.
    /// </summary>
    internal static bool IsErrorLine(string line)
    {
        if (HasErrorPrefix(line, "Error:", StringComparison.InvariantCulture))
            return true;
        return new[] { "エラー：", "错误：" } // ja, zh-CHS
            .Any(p => HasErrorPrefix(line, p, StringComparison.CurrentCulture));
    }

    private static bool HasErrorPrefix(string line, string prefix, StringComparison comparison)
    {
        var i = line.IndexOf(prefix, comparison);
        return i == 0 || (i > 0 && line[i - 1] == '\t');
    }

    /// <summary>How <see cref="WaitForConnection"/> ended - and, for the two failures, whose fault it is.</summary>
    private enum StartupResult
    {
        /// <summary>Skyline connected; the arguments can be written.</summary>
        Connected,

        /// <summary>No Skyline appeared at all - a launch that failed, not one that is slow.</summary>
        NeverStarted,

        /// <summary>Skyline is running but stayed silent past <see cref="_maxStartupTimeout"/>.</summary>
        StartedButNeverConnected,
    }

    /// <summary>
    /// Wait for Skyline to connect back on the command pipe.
    ///
    /// <para><b>Why the async overload.</b> The synchronous <c>WaitForConnection()</c> has no timeout and
    /// no cancellation, so giving up on it means blocking a worker thread and then releasing it by
    /// connecting a dummy client to our own pipe. The waiter cannot tell that dummy from Skyline: it
    /// records a connection, the wait reports success, and <see cref="RunCore"/> writes the arguments to a
    /// client that has already been disposed - failing with <c>IOException: Pipe is broken.</c> That is a
    /// timeout wearing a disguise, and it was diagnosed as a Skyline fault for exactly that reason. Worse,
    /// it is a race, so two calls failing for the SAME reason reported it two different ways: "Pipe is
    /// broken." from the first, the honest "did not start" from the second.</para>
    ///
    /// <para><b>Why the deadline extends.</b> A flat wall-clock limit cannot distinguish a Skyline that is
    /// slow to start from one that is never coming, and gets the answer wrong under exactly the load PRISM
    /// creates for itself by exporting several documents at once. The Skyline we launched is not our
    /// child - the cmd.exe that activates the ClickOnce shortcut exits immediately - but it is visible in
    /// the process list, so its presence answers the question directly: keep waiting while it is running,
    /// give up when it is not.</para>
    /// </summary>
    private StartupResult WaitForConnection(
        NamedPipeServerStream serverStream, HashSet<int> preexisting, Action<string> log,
        CancellationToken cancellationToken, TimeSpan? timeout)
    {
        // The caller's bound caps BOTH startup deadlines. Without this the extending wait below answers
        // only to its own limits, so a caller that documented "give up after this long" - the isolation
        // importer passes 5 min precisely so it can never hold a run up - waited the full base plus
        // extended startup before its deadline was consulted for the first time.
        var baseDeadline = Min(_startupTimeout, timeout);
        var maxDeadline = Min(_maxStartupTimeout, timeout);

        using var giveUp = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var connecting = serverStream.WaitForConnectionAsync(giveUp.Token);

        var start = DateTime.UtcNow;
        var extended = false;
        while (true)
        {
            // Checked BEFORE the wait, and the wait is given no token, so a Stop always leaves here as an
            // OperationCanceledException. Passing the token to Wait instead makes the exception TYPE a
            // race: `giveUp` is linked to the caller's token, so a Stop signals the token and cancels the
            // task at the same instant, and whichever Wait observes first decides. Observing the task
            // gives AggregateException(TaskCanceledException), which slips past every
            // `catch (OperationCanceledException)` between here and the caller - so the export is logged
            // as "Parquet export failed" and the CSV fallback launches Skyline again. Stop that does not
            // stop is worse than Stop that takes a second.
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                if (connecting.Wait((int)StartupPollInterval.TotalMilliseconds))
                    return StartupResult.Connected;
            }
            catch (AggregateException ex)
            {
                giveUp.Cancel();
                cancellationToken.ThrowIfCancellationRequested(); // cancelled mid-wait
                throw new InvalidOperationException(
                    $"{_installation.AppName}: waiting on the command pipe failed "
                    + $"({ex.GetBaseException().Message}).", ex.GetBaseException());
            }

            var waited = DateTime.UtcNow - start;
            if (waited < baseDeadline)
                continue;

            // Past the base deadline. Only a Skyline that is actually up earns more time.
            var running = StartedSkylineIsRunning(preexisting);
            if (!running || waited >= maxDeadline)
            {
                giveUp.Cancel();
                return running ? StartupResult.StartedButNeverConnected : StartupResult.NeverStarted;
            }
            if (!extended)
            {
                log($"    {_installation.AppName} is taking longer than "
                    + $"{baseDeadline.TotalSeconds:F0}s to start but is running; still waiting "
                    + $"(up to {maxDeadline.TotalMinutes:F1} min).");
                extended = true;
            }
        }
    }

    /// <summary>The shorter of a fixed startup deadline and the caller's overall bound, if it gave one.</summary>
    private static TimeSpan Min(TimeSpan fixedDeadline, TimeSpan? callerBound) =>
        callerBound is null || callerBound.Value > fixedDeadline ? fixedDeadline : callerBound.Value;

    /// <summary>
    /// What is left of the caller's overall bound, or null when it gave none. Never negative: an
    /// exhausted budget is zero, which every consumer here reads as "give up now" rather than as a
    /// negative timeout they would have to special-case.
    /// </summary>
    private static TimeSpan? Remaining(DateTime? deadline)
    {
        if (deadline is null)
            return null;
        var left = deadline.Value - DateTime.UtcNow;
        return left > TimeSpan.Zero ? left : TimeSpan.Zero;
    }

    /// <summary>
    /// Drain Skyline's output until it closes the pipe, collecting the <c>Error:</c> lines that are the
    /// protocol's only failure signal, and giving up if it goes silent or overruns the caller's bound.
    ///
    /// <para>Separated from <see cref="RunCore"/> so the two deadlines can be tested against a fake
    /// clock. They are the whole point of this method and neither is reachable from a test that has to
    /// start a real Skyline and wait twenty minutes for it to say nothing.</para>
    /// </summary>
    /// <param name="utcNow">Injected clock; defaults to the real one.</param>
    /// <param name="totalBound">
    /// The caller's overall bound, carried alongside <paramref name="deadline"/> only so the message can
    /// name it. Reporting "did not finish within the time allowed" without the number leaves a reader
    /// unable to tell a 5-minute bound from a 60-minute one, or which caller set it.
    /// </param>
    internal static StringBuilder ConsumeOutput(
        BlockingCollection<string> lines, Action<string> log, string appName, TimeSpan idleTimeout,
        DateTime? deadline, CancellationToken cancellationToken, Func<DateTime>? utcNow = null,
        TimeSpan? totalBound = null)
    {
        var now = utcNow ?? (() => DateTime.UtcNow);
        var idleWatch = new SkylineIdleWatchdog(appName, idleTimeout, now);
        var errors = new StringBuilder();

        while (true)
        {
            // Wake regularly even when Skyline is silent, so cancellation and both deadlines are
            // observed while nothing is being printed - which is exactly the stuck case.
            if (lines.TryTake(out var line, (int)PollInterval.TotalMilliseconds, cancellationToken))
            {
                log("    " + line);
                idleWatch.SawOutput();
                if (IsErrorLine(line))
                    errors.AppendLine(line.Trim());
                continue;
            }
            if (lines.IsCompleted)
                return errors;

            if (deadline is not null && now() > deadline)
                throw new TimeoutException(
                    $"{appName} did not finish within "
                    + (totalBound is null ? "the time allowed" : $"{totalBound.Value.TotalMinutes:F0} min")
                    + " and was stopped.");

            if (idleWatch.CheckStalled(log) is { } stalled)
                throw stalled;
        }
    }

    /// <summary>
    /// Less budget than this left for the output pipe means the caller's bound is spent, not that Skyline
    /// is unresponsive. Skyline opens the pipe within milliseconds of connecting on the command pipe, so
    /// a second is generous for the real case and still catches an exhausted budget before it can be
    /// mistaken for one.
    /// </summary>
    private static readonly TimeSpan MinimumOutPipeBudget = TimeSpan.FromSeconds(1);

    /// <summary>How often the connection wait wakes to re-check the deadline and the process list.</summary>
    private static readonly TimeSpan StartupPollInterval = TimeSpan.FromSeconds(1);

    /// <summary>Whether a Skyline that was not running before we launched is running now.</summary>
    private static bool StartedSkylineIsRunning(HashSet<int> preexisting) =>
        SkylineProcessIds().Any(id => !preexisting.Contains(id));

    /// <summary>
    /// cmd.exe needs ^ and &amp; escaped even inside quotes (a user name like "V&amp;V" otherwise breaks the
    /// command). Once escaping starts, spaces must be escaped too. Ported from SkylineRunner.
    /// </summary>
    internal static string EscapeForCmd(string path)
    {
        var escapeChars = new[] { '^', '&' }; // caret first, or it double-escapes
        if (path.IndexOfAny(escapeChars) < 0)
            return path;
        foreach (var ch in escapeChars.Append(' '))
            path = path.Replace(ch.ToString(), "^" + ch);
        return path;
    }
}
