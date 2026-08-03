#nullable enable

using System;
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

    public SkylineAppRunner(Installation installation, TimeSpan? startupTimeout = null)
    {
        _installation = installation;
        // A cold ClickOnce start (plus the update check Skyline does on launch) can take a while; the
        // official runner waits 15 s, which is too tight on a first run of the day.
        _startupTimeout = startupTimeout ?? TimeSpan.FromSeconds(90);
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

    public void Run(string[] args, Action<string> log, CancellationToken cancellationToken)
    {
        var suffix = "-" + Guid.NewGuid();
        var inPipeName = "SkylineInputPipe" + suffix;
        var outPipeName = "SkylineOutputPipe" + suffix;

        log($"  > {_installation.AppName} {string.Join(" ", args)}");

        // cmd.exe /c is how the shortcut gets launched with an argument; .appref-ms is not directly
        // executable. Paths containing ^ or & must be escaped for cmd even though they are quoted.
        var psi = new ProcessStartInfo("cmd.exe")
        {
            CreateNoWindow = true,
            UseShellExecute = false,
            Arguments = $"/c \"{EscapeForCmd(_installation.ShortcutPath)}\" CMD{suffix}",
        };

        using var serverStream = new NamedPipeServerStream(inPipeName);
        using (var launcher = Process.Start(psi))
        {
            if (launcher is null)
                throw new InvalidOperationException($"Could not launch {_installation.AppName}.");
        }

        if (!WaitForConnection(serverStream, inPipeName, cancellationToken))
        {
            throw new InvalidOperationException(
                $"{_installation.AppName} did not start within {_startupTimeout.TotalSeconds:F0}s "
                + "(no connection on the command pipe).");
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
        var errors = new StringBuilder();
        using var outPipe = new NamedPipeClientStream(outPipeName);
        try
        {
            outPipe.Connect((int)_startupTimeout.TotalMilliseconds);
        }
        catch (TimeoutException)
        {
            throw new InvalidOperationException(
                $"{_installation.AppName} started but never opened its output pipe.");
        }

        using (var reader = new StreamReader(outPipe))
        {
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                cancellationToken.ThrowIfCancellationRequested();
                log("    " + line);
                if (IsErrorLine(line))
                    errors.AppendLine(line.Trim());
            }
        }

        if (errors.Length > 0)
            throw new InvalidOperationException(errors.ToString().Trim());
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

    private bool WaitForConnection(
        NamedPipeServerStream serverStream, string inPipeName, CancellationToken cancellationToken)
    {
        // WaitForConnection has no cancellable/timeout overload, so run it on a worker and give up on the
        // deadline. The stray waiter is then released by connecting to our own pipe.
        var connected = false;
        var done = new ManualResetEventSlim(false);
        var waiter = new Thread(() =>
        {
            try
            {
                serverStream.WaitForConnection();
                connected = true;
            }
            catch (Exception)
            {
                // Disposed or aborted while waiting; treated as "not connected".
            }
            finally
            {
                done.Set();
            }
        }) { IsBackground = true };
        waiter.Start();

        var deadline = _startupTimeout;
        if (done.Wait(deadline, cancellationToken) && connected)
            return true;

        // Unblock the waiting thread so the pipe can be disposed cleanly.
        try
        {
            using var fake = new NamedPipeClientStream(inPipeName);
            fake.Connect(10);
        }
        catch (Exception)
        {
            // Nothing listening any more - fine, that was the point.
        }
        return connected;
    }

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
