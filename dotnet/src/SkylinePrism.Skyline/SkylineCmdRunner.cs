#nullable enable

using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;

namespace SkylinePrism.Skyline;

/// <summary>
/// Runs Skyline command-line arguments through <c>SkylineCmd.exe</c>. Faster to start than
/// <see cref="SkylineAppRunner"/>, but it cannot export parquet: <c>SkylineCmd.exe.config</c> ships
/// without the Parquet.Net assembly bindings that <c>Skyline.exe.config</c> has, so a
/// <c>.parquet</c> output fails with "Could not load file or assembly 'Parquet'". Used as the fallback
/// when no installed Skyline shortcut is found.
/// </summary>
public sealed class SkylineCmdRunner : ISkylineCommandRunner
{
    private readonly string _exePath;

    public SkylineCmdRunner(string exePath)
    {
        if (string.IsNullOrWhiteSpace(exePath))
            throw new ArgumentException("SkylineCmd path is required.", nameof(exePath));
        _exePath = exePath;
    }

    public string Description => $"SkylineCmd ({_exePath})";

    /// <summary>False - see the class remarks; the caller must not ask this runner for parquet.</summary>
    public bool SupportsParquet => false;

    /// <summary>Locate SkylineCmd.exe and wrap it, or null when it is not installed.</summary>
    public static SkylineCmdRunner? Find(string? explicitPath = null, Action<string>? log = null)
    {
        var exe = SkylineCmdLocator.Find(explicitPath, log);
        return exe is null ? null : new SkylineCmdRunner(exe);
    }

    /// <summary>
    /// Run SkylineCmd, streaming its output into <paramref name="log"/>.
    /// <para>
    /// <paramref name="timeout"/> defaults to none, because loading a large document and writing a
    /// transition report legitimately takes many minutes and the caller cancels. Pass a bound for work
    /// that is merely an enrichment, where waiting indefinitely turns an optional extra into a hang -
    /// see <see cref="ISkylineCommandRunner.Run"/>. On expiry the process is killed and a
    /// <see cref="TimeoutException"/> is thrown.
    /// </para>
    /// </summary>
    public void Run(string[] args, Action<string> log, CancellationToken cancellationToken,
        TimeSpan? timeout = null)
    {
        var deadline = timeout.HasValue ? DateTime.UtcNow + timeout.Value : (DateTime?)null;
        var psi = new ProcessStartInfo
        {
            FileName = _exePath,
            WorkingDirectory = Path.GetDirectoryName(_exePath) ?? AppContext.BaseDirectory,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var a in args)
            psi.ArgumentList.Add(a); // ArgumentList quotes paths with spaces correctly

        log("  > SkylineCmd " + string.Join(" ", args));

        using var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        var stderr = new StringBuilder();
        process.OutputDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
                log("    " + e.Data);
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (string.IsNullOrWhiteSpace(e.Data))
                return;
            log("    " + e.Data);
            stderr.AppendLine(e.Data);
        };

        if (!process.Start())
            throw new InvalidOperationException($"Could not start {_exePath}.");
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        // Poll so cancellation is observed without blocking on an async wait (Run may be called from any
        // thread). The final no-arg WaitForExit flushes the async stdout/stderr handlers.
        while (!process.WaitForExit(250))
        {
            if (cancellationToken.IsCancellationRequested)
            {
                TryKill(process, log);
                cancellationToken.ThrowIfCancellationRequested();
            }
            if (deadline is not null && DateTime.UtcNow > deadline)
            {
                TryKill(process, log);
                throw new TimeoutException(
                    $"SkylineCmd did not finish within {timeout!.Value.TotalMinutes:F0} min "
                    + "and was stopped.");
            }
        }
        process.WaitForExit();

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"SkylineCmd exited with code {process.ExitCode}."
                + (stderr.Length > 0 ? " " + stderr.ToString().Trim() : ""));
        }
    }

    private static void TryKill(Process process, Action<string> log)
    {
        try
        {
            if (!process.HasExited)
                process.Kill(entireProcessTree: true);
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or SystemException)
        {
            log("(could not stop SkylineCmd: " + ex.Message + ")");
        }
    }
}
