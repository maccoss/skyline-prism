#nullable enable

using System;
using System.Threading;

namespace SkylinePrism.Skyline;

/// <summary>
/// Runs a batch of Skyline command-line arguments against a document that is NOT open, and reports
/// whether it succeeded. Two implementations exist because they differ in one decisive way:
/// <list type="bullet">
/// <item><description><see cref="SkylineAppRunner"/> drives the INSTALLED <c>Skyline.exe</c> headlessly
/// (the SkylineRunner mechanism). It therefore gets <c>Skyline.exe.config</c> - the one carrying the
/// Parquet.Net assembly binding - so it can export parquet.</description></item>
/// <item><description><see cref="SkylineCmdRunner"/> launches <c>SkylineCmd.exe</c>, which starts faster
/// but ships without those bindings, so its parquet export fails.</description></item>
/// </list>
/// <see cref="HeadlessSkylineExporter"/> prefers the first and falls back to the second.
/// </summary>
public interface ISkylineCommandRunner
{
    /// <summary>How this runner reaches Skyline, for logs and error messages.</summary>
    string Description { get; }

    /// <summary>
    /// True when this runner executes the full Skyline application (and so can write parquet), false for
    /// the stripped SkylineCmd host.
    /// </summary>
    bool SupportsParquet { get; }

    /// <summary>
    /// Run <paramref name="args"/>, streaming Skyline's output into <paramref name="log"/>. Throws
    /// <see cref="InvalidOperationException"/> when Skyline reports an error, and
    /// <see cref="OperationCanceledException"/> if cancelled.
    /// </summary>
    /// <param name="timeout">
    /// Give up after this long with the command still running, killing the Skyline it started.
    /// <c>null</c> means wait indefinitely, which is right for the report export - it is the point of
    /// the run, it can legitimately take an hour on a large document, and abandoning it accomplishes
    /// nothing. Pass a bound for work that is merely an enrichment, where waiting forever turns an
    /// optional extra into a hang.
    /// </param>
    void Run(string[] args, Action<string> log, CancellationToken cancellationToken,
        TimeSpan? timeout = null);
}
