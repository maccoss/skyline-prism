using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// Stop has to actually stop. A token that is only checked between stages is not much use when a
/// single stage runs for twenty minutes, so these pin that cancellation is observed and that it
/// surfaces as cancellation rather than as a crash the UI would report as a failed run.
/// </summary>
public class PipelineCancellationTests
{
    private static string[] Inputs() => new[]
    {
        Path.Combine(Fixtures.Path2("mini", "merge"), "mini_plate1.csv"),
        Path.Combine(Fixtures.Path2("mini", "merge"), "mini_plate2.csv"),
    };

    /// <summary>Cancelled before it starts: nothing should run at all.</summary>
    [Fact]
    public void AlreadyCancelled_StopsImmediately()
    {
        var outDir = NewOutputDir();
        try
        {
            using var cts = new CancellationTokenSource();
            cts.Cancel();

            var stages = 0;
            Assert.ThrowsAny<OperationCanceledException>(() => PrismPipeline.Run(
                Inputs(), outDir, Config(), metadataPaths: null,
                log: _ => Interlocked.Increment(ref stages),
                forceReprocess: true, cancellationToken: cts.Token));

            // The merge is the first thing that happens; it must not have produced its output.
            Assert.False(File.Exists(Path.Combine(outDir, "merged_data.parquet")));
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    /// <summary>
    /// Cancelled from another thread partway through: the run must end promptly and as a
    /// cancellation. The fixture is tiny, so "promptly" here is really "at the first checkpoint it
    /// reaches" - the point being that some checkpoint exists on every path, not that this timing is
    /// meaningful.
    /// </summary>
    [Fact]
    public void CancelledMidRun_EndsAsCancellationNotFailure()
    {
        var outDir = NewOutputDir();
        try
        {
            using var cts = new CancellationTokenSource();
            var reachedPipeline = new ManualResetEventSlim(false);

            void Log(string line)
            {
                // Cancel as soon as the run is genuinely under way.
                if (line.Contains("Stage 1", StringComparison.Ordinal))
                    reachedPipeline.Set();
            }

            var watchdog = Stopwatch.StartNew();
            var thread = new Thread(() =>
            {
                reachedPipeline.Wait(TimeSpan.FromSeconds(30));
                cts.Cancel();
            }) { IsBackground = true };
            thread.Start();

            var ex = Record.Exception(() => PrismPipeline.Run(
                Inputs(), outDir, Config(), metadataPaths: null, log: Log,
                forceReprocess: true, cancellationToken: cts.Token));
            watchdog.Stop();

            // Either it finished before the cancel landed (the fixture is 5 peptides) or it cancelled.
            // What must NOT happen is some other exception - that is what the UI shows as a crash.
            if (ex is not null)
                Assert.True(IsCancellation(ex), $"expected cancellation, got {ex.GetType().Name}: {ex.Message}");
            Assert.True(watchdog.Elapsed < TimeSpan.FromMinutes(2), "run did not end promptly after Stop");
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    private static bool IsCancellation(Exception ex) => ex switch
    {
        OperationCanceledException => true,
        AggregateException agg => agg.InnerExceptions.Count > 0
                                  && System.Linq.Enumerable.All(agg.InnerExceptions, IsCancellation),
        _ => false,
    };

    private static PrismConfig Config() =>
        PrismConfig.Load(Path.Combine(Fixtures.Path2("mini", "e2e-sum"), "config.yaml"));

    private static string NewOutputDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_cancel_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void Cleanup(string dir)
    {
        try
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
        catch (IOException)
        {
            // A cancelled DuckDB query can still be releasing its files; leaving a temp dir behind is
            // not worth failing a test over.
        }
    }
}
