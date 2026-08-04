using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Guards the invariant that makes parallel export of several Skyline documents safe.
///
/// <para>The pipeline pairs metadata files to inputs <b>positionally</b> - metadata[N] describes
/// input[N] - which is what lets identically named replicates keep their own document's sample types
/// and batch. Exporting documents concurrently means they finish in an arbitrary order, so results must
/// be stored BY INDEX and only flattened afterwards. Appending as each finishes would silently shuffle
/// the pairing: no error, no crash, just every plate's replicates attributed to the wrong document.</para>
///
/// <para>These tests exercise the collect-by-index pattern that <c>MainWindow.RunPipeline</c> uses,
/// with deliberately inverted completion order.</para>
/// </summary>
public class ParallelExportOrderingTests
{
    /// <summary>Runs work with the same shape as the export loop and returns results in input order.</summary>
    private static string[] CollectByIndex(int count, int degree, Func<int, string> work)
    {
        var results = new string?[count];
        Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = degree },
            i => results[i] = work(i));
        return results.Select(r => r!).ToArray();
    }

    [Fact]
    public void ResultsStayInInputOrder_EvenWhenCompletionOrderIsReversed()
    {
        const int n = 6;
        // Item 0 is slowest, item n-1 fastest: completion order is the exact reverse of input order.
        var results = CollectByIndex(n, degree: n, i =>
        {
            Thread.Sleep(20 * (n - i));
            return $"doc{i}";
        });

        Assert.Equal(Enumerable.Range(0, n).Select(i => $"doc{i}").ToArray(), results);
    }

    [Fact]
    public void CompletionOrderReallyDoesDiffer_SoTheTestAboveIsMeaningful()
    {
        const int n = 6;
        var completion = new ConcurrentQueue<int>();
        CollectByIndex(n, degree: n, i =>
        {
            Thread.Sleep(20 * (n - i));
            completion.Enqueue(i);
            return $"doc{i}";
        });

        // If everything happened to run sequentially the ordering test would prove nothing.
        Assert.NotEqual(Enumerable.Range(0, n).ToArray(), completion.ToArray());
    }

    [Fact]
    public void AppendingOnCompletionWouldMisPairMetadata()
    {
        // Demonstrates the bug this design avoids: the naive "add as each finishes" approach produces a
        // report/metadata pairing that does not match the inputs.
        const int n = 5;
        var appended = new ConcurrentQueue<string>();
        Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = n }, i =>
        {
            Thread.Sleep(20 * (n - i));
            appended.Enqueue($"doc{i}");
        });

        var inOrder = Enumerable.Range(0, n).Select(i => $"doc{i}").ToArray();
        Assert.NotEqual(inOrder, appended.ToArray());          // append order is wrong...
        Assert.Equal(inOrder.OrderBy(x => x), appended.OrderBy(x => x)); // ...though the same set
    }

    [Fact]
    public void SequentialDegreeStillPreservesOrder()
    {
        var results = CollectByIndex(4, degree: 1, i => $"doc{i}");
        Assert.Equal(new[] { "doc0", "doc1", "doc2", "doc3" }, results);
    }

    [Fact]
    public void AWorkerFailureSurfacesAsTheOriginalException()
    {
        // MainWindow unwraps AggregateException so the user sees which input failed, not
        // "One or more errors occurred."
        var ex = Assert.Throws<AggregateException>(() =>
            Parallel.For(0, 4, new ParallelOptions { MaxDegreeOfParallelism = 4 }, i =>
            {
                if (i == 2)
                    throw new InvalidOperationException("Input 'PlateC' (batch 'PlateC') failed: boom");
            }));

        var first = ex.Flatten().InnerExceptions[0];
        Assert.IsType<InvalidOperationException>(first);
        Assert.Contains("PlateC", first.Message);
    }
}
