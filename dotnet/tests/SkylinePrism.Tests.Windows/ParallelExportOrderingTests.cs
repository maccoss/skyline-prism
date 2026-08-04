using System;
using System.Collections.Generic;
using System.Linq;
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
/// <para>Completion order is <b>simulated explicitly</b> rather than provoked with sleeps: a CI runner
/// with few cores can serialise a <c>Parallel.For</c> no matter what degree is requested, so any test
/// asserting "these finished out of order" is inherently flaky. The invariant under test - index-based
/// collection survives arbitrary completion order - does not need real concurrency to verify.</para>
/// </summary>
public class ParallelExportOrderingTests
{
    /// <summary>An arbitrary, fixed completion order that is not the input order.</summary>
    private static readonly int[] CompletionOrder = { 3, 0, 5, 1, 4, 2 };

    private const int Count = 6;

    private static string Doc(int i) => $"doc{i}";

    [Fact]
    public void CollectingByIndex_PreservesInputOrder_ForAnyCompletionOrder()
    {
        // What RunPipeline does: write into a pre-sized slot, flatten afterwards.
        var slots = new string?[Count];
        foreach (var i in CompletionOrder)
            slots[i] = Doc(i);

        var flattened = slots.Where(s => s is not null).Select(s => s!).ToArray();

        Assert.Equal(Enumerable.Range(0, Count).Select(Doc).ToArray(), flattened);
    }

    [Fact]
    public void AppendingOnCompletion_MisPairsMetadata()
    {
        // The bug this design avoids: same work, same results, wrong pairing.
        var appended = new List<string>();
        foreach (var i in CompletionOrder)
            appended.Add(Doc(i));

        var inInputOrder = Enumerable.Range(0, Count).Select(Doc).ToArray();
        Assert.NotEqual(inInputOrder, appended);                              // order is wrong...
        Assert.Equal(inInputOrder.OrderBy(x => x), appended.OrderBy(x => x)); // ...though the same set
    }

    [Fact]
    public void ReportsAndMetadataStayAlignedWithEachOther()
    {
        // The pairing that actually matters: report[N] and metadata[N] must describe the same document,
        // whatever order the exports finished in.
        var reports = new string?[Count];
        var metadata = new string?[Count];
        foreach (var i in CompletionOrder)
        {
            reports[i] = $"{Doc(i)}.parquet";
            metadata[i] = $"{Doc(i)}.metadata.csv";
        }

        for (var i = 0; i < Count; i++)
        {
            Assert.Equal(
                System.IO.Path.GetFileNameWithoutExtension(reports[i]),
                metadata[i]!.Replace(".metadata.csv", ""));
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void ResultsStayInInputOrder_UnderRealParallelism(int degree)
    {
        // Exercises the real Parallel.For path. Order preservation holds whether the runner actually
        // overlaps the work or serialises it, so this is safe on any machine.
        var slots = new string?[Count];
        Parallel.For(0, Count, new ParallelOptions { MaxDegreeOfParallelism = degree },
            i => slots[i] = Doc(i));

        Assert.Equal(Enumerable.Range(0, Count).Select(Doc).ToArray(), slots.Select(s => s!).ToArray());
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
