using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace Stage2Bench;

/// <summary>
/// Compares the candidate Stage 2 read strategies on one partition of a real merged dataset.
/// <para>
/// Two things here are deliberate, both learned the hard way in the work that produced
/// <c>dotnet/STAGE2_THROUGHPUT.md</c>:
/// </para>
/// <list type="number">
/// <item><b>Arms are interleaved</b> (A,B,C,A,B,C...) rather than run in blocks. Machine state drifts;
/// blocked runs turn that drift into a fake difference between arms, interleaved runs turn it into
/// visible variance within an arm. Results are reported as the MEDIAN of the repeats.</item>
/// <item><b>Every run is watched for contention</b> and labelled. An identical configuration once
/// measured 2.07 and 3.70 minutes hours apart because other software had started in between, and
/// several hours of analysis were built on the gap before anyone looked at the process list.</item>
/// </list>
/// <para>
/// Arms must also agree on what they read - rows and peptides are compared across arms and a mismatch
/// is reported loudly, because a faster arm that reads different data is not a faster arm.
/// </para>
/// </summary>
public static class Program
{
    private sealed record Arm(string Name, Func<string, string, int, ReadResult> Run);

    private sealed record Timing(string Arm, double Seconds, double AllocGb, double PeakWsGb,
        ReadResult Result, MachineLoad.Report Load);

    public static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            Console.WriteLine("""
                Stage2Bench <merged-dataset-dir> [repeats] [budgetMb]

                  merged-dataset-dir   a PRISM merged_data/ directory (partitioned)
                  repeats              times each arm runs, interleaved (default 3)
                  budgetMb             DuckDB memory limit for the arms that use it (default 8192)

                Build one with:  prism merge <report1> <report2> -o <dir>/merged
                Measures the FIRST partition only - the point is the per-partition read cost.
                """);
            return args.Length == 0 ? 2 : 0;
        }

        var root = args[0];
        var repeats = args.Length > 1 ? int.Parse(args[1]) : 3;
        var budgetMb = args.Length > 2 ? int.Parse(args[2]) : 8192;

        var partition = FirstPartition(root);
        if (partition is null)
        {
            Console.Error.WriteLine($"No _pep_bucket=* partition found under {root}.");
            return 1;
        }
        var scratch = Path.Combine(Path.GetTempPath(), "stage2bench");
        Directory.CreateDirectory(scratch);

        Console.WriteLine($"partition : {partition}");
        Console.WriteLine($"repeats   : {repeats} (interleaved)   budget: {budgetMb} MB");
        Console.WriteLine();

        var arms = new[]
        {
            new Arm("duckdb-stream", Strategies.DuckDbSortedStream),
            new Arm("nosort-managed", Strategies.ParquetNoSortGrouped),
            new Arm("copy-then-read", Strategies.CopyThenParquetRead),
        };

        var timings = new List<Timing>();
        for (var rep = 1; rep <= repeats; rep++)
        {
            foreach (var arm in arms)
            {
                Console.Write($"  rep {rep} {arm.Name,-16} ... ");
                var t = Measure(arm, partition, scratch, budgetMb);
                timings.Add(t);
                Console.WriteLine($"{t.Seconds,6:n1}s   {t.Load.Describe()}");
            }
        }

        Report(timings, repeats);
        return 0;
    }

    private static Timing Measure(Arm arm, string partition, string scratch, int budgetMb)
    {
        // Settle the heap first so the allocation figure belongs to this arm, not the previous one.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var load = new MachineLoad();
        load.Start();
        var alloc0 = GC.GetTotalAllocatedBytes();
        var proc = Process.GetCurrentProcess();
        var sw = Stopwatch.StartNew();
        var result = arm.Run(partition, scratch, budgetMb);
        sw.Stop();
        proc.Refresh();
        var alloc = (GC.GetTotalAllocatedBytes() - alloc0) / 1024.0 / 1024 / 1024;
        return new Timing(arm.Name, sw.Elapsed.TotalSeconds, alloc,
            proc.PeakWorkingSet64 / 1024.0 / 1024 / 1024, result, load.Stop());
    }

    private static void Report(List<Timing> timings, int repeats)
    {
        Console.WriteLine();
        Console.WriteLine($"{"arm",-18} {"median s",9} {"spread",9} {"alloc GB",9} {"rows",14} {"peptides",10} {"values",14}");
        var baseline = double.NaN;
        foreach (var g in timings.GroupBy(t => t.Arm))
        {
            var secs = g.Select(t => t.Seconds).OrderBy(s => s).ToList();
            var median = secs[secs.Count / 2];
            if (double.IsNaN(baseline))
                baseline = median;
            var spread = secs.Count > 1 ? (secs[^1] - secs[0]) / median : 0;
            var first = g.First();
            Console.WriteLine($"{g.Key,-18} {median,9:n1} {spread,8:p0} {g.Max(t => t.AllocGb),9:n2} "
                + $"{first.Result.Rows,14:n0} {first.Result.Peptides,10:n0} {first.Result.Values,14:n0}"
                + (median > 0 && !double.IsNaN(baseline) ? $"   {baseline / median,5:n2}x" : ""));
        }

        // Correctness before speed: arms that disagree on the data are not comparable.
        var rows = timings.Select(t => t.Result.Rows).Distinct().ToList();
        var peps = timings.Select(t => t.Result.Peptides).Distinct().ToList();
        var vals = timings.Select(t => t.Result.Values).Distinct().ToList();
        Console.WriteLine();
        if (rows.Count > 1 || peps.Count > 1 || vals.Count > 1)
        {
            Console.WriteLine("*** ARMS DISAGREE — timings are not comparable ***");
            foreach (var g in timings.GroupBy(t => t.Arm))
                Console.WriteLine($"    {g.Key,-18} rows {g.First().Result.Rows:n0}  "
                    + $"peptides {g.First().Result.Peptides:n0}  values {g.First().Result.Values:n0}");
        }
        else
        {
            Console.WriteLine($"arms agree: {rows[0]:n0} rows, {peps[0]:n0} peptides, {vals[0]:n0} values accumulated");
        }

        var contended = timings.Count(t => t.Load.Contended);
        if (contended > 0)
        {
            Console.WriteLine();
            Console.WriteLine($"*** {contended} of {timings.Count} runs were CONTENDED — treat absolute");
            Console.WriteLine("    numbers as indicative only, and prefer the ratios, which contention");
            Console.WriteLine("    affects roughly equally across interleaved arms. Re-run on a quiet machine");
            Console.WriteLine("    before designing anything around these figures.");
        }
        if (repeats < 3)
            Console.WriteLine("(fewer than 3 repeats: the spread column is not meaningful)");
    }

    private static string? FirstPartition(string root)
    {
        if (File.Exists(root))
            return root;
        if (!Directory.Exists(root))
            return null;
        var part = Directory.GetDirectories(root, "_pep_bucket=*")
            .OrderBy(d => d, StringComparer.Ordinal)
            .FirstOrDefault();
        return part is null ? null : Path.Combine(part, "*.parquet");
    }
}
