using System;
using System.IO;
using DuckDB.NET.Data;

namespace SkylinePrism.Core.IO;

/// <summary>
/// The DuckDB settings PRISM cannot afford to leave at their defaults, in one place: a bounded buffer
/// pool with somewhere to spill, a partitioned writer that cannot outgrow it, and commands that hand
/// rows back as they are produced instead of materializing the whole result first.
/// <para>
/// Both exist because DuckDB runs <b>in-process</b> and allocates native memory the .NET GC neither
/// sees nor collects. A connection left at the defaults gets a buffer pool of 80% of physical RAM,
/// which on a workstation that is also running the Skyline instance PRISM was launched from is not a
/// safety margin - it is a promise to page Skyline out.
/// </para>
/// </summary>
internal static class DuckDbTuning
{
    /// <summary>
    /// Bound a connection's buffer pool and give it a spill directory. Work beyond
    /// <paramref name="memoryBudgetMb"/> spills to <paramref name="tempDir"/>, so a small budget is
    /// slower, never wrong - but a connection with NO <c>temp_directory</c> cannot spill at all and
    /// fails outright when it runs out, which is why these two are always set together.
    /// </summary>
    public static void Apply(DuckDBConnection conn, int memoryBudgetMb, string tempDir)
    {
        Directory.CreateDirectory(tempDir);
        Set(conn, $"SET memory_limit='{memoryBudgetMb}MB'");
        Set(conn, $"SET temp_directory='{tempDir.Replace("'", "''")}'");
        // Nothing in PRISM depends on a result arriving in file order - every consumer either
        // aggregates, or asks for an explicit ORDER BY - so let DuckDB skip the reordering buffers
        // it would otherwise keep to preserve it.
        Set(conn, "SET preserve_insertion_order=false");
        Set(conn, $"SET threads={Environment.ProcessorCount}");
    }

    /// <summary>
    /// The extra bounds a <c>PARTITION_BY</c> write needs.
    /// <para>
    /// <b>This is not optional at scale.</b> DuckDB's partitioned writer buffers rows per WRITING
    /// THREAD, and those buffers live OUTSIDE the spillable buffer pool - <c>memory_limit</c> watches
    /// them fill it but <c>temp_directory</c> cannot rescue them, so the write dies with "failed to pin
    /// block" rather than spilling. The footprint is therefore <i>threads x flush threshold</i>: at the
    /// defaults on a 32-core machine that is 32 x 524,288 = 16.8M rows in flight, which exhausted an
    /// 8 GB budget in seconds on a 20-document cohort.
    /// </para>
    /// <para>
    /// So the lever is the THREAD COUNT, not the partition count - and capping it is not merely a
    /// memory concession, it is also faster. Each thread keeps an open buffer per partition it has
    /// seen, and rows are hash-assigned, so every thread touches every partition: 32 threads x 155
    /// partitions is ~5,000 live buffers, each flushed in slivers far smaller than a row group.
    /// Measured on 186M rows into 16 partitions, dropping 32 threads to 8 took the write from
    /// 1.62 min / 7.9 GB to <b>0.84 min / 2.4 GB</b>. 16 threads was no faster than 8 and heavier;
    /// 4 was slower. The partition count then costs only file handles: 256 partitions at 8 threads
    /// still writes in 2.09 min at 7.8 GB.
    /// </para>
    /// </summary>
    public static void ApplyPartitionedWrite(DuckDBConnection conn, int partitions)
    {
        // Hold every partition open. Rows are hash-assigned, so all partitions are live throughout the
        // scan; with fewer slots DuckDB evicts and reopens constantly - a rejected experiment at 16
        // slots over 155 partitions produced 1,048,920 files and took 82 minutes.
        Set(conn, $"SET partitioned_write_max_open_files={Math.Max(1, partitions)}");

        var threads = WriteThreads();
        Set(conn, $"SET threads={threads}");
        Set(conn, $"SET partitioned_write_flush_threshold={TargetBufferedRows / threads}");
    }

    /// <summary>
    /// Threads for a partitioned write. A flat cap rather than a fraction of the machine: what it
    /// bounds is live partition buffers (threads x partitions), which has nothing to do with how many
    /// cores are available, and past ~8 the extra threads buy no throughput on a write that is
    /// compression- and IO-bound.
    /// </summary>
    private static int WriteThreads() => Math.Max(1, Math.Min(8, Environment.ProcessorCount));

    /// <summary>
    /// Rows the partitioned writer may hold in flight across all threads. At ~26 wide columns this is
    /// a few hundred MB - large enough that each flush is a real chunk rather than a sliver, small
    /// enough to leave the budget to the readers feeding it.
    /// </summary>
    private const int TargetBufferedRows = 2_000_000;

    /// <summary>
    /// A command that streams its result.
    /// <para>
    /// <b>This is not the DuckDB.NET default.</b> <c>DuckDBCommand.UseStreamingMode</c> is
    /// <c>false</c> unless you set it, and a non-streaming <c>ExecuteReader</c> runs the query to
    /// completion and materializes <i>every row</i> into a client-side result before
    /// <c>reader.Read()</c> returns once. That result is not part of the buffer pool, so
    /// <c>memory_limit</c> does not cap it and <c>temp_directory</c> cannot spill it: on a
    /// transition-level cohort it is simply hundreds of millions of rows resident at once. Any query
    /// whose result set scales with the size of the data must use this.
    /// </para>
    /// </summary>
    public static DuckDBCommand StreamingCommand(DuckDBConnection conn, string sql)
    {
        var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        cmd.UseStreamingMode = true;
        return cmd;
    }

    private static void Set(DuckDBConnection conn, string sql)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }
}
