using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using DuckDB.NET.Data;
using SkylinePrism.Core.Rollup;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Streams the merged transition-level data grouped by peptide, one <see cref="PeptideBlock"/> at a
/// time (mirroring chunked_processing.rollup_transitions_sorted).
/// <para>
/// Grouping comes from sorting each <see cref="MergedDataset"/> partition separately, never the cohort
/// as a whole. The merge has already hashed every row of a peptide into the same bucket, so a
/// per-partition <c>ORDER BY</c> produces exactly the same blocks a global one would - at a fraction of
/// the footprint, and a fraction that does not grow when more documents are added. Sorting the whole
/// dataset here instead would reintroduce the blocking operator the partitioning exists to remove.
/// </para>
/// <para>
/// Streaming is also <b>opt-in and easy to lose</b>: <c>DuckDBCommand.UseStreamingMode</c> defaults to
/// <c>false</c>, and without it <c>ExecuteReader</c> materializes the entire result set before the
/// first <c>Read()</c> returns - resident all at once, outside the buffer pool where neither
/// <c>memory_limit</c> nor <c>temp_directory</c> can touch it. Every query below therefore goes through
/// <see cref="DuckDbTuning.StreamingCommand"/>, and the connections are bounded by
/// <see cref="DuckDbTuning.Apply"/>. This class is the producer for the whole rollup, so a regression
/// here is a regression in PRISM's peak memory, whatever the rest of the stage does.
/// </para>
/// </summary>
public static class MergedParquetReader
{
    /// <summary>Distinct, sorted, non-null values of the sample column (the wide output columns).</summary>
    public static List<string> GetSortedSamples(
        MergedDataset dataset, string sampleCol, int memoryBudgetMb = 0)
    {
        using var conn = OpenBounded(dataset, memoryBudgetMb);
        using var cmd = DuckDbTuning.StreamingCommand(conn,
            $"SELECT DISTINCT \"{sampleCol}\" AS s FROM {Scan(dataset.ScanTarget)} " +
            "WHERE s IS NOT NULL ORDER BY s");
        using var reader = cmd.ExecuteReader();
        var samples = new List<string>();
        while (reader.Read())
            samples.Add(reader.GetString(0));
        return samples;
    }

    /// <summary>
    /// Stream <see cref="PeptideBlock"/>s grouped by peptide. Consecutive rows sharing a peptide value
    /// are grouped into one block, and partitions are processed in turn - safe because the merge
    /// guarantees a peptide never spans two of them, so a block is always complete when it is yielded.
    /// </summary>
    public static IEnumerable<PeptideBlock> StreamPeptideBlocks(
        MergedDataset dataset, SkylineColumns cols, IReadOnlyList<string> samples, bool includeProductMz = false,
        bool includeShapeCorr = false, int memoryBudgetMb = 0)
    {
        var withMz = includeProductMz && cols.ProductMz is not null;
        var withShape = includeShapeCorr && cols.ShapeCorrelation is not null;
        var extra = "";
        if (withMz)
            extra += $", p.\"{cols.ProductMz}\" AS mz";
        if (withShape)
            extra += $", p.\"{cols.ShapeCorrelation}\" AS shape";
        var mzIdx = withMz ? 7 : -1;
        var shapeIdx = withShape ? (withMz ? 8 : 7) : -1;

        var pool = new StringPool();

        // The transition id and the sample index are built by DuckDB rather than per row here. Both
        // were pure allocation: a fragment-ion string, two charge strings and a ~45-character sample id
        // per ROW, all of which the pool then deduplicated and threw away. Measured on a 15.5M-row
        // partition, moving them into the query took the read from 5.0 s / 4.5 GB allocated to
        // 3.3 s / 1.5 GB. (A fully allocation-free shape using dense_rank() to return ids and read the
        // strings only on change was also measured, and was SLOWER - 6.3 s - because the window
        // functions cost more than the strings they saved. Measure before assuming.)
        var tidSql = TransitionIdSql(dataset, cols);
        var sampleSql = SampleIndexSql(samples);

        foreach (var partition in dataset.Partitions)
        {
            // A connection PER PARTITION, deliberately - not one reused for the whole stage.
            // Running a second streaming command on a connection whose previous streaming reader has
            // been disposed corrupts the managed heap: it crashed ~2 runs in 3 with an
            // AccessViolationException, surfacing at whatever allocated next (String.Concat building
            // the very SQL below), always at the first partition boundary. Opening a connection is
            // milliseconds against the ~15 s a partition takes, and one command per connection is the
            // pattern every other reader here already used.
            using var conn = OpenBounded(dataset, memoryBudgetMb);
            using var cmd = DuckDbTuning.StreamingCommand(conn,
                "SELECT " +
                $"p.\"{cols.Peptide}\" AS pep, " +
                $"{tidSql} AS tid, " +
                $"{IsPrecursorSql(cols.Transition, "p")} AS isprec, " +
                $"TRY_CAST(p.\"{cols.PrecursorCharge}\" AS INTEGER) AS pz, " +
                "m.s_idx AS samp, " +
                $"p.\"{cols.Abundance}\" AS area, " +
                $"p.\"{cols.RetentionTime}\" AS rt" +
                extra + " " +
                $"FROM {Scan(partition)} p " +
                // INNER join: a row whose sample is not in the run's sample list is dropped here,
                // exactly as the old per-row dictionary lookup dropped it.
                $"JOIN {sampleSql} m ON m.s_name = p.\"{cols.Sample}\" " +
                $"ORDER BY p.\"{cols.Peptide}\"");
            using var reader = cmd.ExecuteReader();

            PeptideBlock? current = null;
            while (reader.Read())
            {
                var pep = reader.IsDBNull(0) ? string.Empty : reader.GetString(0);
                if (current is null || !string.Equals(current.Peptide, pep, StringComparison.Ordinal))
                {
                    if (current is not null)
                        yield return current;
                    current = new PeptideBlock { Peptide = pep };
                }

                current.TransitionId.Add(pool.Get(reader, 1));
                current.IsPrecursor.Add(!reader.IsDBNull(2) && reader.GetBoolean(2));
                current.PrecursorCharge.Add(reader.IsDBNull(3) ? 0 : reader.GetInt32(3));
                current.SampleIndex.Add(reader.GetInt32(4));
                current.Area.Add(ToDouble(reader.GetValue(5)));
                current.RetentionTime.Add(ToDouble(reader.GetValue(6)));
                if (withMz)
                    current.ProductMz.Add(ToDouble(reader.GetValue(mzIdx)));
                if (withShape)
                    current.ShapeCorrelation.Add(ToDouble(reader.GetValue(shapeIdx)));
            }

            // End of partition ends the peptide: nothing in a later partition can extend this block.
            if (current is not null)
                yield return current;
        }
    }

    /// <summary>
    /// SQL producing the transition id, <c>ion_z{precursor}_{product}</c>.
    /// <para>
    /// This string is written to <c>peptides_rollup_residuals.parquet</c>, so it is an output contract and has
    /// to render byte-identically to the C# concatenation it replaces. It does for the types that
    /// occur: charges are INTEGER in a Skyline export, and <c>CAST(2 AS VARCHAR)</c> is <c>"2"</c> just
    /// as <c>2.ToString(InvariantCulture)</c> is; a VARCHAR charge column (Skyline writes <c>#N/A</c>
    /// into numeric columns) casts to itself. A FLOATING-POINT charge column is the one that would not
    /// match - DuckDB renders <c>2.0</c> where .NET renders <c>2</c> - so <see cref="ChargePart"/>
    /// casts that case through BIGINT first. That is exact because charge states are whole numbers
    /// whatever type an export stores them in; it is not a general double-to-string equivalence.
    /// </para>
    /// </summary>
    private static string TransitionIdSql(MergedDataset dataset, SkylineColumns cols)
    {
        var ion = $"COALESCE(p.\"{cols.Transition}\", 'nan')";
        var pz = ChargePart(dataset, cols.PrecursorCharge);
        var zz = ChargePart(dataset, cols.ProductCharge);
        return $"({ion} || '_z' || {pz} || '_' || {zz})";
    }

    private static string ChargePart(MergedDataset dataset, string column) =>
        IsFloatingPoint(dataset, column)
            // A floating-point charge column still holds whole numbers - charge states are integers,
            // whatever type the export wrote them as. Going through BIGINT reproduces .NET's rendering
            // ("2"); casting the double straight to VARCHAR would give "2.0" and silently change every
            // transition id in peptides_rollup_residuals.parquet.
            ? $"COALESCE(CAST(CAST(p.\"{column}\" AS BIGINT) AS VARCHAR), 'nan')"
            : $"COALESCE(CAST(p.\"{column}\" AS VARCHAR), 'nan')";

    private static bool IsFloatingPoint(MergedDataset dataset, string column)
    {
        try
        {
            using var conn = new DuckDBConnection("Data Source=:memory:");
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText =
                $"SELECT column_type FROM (DESCRIBE SELECT * FROM {Scan(dataset.Partitions[0])}) "
                + $"WHERE column_name = '{column.Replace("'", "''")}'";
            var type = cmd.ExecuteScalar() as string ?? "";
            return type.Contains("DOUBLE", StringComparison.OrdinalIgnoreCase)
                || type.Contains("FLOAT", StringComparison.OrdinalIgnoreCase)
                || type.Contains("DECIMAL", StringComparison.OrdinalIgnoreCase);
        }
        catch (Exception)
        {
            // Unreadable schema. Assume not floating point, which is the overwhelmingly common case and
            // the same expression the pre-SQL code produced; if the dataset is genuinely broken, the
            // read below fails with a better message than a schema probe could give.
            return false;
        }
    }

    /// <summary>
    /// SQL mapping each sample name to its index in the run's sample list - the same list that becomes
    /// the output columns, so the index IS the output column. Emitted as a literal VALUES table
    /// (a few hundred to a few thousand rows) that DuckDB hash-joins once per partition, which is far
    /// cheaper than materializing the ~45-character sample id on every transition row.
    /// </summary>
    private static string SampleIndexSql(IReadOnlyList<string> samples)
    {
        // An empty list would emit "VALUES )" and fail as a SQL syntax error deep in the read, which is
        // a terrible way to learn that the merged data has no usable sample column. Say so here.
        if (samples.Count == 0)
            throw new InvalidOperationException(
                "No samples were found in the merged data, so the transition rollup has nothing to roll "
                + "up to. Check that the sample column was detected correctly (see the 'Columns:' line "
                + "in the run log) and that the input reports contain replicate names.");

        var sb = new StringBuilder("(SELECT * FROM (VALUES ");
        for (var i = 0; i < samples.Count; i++)
        {
            if (i > 0)
                sb.Append(", ");
            sb.Append('(').Append('\'').Append(samples[i].Replace("'", "''")).Append("', ").Append(i).Append(')');
        }
        return sb.Append(") AS t(s_name, s_idx))").ToString();
    }

    /// <summary>
    /// A <c>read_parquet</c> call for a dataset or partition target. <c>hive_partitioning=false</c>
    /// keeps the bucket column out of the result, so the schema PRISM sees is exactly the schema it saw
    /// when the merge wrote a single file.
    /// </summary>
    /// <summary>
    /// The SQL predicate for "this row is a precursor (MS1) row, not a fragment".
    ///
    /// <para>Shared because Stage 2 excludes these rows from the rollup (<c>ExcludePrecursor</c>,
    /// default on) and MS2 signal accounting has to exclude exactly the same set - a second copy of
    /// the rule would be free to drift, and the symptom would be an assigned fraction that is quietly
    /// wrong rather than an error. Skyline labels these <c>precursor</c>, <c>precursor [M+1]</c>,
    /// <c>precursor [M+2]</c>, so the test is a prefix.</para>
    /// </summary>
    /// <param name="transitionColumn">The fragment-ion column's name in this export.</param>
    /// <param name="alias">Table alias the column is qualified with, or null for none.</param>
    internal static string IsPrecursorSql(string transitionColumn, string? alias = null)
    {
        var qualified = string.IsNullOrEmpty(alias)
            ? $"\"{transitionColumn}\""
            : $"{alias}.\"{transitionColumn}\"";
        return $"starts_with({qualified}, 'precursor')";
    }

    internal static string Scan(string target) =>
        $"read_parquet('{Esc(target)}', hive_partitioning=false)";

    /// <summary>
    /// A connection whose buffer pool is bounded and which can spill, sharing the merge's budget
    /// (<c>processing.merge_memory_mb</c>, 0 = auto) and scratch directory. The DISTINCT and the
    /// per-partition ORDER BY are both blocking operators: bounded, they spill; unbounded, they take
    /// whatever the machine has.
    /// </summary>
    private static DuckDBConnection OpenBounded(MergedDataset dataset, int memoryBudgetMb)
    {
        // Note for anyone adding concurrency here: DuckDB.NET keys a static
        // Connection.ConnectionManager cache by connection string, so every "Data Source=:memory:" in
        // the process is refcounted references to ONE database instance - one buffer pool, one
        // memory_limit (a database-level setting, not a connection one). Two threads opening and
        // closing these independently can tear the instance down under each other. See
        // TransitionRollup.RunParallel.
        var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        DuckDbTuning.Apply(
            conn,
            memoryBudgetMb > 0 ? memoryBudgetMb : DuckDbMerge.AutoMemoryBudgetMb(),
            DuckDbMerge.ResolveTempDirectory(dataset.Root));
        return conn;
    }

    /// <summary>
    /// Hands back one shared instance per distinct string instead of a fresh allocation per row.
    /// <para>
    /// These three columns are drawn from tiny domains - the sample id is one of N replicates, the
    /// fragment ion one of a few hundred names, the charges single digits - but the reader sees them
    /// once per TRANSITION ROW, and a block spans every sample. Without pooling, one peptide's block on
    /// a 100-document cohort is ~60,000 rows x 4 freshly allocated strings; the sample id alone is a
    /// 45-character "<c>replicate__@__document</c>". With <c>dop*4</c> blocks in flight that is the
    /// difference between ~1 GB of live strings and a few MB. The pool is per-stream, so it dies with
    /// the enumeration rather than living as a static cache.
    /// </para>
    /// </summary>
    private sealed class StringPool
    {
        private readonly Dictionary<string, string> _byValue = new(StringComparer.Ordinal);

        public string Get(DuckDBDataReader reader, int ordinal)
            => reader.IsDBNull(ordinal) ? string.Empty : Intern(reader.GetString(ordinal));

        public string GetKey(object? value) => Intern(FormatKey(value));

        private string Intern(string value)
        {
            if (_byValue.TryGetValue(value, out var existing))
                return existing;
            _byValue[value] = value;
            return value;
        }
    }

    // Format a charge value for the transition-id key. Only distinctness matters for the
    // transition count, and (ion, prec, prod) -> string is injective, so plain invariant
    // formatting is sufficient.
    private static string FormatKey(object? v) => v switch
    {
        null => "nan",
        DBNull => "nan",
        long l => l.ToString(CultureInfo.InvariantCulture),
        int n => n.ToString(CultureInfo.InvariantCulture),
        short s => s.ToString(CultureInfo.InvariantCulture),
        double d => d.ToString(CultureInfo.InvariantCulture),
        _ => Convert.ToString(v, CultureInfo.InvariantCulture) ?? "nan",
    };

    // Skyline exports missing/non-detected values as tokens like "#N/A", so numeric columns
    // can arrive as text (DuckDB infers VARCHAR). Parse tolerantly: unparseable tokens become
    // NaN, which the rollup imputes exactly as a genuine missing measurement.
    private static double ToDouble(object? v) => v switch
    {
        null or DBNull => double.NaN,
        double d => d,
        float f => f,
        long l => l,
        int n => n,
        short s => s,
        decimal m => (double)m,
        string str => ParseNumber(str),
        _ => ParseNumber(Convert.ToString(v, CultureInfo.InvariantCulture)),
    };

    private static double ParseNumber(string? str) =>
        double.TryParse(str, NumberStyles.Any, CultureInfo.InvariantCulture, out var r) ? r : double.NaN;

    private static string Esc(string path) => path.Replace("'", "''");
}
