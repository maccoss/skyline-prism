using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Tests.TestSupport;

/// <summary>
/// A per-column fingerprint of a pipeline output, taken over the EXACT bits of every value.
/// <para>
/// This is deliberately not the same thing as the cross-language parity tests. Those compare C#
/// against the committed <b>Python</b> goldens with a tolerance (1e-9 for the deterministic core,
/// looser for ComBat), because two independent implementations cannot be expected to agree to the
/// last bit. That tolerance is correct there, but it means a C# change that moves a quantity by
/// less than the tolerance passes silently. This digest closes that gap: it compares C# against
/// its own committed reference, so <b>any</b> change to <b>any</b> quantity fails.
/// </para>
/// <para>
/// Doubles are hashed by <see cref="BitConverter.DoubleToInt64Bits"/> rather than by their text
/// form, so NaN payloads, signed zero, and last-ulp differences are all caught - formatting would
/// round them away. Rows are ordered by the key column so the digest does not depend on output row
/// order, which is explicitly not a parity contract (see CLAUDE.md, "C# Stage 1 partitions").
/// </para>
/// </summary>
public static class QuantityDigest
{
    /// <summary>Outputs fingerprinted for every fixture, in a fixed order.</summary>
    public static readonly string[] Files =
    {
        "peptides_rollup.parquet",
        "peptides_log2_internal.parquet",
        "proteins_raw.parquet",
        "corrected_peptides.parquet",
        "corrected_proteins.parquet",
    };

    /// <summary>
    /// Fingerprint every column of every output in <paramref name="outputDir"/>, as
    /// "file\tcolumn\tsha256" lines sorted by file then column.
    /// </summary>
    public static List<string> Compute(string outputDir)
    {
        var lines = new List<string>();
        foreach (var file in Files)
        {
            var path = Path.Combine(outputDir, file);
            if (!File.Exists(path))
                continue;

            var table = ParquetTable.Load(path);
            var order = RowOrder(table);
            foreach (var col in table.ColumnNames.OrderBy(c => c, StringComparer.Ordinal))
                lines.Add($"{file}\t{col}\t{HashColumn(table.Column(col), order)}");
        }
        return lines;
    }

    /// <summary>
    /// Row indices ordered by the first column's value. The outputs are keyed tables (peptide or
    /// protein group), so that is a total order; ties fall back to the original index to stay
    /// deterministic if a key ever repeats.
    /// <para>
    /// Every comparison of two output tables must go through this. Row ORDER is explicitly not a
    /// parity contract - Stage 1 writes hash partitions and does not sort (CLAUDE.md, "C# Stage 1
    /// partitions, and does NOT sort") - so comparing by raw row index is a latent flake that
    /// happens to pass whenever two runs coincide.
    /// </para>
    /// </summary>
    public static int[] RowOrder(ParquetTable table)
    {
        var key = table.Column(table.ColumnNames[0]);
        return Enumerable.Range(0, table.RowCount)
            .OrderBy(i => Fixtures.FormatCell(key.GetValue(i)), StringComparer.Ordinal)
            .ThenBy(i => i)
            .ToArray();
    }

    private static string HashColumn(Array col, int[] order)
    {
        using var sha = SHA256.Create();
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            foreach (var i in order)
            {
                var v = col.GetValue(i);
                switch (v)
                {
                    case null:
                        w.Write((byte)0);
                        break;
                    case double d:
                        w.Write((byte)1);
                        w.Write(BitConverter.DoubleToInt64Bits(d));
                        break;
                    case float f:
                        w.Write((byte)2);
                        w.Write(BitConverter.SingleToInt32Bits(f));
                        break;
                    case bool b:
                        w.Write((byte)3);
                        w.Write(b);
                        break;
                    default:
                        w.Write((byte)4);
                        w.Write(Convert.ToString(v, CultureInfo.InvariantCulture) ?? string.Empty);
                        break;
                }
            }
        }
        return Convert.ToHexString(sha.ComputeHash(ms.ToArray())).ToLowerInvariant();
    }
}
