using System;
using System.IO;
using System.Linq;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// Streams a small parquet through MergedParquetReader: peptide-block grouping, the tolerant
/// numeric parsing (VARCHAR area with a "#N/A" token -> NaN), and the optional product-m/z +
/// shape-correlation columns - the reader's low-branch parsing paths.
/// </summary>
public class MergedParquetReaderTests
{
    private static readonly SkylineColumns Cols = new()
    {
        Peptide = "pep", Sample = "samp", Abundance = "area", Transition = "ion",
        PrecursorCharge = "pz", ProductCharge = "zz", RetentionTime = "rt",
        ProductMz = "mz", ShapeCorrelation = "shape",
    };

    // area is VARCHAR (mixes a number with "#N/A"); pz/zz are integers; rt/mz/shape are doubles.
    private static string WriteParquet()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_mpr_" + Guid.NewGuid().ToString("N") + ".parquet");
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            "COPY (SELECT * FROM (VALUES "
            + "('PEPA', 'y1', 2, 1, 'S1', '100.5', 10.0, 200.1, 0.95), "
            + "('PEPA', 'y2', 2, 1, 'S1', '#N/A',  10.0, 300.2, 0.80), "
            + "('PEPB', 'y1', 2, 1, 'S2', '50.0',  12.0, 200.1, 0.90)) "
            + "AS t(pep, ion, pz, zz, samp, area, rt, mz, shape)) "
            + $"TO '{path.Replace("'", "''")}' (FORMAT PARQUET)";
        cmd.ExecuteNonQuery();
        return path;
    }

    [Fact]
    public void GetSortedSamples_ReturnsDistinctSorted()
    {
        var path = WriteParquet();
        try
        {
            Assert.Equal(new[] { "S1", "S2" }, MergedParquetReader.GetSortedSamples(MergedDataset.Open(path), "samp"));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void StreamPeptideBlocks_GroupsByPeptide_ParsesNaN_AndOptionalColumns()
    {
        var path = WriteParquet();
        try
        {
            var blocks = MergedParquetReader
                .StreamPeptideBlocks(MergedDataset.Open(path), Cols, includeProductMz: true, includeShapeCorr: true)
                .ToList();

            Assert.Equal(2, blocks.Count);
            var a = blocks.Single(b => b.Peptide == "PEPA");
            var b = blocks.Single(x => x.Peptide == "PEPB");

            Assert.Equal(2, a.Area.Count);
            Assert.Equal(100.5, a.Area[0], 9);
            Assert.True(double.IsNaN(a.Area[1]));      // "#N/A" -> NaN
            Assert.Equal(new[] { 200.1, 300.2 }, a.ProductMz);
            Assert.Equal(new[] { 0.95, 0.80 }, a.ShapeCorrelation);
            Assert.Single(b.Area);
            Assert.Equal(50.0, b.Area[0], 9);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void StreamPeptideBlocks_WithoutOptionalColumns_LeavesThemEmpty()
    {
        var path = WriteParquet();
        try
        {
            var a = MergedParquetReader.StreamPeptideBlocks(MergedDataset.Open(path), Cols).First(x => x.Peptide == "PEPA");
            Assert.Empty(a.ProductMz);
            Assert.Empty(a.ShapeCorrelation);
            Assert.Equal(2, a.Area.Count); // still reads the core columns
        }
        finally { File.Delete(path); }
    }
}
