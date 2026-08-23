using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// Skyline exports the PRISM report as either CSV or parquet, and the tool prefers parquet
/// (typed, and far faster than paginating a CSV). Both must give the same science.
/// <para>
/// This is not a formality. The two exports differ in more than encoding:
/// </para>
/// <list type="bullet">
/// <item><b>Column names differ.</b> CSV writes spaced English ("Protein Accession",
/// "Peptide Modified Sequence Unimod Ids"); parquet writes PascalCase with no spaces
/// ("ProteinAccession", "PeptideModifiedSequenceUnimodIds"), and renames Total Ion Current Area to
/// TicArea. <see cref="SkylineColumns.FindColumn"/> normalizes case, spaces and underscores, which
/// covers all of these except TicArea - and nothing in Core reads TicArea.</item>
/// <item><b>Physical types differ.</b> Skyline's parquet writes Area as Double and the charges as
/// Int32. On a CSV, DuckDB infers types from the data, so a slice whose areas are all whole
/// numbers - like these fixtures - comes back as Int64. The rollup therefore sees integer columns
/// on the CSV path and floating-point ones on the parquet path.</item>
/// </list>
/// <para>
/// Before this test the parquet branches of <see cref="DuckDbMerge"/> (read_parquet, and DESCRIBE
/// for the header) had no coverage at all: every fixture and every test fed CSV, while production
/// feeds parquet.
/// </para>
/// </summary>
public class ExportFormatParityTests
{
    /// <summary>
    /// The same cohort, exported both ways, must produce bit-identical quantities - not merely
    /// close ones, since this is the same code reading the same numbers.
    /// </summary>
    [Theory]
    [InlineData("e2e-medpolish")]
    [InlineData("e2e-sum")]
    public void ParquetAndCsvExports_ProduceIdenticalQuantities(string fixture)
    {
        var mergeDir = Fixtures.Path2("mini", "merge");
        var config = PrismConfig.Load(Path.Combine(Fixtures.Path2("mini", fixture), "config.yaml"));

        var csvOut = Path.Combine(Path.GetTempPath(), "prism_fc_" + Guid.NewGuid().ToString("N"));
        var pqOut = Path.Combine(Path.GetTempPath(), "prism_fp_" + Guid.NewGuid().ToString("N"));
        try
        {
            PrismPipeline.Run(
                new[] { Path.Combine(mergeDir, "mini_plate1.csv"), Path.Combine(mergeDir, "mini_plate2.csv") },
                csvOut, config);
            PrismPipeline.Run(
                new[] { Path.Combine(mergeDir, "mini_plate1.parquet"), Path.Combine(mergeDir, "mini_plate2.parquet") },
                pqOut, config);

            var compared = 0;
            foreach (var file in QuantityDigest.Files)
            {
                var a = Path.Combine(csvOut, file);
                var b = Path.Combine(pqOut, file);
                if (!File.Exists(a) || !File.Exists(b))
                    continue;
                AssertBitIdentical(file, ParquetTable.Load(a), ParquetTable.Load(b));
                compared++;
            }
            Assert.True(compared >= 3, $"Only {compared} outputs compared; expected the pipeline to write more.");
        }
        finally
        {
            foreach (var d in new[] { csvOut, pqOut })
                if (Directory.Exists(d))
                    Directory.Delete(d, recursive: true);
        }
    }

    /// <summary>
    /// The batch label is the input file's stem, and both fixtures share the stem
    /// (<c>mini_plate1</c>), so sample columns - and therefore every downstream grouping - must
    /// come out identically named regardless of which extension was read.
    /// </summary>
    [Fact]
    public void ParquetExport_YieldsTheSameSampleColumns()
    {
        var mergeDir = Fixtures.Path2("mini", "merge");
        var config = PrismConfig.Load(Path.Combine(Fixtures.Path2("mini", "e2e-medpolish"), "config.yaml"));

        var csvOut = Path.Combine(Path.GetTempPath(), "prism_sc_" + Guid.NewGuid().ToString("N"));
        var pqOut = Path.Combine(Path.GetTempPath(), "prism_sp_" + Guid.NewGuid().ToString("N"));
        try
        {
            PrismPipeline.Run(
                new[] { Path.Combine(mergeDir, "mini_plate1.csv"), Path.Combine(mergeDir, "mini_plate2.csv") },
                csvOut, config);
            PrismPipeline.Run(
                new[] { Path.Combine(mergeDir, "mini_plate1.parquet"), Path.Combine(mergeDir, "mini_plate2.parquet") },
                pqOut, config);

            var a = ParquetTable.Load(Path.Combine(csvOut, "peptides_rollup.parquet"));
            var b = ParquetTable.Load(Path.Combine(pqOut, "peptides_rollup.parquet"));

            // Skip index 0: the key column's NAME is inherited from the input schema, so it is
            // "Peptide Modified Sequence Unimod Ids" on the CSV path and
            // "PeptideModifiedSequenceUnimodIds" on the parquet path. Its VALUES are compared by
            // ParquetAndCsvExports_ProduceIdenticalQuantities.
            Assert.Equal(a.ColumnNames.Skip(1), b.ColumnNames.Skip(1));
        }
        finally
        {
            foreach (var d in new[] { csvOut, pqOut })
                if (Directory.Exists(d))
                    Directory.Delete(d, recursive: true);
        }
    }

    /// <summary>
    /// Compare two tables column by column, by POSITION. Values must match bit-for-bit; only the
    /// key column's name is allowed to differ (see above).
    /// </summary>
    private static void AssertBitIdentical(string file, ParquetTable a, ParquetTable b)
    {
        Assert.Equal(a.ColumnNames.Count, b.ColumnNames.Count);
        Assert.Equal(a.RowCount, b.RowCount);

        for (var c = 0; c < a.ColumnNames.Count; c++)
        {
            var colA = a.Column(a.ColumnNames[c]);
            var colB = b.Column(b.ColumnNames[c]);
            for (var r = 0; r < a.RowCount; r++)
            {
                var x = colA.GetValue(r);
                var y = colB.GetValue(r);
                Assert.True(BitEqual(x, y),
                    $"{file} column {c} ('{a.ColumnNames[c]}' / '{b.ColumnNames[c]}') row {r}: "
                    + $"csv={Fixtures.FormatCell(x)} parquet={Fixtures.FormatCell(y)}");
            }
        }
    }

    /// <summary>Bit equality, so NaN matches NaN and +0.0 does not match -0.0.</summary>
    private static bool BitEqual(object? x, object? y)
    {
        if (x is null || y is null)
            return x is null && y is null;
        if (x is double dx && y is double dy)
            return BitConverter.DoubleToInt64Bits(dx) == BitConverter.DoubleToInt64Bits(dy);
        if (x is float fx && y is float fy)
            return BitConverter.SingleToInt32Bits(fx) == BitConverter.SingleToInt32Bits(fy);
        return Equals(x, y);
    }
}
