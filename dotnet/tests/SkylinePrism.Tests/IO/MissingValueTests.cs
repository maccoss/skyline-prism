using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Rollup;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// Real Skyline exports write "#N/A" for missing measurements, which DuckDB infers as text.
/// The reader must treat such tokens as NaN (imputed downstream) rather than throwing.
/// Regression test for the crash on the SEA-AD MTG dataset.
/// </summary>
public class MissingValueTests
{
    [Fact]
    public void Rollup_TreatsNaTokensAsMissing_DoesNotThrow()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_na_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var csv = Path.Combine(dir, "mini_missing.csv");
            File.WriteAllText(csv,
                "Protein,Protein Accession,Protein Gene,Peptide Modified Sequence Unimod Ids,Fragment Ion,Precursor Charge,Product Charge,Product Mz,Area,Retention Time,Replicate Name\n" +
                "sp|P1|P1_HUMAN,P1,GeneA,PEPTIDER,y3,2,1,400.1,1000,10.5,Sample1\n" +
                "sp|P1|P1_HUMAN,P1,GeneA,PEPTIDER,y4,2,1,500.1,#N/A,10.6,Sample1\n");

            var merged = Path.Combine(dir, "merged.parquet");
            DuckDbMerge.Merge(new[] { csv }, merged);

            var cols = SkylineColumns.Detect(Fixtures.LoadMerged(merged).ColumnNames.ToHashSet());
            var cfg = new TransitionRollupConfig
            {
                Method = TransitionRollupMethod.Sum,
                MinTransitions = 1,
                UseMs1 = false,
            };
            var outPath = Path.Combine(dir, "peptides_rollup.parquet");
            var result = TransitionRollup.Run(MergedDataset.Open(merged), cols, cfg, outPath);

            Assert.Equal(1, result.NPeptides);
            var table = ParquetTable.Load(outPath);
            var sampleCol = table.ColumnNames.First(c => c.Contains("__@__"));
            var val = table.GetDouble(sampleCol)[0];
            Assert.True(val.HasValue && !double.IsNaN(val.Value) && double.IsFinite(val.Value),
                $"expected a finite imputed abundance, got {val}");
            // 2 transitions: 1000 measured + missing imputed to 500 -> sum 1500 -> log2 ~10.55.
            Assert.Equal(Math.Log2(1500.0), val!.Value, 2);
            Assert.Equal(2, table.GetLong("n_transitions")[0]);
        }
        finally
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
    }
}
