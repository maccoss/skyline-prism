using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.Rollup;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Re-rolling the Dynamic Range plot's proteins under a method the run did not use. The point of the
/// feature is that the method CHANGES the picture - median polish gives a typical peptide's level, sum
/// scales with peptide count, iBAQ divides by the theoretical count - so these tests pin the arithmetic
/// of each choice and the group membership it is applied over.
/// </summary>
public class DynamicRangeRollupTests
{
    /// <summary>
    /// A corrected_peptides-shaped parquet: LINEAR values, and protein_group / leading_* as the pipeline
    /// writes them - ';'-separated and index-aligned when a peptide is shared between groups.
    /// </summary>
    private static string WritePeptides(
        string dir, string[] peptides, string[] groups, string[] accessions, string[] genes,
        params double[][] replicates)
    {
        var path = Path.Combine(dir, "corrected_peptides.parquet");
        var meta = new List<ParquetWideWriter.MetaColumn>
        {
            ParquetWideWriter.Strings("Peptide Modified Sequence", peptides),
            ParquetWideWriter.Strings("protein_group", groups),
            ParquetWideWriter.Strings("leading_protein", accessions),
            ParquetWideWriter.Strings("leading_name", accessions),
            ParquetWideWriter.Strings("leading_gene_name", genes),
        };
        var samples = Enumerable.Range(1, replicates.Length).Select(i => "R" + i).ToList();
        ParquetWideWriter.Write(path, meta, samples, replicates, peptides.Length);
        return path;
    }

    private static string TempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism-range-rollup-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>
    /// Two proteins: one with more, dimmer peptides and one with fewer, brighter ones. This is exactly
    /// the case the drop-down exists for - the gap between them is 2.25x under sum and 3x under median
    /// polish, because a summing method carries the peptide count into the answer and a polish does not.
    /// Neither is wrong; they are different quantities, and the plot cannot say which one it is showing
    /// unless the user can switch.
    /// </summary>
    [Fact]
    public void SumAndMedianPolishRankTheSameDataDifferently()
    {
        var dir = TempDir();
        try
        {
            // MANY: four peptides at 1,000 each. BRIGHT: three at 3,000 each.
            var path = WritePeptides(
                dir,
                peptides: new[] { "AAAK", "BBBK", "CCCK", "DDDK", "EEEK", "FFFK", "GGGK" },
                groups: new[] { "PG1", "PG1", "PG1", "PG1", "PG2", "PG2", "PG2" },
                accessions: new[] { "P1", "P1", "P1", "P1", "P2", "P2", "P2" },
                genes: new[] { "MANY", "MANY", "MANY", "MANY", "FEW", "FEW", "FEW" },
                new[] { 1000.0, 1000, 1000, 1000, 3000, 3000, 3000 },
                new[] { 1000.0, 1000, 1000, 1000, 3000, 3000, 3000 });
            var table = ParquetTable.Load(path);

            var summed = DynamicRangeRollup.Recompute(
                table, new DynamicRangeRollupOptions { Method = ProteinRollupMethod.Sum });
            var polished = DynamicRangeRollup.Recompute(
                table, new DynamicRangeRollupOptions { Method = ProteinRollupMethod.MedianPolish });

            // Sum totals the peptides: 4 x 1,000 against 3 x 3,000.
            Assert.Equal(4000.0, summed.Single(e => e.Label == "MANY").MeanAbundance, 3);
            Assert.Equal(9000.0, summed.Single(e => e.Label == "FEW").MeanAbundance, 3);

            // Median polish estimates the level of a typical peptide, so it does not multiply by count:
            // 1,000 against 3,000.
            Assert.Equal(1000.0, polished.Single(e => e.Label == "MANY").MeanAbundance, 3);
            Assert.Equal(3000.0, polished.Single(e => e.Label == "FEW").MeanAbundance, 3);

            // ...and the ratio between the two proteins is 2.25x under sum, 3x under median polish. That
            // difference is the whole reason for letting the user switch.
            Assert.Equal(
                2.25,
                summed.Single(e => e.Label == "FEW").MeanAbundance
                    / summed.Single(e => e.Label == "MANY").MeanAbundance,
                3);
            Assert.Equal(
                3.0,
                polished.Single(e => e.Label == "FEW").MeanAbundance
                    / polished.Single(e => e.Label == "MANY").MeanAbundance,
                3);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// iBAQ divides by the theoretical peptide count, which is what makes it the one method meant for
    /// comparing one protein against another. Without a FASTA it divides by the OBSERVED count instead -
    /// supported, but a different quantity, which is why the UI says so rather than presenting them as
    /// the same view.
    /// </summary>
    [Fact]
    public void IbaqDividesByTheTheoreticalCount_AndFallsBackToObservedWithout()
    {
        var dir = TempDir();
        try
        {
            var path = WritePeptides(
                dir,
                peptides: new[] { "AAAK", "BBBK", "CCCK" },
                groups: new[] { "PG1", "PG1", "PG1" },
                accessions: new[] { "P1", "P1", "P1" },
                genes: new[] { "AAA", "AAA", "AAA" },
                new[] { 1000.0, 1000, 1000 });
            var table = ParquetTable.Load(path);

            var withFasta = DynamicRangeRollup.Recompute(table, new DynamicRangeRollupOptions
            {
                Method = ProteinRollupMethod.Ibaq,
                TheoreticalCounts = new Dictionary<string, int> { ["P1"] = 30 },
            });
            var without = DynamicRangeRollup.Recompute(
                table, new DynamicRangeRollupOptions { Method = ProteinRollupMethod.Ibaq });

            Assert.Equal(3000.0 / 30, withFasta.Single().MeanAbundance, 3);
            Assert.Equal(3000.0 / 3, without.Single().MeanAbundance, 3);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// A shared peptide counts toward EVERY group it names - the pipeline's default all_groups handling.
    /// The ';'-separated identity columns are index-aligned with protein_group, so group 2's accession
    /// and gene come from position 2, not from the first entry.
    /// </summary>
    [Fact]
    public void ASharedPeptideCountsTowardEveryGroupItNames()
    {
        var dir = TempDir();
        try
        {
            var path = WritePeptides(
                dir,
                peptides: new[] { "SHAREDK", "ONLY1K", "ONLY2K" },
                groups: new[] { "PG1;PG2", "PG1", "PG2" },
                accessions: new[] { "P1;P2", "P1", "P2" },
                genes: new[] { "AAA;BBB", "AAA", "BBB" },
                new[] { 500.0, 100, 100 });
            var entries = DynamicRangeRollup.Recompute(
                ParquetTable.Load(path),
                new DynamicRangeRollupOptions { Method = ProteinRollupMethod.Sum });

            Assert.Equal(2, entries.Count);
            // Both groups are the shared peptide plus their own: 500 + 100.
            Assert.All(entries, e => Assert.Equal(600.0, e.MeanAbundance, 3));
            Assert.Equal(new[] { "AAA", "BBB" }, entries.Select(e => e.Label).OrderBy(l => l));
            Assert.Equal(
                new[] { "P1", "P2" },
                entries.Select(e => e.Accession!).OrderBy(a => a));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// The replicate picker applies to a recomputed view too - the rollup runs over the chosen columns
    /// only, so the ranking follows the selection exactly as it does for the run's own matrix.
    /// </summary>
    [Fact]
    public void ARecomputedViewHonoursTheReplicateSelection()
    {
        var dir = TempDir();
        try
        {
            var path = WritePeptides(
                dir,
                peptides: new[] { "AAAK", "BBBK" },
                groups: new[] { "PG1", "PG2" },
                accessions: new[] { "P1", "P2" },
                genes: new[] { "AAA", "BBB" },
                new[] { 100.0, 1000 },      // R1
                new[] { 10000.0, 1000 });   // R2
            var table = ParquetTable.Load(path);
            var options = new DynamicRangeRollupOptions { Method = ProteinRollupMethod.Sum };

            Assert.Equal("AAA", DynamicRangeRollup.Recompute(table, options)[0].Label);
            Assert.Equal("BBB", DynamicRangeRollup.Recompute(table, options, new[] { "R1" })[0].Label);
            Assert.Equal(1, DynamicRangeRollup.Recompute(table, options, new[] { "R1" })[0].SamplesUsed);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// Missing cells stay missing. A peptide's non-positive or NaN value is an unmeasured cell, and
    /// log2 of it must not become a floor value that the sum then counts as real signal.
    /// </summary>
    [Fact]
    public void MissingPeptideValuesAreSkippedRatherThanFlooredToZero()
    {
        var dir = TempDir();
        try
        {
            var path = WritePeptides(
                dir,
                peptides: new[] { "AAAK", "BBBK", "CCCK" },
                groups: new[] { "PG1", "PG1", "PG1" },
                accessions: new[] { "P1", "P1", "P1" },
                genes: new[] { "AAA", "AAA", "AAA" },
                new[] { 1000.0, double.NaN, 0.0 });
            var entry = DynamicRangeRollup.Recompute(
                ParquetTable.Load(path),
                new DynamicRangeRollupOptions { Method = ProteinRollupMethod.Sum }).Single();

            Assert.Equal(1000.0, entry.MeanAbundance, 3);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// A peptide matrix with no protein_group column cannot be rolled up, and says so rather than
    /// silently plotting nothing - a run with output.peptides off, or one from before those columns were
    /// stamped on.
    /// </summary>
    [Fact]
    public void APeptideMatrixWithoutGroupsCannotBeRecomputed()
    {
        var dir = TempDir();
        try
        {
            var path = Path.Combine(dir, "corrected_peptides.parquet");
            ParquetWideWriter.Write(
                path,
                new List<ParquetWideWriter.MetaColumn>
                {
                    ParquetWideWriter.Strings("Peptide Modified Sequence", new[] { "AAAK" }),
                },
                new List<string> { "R1" },
                new List<double[]> { new[] { 1000.0 } },
                1);
            var table = ParquetTable.Load(path);

            Assert.False(DynamicRangeRollup.CanRecompute(table));
            Assert.Empty(DynamicRangeRollup.Recompute(
                table, new DynamicRangeRollupOptions { Method = ProteinRollupMethod.Sum }));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// Progress ends at 1.0 whatever the group count. The bar is there so a several-second rollup does
    /// not read as a hang; one that never reaches full does the opposite.
    /// </summary>
    [Fact]
    public void ProgressReachesOne()
    {
        var dir = TempDir();
        try
        {
            var path = WritePeptides(
                dir,
                peptides: new[] { "AAAK", "BBBK" },
                groups: new[] { "PG1", "PG2" },
                accessions: new[] { "P1", "P2" },
                genes: new[] { "AAA", "BBB" },
                new[] { 100.0, 1000 });

            var reported = new List<double>();
            DynamicRangeRollup.Recompute(
                ParquetTable.Load(path),
                new DynamicRangeRollupOptions { Method = ProteinRollupMethod.Sum },
                sampleColumns: null,
                progress: new SynchronousProgress(reported.Add));

            Assert.NotEmpty(reported);
            Assert.Equal(1.0, reported[^1], 6);
            Assert.All(reported, v => Assert.InRange(v, 0.0, 1.0));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// <see cref="Progress{T}"/> posts to a synchronization context, which a test has none of - the
    /// callbacks would land on the thread pool after the assertions. This one runs them inline.
    /// </summary>
    private sealed class SynchronousProgress : IProgress<double>
    {
        private readonly Action<double> _report;
        private readonly object _lock = new();

        public SynchronousProgress(Action<double> report) => _report = report;

        public void Report(double value)
        {
            lock (_lock)
                _report(value);
        }
    }

    /// <summary>
    /// The accessions iBAQ needs counts for - so the digest reads only the proteins on the plot rather
    /// than the whole FASTA's worth.
    /// </summary>
    [Fact]
    public void LeadingProteinsListsWhatAnIbaqDigestWouldNeed()
    {
        var dir = TempDir();
        try
        {
            var path = WritePeptides(
                dir,
                peptides: new[] { "SHAREDK", "ONLY1K" },
                groups: new[] { "PG1;PG2", "PG1" },
                accessions: new[] { "P1;P2", "P1" },
                genes: new[] { "AAA;BBB", "AAA" },
                new[] { 100.0, 200.0 });

            Assert.Equal(
                new[] { "P1", "P2" },
                DynamicRangeRollup.LeadingProteins(ParquetTable.Load(path)).OrderBy(a => a));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
