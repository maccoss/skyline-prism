using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Rollup;

/// <summary>
/// The residual outputs: which files appear, keyed by what, and for which methods.
/// <para>
/// These exist because the residual files were previously produced by no test at all - every
/// fixture set <c>include_residuals: false</c>, so renaming an output contract passed the entire
/// suite. Both files are now also fingerprinted by <see cref="Pipeline.QuantityRegressionTests"/>;
/// this covers the structure the digest cannot describe (presence, keys, row counts).
/// </para>
/// </summary>
public class ResidualOutputTests
{
    private const string PepCol = "Peptide Modified Sequence Unimod Ids";

    private static string RunFixture(string fixture)
    {
        var mergeDir = Fixtures.Path2("mini", "merge");
        var dir = Fixtures.Path2("mini", fixture);
        var config = PrismConfig.Load(Path.Combine(dir, "config.yaml"));
        var outDir = Path.Combine(Path.GetTempPath(), "prism_res_" + Guid.NewGuid().ToString("N"));
        PrismPipeline.Run(
            new[] { Path.Combine(mergeDir, "mini_plate1.csv"), Path.Combine(mergeDir, "mini_plate2.csv") },
            outDir, config);
        return outDir;
    }

    /// <summary>
    /// median_polish at both stages writes both residual files, each keyed by the pair that
    /// identifies its rows.
    /// </summary>
    [Fact]
    public void MedianPolish_WritesBothResidualFiles()
    {
        var outDir = RunFixture("e2e-medpolish");
        try
        {
            var tr = Path.Combine(outDir, "peptides_rollup_residuals.parquet");
            var pr = Path.Combine(outDir, "proteins_raw_residuals.parquet");
            Assert.True(File.Exists(tr), "transition-stage residuals missing");
            Assert.True(File.Exists(pr), "protein-stage residuals missing");

            // Rows are TRANSITIONS: keyed by (peptide, transition_id).
            var t = ParquetTable.Load(tr);
            Assert.Equal(PepCol, t.ColumnNames[0]);
            Assert.Equal("transition_id", t.ColumnNames[1]);
            Assert.True(t.RowCount > 0);

            // Rows are PEPTIDES: keyed by (protein_group, peptide).
            var p = ParquetTable.Load(pr);
            Assert.Equal("protein_group", p.ColumnNames[0]);
            Assert.Equal(PepCol, p.ColumnNames[1]);
            Assert.True(p.RowCount > 0);

            // Both carry one column per sample beyond their two key columns, and the two files
            // agree on the sample set - they describe the same cohort.
            Assert.Equal(t.ColumnNames.Skip(2), p.ColumnNames.Skip(2));
        }
        finally
        {
            if (Directory.Exists(outDir))
                Directory.Delete(outDir, recursive: true);
        }
    }

    /// <summary>
    /// Only median polish decomposes, so only median polish leaves residuals. e2e-maxlfq is sum at
    /// the transition stage and maxLFQ at the protein stage, so neither file should appear - an
    /// empty or all-NaN file would be worse than none, since it would read as "no interference".
    /// </summary>
    [Fact]
    public void NonPolishMethods_WriteNoResidualFiles()
    {
        var outDir = RunFixture("e2e-maxlfq");
        try
        {
            Assert.False(File.Exists(Path.Combine(outDir, "peptides_rollup_residuals.parquet")));
            Assert.False(File.Exists(Path.Combine(outDir, "proteins_raw_residuals.parquet")));
        }
        finally
        {
            if (Directory.Exists(outDir))
                Directory.Delete(outDir, recursive: true);
        }
    }

    /// <summary>
    /// A group is polished only when it has at least <c>min_peptides</c> peptides; the rest fall
    /// back to a linear sum and contribute no residual rows. So the protein residual file has one
    /// row per (group, peptide) over the polished groups only - never one per peptide overall.
    /// </summary>
    [Fact]
    public void ProteinResiduals_CoverOnlyPolishedGroups()
    {
        var outDir = RunFixture("e2e-medpolish");
        try
        {
            var pr = ParquetTable.Load(Path.Combine(outDir, "proteins_raw_residuals.parquet"));
            var proteins = ParquetTable.Load(Path.Combine(outDir, "proteins_raw.parquet"));
            var nPeptides = proteins.GetLong("n_peptides");
            var minPeptides = PrismConfig
                .Load(Path.Combine(Fixtures.Path2("mini", "e2e-medpolish"), "config.yaml"))
                .ProteinRollup.MinPeptides;

            var expected = nPeptides.Where(n => n >= minPeptides).Sum();
            Assert.Equal(expected, pr.RowCount);

            // Every residual row's group must be a real protein group.
            var groups = new System.Collections.Generic.HashSet<string>(
                proteins.GetString("protein_group").Select(s => s ?? ""), StringComparer.Ordinal);
            foreach (var g in pr.GetString("protein_group"))
                Assert.Contains(g ?? "", groups);
        }
        finally
        {
            if (Directory.Exists(outDir))
                Directory.Delete(outDir, recursive: true);
        }
    }
}
