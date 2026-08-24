using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Rollup;

/// <summary>
/// The batch-corrected residual files: <c>corrected_peptides_residuals.parquet</c> and
/// <c>corrected_proteins_residuals.parquet</c>.
/// <para>
/// A residual is a deviation from a fitted profile, so ComBat's location terms cancel out of it and
/// only its per-batch SCALE applies: <c>e* = e / sqrt(deltaStar[batch, feature])</c>. These tests pin
/// the two properties that matter - the scaling is real when ComBat runs, and the file is a faithful
/// copy when it does not - rather than just asserting the files exist.
/// </para>
/// </summary>
public class CorrectedResidualTests
{
    private const string PepCol = "Peptide Modified Sequence Unimod Ids";

    private static string Run(string fixture, Action<PrismConfig>? tweak = null)
    {
        var mergeDir = Fixtures.Path2("mini", "merge");
        var config = PrismConfig.Load(Path.Combine(Fixtures.Path2("mini", fixture), "config.yaml"));
        tweak?.Invoke(config);
        var outDir = Path.Combine(Path.GetTempPath(), "prism_cres_" + Guid.NewGuid().ToString("N"));
        PrismPipeline.Run(
            new[] { Path.Combine(mergeDir, "mini_plate1.csv"), Path.Combine(mergeDir, "mini_plate2.csv") },
            outDir, config);
        return outDir;
    }

    /// <summary>Both corrected residual files accompany their raw counterparts, with the same shape.</summary>
    [Fact]
    public void CorrectedResiduals_AccompanyTheRawOnes()
    {
        var outDir = Run("e2e-medpolish");
        try
        {
            foreach (var (raw, corrected) in new[]
            {
                ("peptides_rollup_residuals.parquet", "corrected_peptides_residuals.parquet"),
                ("proteins_raw_residuals.parquet", "corrected_proteins_residuals.parquet"),
            })
            {
                var rawPath = Path.Combine(outDir, raw);
                var corPath = Path.Combine(outDir, corrected);
                Assert.True(File.Exists(rawPath), $"missing {raw}");
                Assert.True(File.Exists(corPath), $"missing {corrected}");

                var a = ParquetTable.Load(rawPath);
                var b = ParquetTable.Load(corPath);
                Assert.Equal(a.ColumnNames, b.ColumnNames);
                Assert.Equal(a.RowCount, b.RowCount);
            }
        }
        finally
        {
            if (Directory.Exists(outDir))
                Directory.Delete(outDir, recursive: true);
        }
    }

    /// <summary>
    /// With ComBat off there is nothing to scale by, so the corrected file must equal the raw one
    /// exactly - not approximately. This is what makes it safe for a script to read the corrected
    /// file unconditionally instead of branching on whether correction ran.
    /// </summary>
    [Fact]
    public void WithoutComBat_CorrectedResidualsEqualTheRawOnes()
    {
        var outDir = Run("e2e-medpolish");   // ComBat disabled in this fixture
        try
        {
            foreach (var (raw, corrected) in new[]
            {
                ("peptides_rollup_residuals.parquet", "corrected_peptides_residuals.parquet"),
                ("proteins_raw_residuals.parquet", "corrected_proteins_residuals.parquet"),
            })
            {
                AssertValuesEqual(
                    Path.Combine(outDir, raw), Path.Combine(outDir, corrected), expectEqual: true,
                    because: $"{corrected} should be a faithful copy of {raw} when ComBat does not run");
            }
        }
        finally
        {
            if (Directory.Exists(outDir))
                Directory.Delete(outDir, recursive: true);
        }
    }

    /// <summary>
    /// With ComBat ON the corrected residuals must actually differ from the raw ones - otherwise the
    /// scaling is silently a no-op and the file is misleadingly named. e2e-sum is the ComBat fixture.
    /// </summary>
    [Fact]
    public void WithComBat_ProteinResidualsAreScaled()
    {
        var outDir = Run("e2e-sum");
        try
        {
            var raw = Path.Combine(outDir, "proteins_raw_residuals.parquet");
            var cor = Path.Combine(outDir, "corrected_proteins_residuals.parquet");
            Assert.True(File.Exists(raw) && File.Exists(cor));
            AssertValuesEqual(raw, cor, expectEqual: false,
                because: "protein-level ComBat runs in e2e-sum, so the residuals must be rescaled");
        }
        finally
        {
            if (Directory.Exists(outDir))
                Directory.Delete(outDir, recursive: true);
        }
    }

    /// <summary>
    /// Scaling is per (batch, feature) and multiplicative, so it cannot change a residual's SIGN and
    /// cannot turn a finite value into NaN. Those are the two ways a mis-indexed delta would show up
    /// as plausible-looking numbers rather than an obvious failure.
    /// </summary>
    [Fact]
    public void Scaling_PreservesSignAndFiniteness()
    {
        var outDir = Run("e2e-sum");
        try
        {
            var a = ParquetTable.Load(Path.Combine(outDir, "proteins_raw_residuals.parquet"));
            var b = ParquetTable.Load(Path.Combine(outDir, "corrected_proteins_residuals.parquet"));
            var samples = a.ColumnNames.Where(c => c != "protein_group" && c != PepCol).ToList();
            Assert.NotEmpty(samples);

            var checked_ = 0;
            foreach (var col in samples)
            {
                var x = a.GetDouble(col);
                var y = b.GetDouble(col);
                for (var i = 0; i < a.RowCount; i++)
                {
                    var rawV = x[i] ?? double.NaN;
                    var corV = y[i] ?? double.NaN;
                    if (double.IsNaN(rawV))
                    {
                        Assert.True(double.IsNaN(corV), $"{col} row {i}: NaN became {corV}");
                        continue;
                    }
                    Assert.False(double.IsNaN(corV), $"{col} row {i}: {rawV} became NaN");
                    Assert.True(Math.Sign(rawV) == Math.Sign(corV),
                        $"{col} row {i}: sign flipped, {rawV} -> {corV}");
                    checked_++;
                }
            }
            Assert.True(checked_ > 0, "no finite residuals were compared");
        }
        finally
        {
            if (Directory.Exists(outDir))
                Directory.Delete(outDir, recursive: true);
        }
    }

    private static void AssertValuesEqual(string pathA, string pathB, bool expectEqual, string because)
    {
        var a = ParquetTable.Load(pathA);
        var b = ParquetTable.Load(pathB);
        var meta = new HashSet<string>(new[] { "protein_group", PepCol, "transition_id" }, StringComparer.Ordinal);
        var samples = a.ColumnNames.Where(c => !meta.Contains(c)).ToList();
        Assert.NotEmpty(samples);

        var anyDifferent = false;
        foreach (var col in samples)
        {
            var x = a.GetDouble(col);
            var y = b.GetDouble(col);
            for (var i = 0; i < a.RowCount; i++)
            {
                var xi = x[i] ?? double.NaN;
                var yi = y[i] ?? double.NaN;
                if (double.IsNaN(xi) && double.IsNaN(yi))
                    continue;
                if (BitConverter.DoubleToInt64Bits(xi) != BitConverter.DoubleToInt64Bits(yi))
                {
                    if (expectEqual)
                        Assert.Fail($"{because}: {col} row {i} is {xi} vs {yi}");
                    anyDifferent = true;
                }
            }
        }
        if (!expectEqual)
            Assert.True(anyDifferent, because);
    }
}
