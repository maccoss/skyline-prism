using System;
using System.IO;
using SkylinePrism.Core.Normalization;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Normalization;

/// <summary>
/// Phase A of the Stage 2b/2c streaming work: the per-sample factors computed by a column-at-a-time
/// pass over the parquet must reproduce the whole-matrix <see cref="Normalizer"/> <b>exactly</b> -
/// not approximately. Both see the same values in the same order, so any difference here is a real
/// change in the arithmetic, and the assertions below use exact equality to say so.
/// </summary>
public class NormalizationFactorsTests
{
    [Theory]
    [InlineData("median", 0.0, 0)]
    [InlineData("median", 0.08, 23)]   // missing values + dropped all-NaN rows
    [InlineData("vsn", 0.0, 0)]
    [InlineData("vsn", 0.08, 23)]
    [InlineData("none", 0.0, 0)]
    [InlineData("rt_lowess", 0.0, 0)]
    [InlineData("rt_lowess", 0.08, 23)]
    public void Factors_ReproduceInMemoryNormalizer(string method, double missingFraction, int allNanEvery)
    {
        var root = NewTempDir();
        try
        {
            var cohort = SyntheticCohort.Write(
                Path.Combine(root, "in"), missingFraction: missingFraction, allNanEvery: allNanEvery);
            var (matrix, rt, _) = cohort.LoadKeptMatrix();

            var expected = method switch
            {
                "vsn" => Normalizer.VsnNormalize(matrix),
                "none" => matrix,
                "rt_lowess" => Normalizer.RtLowessNormalize(matrix, rt),
                _ => Normalizer.MedianNormalize(matrix),
            };

            var factors = NormalizationFactors.Compute(
                cohort.InputPath, cohort.Samples, method, SyntheticCohort.RtColumn);
            Assert.NotNull(factors);

            var rows = matrix.GetLength(0);
            var cols = matrix.GetLength(1);
            var changed = 0;
            for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
            {
                var actual = factors!.Apply(j, matrix[i, j], rt[i]);
                Assert.Equal(expected[i, j], actual);
                if (!double.IsNaN(matrix[i, j]) && matrix[i, j] != actual)
                    changed++;
            }

            // "none" is the only method allowed to be a no-op; for the rest, a factor pass that
            // silently computed zeros would otherwise pass this test.
            if (method != "none")
                Assert.True(changed > rows * cols / 2,
                    $"{method} changed only {changed} of {rows * cols} cells - suspiciously close to a no-op.");
        }
        finally
        {
            Cleanup(root);
        }
    }

    /// <summary>Quantile needs the whole column at apply time, so it has no cell-wise factors.</summary>
    [Fact]
    public void Quantile_HasNoFactors()
    {
        var root = NewTempDir();
        try
        {
            var cohort = SyntheticCohort.Write(Path.Combine(root, "in"), nFeatures: 20);
            Assert.Null(NormalizationFactors.Compute(cohort.InputPath, cohort.Samples, "quantile"));
        }
        finally
        {
            Cleanup(root);
        }
    }

    /// <summary>
    /// rt_lowess without an RT column degrades to median, because that is what the in-memory path
    /// does: its rt guard falls through to the method switch, whose default is median.
    /// </summary>
    [Fact]
    public void RtLowess_WithoutRtColumn_FallsBackToMedian()
    {
        var root = NewTempDir();
        try
        {
            var cohort = SyntheticCohort.Write(Path.Combine(root, "in"), nFeatures: 40);
            var (matrix, _, _) = cohort.LoadKeptMatrix();
            var expected = Normalizer.MedianNormalize(matrix);

            var factors = NormalizationFactors.Compute(
                cohort.InputPath, cohort.Samples, "rt_lowess", rtColumn: "no_such_column");
            Assert.NotNull(factors);
            Assert.Equal("median", factors!.Method);

            for (var i = 0; i < matrix.GetLength(0); i++)
            for (var j = 0; j < matrix.GetLength(1); j++)
                Assert.Equal(expected[i, j], factors.Apply(j, matrix[i, j], double.NaN));
        }
        finally
        {
            Cleanup(root);
        }
    }

    private static string NewTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_nf_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void Cleanup(string dir)
    {
        if (Directory.Exists(dir))
            Directory.Delete(dir, recursive: true);
    }
}
