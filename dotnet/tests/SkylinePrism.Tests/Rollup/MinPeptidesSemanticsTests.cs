using System;
using SkylinePrism.Core.Rollup;
using Xunit;

namespace SkylinePrism.Tests.Rollup;

/// <summary>
/// Pins down what <c>protein_rollup.min_peptides</c> actually does, because the name reads like a filter
/// and is not one.
///
/// <para>It is a <b>method-switch threshold</b>: a group with fewer than this many quantified peptides is
/// aggregated by a simple SUM instead of the configured method (median polish is not meaningful on one or
/// two peptides), and is flagged <c>low_confidence</c> in the output. The protein is still reported - no
/// protein is ever dropped for having too few peptides. The tool's Settings tab wording depends on this,
/// so a change here should change that wording too.</para>
/// </summary>
public class MinPeptidesSemanticsTests
{
    /// <summary>[peptides, samples] on the log2 scale.</summary>
    private static double[,] Matrix(params double[][] rows)
    {
        var m = new double[rows.Length, rows[0].Length];
        for (var i = 0; i < rows.Length; i++)
            for (var j = 0; j < rows[i].Length; j++)
                m[i, j] = rows[i][j];
        return m;
    }

    private static double[] Log2SumOfRows(double[,] m)
    {
        var nPep = m.GetLength(0);
        var nCols = m.GetLength(1);
        var result = new double[nCols];
        for (var j = 0; j < nCols; j++)
        {
            var linear = 0.0;
            for (var i = 0; i < nPep; i++)
                linear += Math.Pow(2, m[i, j]);
            result[j] = Math.Log2(Math.Max(linear, 1));
        }
        return result;
    }

    [Fact]
    public void BelowTheThreshold_FallsBackToSum_RatherThanDroppingTheProtein()
    {
        // Two peptides, threshold of 3: median polish is skipped in favour of a linear sum.
        var m = Matrix(new[] { 10.0, 11.0 }, new[] { 12.0, 13.0 });

        var result = ProteinMatrixRollup.Aggregate(m, ProteinRollupMethod.MedianPolish, minPeptides: 3);

        Assert.Equal(2, result.Length);                        // a value per sample - NOT dropped
        Assert.All(result, v => Assert.False(double.IsNaN(v))); // and not blanked out
        var expected = Log2SumOfRows(m);
        Assert.Equal(expected[0], result[0], 9);
        Assert.Equal(expected[1], result[1], 9);
    }

    [Fact]
    public void AtOrAboveTheThreshold_UsesTheConfiguredMethod()
    {
        var m = Matrix(new[] { 10.0, 11.0 }, new[] { 12.0, 13.0 }, new[] { 14.0, 15.0 });

        var polished = ProteinMatrixRollup.Aggregate(m, ProteinRollupMethod.MedianPolish, minPeptides: 3);
        var summed = ProteinMatrixRollup.Aggregate(m, ProteinRollupMethod.Sum, minPeptides: 3);

        // With enough peptides the method actually matters - median polish is not the sum.
        Assert.NotEqual(summed[0], polished[0], 6);
    }

    [Fact]
    public void TheThresholdIsInclusive_ExactlyNPeptidesUsesTheMethod()
    {
        var m = Matrix(new[] { 10.0, 11.0 }, new[] { 12.0, 13.0 }, new[] { 14.0, 15.0 });

        var atThreshold = ProteinMatrixRollup.Aggregate(m, ProteinRollupMethod.MedianPolish, minPeptides: 3);
        var forcedSum = ProteinMatrixRollup.Aggregate(m, ProteinRollupMethod.MedianPolish, minPeptides: 4);

        // n == minPeptides takes the method branch; n < minPeptides takes the sum branch.
        Assert.NotEqual(forcedSum[0], atThreshold[0], 6);
        Assert.Equal(Log2SumOfRows(m)[0], forcedSum[0], 9);
    }

    [Fact]
    public void ASinglePeptideProteinIsStillQuantified()
    {
        var m = Matrix(new[] { 20.0, 21.0 });

        var result = ProteinMatrixRollup.Aggregate(m, ProteinRollupMethod.MedianPolish, minPeptides: 3);

        // The most common "surely this gets filtered out" case: it does not.
        Assert.Equal(20.0, result[0], 6);
        Assert.Equal(21.0, result[1], 6);
    }

    [Theory]
    [InlineData(ProteinRollupMethod.MedianPolish)]
    [InlineData(ProteinRollupMethod.TopN)]
    [InlineData(ProteinRollupMethod.MaxLfq)]
    public void TheFallbackAppliesWhateverMethodWasConfigured(ProteinRollupMethod method)
    {
        var m = Matrix(new[] { 10.0, 11.0 });

        var result = ProteinMatrixRollup.Aggregate(m, method, minPeptides: 5);

        Assert.Equal(Log2SumOfRows(m), result, new DoubleComparer(1e-9));
    }

    private sealed class DoubleComparer : System.Collections.Generic.IEqualityComparer<double>
    {
        private readonly double _tol;
        public DoubleComparer(double tol) => _tol = tol;
        public bool Equals(double a, double b) => Math.Abs(a - b) <= _tol;
        public int GetHashCode(double v) => 0;
    }
}
