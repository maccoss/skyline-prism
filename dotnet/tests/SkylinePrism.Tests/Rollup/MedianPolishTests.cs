using SkylinePrism.Core.Rollup;
using Xunit;

namespace SkylinePrism.Tests.Rollup;

/// <summary>
/// Mirrors tests/test_rollup.py::TestTukeyMedianPolish. The reference values come from
/// R's stats::medpolish on a fixed 5x3 matrix, so this is a cross-language golden test.
/// </summary>
public class MedianPolishTests
{
    // R:
    //   x <- matrix(c(15.77,16.84,17.16, 13.46,14.92,15.36, 15.20,15.43,17.09,
    //                 14.63,16.36,16.41, 13.74,14.62,14.98), nrow=5, byrow=TRUE)
    //   medpolish(x, eps=1e-9, maxiter=50, trace=FALSE)
    private static double[,] ReferenceMatrix() => new double[,]
    {
        { 15.77, 16.84, 17.16 },
        { 13.46, 14.92, 15.36 },
        { 15.20, 15.43, 17.09 },
        { 14.63, 16.36, 16.41 },
        { 13.74, 14.62, 14.98 },
    };

    [Fact]
    public void MatchesRMedpolish_ColEffectsAndOverall()
    {
        var result = TukeyMedianPolish.Run(ReferenceMatrix(), maxIter: 50);

        // R centered col effects [-1.069, 0, 0.361] + overall 16.054.
        double[] expectedColTotal = { 14.985, 16.054, 16.415 };
        Assert.Equal(3, result.ColEffects.Length);
        for (var j = 0; j < 3; j++)
        {
            // Absolute tolerance atol=0.02, matching the Python np.testing.assert_allclose.
            Assert.True(System.Math.Abs(expectedColTotal[j] - result.ColEffects[j]) < 0.02,
                $"col_effect[{j}] {result.ColEffects[j]} not within 0.02 of {expectedColTotal[j]}");
        }

        Assert.True(System.Math.Abs(result.Overall - 16.054) < 0.02,
            $"overall {result.Overall} not within 0.02 of 16.054");
    }

    [Fact]
    public void AdditiveDecomposition_Reconstructs()
    {
        var matrix = ReferenceMatrix();
        var result = TukeyMedianPolish.Run(matrix, maxIter: 50);

        // matrix[i,j] == row_effects[i] + col_effects[j] + residuals[i,j]
        // (col_effects already includes overall).
        for (var i = 0; i < 5; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var recon = result.RowEffects[i] + result.ColEffects[j] + result.Residuals[i, j];
                Assert.Equal(matrix[i, j], recon, 9);
            }
        }
    }

    [Fact]
    public void RowEffects_CenteredToMedianZero()
    {
        var result = TukeyMedianPolish.Run(ReferenceMatrix(), maxIter: 50);
        var median = SkylinePrism.Core.Numerics.Stats.NanMedian(result.RowEffects);
        Assert.Equal(0.0, median, 9);
    }

    [Fact]
    public void SingleRow_ColEffectsMatchInput()
    {
        // With one peptide, col effects should equal the input row (row effect absorbs nothing).
        var matrix = new double[,] { { 20.5, 21.0, 19.75 } };
        var result = TukeyMedianPolish.Run(matrix, maxIter: 50);
        Assert.Equal(20.5, result.ColEffects[0], 9);
        Assert.Equal(21.0, result.ColEffects[1], 9);
        Assert.Equal(19.75, result.ColEffects[2], 9);
    }
}
