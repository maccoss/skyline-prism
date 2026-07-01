using SkylinePrism.Core.Numerics;
using Xunit;

namespace SkylinePrism.Tests.Numerics;

/// <summary>
/// Locks the linear-algebra wrappers ComBat relies on (np.linalg.solve, matrix_rank).
/// </summary>
public class LinAlgTests
{
    [Fact]
    public void Solve_2x2_MatchesHandComputed()
    {
        // 3x + 2y = 5 ; x + 2y = 5  ->  x = 0, y = 2.5
        var a = new double[,] { { 3.0, 2.0 }, { 1.0, 2.0 } };
        var b = new[] { 5.0, 5.0 };
        var x = LinAlg.Solve(a, b);
        Assert.Equal(0.0, x[0], 12);
        Assert.Equal(2.5, x[1], 12);
    }

    [Fact]
    public void Solve_MultipleRhs()
    {
        var a = new double[,] { { 2.0, 0.0 }, { 0.0, 4.0 } };
        var b = new double[,] { { 2.0, 6.0 }, { 8.0, 4.0 } };
        var x = LinAlg.Solve(a, b);
        // Row-wise: [2/2, 6/2] and [8/4, 4/4]
        Assert.Equal(1.0, x[0, 0], 12);
        Assert.Equal(3.0, x[0, 1], 12);
        Assert.Equal(2.0, x[1, 0], 12);
        Assert.Equal(1.0, x[1, 1], 12);
    }

    [Fact]
    public void MatrixRank_FullRank()
    {
        var a = new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        Assert.Equal(2, LinAlg.MatrixRank(a));
    }

    [Fact]
    public void MatrixRank_Singular()
    {
        // Rows are linearly dependent -> rank 1
        var a = new double[,] { { 1.0, 2.0 }, { 2.0, 4.0 } };
        Assert.Equal(1, LinAlg.MatrixRank(a));
    }
}
