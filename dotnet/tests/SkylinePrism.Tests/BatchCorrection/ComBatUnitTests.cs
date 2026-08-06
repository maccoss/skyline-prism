using System;
using SkylinePrism.Core.BatchCorrection;
using Xunit;

namespace SkylinePrism.Tests.BatchCorrection;

/// <summary>
/// Controlled ComBat parity: a fixed 3x6 / 2-batch matrix whose expected output was
/// produced by the Python batch_correction.combat. Locks ComBat independent of the
/// pipeline fixtures.
/// </summary>
public class ComBatUnitTests
{
    [Fact]
    public void Combat_SmallControlled_MatchesPython()
    {
        var data = new double[,]
        {
            { 10.0, 11.0, 10.5, 12.0, 13.0, 12.5 },
            { 5.0, 5.5, 6.0, 7.0, 6.5, 7.5 },
            { 20.0, 21.0, 19.0, 22.0, 23.0, 21.0 },
        };
        var batch = new[] { "1", "1", "1", "2", "2", "2" };

        // Regenerated from skyline_prism.batch_correction.combat after var_pooled switched to
        // ddof 0 (sva's dense denominator - see ComBat.VarPooledDdof). Row 1 is unchanged because
        // it is symmetric enough that the denominator cancels.
        var expected = new double[,]
        {
            { 10.975109340832176, 11.791605921759903, 11.38335763129604, 11.208394078240097, 12.024890659167824, 11.61664236870396 },
            { 5.841751709536137, 6.25, 6.658248290463863, 6.25, 5.841751709536137, 6.658248290463863 },
            { 21.23328473740792, 22.049781318335647, 20.416788156480195, 20.76671526259208, 21.583211843519805, 19.950218681664353 },
        };

        var actual = ComBat.Run(data, batch);

        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 6; j++)
                Assert.Equal(expected[i, j], actual[i, j], 9);
    }

    [Fact]
    public void Combat_SingleSampleBatch_Aborts()
    {
        // Batch "2" has a single sample: ComBat cannot estimate its effect, so it must abort
        // (matching Python's _check_inputs) rather than silently degrade to a mean-only correction.
        var data = new double[,]
        {
            { 10.0, 11.0, 10.5, 12.0 },
            { 5.0, 5.5, 6.0, 7.0 },
        };
        var batch = new[] { "1", "1", "1", "2" };

        var ex = Assert.Throws<InvalidOperationException>(() => ComBat.Run(data, batch));
        Assert.Contains("single sample", ex.Message);
    }
}
