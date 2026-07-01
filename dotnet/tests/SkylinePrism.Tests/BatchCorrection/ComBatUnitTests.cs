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

        var expected = new double[,]
        {
            { 10.983495716930463, 11.767790128856502, 11.375642922893483, 11.232209871143498, 12.016504283069537, 11.624357077106517 },
            { 5.841751709536137, 6.25, 6.658248290463863, 6.25, 5.841751709536137, 6.658248290463863 },
            { 21.248714154213033, 22.033008566139074, 20.464419742286996, 20.751285845786967, 21.535580257713004, 19.966991433860926 },
        };

        var actual = ComBat.Run(data, batch);

        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 6; j++)
                Assert.Equal(expected[i, j], actual[i, j], 9);
    }
}
