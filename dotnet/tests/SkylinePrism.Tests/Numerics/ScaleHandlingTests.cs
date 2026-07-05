using System;
using Xunit;

namespace SkylinePrism.Tests.Numerics;

/// <summary>
/// Mirrors the Python tests/test_scale_handling.py invariants. The LINEAR vs LOG2
/// boundary discipline is load-bearing for the whole pipeline, so these lock the
/// conversions the C# code must use.
/// </summary>
public class ScaleHandlingTests
{
    [Theory]
    [InlineData(10.0, 1024.0)]
    [InlineData(12.0, 4096.0)]
    [InlineData(14.0, 16384.0)]
    [InlineData(16.0, 65536.0)]
    public void Pow2_LogToLinear(double log2Value, double expectedLinear)
    {
        Assert.Equal(expectedLinear, Math.Pow(2.0, log2Value), 9);
    }

    [Fact]
    public void Log2_LinearToLog_RoundTrips()
    {
        foreach (var linear in new[] { 1024.0, 4096.0, 16384.0, 65536.0 })
        {
            var log2 = Math.Log2(linear);
            Assert.Equal(linear, Math.Pow(2.0, log2), 6);
        }
    }

    [Fact]
    public void CvOnLinear_IsLargerThanCvOnLog2()
    {
        // On log scale variance is compressed: the linear-scale CV must exceed the
        // (meaningless) log2-scale CV for the same data. tests/test_scale_handling.py
        // asserts the same relationship.
        double[] log2Data = { 10.0, 11.0, 12.0, 13.0 };
        var linear = new double[log2Data.Length];
        for (var i = 0; i < log2Data.Length; i++)
            linear[i] = Math.Pow(2.0, log2Data[i]);

        var cvLinear = Cv(linear);
        var cvLog2 = Cv(log2Data);

        Assert.True(cvLinear > 3.0 * cvLog2,
            $"expected linear CV ({cvLinear}) >> log2 CV ({cvLog2})");
    }

    private static double Cv(double[] values)
    {
        double sum = 0.0;
        foreach (var v in values)
            sum += v;
        var mean = sum / values.Length;
        double ss = 0.0;
        foreach (var v in values)
            ss += (v - mean) * (v - mean);
        var std = Math.Sqrt(ss / (values.Length - 1));
        return std / mean * 100.0;
    }
}
