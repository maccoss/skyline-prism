using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>Dual-control validation verdict: detects QC CV improvement vs. a QC CV increase.</summary>
public class ValidationStatusTests
{
    // 12 features; samples 0,1 = reference, 2,3 = QC (QC material offset +2 so controls stay distinct).
    private static double[,] Build(double controlSpread)
    {
        var m = new double[12, 4];
        for (var f = 0; f < 12; f++)
        {
            var baseVal = 10 + f;
            m[f, 0] = baseVal + controlSpread;
            m[f, 1] = baseVal - controlSpread;
            m[f, 2] = baseVal + 2 + controlSpread;
            m[f, 3] = baseVal + 2 - controlSpread;
        }
        return m;
    }

    private static readonly int[] Ref = { 0, 1 };
    private static readonly int[] Qc = { 2, 3 };

    [Fact]
    public void DetectsQcCvImprovement()
    {
        var before = Build(1.5);
        var after = Build(0.3); // tighter controls after correction
        var v = ValidationStatus.Compute(before, after, Ref, Qc);
        Assert.NotNull(v);
        Assert.True(v!.QcCvImprovement > 0, $"QC CV should improve, got {v.QcCvImprovement}");
        Assert.DoesNotContain(v.Warnings, w => w.Contains("QC CV increased"));
    }

    [Fact]
    public void FlagsQcCvIncrease()
    {
        var before = Build(0.3);
        var after = Build(1.5); // controls got worse
        var v = ValidationStatus.Compute(before, after, Ref, Qc);
        Assert.NotNull(v);
        Assert.True(v!.QcCvImprovement < 0);
        Assert.Contains(v.Warnings, w => w.Contains("QC CV increased"));
        Assert.False(v.Passed);
    }

    [Fact]
    public void ReturnsNull_WithoutBothControlGroups()
    {
        var m = Build(1.0);
        Assert.Null(ValidationStatus.Compute(m, m, Ref, new[] { 2 })); // only 1 QC
    }
}
