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
        Assert.Contains(v.Failures, w => w.Contains("QC CV increased"));

        // The report names the reason in the banner itself, not only in a WARNINGS box below the
        // table, and does not repeat it there.
        var sb = new System.Text.StringBuilder();
        QcReport.AppendValidation(sb, v, 2, 2, 4);
        var html = sb.ToString();
        var banner = html.Substring(0, html.IndexOf("</div>", StringComparison.Ordinal));
        Assert.Contains("Validation: FAILED", banner);
        Assert.Contains("QC CV increased", banner);
        Assert.Equal(1, html.Split("QC CV increased").Length - 1);
        // The reference got worse too here: that is a warning, below, and not in the banner.
        Assert.DoesNotContain("Reference CV increased", banner);
        Assert.Contains("Reference CV increased", html);
    }

    [Fact]
    public void PcaDistanceShrinking_IsANote_AndDoesNotFailTheVerdict()
    {
        // The shape of a real report that read FAILED over two green improvements: both control CVs
        // improve, but the QC-reference distance shrinks below 0.5. Here the reference and QC differ
        // only by a uniform +2 offset before (a loading difference) and not at all after - which is
        // normalization working, so it must not fail the run.
        var before = new double[12, 4];
        var after = new double[12, 4];
        for (var f = 0; f < 12; f++)
        {
            var baseVal = 10 + f;
            var wobble = (f % 3 - 1) * 0.4;
            before[f, 0] = baseVal + 1.0 + wobble;
            before[f, 1] = baseVal - 1.0 - wobble;
            before[f, 2] = baseVal + 2 + 1.0 - wobble;
            before[f, 3] = baseVal + 2 - 1.0 + wobble;
            after[f, 0] = baseVal + 0.2 + wobble;
            after[f, 1] = baseVal - 0.2 - wobble;
            after[f, 2] = baseVal + 0.2 - wobble;
            after[f, 3] = baseVal - 0.2 + wobble;
        }

        var v = ValidationStatus.Compute(before, after, Ref, Qc);
        Assert.NotNull(v);
        Assert.True(v!.QcCvImprovement > 0 && v.ReferenceCvImprovement > 0);
        Assert.True(v.PcaDistanceShrank, $"expected the distance to shrink, ratio {v.PcaDistanceRatio}");
        Assert.True(v.Passed);
        Assert.Empty(v.Failures);
        Assert.Empty(v.Warnings);
        Assert.Contains(v.Notes, n => n.Contains("PCA", StringComparison.Ordinal)
            && n.Contains("loading", StringComparison.Ordinal));

        var sb = new System.Text.StringBuilder();
        QcReport.AppendValidation(sb, v, 2, 2, 4);
        var html = sb.ToString();
        Assert.Contains("Validation: PASSED", html);
        Assert.Contains("<strong>NOTES</strong>", html);
    }

    [Fact]
    public void ReferenceCvIncrease_IsAWarningButNotAFailure()
    {
        // Reference spread grows, QC spread shrinks, groups stay apart: the verdict passes, and the
        // reference warning is reported without being presented as the reason for a failure.
        var before = Build(1.0);
        var after = Build(1.0);
        for (var f = 0; f < 12; f++)
        {
            after[f, 0] = 10 + f + 1.5;
            after[f, 1] = 10 + f - 1.5;
            after[f, 2] = 10 + f + 2 + 0.3;
            after[f, 3] = 10 + f + 2 - 0.3;
        }

        var v = ValidationStatus.Compute(before, after, Ref, Qc);
        Assert.NotNull(v);
        Assert.Contains(v!.Warnings, w => w.Contains("Reference CV increased"));
        Assert.DoesNotContain(v.Failures, w => w.Contains("Reference CV increased"));
        Assert.Equal(v.Failures.Count == 0, v.Passed);
    }

    [Fact]
    public void ReturnsNull_WithoutBothControlGroups()
    {
        var m = Build(1.0);
        Assert.Null(ValidationStatus.Compute(m, m, Ref, new[] { 2 })); // only 1 QC
    }
}
