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
        Assert.DoesNotContain(v.Failures, w => w.Contains("QC CV increased"));
        Assert.True(v.Passed);
    }

    [Fact]
    public void FlagsQcCvIncrease()
    {
        var before = Build(0.3);
        var after = Build(1.5); // controls got worse
        var v = ValidationStatus.Compute(before, after, Ref, Qc);
        Assert.NotNull(v);
        Assert.True(v!.QcCvImprovement < 0);
        Assert.False(v.Passed);
        Assert.Contains(v.Failures, w => w.Contains("QC CV increased"));
        Assert.DoesNotContain(v.Warnings, w => w.Contains("QC CV increased"));

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
        // The banner list is new markup in a page Panorama publishes; it must stay acceptable there.
        Assert.Empty(SkylinePrism.Core.Panorama.WikiReport.Refusals(html));
    }

    // Reference and QC differ by a uniform +2 offset in "apart" and not at all in "together"; both
    // control CVs are tighter in "together".
    private static (double[,] Apart, double[,] Together) OffsetRemoved()
    {
        var apart = new double[12, 4];
        var together = new double[12, 4];
        for (var f = 0; f < 12; f++)
        {
            var baseVal = 10 + f;
            var wobble = (f % 3 - 1) * 0.4;
            apart[f, 0] = baseVal + 1.0 + wobble;
            apart[f, 1] = baseVal - 1.0 - wobble;
            apart[f, 2] = baseVal + 2 + 1.0 - wobble;
            apart[f, 3] = baseVal + 2 - 1.0 + wobble;
            together[f, 0] = baseVal + 0.2 + wobble;
            together[f, 1] = baseVal - 0.2 - wobble;
            together[f, 2] = baseVal + 0.2 - wobble;
            together[f, 3] = baseVal - 0.2 + wobble;
        }
        return (apart, together);
    }

    [Fact]
    public void PcaDistanceShrinkingFromRaw_IsANote_AndDoesNotFailTheVerdict()
    {
        // The shape of a real report that read FAILED over two green improvements: both control CVs
        // improve, but the QC-reference distance shrinks below 0.5. Measured from raw data, a +2
        // offset that is gone after processing is a loading difference normalization removed - with
        // no normalized matrix to tell it apart from a collapse, it must not fail the run.
        var (before, after) = OffsetRemoved();

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
        Assert.Contains("for information only", html);
    }

    [Fact]
    public void PcaDistanceShrinkingInNormalization_PassesWithoutANote()
    {
        // The same raw -> output change, but the normalized matrix shows the offset was already gone
        // before batch correction: normalization removed it, and ComBat moved nothing.
        var (raw, after) = OffsetRemoved();

        var v = ValidationStatus.Compute(raw, after, Ref, Qc, normalized: after);
        Assert.NotNull(v);
        Assert.True(v!.PcaFromNormalized);
        Assert.Equal(1.0, v.PcaDistanceRatio, 12);
        Assert.True(v.Passed);
        Assert.DoesNotContain(v.Notes, n => n.Contains("PCA", StringComparison.Ordinal));
    }

    [Fact]
    public void PcaCollapseInBatchCorrection_FailsTheVerdict_AndNamesIt()
    {
        // The two materials are still apart after normalization and together after batch correction:
        // ComBat removed a real difference, which is what the check exists to catch. Both CVs improve,
        // so the collapse is the only reason, and the banner has to say so.
        var (apart, together) = OffsetRemoved();

        var v = ValidationStatus.Compute(apart, together, Ref, Qc, normalized: apart);
        Assert.NotNull(v);
        Assert.True(v!.QcCvImprovement > 0 && v.ReferenceCvImprovement > 0);
        Assert.True(v.PcaDistanceShrank, $"expected the distance to shrink, ratio {v.PcaDistanceRatio}");
        Assert.False(v.Passed);
        var failure = Assert.Single(v.Failures);
        Assert.Contains("Batch correction moved the QC and reference groups", failure);

        var sb = new System.Text.StringBuilder();
        QcReport.AppendValidation(sb, v, 2, 2, 4);
        var html = sb.ToString();
        var banner = html.Substring(0, html.IndexOf("</div>", StringComparison.Ordinal));
        Assert.Contains("Validation: FAILED", banner);
        Assert.Contains("Batch correction moved", banner);
        Assert.Contains("fails below 0.5", html);
        Assert.Empty(SkylinePrism.Core.Panorama.WikiReport.Refusals(html));
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
        Assert.True(v.Passed);
    }

    [Fact]
    public void ReturnsNull_WithoutBothControlGroups()
    {
        var m = Build(1.0);
        Assert.Null(ValidationStatus.Compute(m, m, Ref, new[] { 2 })); // only 1 QC
    }
}
