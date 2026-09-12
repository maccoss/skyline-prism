using SkylinePrism.Core.Pipeline;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// What the run log says about ComBat. The ORDER of the cases is the content.
/// </summary>
public class BatchCorrectionNoteTests
{
    /// <summary>
    /// The bug this file exists for: a correction the user turned OFF must not be reported as one
    /// the data refused.
    ///
    /// <para>Reported from a real run - "Batches: 1 from single label (no batch annotation); ComBat
    /// skipped (needs &gt;= 2 batches)" - with the GUI option unchecked. It reads as "PRISM wanted to
    /// correct and your data would not let it", which sends someone looking for a Batch column they
    /// never wanted and hides that the setting was honoured.</para>
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(12)]
    public void TurnedOffNeverBlamesTheBatchCount(int batches)
    {
        var note = BatchCorrectionNote.For(
            enabled: false, peptideLevel: true, proteinLevel: true, batchCount: batches);

        Assert.Equal("off (not requested)", note);
        Assert.DoesNotContain("skipped", note);
        Assert.DoesNotContain("batches", note);
    }

    /// <summary>
    /// Both arms off is also "not requested", whatever the master switch says - there is no
    /// correction to report on.
    /// </summary>
    [Fact]
    public void BothArmsOffIsNotRequestedEvenWhenEnabled()
    {
        Assert.Equal(
            "off (not requested)",
            BatchCorrectionNote.For(
                enabled: true, peptideLevel: false, proteinLevel: false, batchCount: 4));
    }

    /// <summary>Asked for and impossible: say both, and say how many batches there actually were.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void RequestedButTooFewBatchesSaysBoth(int batches)
    {
        var note = BatchCorrectionNote.For(
            enabled: true, peptideLevel: true, proteinLevel: true, batchCount: batches);

        Assert.Contains("requested", note);
        Assert.Contains("skipped", note);
        Assert.Contains($"found {batches}", note);
    }

    /// <summary>Which arms were asked for is carried through, so a partial request is not read as both.</summary>
    [Theory]
    [InlineData(true, false, "peptide only")]
    [InlineData(false, true, "protein only")]
    [InlineData(true, true, "peptide and protein")]
    public void TheArmsAskedForAreNamed(bool peptide, bool protein, string expected)
    {
        Assert.Equal(
            $"on ({expected})",
            BatchCorrectionNote.For(
                enabled: true, peptideLevel: peptide, proteinLevel: protein, batchCount: 3));

        Assert.Contains(
            expected,
            BatchCorrectionNote.For(
                enabled: true, peptideLevel: peptide, proteinLevel: protein, batchCount: 1));
    }
}
