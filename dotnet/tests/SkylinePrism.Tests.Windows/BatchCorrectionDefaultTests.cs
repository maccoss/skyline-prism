using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The rule behind the ComBat boxes ticking themselves.
/// </summary>
public class BatchCorrectionDefaultTests
{
    /// <summary>
    /// THE case to protect: turning the correction off keeps it off, however many documents are
    /// added afterwards.
    ///
    /// <para>ComBat changes every reported abundance, so a box that re-ticked itself when the next
    /// input arrived would silently reinstate a correction someone had deliberately declined. The
    /// boxes are also never disabled, so the choice is always available to make.</para>
    /// </summary>
    [Fact]
    public void TurningItOffStaysOffHoweverManyDocumentsAreAdded()
    {
        // The user unticked it, so nothing suggests a value again - at any input count.
        Assert.Null(BatchCorrectionDefault.Suggest(1, null, userChose: true));
        Assert.Null(BatchCorrectionDefault.Suggest(2, null, userChose: true));
        Assert.Null(BatchCorrectionDefault.Suggest(12, null, userChose: true));
        Assert.Null(BatchCorrectionDefault.Suggest(12, "Plate", userChose: true));
    }

    /// <summary>Ticking it on a single document also sticks - the override runs both ways.</summary>
    [Fact]
    public void TurningItOnAlsoSticks()
    {
        Assert.Null(BatchCorrectionDefault.Suggest(1, null, userChose: true));
    }

    /// <summary>One document and no Batch column is one batch, and ComBat is skipped anyway.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void OneDocumentWithNoBatchColumnSuggestsOff(int inputs)
    {
        Assert.False(BatchCorrectionDefault.Suggest(inputs, null, userChose: false));
        Assert.False(BatchCorrectionDefault.Suggest(inputs, "   ", userChose: false));
        Assert.False(BatchCorrectionDefault.HaveBatches(inputs, null));
    }

    /// <summary>A second input is a second batch, by source document.</summary>
    [Fact]
    public void ASecondDocumentSuggestsOn()
    {
        Assert.True(BatchCorrectionDefault.Suggest(2, null, userChose: false));
        Assert.True(BatchCorrectionDefault.HaveBatches(2, null));
    }

    /// <summary>
    /// A Batch column names batches WITHIN one document, so a single annotated plate map has them.
    /// Tying this to the input count alone would leave that user unable to keep the box ticked.
    /// </summary>
    [Fact]
    public void OneDocumentWithABatchColumnSuggestsOn()
    {
        Assert.True(BatchCorrectionDefault.Suggest(1, "Plate", userChose: false));
        Assert.True(BatchCorrectionDefault.HaveBatches(1, "Plate"));
    }
}
