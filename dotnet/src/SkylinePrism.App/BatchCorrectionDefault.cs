namespace SkylinePrism.App;

/// <summary>
/// Whether the ComBat boxes should be ticked, as a rule that can be tested without a window.
/// </summary>
/// <remarks>
/// <para>Extracted from <c>MainWindow</c> for the same reason <c>VizNavigation</c> was: it is a
/// decision with cases, it lives in a file at zero coverage, and the interesting case - that turning
/// the correction OFF stays off when another document is added - is exactly the kind of stateful
/// behaviour that a later change breaks silently.</para>
/// </remarks>
internal static class BatchCorrectionDefault
{
    /// <summary>
    /// Whether the run has more than one batch for ComBat to correct BETWEEN.
    /// </summary>
    /// <remarks>
    /// Two ways to get batches, and the second is why this is not just the input count: separate
    /// inputs become separate batches by source document, and a Batch column names batches WITHIN
    /// one document. Someone with a single annotated plate map has batches and no second input.
    /// </remarks>
    public static bool HaveBatches(int inputCount, string? batchColumn) =>
        inputCount > 1 || !string.IsNullOrWhiteSpace(batchColumn);

    /// <summary>
    /// What the boxes should be set to, or null to leave them exactly as they are.
    /// </summary>
    /// <param name="userChose">
    /// The user has ticked or unticked a box themselves, or a loaded config stated a value. Null is
    /// returned from then on: a default that reasserted itself after being overruled is worse than
    /// one that is simply wrong, and "I turned this off and it came back" is the complaint that
    /// makes people stop trusting the window.
    /// </param>
    public static bool? Suggest(int inputCount, string? batchColumn, bool userChose) =>
        userChose ? null : HaveBatches(inputCount, batchColumn);
}
