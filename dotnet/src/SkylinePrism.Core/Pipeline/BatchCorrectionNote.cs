namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// What the run log says about ComBat, in one line.
/// </summary>
/// <remarks>
/// <para>Its own type because the ORDER of the cases is the whole content, and getting that order
/// wrong is not a cosmetic bug. The first version tested the batch count before the setting, so a
/// run with one batch always reported "skipped (needs &gt;= 2 batches)" - including when the user had
/// deliberately turned the correction off. That reads as "PRISM wanted to correct and your data
/// would not let it", which sends someone looking for a Batch column they never wanted, and it hides
/// the fact that the setting was honoured.</para>
///
/// <para>So the user's intent is reported first, and the data's capacity to satisfy it second. They
/// answer different questions: "was this asked for" and "could it be done".</para>
/// </remarks>
internal static class BatchCorrectionNote
{
    public static string For(bool enabled, bool peptideLevel, bool proteinLevel, int batchCount)
    {
        // Not requested at all - by the master switch, or by both arms being off. Nothing about the
        // batch count is relevant to a correction nobody asked for.
        if (!enabled || (!peptideLevel && !proteinLevel))
            return "off (not requested)";

        var wanted = (peptideLevel, proteinLevel) switch
        {
            (true, true) => "peptide and protein",
            (true, false) => "peptide only",
            _ => "protein only",
        };

        // Asked for, and the data cannot support it. ComBat estimates a per-batch effect, so one
        // batch leaves nothing to estimate.
        if (batchCount < 2)
            return $"requested ({wanted}) but skipped - needs >= 2 batches, found {batchCount}";

        return $"on ({wanted})";
    }
}
