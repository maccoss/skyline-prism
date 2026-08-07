using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.BatchCorrection;

/// <summary>
/// Reference-anchored ComBat: estimate each batch's technical effect from the REFERENCE samples
/// only - identical material run in every batch - and apply it to every sample in that batch.
///
/// <para>Standard ComBat aligns each batch to the across-batch grand mean, which assumes the batches
/// hold comparable biology; where they do not (different case/control ratios per plate, say) it
/// removes real signal. Anchoring on material that is the same everywhere makes the per-batch
/// difference purely technical by construction, so no biology is at risk and the batches need not be
/// comparable at all.</para>
///
/// <para>This is <see cref="ComBatCore"/> with a different <see cref="ComBatPlan"/> - the estimator,
/// the empirical-Bayes shrinkage, the NaN handling and the refusal to invent an unestimable scale
/// are literally the same code as standard ComBat. What differs is captured in the plan:</para>
/// <list type="bullet">
/// <item><b>Fit set</b> = the batch's reference samples; apply set = all of its samples.</item>
/// <item><b>Pooled scale</b> = <see cref="PooledScaleRule.PooledWithinBatch"/>: the fit set is
/// repeated measurements of one material, so the honest scale is their pooled within-batch variance
/// over <c>sum(n_i - 1)</c>, not the residual variance of a whole cohort.</item>
/// <item><b>Location-only batches</b> where there are fewer than 2 references: no replicates, so no
/// scale - and under <c>noReferenceBatch = "fallback"</c> the batch's own samples stand in for its
/// references, where the spread is biological and estimating a scale from it would shrink real
/// signal.</item>
/// </list>
///
/// <para>LOG2 [nFeatures, nSamples] in and out.</para>
/// </summary>
public static class ReferenceAnchoredComBat
{
    public static double[,] Run(
        double[,] data,
        IReadOnlyList<string> batchLabels,
        IReadOnlyList<bool> referenceMask,
        bool parPrior = true,
        string noReferenceBatch = "fallback",
        ComBatDiagnostics? diagnostics = null)
    {
        var nSamples = data.GetLength(1);
        if (batchLabels.Count != nSamples)
            throw new ArgumentException("batchLabels length must equal number of samples.");
        if (referenceMask.Count != nSamples)
            throw new ArgumentException("referenceMask length must equal number of samples.");
        if (noReferenceBatch is not ("fallback" or "skip" or "error"))
            throw new ArgumentException("noReferenceBatch must be 'fallback', 'skip', or 'error'.");
        if (!referenceMask.Any(m => m))
            throw new ArgumentException(
                "Reference-anchored ComBat requires at least one reference sample.");

        var (_, batches, batchOfSample) = ComBat.SortedBatches(batchLabels);
        var nBatch = batches.Length;
        if (nBatch < 2)
            return (double[,])data.Clone();

        var references = new List<int>[nBatch];
        for (var i = 0; i < nBatch; i++)
            references[i] = batches[i].Where(s => referenceMask[s]).ToList();

        if (noReferenceBatch == "error" && references.Any(r => r.Count == 0))
            throw new ArgumentException(
                "Some batches have no reference samples; cannot reference-anchor. "
                + "Set noReferenceBatch='fallback'/'skip' or provide references in every batch.");

        var fit = new List<int>[nBatch];
        var locationOnly = new bool[nBatch];
        for (var i = 0; i < nBatch; i++)
        {
            if (references[i].Count > 0)
            {
                fit[i] = references[i];
                // One reference is a level, not a spread: correct where it sits, do not rescale.
                locationOnly[i] = references[i].Count < 2;
            }
            else if (noReferenceBatch == "fallback")
            {
                // No anchor: fall back to the batch's own centre. Its spread is biological, so this
                // is a location correction only - the "assume comparable biology" assumption that
                // reference anchoring exists to avoid, taken deliberately and only for this batch.
                fit[i] = batches[i];
                locationOnly[i] = true;
            }
            else
            {
                fit[i] = new List<int>(); // "skip": left exactly as it came in
            }
        }

        return ComBatCore.Run(data, new ComBatPlan
        {
            Apply = batches,
            Fit = fit,
            BatchOfSample = batchOfSample,
            LocationOnly = locationOnly,
            PooledScale = PooledScaleRule.PooledWithinBatch,
            MeanOnly = false,
            ParPrior = parPrior,
        }, diagnostics);
    }
}
