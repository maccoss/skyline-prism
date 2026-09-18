using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>Result of <see cref="DifferentialPca.Compute"/>.</summary>
public sealed class PcaResult
{
    internal PcaResult(string[] sampleIds, double[,] scores, double[] varianceRatio, int nFeaturesUsed)
    {
        SampleIds = sampleIds;
        Scores = scores;
        VarianceRatio = varianceRatio;
        NFeaturesUsed = nFeaturesUsed;
    }

    /// <summary>Sample ids in row order of <see cref="Scores"/>.</summary>
    public string[] SampleIds { get; }

    /// <summary>Principal-coordinate scores, <c>[sample, component]</c>. Signs per component are
    /// arbitrary (a component and its negation are equivalent).</summary>
    public double[,] Scores { get; }

    /// <summary>Fraction of total variance per component.</summary>
    public double[] VarianceRatio { get; }

    /// <summary>Features used (complete across every selected sample).</summary>
    public int NFeaturesUsed { get; }
}

/// <summary>
/// Sample-space PCA for the differential explorer, ported from the Python <c>compute_pca</c>
/// (prism_diff_explorer.py). Operates on a LOG2 matrix: keeps only features observed in every
/// selected sample (complete case), centers each feature across the selected samples (no scaling),
/// and takes the principal coordinates <c>U * S</c> of the singular value decomposition. Computed via
/// the eigendecomposition of the small sample x sample Gram matrix (X X^T = U S^2 U^T), which avoids
/// forming a feature-space SVD.
/// </summary>
public static class DifferentialPca
{
    /// <summary>
    /// Compute PCA over the samples in <paramref name="sampleColumns"/> (column indices into
    /// <paramref name="exprLog2FeaturesBySamples"/>), returning up to <paramref name="nComponents"/>
    /// components. Throws if fewer than two samples or fewer than two complete features remain.
    /// </summary>
    public static PcaResult Compute(
        double[,] exprLog2FeaturesBySamples,
        IReadOnlyList<string> sampleIds,
        IReadOnlyList<int> sampleColumns,
        int nComponents = 6)
    {
        var nFeatures = exprLog2FeaturesBySamples.GetLength(0);
        var nColumns = exprLog2FeaturesBySamples.GetLength(1);
        if (sampleIds.Count != nColumns)
            throw new ArgumentException(
                $"sampleIds has {sampleIds.Count} entries but the matrix has {nColumns} columns.",
                nameof(sampleIds));

        var nSamples = sampleColumns.Count;
        if (nSamples < 2)
            throw new ArgumentException("PCA needs at least 2 samples.", nameof(sampleColumns));

        var cols = new int[nSamples];
        var ids = new string[nSamples];
        for (var s = 0; s < nSamples; s++)
        {
            var c = sampleColumns[s];
            if (c < 0 || c >= nColumns)
                throw new ArgumentException($"Sample column index {c} is out of range.");
            cols[s] = c;
            ids[s] = sampleIds[c];
        }

        // Gram matrix over complete-case, per-feature-centered columns.
        var gram = new DenseMatrix(nSamples, nSamples);
        var col = new double[nSamples];
        var used = 0;
        for (var f = 0; f < nFeatures; f++)
        {
            var complete = true;
            double sum = 0;
            for (var s = 0; s < nSamples; s++)
            {
                var v = exprLog2FeaturesBySamples[f, cols[s]];
                if (double.IsNaN(v))
                {
                    complete = false;
                    break;
                }

                col[s] = v;
                sum += v;
            }

            if (!complete)
                continue;

            var mean = sum / nSamples;
            for (var s = 0; s < nSamples; s++)
                col[s] -= mean;

            for (var i = 0; i < nSamples; i++)
            {
                var ci = col[i];
                for (var j = 0; j < nSamples; j++)
                    gram[i, j] += ci * col[j];
            }

            used++;
        }

        if (used < 2)
            throw new ArgumentException("PCA needs at least 2 features complete across the samples.");

        var evd = gram.Evd(Symmetricity.Symmetric);
        var evals = evd.EigenValues;   // ascending
        var evecs = evd.EigenVectors;  // columns are eigenvectors

        double totalVar = 0;
        for (var i = 0; i < nSamples; i++)
            totalVar += evals[i].Real;

        // numpy's svd(full_matrices=False) yields min(nSamples, nFeatures) singular values; matching it
        // avoids padding the tail with zero-rank components (which show as "-0.0%" variance).
        var k = Math.Min(nComponents, Math.Min(nSamples, used));
        var scores = new double[nSamples, k];
        var varianceRatio = new double[k];
        for (var j = 0; j < k; j++)
        {
            var idx = nSamples - 1 - j; // descending
            var lambda = evals[idx].Real;
            var s = Math.Sqrt(Math.Max(lambda, 0.0));
            for (var i = 0; i < nSamples; i++)
                scores[i, j] = evecs[i, idx] * s;
            varianceRatio[j] = totalVar > 0 ? lambda / totalVar : 0.0;
        }

        return new PcaResult(ids, scores, varianceRatio, used);
    }
}
