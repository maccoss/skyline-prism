using System;
using System.Collections.Generic;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// Benjamini-Hochberg false-discovery-rate correction, ported from the PRISM Differential
/// Explorer's <c>_bh</c> (prism_diff_explorer.py). NaN-safe: NaN p-values are excluded from the
/// correction (m counts only finite p-values) and passed through as NaN, bit-identical to that
/// reference. This is the standalone BH used for the "family" significance basis.
///
/// The explorer's genome-wide <c>adj.P.Val</c> instead comes from statsmodels
/// <c>multipletests(method="fdr_bh")</c>, which agrees with this to within rounding on finite
/// p-values (it scales by <c>rank/m</c> rather than <c>m/rank</c>) but, unlike this, returns all
/// NaN if any input is NaN and counts NaN toward m. The moderated-t path feeds only finite
/// p-values (complete rows, positive posterior variance), where the two are equivalent.
/// </summary>
public static class Fdr
{
    /// <summary>
    /// Benjamini-Hochberg adjusted p-values, same order as the input. NaN entries stay NaN and do
    /// not count toward the number of tests. Adjusted values are the monotone (reverse cumulative
    /// minimum) step-up transform, clipped to [0, 1].
    /// </summary>
    public static double[] BenjaminiHochberg(ReadOnlySpan<double> pValues)
    {
        var n = pValues.Length;
        var adj = new double[n];

        // Indices of the finite (testable) p-values; NaN positions are set aside as NaN.
        var okIdx = new List<int>(n);
        for (var i = 0; i < n; i++)
        {
            if (double.IsNaN(pValues[i]))
                adj[i] = double.NaN;
            else
                okIdx.Add(i);
        }

        var m = okIdx.Count;
        if (m == 0)
            return adj;

        var p = new double[m];
        for (var k = 0; k < m; k++)
            p[k] = pValues[okIdx[k]];

        // order[r] = index into p of the r-th smallest p-value (ascending).
        var order = Stats.ArgSort(p);

        // ranked[r] = p_(r) * m / (r+1)  -- the raw step-up values in ascending-p order.
        var ranked = new double[m];
        for (var r = 0; r < m; r++)
            ranked[r] = p[order[r]] * m / (r + 1);

        // Enforce monotonicity via the reverse cumulative minimum (np.minimum.accumulate on the
        // reversed array), so a smaller downstream value pulls earlier ones down.
        var cmin = double.PositiveInfinity;
        for (var r = m - 1; r >= 0; r--)
        {
            if (ranked[r] < cmin)
                cmin = ranked[r];
            ranked[r] = cmin;
        }

        // Clip to [0, 1] and scatter back to the original positions.
        for (var r = 0; r < m; r++)
        {
            var v = ranked[r];
            if (v < 0.0)
                v = 0.0;
            else if (v > 1.0)
                v = 1.0;
            adj[okIdx[order[r]]] = v;
        }

        return adj;
    }
}
