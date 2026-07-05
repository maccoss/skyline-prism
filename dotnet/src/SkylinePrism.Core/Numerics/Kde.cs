using System;
using System.Collections.Generic;

namespace SkylinePrism.Core.Numerics;

/// <summary>
/// Gaussian kernel density estimate on a fixed grid (the per-sample density curves in the QC
/// intensity-distribution plot). Silverman's rule-of-thumb bandwidth. The data is binned into a fine
/// histogram first and each grid point only sums nearby bins, so the cost is ~O(grid) rather than
/// O(grid * n) - important for the tens of thousands of peptides/proteins per sample.
/// </summary>
public static class Kde
{
    public static double[] Estimate(IReadOnlyList<double> values, double[] grid, int bins = 512)
    {
        var density = new double[grid.Length];
        var n = values.Count;
        if (n < 2 || grid.Length == 0)
            return density;

        double mean = 0;
        for (var i = 0; i < n; i++) mean += values[i];
        mean /= n;
        double ss = 0;
        for (var i = 0; i < n; i++) { var d = values[i] - mean; ss += d * d; }
        var std = Math.Sqrt(ss / (n - 1));
        if (std <= 0)
            return density;
        var h = 1.06 * std * Math.Pow(n, -0.2); // Silverman
        if (h <= 0)
            return density;

        var gmin = grid[0];
        var gmax = grid[^1];
        if (gmax <= gmin)
            return density;
        var binW = (gmax - gmin) / bins;
        var counts = new double[bins];
        for (var i = 0; i < n; i++)
        {
            var b = (int)((values[i] - gmin) / binW);
            if (b < 0) b = 0;
            else if (b >= bins) b = bins - 1;
            counts[b]++;
        }

        var norm = 1.0 / (n * h * Math.Sqrt(2.0 * Math.PI));
        var inv2h2 = 1.0 / (2.0 * h * h);
        var reach = Math.Max(1, (int)Math.Ceiling(5.0 * h / binW)); // Gaussian is negligible beyond ~5h
        for (var gi = 0; gi < grid.Length; gi++)
        {
            var x = grid[gi];
            var centerBin = (int)((x - gmin) / binW);
            var lo = Math.Max(0, centerBin - reach);
            var hi = Math.Min(bins - 1, centerBin + reach);
            var sum = 0.0;
            for (var b = lo; b <= hi; b++)
            {
                if (counts[b] == 0) continue;
                var d = x - (gmin + (b + 0.5) * binW);
                sum += counts[b] * Math.Exp(-d * d * inv2h2);
            }
            density[gi] = sum * norm;
        }
        return density;
    }
}
