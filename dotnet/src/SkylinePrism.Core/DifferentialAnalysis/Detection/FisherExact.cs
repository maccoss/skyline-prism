using System;
using MathNet.Numerics;

namespace SkylinePrism.Core.DifferentialAnalysis.Detection;

/// <summary>
/// Two-sided Fisher exact test for a 2x2 contingency table, matching scipy.stats.fisher_exact. The
/// two-sided p-value is the sum of hypergeometric probabilities over all tables with the same margins
/// whose probability does not exceed the observed table's (within a 1 + 1e-7 tolerance), clamped to 1.
/// </summary>
public static class FisherExact
{
    /// <summary>
    /// Two-sided p-value for the table <c>[[a, b], [c, d]]</c> (all counts non-negative).
    /// </summary>
    public static double TwoSidedP(int a, int b, int c, int d)
    {
        if (a < 0 || b < 0 || c < 0 || d < 0)
            throw new ArgumentException("Cell counts must be non-negative.");

        var total = a + b + c + d;
        if (total == 0)
            return 1.0;

        var nGood = a + c;      // column-1 total
        var nSample = a + b;    // row-1 total
        var logDen = LogChoose(total, nSample);

        double Pmf(int k) =>
            Math.Exp(LogChoose(nGood, k) + LogChoose(total - nGood, nSample - k) - logDen);

        var kLo = Math.Max(0, nSample - (total - nGood));
        var kHi = Math.Min(nSample, nGood);

        var pObserved = Pmf(a);
        var cutoff = pObserved * (1.0 + 1e-7);

        double sum = 0;
        for (var k = kLo; k <= kHi; k++)
        {
            var pk = Pmf(k);
            if (pk <= cutoff)
                sum += pk;
        }

        return Math.Min(sum, 1.0);
    }

    private static double LogChoose(int n, int k)
    {
        if (k < 0 || k > n)
            return double.NegativeInfinity;
        return SpecialFunctions.GammaLn(n + 1) - SpecialFunctions.GammaLn(k + 1)
            - SpecialFunctions.GammaLn(n - k + 1);
    }
}
