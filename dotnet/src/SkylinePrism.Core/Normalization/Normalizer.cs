using System;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.Normalization;

/// <summary>
/// Global (sample-level) normalization on LOG2 matrices, ported from the inline Stage 2b/4b
/// logic in cli.py. Matrices are [nFeatures, nSamples]; NaN is preserved.
/// </summary>
public static class Normalizer
{
    /// <summary>
    /// Median normalization (cli.py ~1908): for each sample column subtract
    /// (median(col) - median(all sample medians)). Equalizes every sample's median to the
    /// global median. Operates on LOG2. pandas median skips NaN.
    /// </summary>
    public static double[,] MedianNormalize(double[,] log2Matrix)
    {
        var nRows = log2Matrix.GetLength(0);
        var nCols = log2Matrix.GetLength(1);

        var sampleMedians = new double[nCols];
        var colBuf = new double[nRows];
        for (var j = 0; j < nCols; j++)
        {
            for (var i = 0; i < nRows; i++)
                colBuf[i] = log2Matrix[i, j];
            sampleMedians[j] = Stats.NanMedian(colBuf);
        }

        var globalMedian = Stats.NanMedian(sampleMedians);

        var result = new double[nRows, nCols];
        for (var j = 0; j < nCols; j++)
        {
            var normFactor = sampleMedians[j] - globalMedian;
            for (var i = 0; i < nRows; i++)
                result[i, j] = log2Matrix[i, j] - normFactor; // NaN - x = NaN
        }
        return result;
    }
}
