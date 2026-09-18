using System;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Parity micro-tests for <see cref="LinearModel"/> (limma lmFit). Expected values are hand-computed
/// for a two-group design and cross-checkable by hand, so they lock the OLS primitives (coefficients,
/// residual SD, Amean, unscaled SE, residual df) the moderated-t step builds on.
/// </summary>
public class LinearModelTests
{
    // Two groups of two samples: intercept + groupB indicator.
    //   sample: A A B B
    private static readonly double[,] Design =
    {
        { 1, 0 },
        { 1, 0 },
        { 1, 1 },
        { 1, 1 },
    };

    [Fact]
    public void Fit_TwoGroups_RecoversCoefficientsSigmaAmean()
    {
        // feature0 = [1,3,10,14]: group A mean 2, group B mean 12 -> intercept 2, groupB 10.
        //   fitted [2,2,12,12], residuals [-1,1,-2,2], SSR = 10, df = 2, sigma = sqrt(5).
        //   Amean = (1+3+10+14)/4 = 7.
        // feature1 = [5,5,5,5]: intercept 5, groupB 0, SSR 0, sigma 0, Amean 5.
        var data = new[,]
        {
            { 1.0, 3.0, 10.0, 14.0 },
            { 5.0, 5.0, 5.0, 5.0 },
        };

        var fit = LinearModel.Fit(data, Design);

        Assert.Equal(2.0, fit.DfResidual, 12);

        Assert.Equal(2.0, fit.Coefficients[0, 0], 12);
        Assert.Equal(10.0, fit.Coefficients[0, 1], 12);
        Assert.Equal(5.0, fit.Coefficients[1, 0], 12);
        Assert.Equal(0.0, fit.Coefficients[1, 1], 12);

        Assert.Equal(Math.Sqrt(5.0), fit.Sigma[0], 12);
        Assert.Equal(0.0, fit.Sigma[1], 12);

        Assert.Equal(7.0, fit.Amean[0], 12);
        Assert.Equal(5.0, fit.Amean[1], 12);
    }

    [Fact]
    public void Fit_UnscaledStandardErrors_DependOnDesignOnly()
    {
        // (X^T X) = [[4,2],[2,2]], det 4, (X^T X)^-1 = [[0.5,-0.5],[-0.5,1.0]].
        //   diag = [0.5, 1.0] -> stdev_unscaled = [sqrt(0.5), 1.0].
        var data = new[,] { { 1.0, 3.0, 10.0, 14.0 } };
        var fit = LinearModel.Fit(data, Design);

        Assert.Equal(Math.Sqrt(0.5), fit.StdevUnscaled[0], 12);
        Assert.Equal(1.0, fit.StdevUnscaled[1], 12);
    }

    [Fact]
    public void Fit_RejectsDesignWithTooFewSamples()
    {
        var data = new[,] { { 1.0, 2.0 } };
        var design = new[,] { { 1.0, 0.0 }, { 1.0, 1.0 } }; // 2 samples, 2 coefficients -> df 0
        Assert.Throws<ArgumentException>(() => LinearModel.Fit(data, design));
    }

    [Fact]
    public void Fit_RejectsRankDeficientDesign()
    {
        // [intercept, groupB, 1-groupB]: the third column is collinear with the first two, so the
        // design is rank 2 with 3 columns. The Python reference refuses before fitting; we must not
        // return NaN/Infinity from a singular normal-equations inverse.
        var data = new[,] { { 1.0, 3.0, 10.0, 14.0 } };
        var design = new[,]
        {
            { 1.0, 0.0, 1.0 },
            { 1.0, 0.0, 1.0 },
            { 1.0, 1.0, 0.0 },
            { 1.0, 1.0, 0.0 },
        };
        Assert.Throws<ArgumentException>(() => LinearModel.Fit(data, design));
    }
}
