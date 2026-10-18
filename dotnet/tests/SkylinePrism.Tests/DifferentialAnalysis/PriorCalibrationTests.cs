using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// The property the default prior was chosen for: calibrated BY INTENSITY, not just overall. A prior
/// can give 5% of null p-values below 0.05 on average while being optimistic for bright features and
/// conservative for dim ones, the two cancelling - which is what the pool-shaped intensity trend did,
/// and what every earlier calibration check, measuring the overall rate only, missed. See "Why
/// limma-trend is the default" in docs/differential-analysis.md.
/// </summary>
public class PriorCalibrationTests
{
    private readonly ITestOutputHelper _out;

    public PriorCalibrationTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// A null with the shape real data has. Technical variance on the log2 scale falls with intensity
    /// towards a floor, as a CV does; the QC and reference pools carry only that. Study samples add
    /// biological variance that is flat on the log scale (a constant CV). Each component varies per
    /// feature (a scaled inverse chi-square on 10 df), so the true prior df is finite.
    /// </summary>
    /// <remarks>
    /// Six seeded draws of 2,000 features, unpaired 5 vs 5, give 2,400 features per intensity quintile:
    /// the binomial sd of a 5% rate is then 0.44%, so the 3.5-6.5% bounds on limma-trend are more than
    /// three sd wide on each side. Measured: limma-trend 4.7-5.8% in every quintile; the pool-shaped
    /// trend 1.9% in the dimmest and 7.4% in the brightest, so the 3-point bound on its tilt has
    /// about 2.5 points of margin.
    /// </remarks>
    [Fact]
    public void UnderTheNull_LimmaTrendIsCalibratedByIntensity_AndThePoolShapedTrendIsNot()
    {
        const int nF = 2000, n = 5, nCtrl = 8, draws = 6;
        var limma = new (int Hits, int N)[5];
        var pools = new (int Hits, int N)[5];

        for (var draw = 0; draw < draws; draw++)
        {
            var rng = new Random(20261018 + draw);
            double Normal() => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
            double InvChi2(int df)
            {
                double s = 0;
                for (var i = 0; i < df; i++)
                {
                    var z = Normal();
                    s += z * z;
                }

                return df / s;
            }

            var width = 2 * n + 2 * nCtrl;
            var expr = new double[nF, width];
            var level = new double[nF];
            for (var f = 0; f < nF; f++)
            {
                var l = 14 + 14 * rng.NextDouble();
                level[f] = l;
                var tech = (0.0049 + 0.25 * Math.Exp(-(l - 14) / 2.5)) * InvChi2(10);
                var bio = 0.09 * InvChi2(10);
                for (var j = 0; j < 2 * n; j++)
                    expr[f, j] = l + Math.Sqrt(bio) * Normal() + Math.Sqrt(tech) * Normal();
                for (var j = 0; j < nCtrl; j++)
                {
                    expr[f, 2 * n + j] = l + Math.Sqrt(tech) * Normal();
                    expr[f, 2 * n + nCtrl + j] = l - 1 + Math.Sqrt(tech) * Normal();
                }
            }

            var order = Enumerable.Range(0, nF).OrderBy(f => level[f]).ToArray();
            var quintile = new int[nF];
            for (var k = 0; k < nF; k++)
                quintile[order[k]] = k * 5 / nF;
            var ids = Enumerable.Range(0, nF).Select(f => "F" + f).ToArray();
            var a = Enumerable.Range(0, n).ToArray();
            var b = Enumerable.Range(n, n).ToArray();
            var controls = new IReadOnlyList<int>[]
            {
                Enumerable.Range(2 * n, nCtrl).ToArray(),
                Enumerable.Range(2 * n + nCtrl, nCtrl).ToArray(),
            };

            void Tally((int Hits, int N)[] into, DifferentialResult result)
            {
                foreach (var row in result.Rows)
                {
                    var q = quintile[int.Parse(row.FeatureId.AsSpan(1))];
                    into[q] = (into[q].Hits + (row.PValue < 0.05 ? 1 : 0), into[q].N + 1);
                }
            }

            Tally(limma, Differential.Run(expr, ids, a, b, new DifferentialOptions
            {
                Prior = VariancePrior.LimmaTrend,
                Correction = MultipleTesting.None,
            }));
            Tally(pools, Differential.Run(expr, ids, a, b, new DifferentialOptions
            {
                Prior = VariancePrior.IntensityTrend,
                PriorGroupColumns = controls,
                Correction = MultipleTesting.None,
            }));
        }

        double Rate((int Hits, int N) t) => (double)t.Hits / t.N;
        _out.WriteLine("quintile  limma-trend  intensity trend from pools");
        for (var q = 0; q < 5; q++)
            _out.WriteLine($"Q{q + 1}        {Rate(limma[q]):0.0000}       {Rate(pools[q]):0.0000}");

        foreach (var t in limma)
            Assert.InRange(Rate(t), 0.035, 0.065);
        Assert.True(Rate(pools[4]) - Rate(pools[0]) > 0.03,
            $"the pool-shaped trend no longer tilts: {Rate(pools[0]):0.000} dimmest, {Rate(pools[4]):0.000} brightest");
    }
}
