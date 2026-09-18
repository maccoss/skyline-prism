using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Double;
using SkylinePrism.Core.Numerics;

namespace SkylinePrism.Core.DifferentialAnalysis.Detection;

/// <summary>One peptide's covariate-adjusted detection GLM result.</summary>
public sealed record DetectionGlmRow(
    string PeptideId,
    int DetA,
    int NA,
    int DetB,
    int NB,
    double RateA,
    double RateB,
    double LogOr,
    double P,
    double Q);

/// <summary>Result of <see cref="DetectionGlm.Run"/>: rows plus identifiability diagnostics.</summary>
public sealed class DetectionGlmResult
{
    internal DetectionGlmResult(IReadOnlyList<DetectionGlmRow> rows, bool identifiable,
        double groupCollinearityR2, int nParams, int nA, int nB,
        IReadOnlyList<string> covariatesUsed, IReadOnlyList<string> dropped, int nNonConverged,
        string? warning)
    {
        Rows = rows;
        Identifiable = identifiable;
        GroupCollinearityR2 = groupCollinearityR2;
        NParams = nParams;
        NA = nA;
        NB = nB;
        CovariatesUsed = covariatesUsed;
        Dropped = dropped;
        NNonConverged = nNonConverged;
        Warning = warning;
    }

    /// <summary>Per-peptide rows sorted by p (empty when the design is not identifiable).</summary>
    public IReadOnlyList<DetectionGlmRow> Rows { get; }

    /// <summary>Whether the group term is separable from the covariates and the design is full rank for n.</summary>
    public bool Identifiable { get; }

    /// <summary>R^2 of regressing the group indicator on the reduced (covariate) design.</summary>
    public double GroupCollinearityR2 { get; }

    /// <summary>Number of parameters in the full design.</summary>
    public int NParams { get; }

    /// <summary>Group A / group B sample counts.</summary>
    public int NA { get; }

    /// <summary>Group A / group B sample counts.</summary>
    public int NB { get; }

    /// <summary>Covariate design-column names used.</summary>
    public IReadOnlyList<string> CovariatesUsed { get; }

    /// <summary>Covariates or levels dropped during design construction.</summary>
    public IReadOnlyList<string> Dropped { get; }

    /// <summary>Peptides where a Firth fit did not converge.</summary>
    public int NNonConverged { get; }

    /// <summary>Set when the design is not identifiable.</summary>
    public string? Warning { get; }
}

/// <summary>
/// Covariate-adjusted peptide detection test, ported from the explorer's <c>detection_test_glm</c>.
/// Tests <c>detected ~ group [+ covariates]</c> with a Firth-penalized likelihood-ratio test on the
/// group term (group B = 1, so <c>logOR &gt; 0</c> means more detected in B), so it can adjust for
/// confounders (cohort, age, ...) a plain Fisher test cannot. An identifiability guard refuses a
/// contrast where the group is not separable from the covariates (a confounded design).
/// </summary>
public static class DetectionGlm
{
    /// <summary>
    /// Run the adjusted detection test over <paramref name="detectionMatrix"/> (peptides x samples,
    /// binary) for <paramref name="groupAColumns"/> vs <paramref name="groupBColumns"/>, adjusting for
    /// <paramref name="covariates"/>. Returns an empty result with a warning if the design is not
    /// identifiable.
    /// </summary>
    public static DetectionGlmResult Run(
        double[,] detectionMatrix,
        IReadOnlyList<string> peptideIds,
        IReadOnlyList<int> groupAColumns,
        IReadOnlyList<int> groupBColumns,
        IReadOnlyList<Covariate>? covariates = null)
    {
        var nA = groupAColumns.Count;
        var nB = groupBColumns.Count;
        var samples = groupAColumns.Concat(groupBColumns).ToArray();
        var nS = samples.Length;
        var grp = new double[nS];
        for (var s = nA; s < nS; s++)
            grp[s] = 1.0;

        var (xcov, covNames, dropped) = BuildDetectionDesign(covariates, samples, grp);
        var nCov = xcov?.GetLength(1) ?? 0;

        // x_red = [1 | xcov]; x_full = [1 | grp | xcov].
        var xRed = new double[nS, 1 + nCov];
        var xFull = new double[nS, 2 + nCov];
        for (var s = 0; s < nS; s++)
        {
            xRed[s, 0] = 1.0;
            xFull[s, 0] = 1.0;
            xFull[s, 1] = grp[s];
            for (var c = 0; c < nCov; c++)
            {
                xRed[s, 1 + c] = xcov![s, c];
                xFull[s, 2 + c] = xcov![s, c];
            }
        }

        var nParams = xFull.GetLength(1);
        var rFull = LinAlg.MatrixRank(xFull);
        var rRed = LinAlg.MatrixRank(xRed);
        var identifiable = rFull > rRed && rFull == nParams && nS > nParams;

        var groupR2 = 0.0;
        if (xRed.GetLength(1) > 1 && NumpyMath.Var(grp, 0) > 0)
        {
            // SVD (min-norm least squares) matches numpy.linalg.lstsq and, unlike QR, stays finite when
            // the reduced design is rank-deficient (nested categorical covariates).
            var xRedMatrix = DenseMatrix.OfArray(xRed);
            var coef = xRedMatrix.Svd().Solve(DenseVector.OfArray(grp));
            var fitted = xRedMatrix * coef;
            var resid = new double[nS];
            for (var s = 0; s < nS; s++)
                resid[s] = grp[s] - fitted[s];
            groupR2 = Math.Max(0.0, 1.0 - NumpyMath.Var(resid, 0) / NumpyMath.Var(grp, 0));
        }

        if (!identifiable)
        {
            var warning = "Group is not separable from the covariates (or n is too small for the " +
                "design) - the adjusted detection test is not identifiable. The contrast is confounded; " +
                "use the unadjusted view with caution or balance the groups.";
            return new DetectionGlmResult(Array.Empty<DetectionGlmRow>(), false, groupR2, nParams,
                nA, nB, covNames, dropped, 0, warning);
        }

        var nPeptides = detectionMatrix.GetLength(0);
        var pValues = new double[nPeptides];
        var logOr = new double[nPeptides];
        var detA = new int[nPeptides];
        var detB = new int[nPeptides];
        var nNonConverged = 0;
        for (var pep = 0; pep < nPeptides; pep++)
        {
            var y = new double[nS];
            double sum = 0;
            for (var s = 0; s < nS; s++)
            {
                y[s] = detectionMatrix[pep, samples[s]] >= 0.5 ? 1.0 : 0.0;
                sum += y[s];
                if (s < nA)
                    detA[pep] += (int)y[s];
                else
                    detB[pep] += (int)y[s];
            }

            if (sum == 0.0 || sum == nS)
            {
                logOr[pep] = 0.0;
                pValues[pep] = 1.0;
                continue;
            }

            try
            {
                var full = FirthLogit.Fit(xFull, y);
                var red = FirthLogit.Fit(xRed, y);
                var stat = Math.Max(2.0 * (full.PenalizedLogLik - red.PenalizedLogLik), 0.0);
                pValues[pep] = Chi2SurvivalDf1(stat);
                logOr[pep] = full.Beta[1];
                if (!(full.Converged && red.Converged))
                    nNonConverged++;
            }
            catch (Exception e) when (e is InvalidOperationException or ArithmeticException)
            {
                logOr[pep] = double.NaN;
                pValues[pep] = double.NaN;
            }
        }

        var q = Fdr.BenjaminiHochberg(pValues);

        var rows = new DetectionGlmRow[nPeptides];
        for (var pep = 0; pep < nPeptides; pep++)
            rows[pep] = new DetectionGlmRow(peptideIds[pep], detA[pep], nA, detB[pep], nB,
                nA > 0 ? detA[pep] / (double)nA : double.NaN,
                nB > 0 ? detB[pep] / (double)nB : double.NaN,
                logOr[pep], pValues[pep], q[pep]);

        // Sort ascending by p with NaN p-values last (matching pandas sort_values default).
        var ordered = rows.OrderBy(r => double.IsNaN(r.P) ? 1 : 0).ThenBy(r => r.P).ToArray();

        return new DetectionGlmResult(ordered, true, groupR2, nParams, nA, nB, covNames, dropped,
            nNonConverged, null);
    }

    /// <summary>chi-square survival function for df = 1: <c>P(X &gt; x) = erfc(sqrt(x/2))</c>, computed
    /// tail-accurately as <c>2 * Phi(-sqrt(x))</c> to match scipy.stats.chi2.sf(x, 1).</summary>
    private static double Chi2SurvivalDf1(double x)
    {
        if (x <= 0.0)
            return 1.0;
        return 2.0 * Normal.CDF(0.0, 1.0, -Math.Sqrt(x));
    }

    /// <summary>
    /// Detection covariate design over the given sample columns: numeric (dropped on any missing,
    /// skipped if constant, else mean-centered) and categorical (NaN as its own "NA" level, drop-first
    /// dummies, rare levels with fewer than 3 in either direction pooled into the reference, and dummies
    /// collinear with the group dropped).
    /// </summary>
    private static (double[,]? X, List<string> Names, List<string> Dropped) BuildDetectionDesign(
        IReadOnlyList<Covariate>? covariates, int[] cols, double[] grp)
    {
        var nS = cols.Length;
        var columns = new List<double[]>();
        var names = new List<string>();
        var dropped = new List<string>();

        if (covariates != null)
        {
            foreach (var cov in covariates)
            {
                if (cov is NumericCovariate num)
                {
                    var v = new double[nS];
                    var missing = false;
                    for (var s = 0; s < nS; s++)
                    {
                        v[s] = num.Values[cols[s]];
                        if (double.IsNaN(v[s]))
                            missing = true;
                    }

                    if (missing)
                    {
                        dropped.Add($"{num.Name} (missing numeric values)");
                        continue;
                    }

                    if (Distinct(v) < 2)
                        continue;

                    var mean = NumpyMath.Mean(v);
                    var centered = new double[nS];
                    for (var s = 0; s < nS; s++)
                        centered[s] = v[s] - mean;
                    columns.Add(centered);
                    names.Add(num.Name);
                }
                else if (cov is CategoricalCovariate cat)
                {
                    var vals = new string[nS];
                    for (var s = 0; s < nS; s++)
                        vals[s] = cat.Values[cols[s]] ?? "NA"; // NaN -> its own level

                    var levels = vals.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToArray();
                    for (var li = 1; li < levels.Length; li++)
                    {
                        var level = levels[li];
                        var cv = new double[nS];
                        double count = 0;
                        for (var s = 0; s < nS; s++)
                        {
                            cv[s] = vals[s] == level ? 1.0 : 0.0;
                            count += cv[s];
                        }

                        if (Distinct(cv) < 2)
                            continue;
                        if (count < 3 || count > nS - 3)
                        {
                            dropped.Add($"{cat.Name}_{level} (rare level, n<3 - pooled into reference)");
                            continue;
                        }

                        if (AllClose(cv, grp, complement: false) || AllClose(cv, grp, complement: true))
                        {
                            dropped.Add($"{cat.Name}_{level} (confounded with group)");
                            continue;
                        }

                        columns.Add(cv);
                        names.Add($"{cat.Name}_{level}");
                    }
                }
            }
        }

        if (columns.Count == 0)
            return (null, names, dropped);

        var x = new double[nS, columns.Count];
        for (var s = 0; s < nS; s++)
        for (var c = 0; c < columns.Count; c++)
            x[s, c] = columns[c][s];
        return (x, names, dropped);
    }

    private static int Distinct(double[] v)
    {
        var set = new HashSet<double>();
        foreach (var x in v)
            set.Add(x);
        return set.Count;
    }

    private static bool AllClose(double[] a, double[] grp, bool complement)
    {
        for (var i = 0; i < a.Length; i++)
        {
            var b = complement ? 1.0 - grp[i] : grp[i];
            if (Math.Abs(a[i] - b) > 1e-8 + 1e-5 * Math.Abs(b))
                return false;
        }

        return true;
    }
}
