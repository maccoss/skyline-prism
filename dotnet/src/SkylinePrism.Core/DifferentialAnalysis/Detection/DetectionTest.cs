using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.DifferentialAnalysis.Detection;

/// <summary>One peptide's detection-frequency test result.</summary>
public sealed record DetectionRow(
    string PeptideId,
    int DetA,
    int NA,
    int DetB,
    int NB,
    double RateA,
    double RateB,
    double P,
    double Q);

/// <summary>
/// Peptide detection-frequency test, ported from the explorer's <c>detection_test</c>. For each
/// peptide it compares how often the peptide is detected (present/absent) between two sample groups
/// with a two-sided Fisher exact test, then Benjamini-Hochberg adjusts across peptides. This is the
/// appropriate on/off analysis for cryptic peptides, where the dense abundance matrix would conflate
/// detection with abundance. Input is a binary peptide x sample detection matrix (>= 0.5 means
/// detected).
/// </summary>
public static class DetectionTest
{
    /// <summary>
    /// Run the per-peptide Fisher test over <paramref name="detectionMatrix"/> (peptides x samples,
    /// binary) for the samples in <paramref name="groupAColumns"/> vs <paramref name="groupBColumns"/>,
    /// returning rows sorted ascending by p-value.
    /// </summary>
    public static IReadOnlyList<DetectionRow> Run(
        double[,] detectionMatrix,
        IReadOnlyList<string> peptideIds,
        IReadOnlyList<int> groupAColumns,
        IReadOnlyList<int> groupBColumns)
    {
        var nPeptides = detectionMatrix.GetLength(0);
        if (peptideIds.Count != nPeptides)
            throw new ArgumentException(
                $"peptideIds has {peptideIds.Count} entries but the matrix has {nPeptides} peptides.",
                nameof(peptideIds));

        var nA = groupAColumns.Count;
        var nB = groupBColumns.Count;
        var pValues = new double[nPeptides];
        var detA = new int[nPeptides];
        var detB = new int[nPeptides];
        for (var p = 0; p < nPeptides; p++)
        {
            foreach (var c in groupAColumns)
                if (detectionMatrix[p, c] >= 0.5)
                    detA[p]++;
            foreach (var c in groupBColumns)
                if (detectionMatrix[p, c] >= 0.5)
                    detB[p]++;

            pValues[p] = FisherExact.TwoSidedP(detA[p], nA - detA[p], detB[p], nB - detB[p]);
        }

        var q = Fdr.BenjaminiHochberg(pValues);

        var rows = new DetectionRow[nPeptides];
        for (var p = 0; p < nPeptides; p++)
            rows[p] = new DetectionRow(peptideIds[p], detA[p], nA, detB[p], nB,
                nA > 0 ? detA[p] / (double)nA : double.NaN,
                nB > 0 ? detB[p] / (double)nB : double.NaN,
                pValues[p], q[p]);

        return rows.OrderBy(r => r.P).ToArray();
    }
}
