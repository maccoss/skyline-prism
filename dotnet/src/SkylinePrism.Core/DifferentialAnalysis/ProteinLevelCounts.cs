using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// The check every peptide-count prior makes before it runs: that the input is protein-level, one
/// row per protein with the number of peptides it was rolled up from. A peptide has no peptide count,
/// so these priors refuse anything else rather than run on it - or fall back to another prior, which
/// would be a silently different p-value.
/// </summary>
/// <remarks>
/// The same checks, in the same order, as <c>proteomics-toolkit</c>'s
/// <c>_protein_level_peptide_counts</c>, so the two tools refuse the same input. It refuses when:
/// <list type="bullet">
/// <item>there are no counts (PRISM writes <c>n_peptides</c> to the protein parquet only);</item>
/// <item>a count is missing, not a whole number, or below 1;</item>
/// <item>every protein has the same count - what a peptide-level table looks like;</item>
/// <item>a feature id repeats - rows that are not one per protein, which is what peptide rows look
/// like after each has been given its protein's count.</item>
/// </list>
/// It cannot see a peptide-level table with unique peptide ids and a varying count added by hand;
/// nothing in such a table says which level it is.
/// </remarks>
public static class ProteinLevelCounts
{
    /// <summary>Throws <see cref="ArgumentException"/>, saying why, unless the counts are protein-level.</summary>
    public static void Check(IReadOnlyList<double>? counts, IReadOnlyList<string> featureIds, VariancePrior prior)
    {
        var name = DifferentialTokens.Prior(prior);
        if (counts is null)
            throw new ArgumentException(
                $"The {name} prior is for protein-level data and needs each protein's peptide count "
                + "(n_peptides), which only the protein-level matrix carries. Use --prior limma-trend "
                + "for a peptide-level analysis.");
        if (counts.Count != featureIds.Count)
            throw new ArgumentException(
                $"The {name} prior was given {counts.Count} peptide counts for {featureIds.Count} features.");

        var missing = counts.Count(c => !double.IsFinite(c));
        if (missing > 0)
            throw new ArgumentException(
                $"The {name} prior: {missing} of {counts.Count} features have no peptide count. Every "
                + "protein needs the number of peptides it was rolled up from.");
        if (counts.Any(c => c < 1 || c != Math.Round(c)))
            throw new ArgumentException(
                $"The {name} prior needs whole-number peptide counts of 1 or more (peptides per protein); "
                + $"they run from {counts.Min():G} to {counts.Max():G}, including some that are not.");
        if (counts.Distinct().Count() < 2)
            throw new ArgumentException(
                $"The {name} prior: every feature has n_peptides = {counts[0]:G}, so peptide count carries "
                + "no information. That is what peptide-level input looks like, and this prior is for "
                + "protein-level analysis only; use --prior limma-trend.");

        var seen = new HashSet<string>(StringComparer.Ordinal);
        string? firstRepeat = null;
        var repeats = 0;
        foreach (var id in featureIds)
            if (!seen.Add(id))
            {
                repeats++;
                firstRepeat ??= id;
            }

        if (repeats > 0)
            throw new ArgumentException(
                $"The {name} prior is for protein-level data, one row per protein, but {repeats} rows "
                + $"repeat a feature id already seen (for example '{firstRepeat}'). Peptide-level rows "
                + "carrying their protein's peptide count look like this; use --prior limma-trend for a "
                + "peptide-level analysis.");
    }
}
