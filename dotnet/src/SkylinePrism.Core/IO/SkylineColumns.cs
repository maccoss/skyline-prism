using System.Collections.Generic;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Resolved Skyline/merged-parquet column names, mirroring the auto-detection in
/// cli.py (find_column + the Stage 1 detection block). Column names vary between the
/// CSV export (spaces) and parquet/invariant formats (underscores / no spaces), so all
/// lookups go through <see cref="FindColumn"/>.
/// </summary>
public sealed class SkylineColumns
{
    public required string Peptide { get; init; }
    public required string Sample { get; init; }
    public required string Abundance { get; init; }
    public required string Transition { get; init; }
    public required string PrecursorCharge { get; init; }
    public required string ProductCharge { get; init; }
    public required string RetentionTime { get; init; }
    public string? ShapeCorrelation { get; init; }
    public string? ProductMz { get; init; }
    public string? Protein { get; init; }
    public string? ProteinName { get; init; }
    public string? ProteinGene { get; init; }
    public string? Batch { get; init; }

    /// <summary>
    /// find_column (cli.py:92): try each candidate as-is, then space-&gt;underscore,
    /// underscore-&gt;space, then spaces removed. First match wins; null if none.
    /// </summary>
    public static string? FindColumn(ICollection<string> available, params string[] candidates)
    {
        foreach (var name in candidates)
        {
            if (available.Contains(name))
                return name;
            var underscore = name.Replace(" ", "_");
            if (available.Contains(underscore))
                return underscore;
            var space = name.Replace("_", " ");
            if (available.Contains(space))
                return space;
            var noSpace = name.Replace(" ", "");
            if (available.Contains(noSpace))
                return noSpace;
        }
        return null;
    }

    /// <summary>
    /// Detect the standard column set from the merged parquet's column names, matching
    /// the priority order in cli.py Stage 1 (lines 1206-1312). Sample prefers "Sample ID".
    /// </summary>
    public static SkylineColumns Detect(ICollection<string> available)
    {
        string Require(string logical, params string[] candidates) =>
            FindColumn(available, candidates)
            ?? candidates[^1]; // fall back to last candidate (Python's behaviour on miss)

        var peptide = FindColumn(available,
            "Peptide Modified Sequence Unimod Ids",
            "Peptide Modified Sequence",
            "Peptide")
            ?? "Peptide Modified Sequence";

        var sample = FindColumn(available, "Sample ID")
            ?? FindColumn(available, "Replicate Name", "Replicate_Name")
            ?? "Replicate Name";

        return new SkylineColumns
        {
            Peptide = peptide,
            Sample = sample,
            Abundance = Require("Area", "Area"),
            Transition = Require("Fragment Ion", "Fragment Ion"),
            PrecursorCharge = Require("Precursor Charge", "Precursor Charge"),
            ProductCharge = Require("Product Charge", "Product Charge"),
            RetentionTime = Require("Retention Time", "Retention Time"),
            ShapeCorrelation = FindColumn(available, "Shape Correlation"),
            ProductMz = FindColumn(available, "Product Mz"),
            Protein = FindColumn(available, "Protein Accession"),
            ProteinName = FindColumn(available, "Protein"),
            ProteinGene = FindColumn(available, "Protein Gene"),
            Batch = FindColumn(available, "Batch"),
        };
    }
}
