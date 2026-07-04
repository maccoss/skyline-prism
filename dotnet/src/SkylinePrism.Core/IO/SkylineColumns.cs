using System.Collections.Generic;
using System.Text;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Explicit input-column-name overrides (Python's <c>data.*</c> section). Any value set here wins over
/// auto-detection when it resolves against the available columns; unset falls through to auto-detect.
/// </summary>
public sealed record ColumnOverrides(
    string? Peptide = null,
    string? Protein = null,
    string? ProteinName = null,
    string? Abundance = null,
    string? RetentionTime = null,
    string? Sample = null,
    string? Transition = null,
    string? Batch = null);

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
    public string? AcquiredTime { get; init; }
    public string? Protein { get; init; }
    public string? ProteinName { get; init; }
    public string? ProteinGene { get; init; }
    public string? Batch { get; init; }

    /// <summary>
    /// find_column (cli.py:92), made robust to the different Skyline export conventions: matches each
    /// candidate against the available columns ignoring case, spaces, and underscores - so the English
    /// "Peptide Modified Sequence Unimod Ids", the invariant "PeptideModifiedSequenceUnimodIds", and the
    /// underscore form all resolve to the actual column. First candidate that matches wins; null if none.
    /// </summary>
    public static string? FindColumn(ICollection<string> available, params string[] candidates)
    {
        var lookup = new Dictionary<string, string>();
        foreach (var col in available)
        {
            var norm = Normalize(col);
            if (!lookup.ContainsKey(norm))
                lookup[norm] = col; // first available column with this normalized form wins
        }
        foreach (var name in candidates)
            if (lookup.TryGetValue(Normalize(name), out var actual))
                return actual;
        return null;
    }

    private static string Normalize(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (var c in s)
            if (c != ' ' && c != '_')
                sb.Append(char.ToLowerInvariant(c));
        return sb.ToString();
    }

    /// <summary>
    /// Detect the standard column set from the merged parquet's column names, matching
    /// the priority order in cli.py Stage 1 (lines 1206-1312). Sample prefers "Sample ID".
    /// </summary>
    public static SkylineColumns Detect(ICollection<string> available, ColumnOverrides? overrides = null)
    {
        var o = overrides ?? new ColumnOverrides();

        // An override wins if it resolves against the available columns (config.data.*, matched
        // ignoring case/spaces), else fall through to auto-detection - matching Python (the configured
        // column is a hint that wins if present, otherwise auto-detect).
        string? Ov(string? name) => name is not null ? FindColumn(available, name) : null;
        string Require(string? ov, params string[] candidates) =>
            Ov(ov) ?? FindColumn(available, candidates) ?? candidates[^1];

        var peptide = Ov(o.Peptide)
            ?? FindColumn(available, "Peptide Modified Sequence Unimod Ids", "Peptide Modified Sequence", "Peptide")
            ?? "Peptide Modified Sequence";

        var sample = Ov(o.Sample)
            ?? FindColumn(available, "Sample ID")
            ?? FindColumn(available, "Replicate Name", "Replicate_Name")
            ?? "Replicate Name";

        return new SkylineColumns
        {
            Peptide = peptide,
            Sample = sample,
            Abundance = Require(o.Abundance, "Area"),
            Transition = Require(o.Transition, "Fragment Ion"),
            PrecursorCharge = Require(null, "Precursor Charge"),
            ProductCharge = Require(null, "Product Charge"),
            RetentionTime = Require(o.RetentionTime, "Retention Time"),
            ShapeCorrelation = FindColumn(available, "Shape Correlation"),
            ProductMz = FindColumn(available, "Product Mz"),
            AcquiredTime = FindColumn(available, "Acquired Time"),
            Protein = Ov(o.Protein) ?? FindColumn(available, "Protein Accession"),
            ProteinName = Ov(o.ProteinName) ?? FindColumn(available, "Protein"),
            ProteinGene = FindColumn(available, "Protein Gene"),
            Batch = Ov(o.Batch) ?? FindColumn(available, "Batch"),
        };
    }
}
