using System.Collections.Generic;

namespace SkylinePrism.Core.Parsimony;

/// <summary>
/// A parsimonious protein group (output of Stage 3), mirroring the Python ProteinGroup.
/// Peptide lists distinguish unique, razor, parsimony-assigned (unique + razor), and
/// all-mapped peptides.
/// </summary>
public sealed class ProteinGroup
{
    public required string GroupId { get; init; }
    public required string LeadingProtein { get; init; }
    public string LeadingName { get; init; } = "NA";
    public string LeadingUniProtId { get; init; } = "NA";
    public string LeadingGeneName { get; init; } = "NA";
    public string LeadingDescription { get; init; } = "NA";

    public List<string> MemberProteins { get; init; } = new();
    public List<string> SubsumedProteins { get; init; } = new();

    /// <summary>Peptides assigned by parsimony (unique ∪ razor) = the CSV AllPeptides column.</summary>
    public List<string> Peptides { get; init; } = new();
    public List<string> UniquePeptides { get; init; } = new();
    public List<string> RazorPeptides { get; init; } = new();

    /// <summary>ALL peptides mapping to any member/subsumed protein (used for rollup "all_groups").</summary>
    public List<string> AllMappedPeptides { get; init; } = new();
}
