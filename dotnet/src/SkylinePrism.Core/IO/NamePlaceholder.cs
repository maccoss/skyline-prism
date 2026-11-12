using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.IO;

/// <summary>
/// The one test for "this name is not a name": what an output carries where a gene, accession,
/// entry name or description is unknown.
/// </summary>
/// <remarks>
/// <para>Parsimony writes the literal <c>NA</c> for an unknown gene, entry name and description
/// (<c>ParsimonyEngine</c>, the <c>ProteinGroup</c> defaults), so every output of a non-model organism
/// carries thousands of them, and a file that went through pandas or R can carry the others. Every
/// reader that SHOWS or SUBMITS a name goes through here: before this, each kept its own short list
/// and they disagreed - enrichment dropped "nan" and "none" but sent "NA" to g:Profiler as a gene,
/// and the volcano and Dynamic Range labels printed it.</para>
/// <para>The writer still writes <c>NA</c>: the column is read by users' own scripts and pinned by
/// the bit-exact output digests, so changing what is written is a format change of its own.</para>
/// <para>Separate from <c>CovariateTyping</c>'s missing tokens on purpose: a covariate value is also
/// missing when it is a non-finite number, which says nothing about a name.</para>
/// </remarks>
public static class NamePlaceholder
{
    private static readonly HashSet<string> Tokens =
        new(StringComparer.OrdinalIgnoreCase) { "NA", "#NA", "N/A", "#N/A", "nan", "None", "null" };

    /// <summary>True for an empty or whitespace value and for every placeholder token.</summary>
    public static bool Is(string? value) =>
        string.IsNullOrWhiteSpace(value) || Tokens.Contains(value.Trim());

    /// <summary>
    /// <paramref name="value"/> with each placeholder made empty, member by member of a
    /// ";"-separated list so the members stay aligned with the other identity columns; empty when
    /// nothing in it names anything.
    /// </summary>
    public static string Clean(string? value)
    {
        if (string.IsNullOrEmpty(value))
            return string.Empty;
        if (!value.Contains(';'))
            return Is(value) ? string.Empty : value;
        var members = value.Split(';');
        for (var k = 0; k < members.Length; k++)
            if (Is(members[k]))
                members[k] = string.Empty;
        // A list of nothing but placeholders is no name at all, not a row of separators.
        return members.All(m => m.Length == 0) ? string.Empty : string.Join(";", members);
    }

    /// <summary>
    /// The members of a ";"-separated identity value BY POSITION, placeholders made empty: member k
    /// names group k, so an unknown member stays as an empty string rather than being dropped. A
    /// shared peptide in PG0001 and PG0002 with genes <c>NA;ALB</c> is <c>["", "ALB"]</c>; dropping
    /// the empty slot gave <c>["ALB"]</c>, which pairs ALB with PG0001. Empty when nothing in the
    /// value names anything.
    /// </summary>
    public static IReadOnlyList<string> SplitAligned(string? value)
    {
        var cleaned = Clean(value);
        return cleaned.Length == 0
            ? Array.Empty<string>()
            : cleaned.Split(';').Select(m => m.Trim()).ToArray();
    }

    /// <summary>
    /// The first member of a ";"-separated identity value - the first group's - or null when that
    /// member is unknown. Never a later group's name borrowed in its place.
    /// </summary>
    public static string? FirstMember(string? value) => FirstMember(SplitAligned(value));

    /// <summary><see cref="FirstMember(string?)"/> for members already split by <see cref="SplitAligned"/>.</summary>
    public static string? FirstMember(IReadOnlyList<string> members) =>
        members.Count > 0 && members[0].Length > 0 ? members[0] : null;

    /// <summary>
    /// Members back into one ";"-separated value, keeping each in its place: <c>["", "ALB"]</c> is
    /// <c>;ALB</c>. Empty when no member names anything.
    /// </summary>
    public static string JoinAligned(IReadOnlyList<string>? members) =>
        members is null || members.All(string.IsNullOrEmpty) ? string.Empty : string.Join(";", members);
}
