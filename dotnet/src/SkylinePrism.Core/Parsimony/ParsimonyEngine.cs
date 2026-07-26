using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.Parsimony;

/// <summary>Bidirectional peptide/protein maps + protein metadata (build_peptide_protein_map).</summary>
public sealed class PeptideProteinMap
{
    public Dictionary<string, HashSet<string>> PeptideToProteins { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, HashSet<string>> ProteinToPeptides { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, string> ProteinToName { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, string> ProteinToGene { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, string> ProteinToDescription { get; } = new(StringComparer.Ordinal);
}

/// <summary>
/// Protein parsimony (Stage 3), porting parsimony.py: build_peptide_protein_map (Skyline
/// CSV-based) + compute_protein_groups (subsumable removal, indistinguishable grouping,
/// deterministic iterative razor assignment, PG#### numbering).
/// </summary>
public static class ParsimonyEngine
{
    private static readonly Regex UniProtEntry = new(@"^[sptr]{2}\|[^|]+\|([^\s]+)", RegexOptions.Compiled);

    private static string ExtractEntryName(string identifier)
    {
        var m = UniProtEntry.Match(identifier);
        return m.Success ? m.Groups[1].Value : identifier;
    }

    /// <summary>
    /// Read distinct peptide/protein records from the merged parquet and compute groups. When
    /// <paramref name="fastaPath"/> is set, the peptide-&gt;protein map is built by enzyme-aware
    /// substring search against the FASTA (proper parsimony) instead of the Skyline Protein Accession
    /// column. The <paramref name="enzyme"/>/<paramref name="enzymeSpecificity"/> terminus check
    /// removes phantom assignments to homologs that share the sequence but not the flanking cleavage
    /// site; it is ignored on the Skyline-column path (already enzyme-aware).
    /// </summary>
    public static List<ProteinGroup> Run(
        string mergedParquet, SkylineColumns cols, bool applyParsimony = true, string? fastaPath = null,
        string enzyme = "trypsin", string enzymeSpecificity = "full")
    {
        var records = ReadRecords(mergedParquet, cols);
        var map = string.IsNullOrWhiteSpace(fastaPath)
            ? BuildMap(records)
            : FastaParser.BuildMap(
                fastaPath!, records.Select(r => r.Peptide).Where(p => p.Length > 0).Distinct(),
                enzyme: enzyme, enzymeSpecificity: enzymeSpecificity);
        return applyParsimony ? ComputeProteinGroups(map) : BuildUngroupedGroups(map);
    }

    /// <summary>
    /// One group per protein accession (parsimony disabled): every protein keeps ALL of its
    /// mapped peptides (shared peptides go to each protein), no subsumption/razor.
    /// </summary>
    public static List<ProteinGroup> BuildUngroupedGroups(PeptideProteinMap map)
    {
        var proteins = map.ProteinToPeptides.Keys.OrderBy(x => x, StringComparer.Ordinal).ToList();
        var groups = new List<ProteinGroup>(proteins.Count);
        for (var i = 0; i < proteins.Count; i++)
        {
            var protein = proteins[i];
            var peptides = map.ProteinToPeptides[protein].OrderBy(x => x, StringComparer.Ordinal).ToList();
            if (peptides.Count == 0)
                continue;
            groups.Add(new ProteinGroup
            {
                GroupId = $"PG{i + 1:D4}",
                LeadingProtein = protein,
                LeadingName = map.ProteinToName.GetValueOrDefault(protein, protein),
                LeadingUniProtId = protein,
                LeadingGeneName = map.ProteinToGene.GetValueOrDefault(protein, ""),
                LeadingDescription = map.ProteinToDescription.GetValueOrDefault(protein, ""),
                MemberProteins = new List<string> { protein },
                SubsumedProteins = new List<string>(),
                Peptides = new List<string>(peptides),
                UniquePeptides = new List<string>(peptides),
                RazorPeptides = new List<string>(),
                AllMappedPeptides = new List<string>(peptides),
            });
        }
        return groups;
    }

    public record Record(string Peptide, string ProteinAccession, string Name, string Gene, string Description);

    private static List<Record> ReadRecords(string mergedParquet, SkylineColumns cols)
    {
        var protAcc = cols.Protein ?? cols.ProteinName ?? "Protein";
        var protName = cols.ProteinName ?? protAcc;
        var protGene = cols.ProteinGene;

        var select = new List<string>
        {
            $"\"{cols.Peptide}\" AS pep",
            $"\"{protAcc}\" AS acc",
            $"\"{protName}\" AS name",
            protGene != null ? $"\"{protGene}\" AS gene" : "'' AS gene",
        };

        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"SELECT DISTINCT {string.Join(", ", select)} FROM read_parquet('{mergedParquet.Replace("'", "''")}')";
        using var reader = cmd.ExecuteReader();
        var records = new List<Record>();
        while (reader.Read())
        {
            var pep = reader.IsDBNull(0) ? "" : reader.GetString(0);
            var acc = reader.IsDBNull(1) ? "" : Convert.ToString(reader.GetValue(1), CultureInfo.InvariantCulture) ?? "";
            var name = reader.IsDBNull(2) ? "" : Convert.ToString(reader.GetValue(2), CultureInfo.InvariantCulture) ?? "";
            var gene = reader.IsDBNull(3) ? "" : Convert.ToString(reader.GetValue(3), CultureInfo.InvariantCulture) ?? "";
            // Description column is the same "Protein" column as the name.
            records.Add(new Record(pep, acc, name, gene, name));
        }
        return records;
    }

    public static PeptideProteinMap BuildMap(IEnumerable<Record> records)
    {
        var map = new PeptideProteinMap();

        foreach (var rec in records.Distinct())
        {
            var proteins = SplitList(rec.ProteinAccession);
            var names = SplitList(rec.Name);
            var genes = SplitList(rec.Gene);
            var descriptions = SplitList(rec.Description);

            // Pad names with the corresponding protein accession; genes/descriptions with "".
            while (names.Count < proteins.Count)
                names.Add(proteins[names.Count]);
            while (genes.Count < proteins.Count)
                genes.Add("");
            while (descriptions.Count < proteins.Count)
                descriptions.Add("");

            for (var i = 0; i < proteins.Count; i++)
            {
                var protein = proteins[i];
                var name = i < names.Count ? names[i] : protein;
                var gene = i < genes.Count ? genes[i] : "";
                var description = i < descriptions.Count ? descriptions[i] : "";

                name = ExtractEntryName(name);
                if (string.IsNullOrEmpty(gene))
                    gene = "NA";
                if (string.IsNullOrEmpty(description) || description == protein || description.Contains('|'))
                    description = "NA";

                if (!map.PeptideToProteins.TryGetValue(rec.Peptide, out var ps))
                    map.PeptideToProteins[rec.Peptide] = ps = new HashSet<string>(StringComparer.Ordinal);
                ps.Add(protein);
                if (!map.ProteinToPeptides.TryGetValue(protein, out var peps))
                    map.ProteinToPeptides[protein] = peps = new HashSet<string>(StringComparer.Ordinal);
                peps.Add(rec.Peptide);

                map.ProteinToName.TryAdd(protein, name);
                map.ProteinToGene.TryAdd(protein, gene);
                map.ProteinToDescription.TryAdd(protein, description);
            }
        }
        return map;
    }

    public static List<ProteinGroup> ComputeProteinGroups(PeptideProteinMap map)
    {
        var protToPep = map.ProteinToPeptides;
        var pepToProt = map.PeptideToProteins;

        // Step 1: subsumable proteins (strict subset -> lexicographically smallest superset).
        var proteins = protToPep.Keys.OrderBy(x => x, StringComparer.Ordinal).ToList();
        // Each protein's smallest proper superset depends only on the (read-only) maps, so this
        // parallelizes cleanly and deterministically - each task writes its own key, no contention.
        // A proper superset of a must contain ALL of a's peptides, so it appears in pepToProt for a's
        // rarest peptide; scanning only those candidates (not every protein) makes it near-linear rather
        // than O(proteins^2), keeping the identical result: the lexicographically smallest proper superset.
        var subsumedByConc = new ConcurrentDictionary<string, string>(StringComparer.Ordinal);
        Parallel.ForEach(proteins, a =>
        {
            var pepsA = protToPep[a];
            if (pepsA.Count == 0)
                return;
            string pivot = null!;
            var fewest = int.MaxValue;
            foreach (var pep in pepsA)
            {
                var c = pepToProt[pep].Count;
                if (c < fewest) { fewest = c; pivot = pep; }
            }
            string? smallest = null;
            foreach (var b in pepToProt[pivot])
            {
                if (a == b)
                    continue;
                var pepsB = protToPep[b];
                if (pepsB.Count > pepsA.Count && pepsA.IsProperSubsetOf(pepsB)
                    && (smallest is null || string.CompareOrdinal(b, smallest) < 0))
                    smallest = b;
            }
            if (smallest is not null)
                subsumedByConc[a] = smallest;
        });
        var subsumedBy = new Dictionary<string, string>(subsumedByConc, StringComparer.Ordinal);

        // Build superset -> [subsumed] iterating proteins in sorted order, so the lists are deterministic
        // regardless of the parallel fill order above.
        var subsumingToSubsumed = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (var a in proteins)
        {
            if (!subsumedBy.TryGetValue(a, out var sup))
                continue;
            if (!subsumingToSubsumed.TryGetValue(sup, out var list))
                subsumingToSubsumed[sup] = list = new List<string>();
            list.Add(a);
        }

        var activeProteins = new HashSet<string>(proteins.Where(p => !subsumedBy.ContainsKey(p)), StringComparer.Ordinal);

        // Step 2: indistinguishable proteins (identical peptide sets) among active.
        var bySet = new Dictionary<string, List<string>>();
        foreach (var p in activeProteins)
        {
            var key = string.Join("", protToPep[p].OrderBy(x => x, StringComparer.Ordinal));
            if (!bySet.TryGetValue(key, out var list))
                bySet[key] = list = new List<string>();
            list.Add(p);
        }
        var proteinToIndistGroup = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (var grp in bySet.Values.Where(g => g.Count > 1))
        {
            var set = new HashSet<string>(grp, StringComparer.Ordinal);
            foreach (var p in grp)
                proteinToIndistGroup[p] = set;
        }

        string Canonical(string prot) =>
            proteinToIndistGroup.TryGetValue(prot, out var g)
                ? g.OrderBy(x => x, StringComparer.Ordinal).First()
                : prot;

        // Step 3: unique vs shared peptides (by canonical protein).
        var peptideToCanonical = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (var kv in pepToProt)
        {
            var canon = new HashSet<string>(StringComparer.Ordinal);
            foreach (var p in kv.Value)
                if (activeProteins.Contains(p))
                    canon.Add(Canonical(p));
            peptideToCanonical[kv.Key] = canon;
        }
        var uniquePeptides = peptideToCanonical.Where(kv => kv.Value.Count == 1).Select(kv => kv.Key)
            .ToHashSet(StringComparer.Ordinal);
        var sharedPeptides = peptideToCanonical.Where(kv => kv.Value.Count > 1).Select(kv => kv.Key)
            .ToHashSet(StringComparer.Ordinal);

        var canonicalProteins = activeProteins.Select(Canonical).ToHashSet(StringComparer.Ordinal);
        var canonicalToMembers = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (var can in canonicalProteins)
        {
            canonicalToMembers[can] = proteinToIndistGroup.TryGetValue(can, out var g)
                ? g.OrderBy(x => x, StringComparer.Ordinal).ToList()
                : new List<string> { can };
        }

        var canonicalToUniquePeps = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (var pep in uniquePeptides)
        {
            var can = peptideToCanonical[pep].First();
            if (!canonicalToUniquePeps.TryGetValue(can, out var set))
                canonicalToUniquePeps[can] = set = new HashSet<string>(StringComparer.Ordinal);
            set.Add(pep);
        }

        // Step 4: greedy razor assignment. The selection criterion (highest unique count; ties -> largest
        // total peptide set, matching Osprey's lowest-group-ID-to-largest-set; then lowest canonical
        // accession for determinism) is FIXED - it does not depend on which shared peptides have already
        // been assigned. So the greedy "pick the best candidate each iteration" is equivalent to processing
        // canonical proteins ONCE in that ranked order and giving each whatever shared peptides still remain
        // that it can claim. This replaces the old O(canonical x iterations) per-iteration rescan (which
        // recomputed the candidate set every iteration) with a single sorted pass, and yields the identical
        // assignment (verified by the razor parity tests).
        var remainingShared = new HashSet<string>(sharedPeptides, StringComparer.Ordinal);
        var canonicalToRazorPeps = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        var razorOrder = canonicalProteins
            .OrderByDescending(c => canonicalToUniquePeps.TryGetValue(c, out var s) ? s.Count : 0)
            .ThenByDescending(c => protToPep[canonicalToMembers[c][0]].Count)
            .ThenBy(c => c, StringComparer.Ordinal)
            .ToList();
        foreach (var can in razorOrder)
        {
            if (remainingShared.Count == 0)
                break;
            var canPeps = protToPep[canonicalToMembers[can][0]];
            var toAssign = canPeps.Where(remainingShared.Contains).OrderBy(x => x, StringComparer.Ordinal).ToList();
            if (toAssign.Count == 0)
                continue;
            var razorSet = new HashSet<string>(StringComparer.Ordinal);
            foreach (var p in toAssign)
            {
                razorSet.Add(p);
                remainingShared.Remove(p);
            }
            canonicalToRazorPeps[can] = razorSet;
        }

        // Step 5: build groups.
        var groups = new List<ProteinGroup>();
        var sortedCanonicalAll = canonicalProteins.OrderBy(x => x, StringComparer.Ordinal).ToList();
        for (var i = 0; i < sortedCanonicalAll.Count; i++)
        {
            var can = sortedCanonicalAll[i];
            var members = canonicalToMembers[can];

            var subsumedList = new List<string>();
            foreach (var member in members)
                if (subsumingToSubsumed.TryGetValue(member, out var subs))
                    subsumedList.AddRange(subs);

            var uniquePeps = canonicalToUniquePeps.TryGetValue(can, out var up) ? up : new HashSet<string>(StringComparer.Ordinal);
            var razorPeps = canonicalToRazorPeps.TryGetValue(can, out var rp) ? rp : new HashSet<string>(StringComparer.Ordinal);
            var allPeps = new HashSet<string>(uniquePeps, StringComparer.Ordinal);
            allPeps.UnionWith(razorPeps);

            var allMapped = new HashSet<string>(StringComparer.Ordinal);
            foreach (var member in members)
                if (protToPep.TryGetValue(member, out var mp))
                    allMapped.UnionWith(mp);
            foreach (var sub in subsumedList)
                if (protToPep.TryGetValue(sub, out var sp))
                    allMapped.UnionWith(sp);

            if (allPeps.Count == 0)
                continue;

            var leading = members[0];
            var gene = map.ProteinToGene.GetValueOrDefault(leading, "");
            var description = map.ProteinToDescription.GetValueOrDefault(leading, "");

            groups.Add(new ProteinGroup
            {
                GroupId = $"PG{i + 1:D4}",
                LeadingProtein = leading,
                LeadingName = map.ProteinToName.GetValueOrDefault(leading, leading),
                LeadingUniProtId = leading,
                LeadingGeneName = gene,
                LeadingDescription = description,
                MemberProteins = members,
                SubsumedProteins = subsumedList,
                Peptides = allPeps.ToList(),
                UniquePeptides = uniquePeps.ToList(),
                RazorPeptides = razorPeps.ToList(),
                AllMappedPeptides = allMapped.ToList(),
            });
        }
        return groups;
    }

    private static List<string> SplitList(string value)
    {
        if (string.IsNullOrEmpty(value) || value is "nan" or "None")
            return new List<string>();
        return value.Split(';', StringSplitOptions.None)
            .Select(s => s.Trim())
            .Where(s => s.Length > 0)
            .ToList();
    }
}
