using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.Parsimony;
using Xunit;

namespace SkylinePrism.Tests.Parsimony;

/// <summary>
/// Parsimony must (a) be deterministic regardless of input row order and (b) reproduce Osprey's
/// grouping (maccoss/osprey, crates/osprey-fdr/src/protein.rs / docs/16-protein-parsimony.md):
/// identical-set merging, subset elimination, all-mode shared-to-all, and razor iterative greedy
/// set cover.
/// </summary>
public class ParsimonyOspreyTests
{
    // One record per (protein, peptide). BuildMap unions across records, so a shared peptide is
    // expressed as the same peptide under multiple proteins.
    private static List<ParsimonyEngine.Record> Records(Dictionary<string, string[]> protToPeps)
    {
        var recs = new List<ParsimonyEngine.Record>();
        foreach (var (prot, peps) in protToPeps)
            foreach (var pep in peps)
                recs.Add(new ParsimonyEngine.Record(pep, prot, prot, "", prot));
        return recs;
    }

    private static List<ProteinGroup> Groups(Dictionary<string, string[]> protToPeps)
        => ParsimonyEngine.ComputeProteinGroups(ParsimonyEngine.BuildMap(Records(protToPeps)));

    private static string Signature(List<ProteinGroup> gs) => string.Join("|", gs
        .OrderBy(g => g.LeadingProtein, System.StringComparer.Ordinal)
        .Select(g => $"{g.LeadingProtein}"
            + $":M[{string.Join(",", g.MemberProteins.OrderBy(x => x, System.StringComparer.Ordinal))}]"
            + $":U[{string.Join(",", g.UniquePeptides.OrderBy(x => x, System.StringComparer.Ordinal))}]"
            + $":R[{string.Join(",", g.RazorPeptides.OrderBy(x => x, System.StringComparer.Ordinal))}]"
            + $":A[{string.Join(",", g.AllMappedPeptides.OrderBy(x => x, System.StringComparer.Ordinal))}]"));

    private static ProteinGroup ByLeading(List<ProteinGroup> gs, string leading)
        => gs.Single(g => g.LeadingProtein == leading);

    [Fact]
    public void Grouping_IsOrderIndependent()
    {
        // An ambiguous scenario with subsumption, an indistinguishable pair, unique + shared peptides.
        var scenario = new Dictionary<string, string[]>
        {
            ["P1"] = new[] { "A", "B", "C", "X", "Y" },
            ["P2"] = new[] { "D", "X", "Z" },
            ["P3"] = new[] { "E", "Y", "Z" },
            ["P4"] = new[] { "A", "B", "C", "X", "Y" }, // indistinguishable from P1
            ["P5"] = new[] { "D", "X" },                 // strict subset of P2
        };

        var records = Records(scenario);
        var baseline = Signature(ParsimonyEngine.ComputeProteinGroups(ParsimonyEngine.BuildMap(records)));

        // Several deterministic reorderings of the same records must yield identical groups.
        var orderings = new List<List<ParsimonyEngine.Record>>
        {
            Enumerable.Reverse(records).ToList(),
            records.Skip(7).Concat(records.Take(7)).ToList(),                 // rotate
            records.OrderBy(r => r.Peptide, System.StringComparer.Ordinal).ToList(),
            records.OrderByDescending(r => r.ProteinAccession, System.StringComparer.Ordinal)
                   .ThenBy(r => r.Peptide, System.StringComparer.Ordinal).ToList(),
        };
        foreach (var ordering in orderings)
        {
            var sig = Signature(ParsimonyEngine.ComputeProteinGroups(ParsimonyEngine.BuildMap(ordering)));
            Assert.Equal(baseline, sig);
        }
    }

    [Fact]
    public void IdenticalSets_AreMergedIntoOneGroup()
    {
        // Osprey test_basic_parsimony_grouping: P1,P2 identical {A,B,C}; P3 {D,E}.
        var gs = Groups(new Dictionary<string, string[]>
        {
            ["P1"] = new[] { "A", "B", "C" },
            ["P2"] = new[] { "A", "B", "C" },
            ["P3"] = new[] { "D", "E" },
        });
        Assert.Equal(2, gs.Count);
        var merged = ByLeading(gs, "P1");
        Assert.Equal(new[] { "P1", "P2" }, merged.MemberProteins.OrderBy(x => x, System.StringComparer.Ordinal));
        Assert.Equal(new[] { "A", "B", "C" }, merged.UniquePeptides.OrderBy(x => x, System.StringComparer.Ordinal));
    }

    [Fact]
    public void StrictSubset_IsEliminated()
    {
        // Osprey test_subset_elimination: P1 {A,B,C}, P2 {A,B} subset -> only P1 leads.
        var gs = Groups(new Dictionary<string, string[]>
        {
            ["P1"] = new[] { "A", "B", "C" },
            ["P2"] = new[] { "A", "B" },
        });
        Assert.Single(gs);
        Assert.Equal("P1", gs[0].LeadingProtein);
        Assert.Contains("P2", gs[0].SubsumedProteins); // PRISM records it; Osprey drops it (same grouping)
    }

    [Fact]
    public void AllMode_SharedPeptideMapsToEveryGroup()
    {
        // Osprey test_shared_peptides_all_mode: SHARED in both -> AllMappedPeptides of both groups
        // (this is PRISM's default all_groups path; razor is not consulted).
        var gs = Groups(new Dictionary<string, string[]>
        {
            ["P1"] = new[] { "A", "B", "SHARED" },
            ["P2"] = new[] { "C", "D", "SHARED" },
        });
        Assert.Equal(2, gs.Count);
        Assert.Contains("SHARED", ByLeading(gs, "P1").AllMappedPeptides);
        Assert.Contains("SHARED", ByLeading(gs, "P2").AllMappedPeptides);
    }

    [Fact]
    public void Razor_AssignsSharedToGroupWithMostUnique()
    {
        // Osprey test_shared_peptides_razor_mode / Example 1: P1 has 3 unique, P2 has 1 -> SHARED -> P1.
        var gs = Groups(new Dictionary<string, string[]>
        {
            ["P1"] = new[] { "A", "B", "C", "SHARED" },
            ["P2"] = new[] { "D", "SHARED" },
        });
        Assert.Contains("SHARED", ByLeading(gs, "P1").RazorPeptides);
        Assert.Empty(ByLeading(gs, "P2").RazorPeptides);
    }

    [Fact]
    public void Razor_CascadingAssignment_MatchesOsprey()
    {
        // Osprey Example 2: P1{A,B,C,X,Y}, P2{D,X,Z}, P3{E,Y,Z}.
        // Round 1: P1 (3 unique) claims X,Y. Round 2: P2 claims Z.
        var gs = Groups(new Dictionary<string, string[]>
        {
            ["P1"] = new[] { "A", "B", "C", "X", "Y" },
            ["P2"] = new[] { "D", "X", "Z" },
            ["P3"] = new[] { "E", "Y", "Z" },
        });
        Assert.Equal(new[] { "X", "Y" }, ByLeading(gs, "P1").RazorPeptides.OrderBy(x => x, System.StringComparer.Ordinal));
        Assert.Equal(new[] { "Z" }, ByLeading(gs, "P2").RazorPeptides.OrderBy(x => x, System.StringComparer.Ordinal));
        Assert.Empty(ByLeading(gs, "P3").RazorPeptides);
    }

    [Fact]
    public void Razor_UniqueCountTie_PrefersLargerPeptideSet()
    {
        // Tie on unique count (all 1). Osprey breaks the tie by lowest group ID = largest peptide
        // set, so P2 (the 3-peptide group) claims BOTH shared peptides. The old accession-only
        // tiebreak would have given X -> P1 instead.
        var gs = Groups(new Dictionary<string, string[]>
        {
            ["P1"] = new[] { "A", "X" },
            ["P2"] = new[] { "B", "X", "Y" },
            ["P3"] = new[] { "C", "Y" },
        });
        Assert.Equal(new[] { "X", "Y" }, ByLeading(gs, "P2").RazorPeptides.OrderBy(x => x, System.StringComparer.Ordinal));
        Assert.Empty(ByLeading(gs, "P1").RazorPeptides);
        Assert.Empty(ByLeading(gs, "P3").RazorPeptides);
    }
}
