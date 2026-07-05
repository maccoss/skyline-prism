using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Parsimony;
using Xunit;

namespace SkylinePrism.Tests.Parsimony;

/// <summary>
/// FASTA-based peptide-&gt;protein edges. Osprey does not derive edges from a FASTA - it reads the
/// pre-assigned protein_ids column from the library (osprey-io library/diann.rs split_list), the
/// same model as PRISM's Skyline Protein-Accession path. PRISM's FASTA option re-derives the map by
/// substring + I/L equivalence (the convention library builders use to populate that column). This
/// verifies those edges are correct and feed the parsimony engine.
/// </summary>
public class ParsimonyFastaMapTests
{
    // PROTA contains PEPTIDEAK, SHAREDPEPTIDER, ELVISK. PROTB contains OTHERPEPTIDER, SHAREDPEPTIDER.
    private const string Fasta =
        ">sp|PROTA|PROTA_HUMAN Protein A OS=Homo sapiens GN=GENEA PE=1 SV=1\n" +
        "MKPEPTIDEAKRSHAREDPEPTIDERKELVISKAA\n" +
        ">sp|PROTB|PROTB_HUMAN Protein B OS=Homo sapiens GN=GENEB PE=1 SV=1\n" +
        "MKOTHERPEPTIDERKSHAREDPEPTIDERKQQ\n";

    private static string WriteFasta()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_fasta_" + Guid.NewGuid().ToString("N") + ".fasta");
        File.WriteAllText(path, Fasta);
        return path;
    }

    [Fact]
    public void BuildMap_SubstringAndIlEdges()
    {
        var path = WriteFasta();
        try
        {
            var detected = new[]
            {
                "PEPTIDEAK",        // unique to PROTA
                "OTHERPEPTIDER",    // unique to PROTB
                "SHAREDPEPTIDER",   // in both -> shared
                "ELVLSK",           // protein has ELVISK (I); matches PROTA only via I/L equivalence
                "NOTINANYPROTEIN",  // no match -> dropped
            };
            var map = FastaParser.BuildMap(path, detected);

            Assert.Equal(new[] { "PROTA" }, map.PeptideToProteins["PEPTIDEAK"].OrderBy(x => x, StringComparer.Ordinal));
            Assert.Equal(new[] { "PROTB" }, map.PeptideToProteins["OTHERPEPTIDER"].OrderBy(x => x, StringComparer.Ordinal));
            Assert.Equal(new[] { "PROTA", "PROTB" },
                map.PeptideToProteins["SHAREDPEPTIDER"].OrderBy(x => x, StringComparer.Ordinal));
            Assert.Equal(new[] { "PROTA" }, map.PeptideToProteins["ELVLSK"].OrderBy(x => x, StringComparer.Ordinal));
            Assert.False(map.PeptideToProteins.ContainsKey("NOTINANYPROTEIN")); // unmatched peptides dropped
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void BuildMap_IsOrderIndependent()
    {
        var path = WriteFasta();
        try
        {
            var detected = new[] { "PEPTIDEAK", "OTHERPEPTIDER", "SHAREDPEPTIDER", "ELVLSK" };
            var m1 = FastaParser.BuildMap(path, detected);
            var m2 = FastaParser.BuildMap(path, Enumerable.Reverse(detected).ToArray());
            foreach (var pep in detected)
                Assert.Equal(
                    m1.PeptideToProteins[pep].OrderBy(x => x, StringComparer.Ordinal),
                    m2.PeptideToProteins[pep].OrderBy(x => x, StringComparer.Ordinal));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void FastaEdges_FeedParsimony_IntoOspreyGroups()
    {
        var path = WriteFasta();
        try
        {
            var detected = new[] { "PEPTIDEAK", "OTHERPEPTIDER", "SHAREDPEPTIDER", "ELVLSK" };
            var groups = ParsimonyEngine.ComputeProteinGroups(FastaParser.BuildMap(path, detected));

            Assert.Equal(2, groups.Count);
            var a = groups.Single(g => g.LeadingProtein == "PROTA");
            var b = groups.Single(g => g.LeadingProtein == "PROTB");

            Assert.Equal(new[] { "ELVLSK", "PEPTIDEAK" },
                a.UniquePeptides.OrderBy(x => x, StringComparer.Ordinal));
            Assert.Equal(new[] { "OTHERPEPTIDER" }, b.UniquePeptides.OrderBy(x => x, StringComparer.Ordinal));
            // Default all_groups: the shared peptide belongs to both groups.
            Assert.Contains("SHAREDPEPTIDER", a.AllMappedPeptides);
            Assert.Contains("SHAREDPEPTIDER", b.AllMappedPeptides);
        }
        finally { File.Delete(path); }
    }
}
