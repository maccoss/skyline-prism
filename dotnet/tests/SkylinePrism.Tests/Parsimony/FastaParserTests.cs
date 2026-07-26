using System;
using System.IO;
using SkylinePrism.Core.Parsimony;
using Xunit;

namespace SkylinePrism.Tests.Parsimony;

/// <summary>FASTA parsing, modification stripping, and substring-based peptide->protein mapping.</summary>
public class FastaParserTests
{
    [Fact]
    public void StripModifications_RemovesBracketAndParenMods()
    {
        Assert.Equal("PEPTIDEK", FastaParser.StripModifications("PEPT[+57.02]IDEK"));
        Assert.Equal("MPEPK", FastaParser.StripModifications("M(ox)PEPK"));
        Assert.Equal("PECIDEK", FastaParser.StripModifications("PEC(unimod:4)IDEK"));
    }

    [Fact]
    public void Parse_ReadsUniProtHeaderFields()
    {
        var path = WriteTemp(">sp|P001|PROT1_HUMAN My description OS=Homo sapiens GN=G1 PE=1 SV=1\nPEPTIDEKSAMPLERIEDINK\n");
        try
        {
            var proteins = FastaParser.Parse(path);
            Assert.True(proteins.ContainsKey("P001"));
            var e = proteins["P001"];
            Assert.Equal("PROT1_HUMAN", e.Name);
            Assert.Equal("G1", e.Gene);
            Assert.Equal("My description", e.Description);
            Assert.Equal("PEPTIDEKSAMPLERIEDINK", e.Sequence);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void BuildMap_EnzymeAware_SubstringMatch_WithIlEquivalence()
    {
        var path = WriteTemp(
            ">sp|P001|PROT1 desc1 GN=G1\nPEPTIDEKSAMPLERIEDINK\n" +
            ">sp|P002|PROT2 desc2 GN=G2\nAAAPEPTIDEKBBB\n");
        try
        {
            // PEPTIDEK is a valid tryptic peptide of P001 (protein N-terminus .. K) but only a
            // SUBSTRING of P002 (there it is preceded by A, not a cleavage site) -> P001 only.
            // LEDLNK is the I/L-normalized form of IEDINK (in P001 only); NOTHERE matches nothing.
            var map = FastaParser.BuildMap(path, new[] { "PEPTIDEK", "LEDLNK", "NOTHERE" });

            Assert.Equal(new[] { "P001" }, Sorted(map.PeptideToProteins["PEPTIDEK"]));
            Assert.Equal(new[] { "P001" }, Sorted(map.PeptideToProteins["LEDLNK"]));
            Assert.False(map.PeptideToProteins.ContainsKey("NOTHERE"));

            Assert.Contains("PEPTIDEK", map.ProteinToPeptides["P001"]);
            Assert.Contains("LEDLNK", map.ProteinToPeptides["P001"]);
            Assert.Equal("G1", map.ProteinToGene["P001"]);
            Assert.Equal("PROT1", map.ProteinToName["P001"]);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void BuildMap_SpecificityNone_RestoresPureSubstring()
    {
        var path = WriteTemp(
            ">sp|P001|PROT1 desc1 GN=G1\nPEPTIDEKSAMPLERIEDINK\n" +
            ">sp|P002|PROT2 desc2 GN=G2\nAAAPEPTIDEKBBB\n");
        try
        {
            // With the enzyme check disabled, PEPTIDEK is a substring of both proteins (legacy behavior).
            var map = FastaParser.BuildMap(path, new[] { "PEPTIDEK" }, enzymeSpecificity: "none");
            Assert.Equal(new[] { "P001", "P002" }, Sorted(map.PeptideToProteins["PEPTIDEK"]));
        }
        finally { File.Delete(path); }
    }

    // Alpha-synuclein (SNCA) / beta-synuclein (SNCB) N-termini. AKEGVVAAAEK is a substring of BOTH but
    // tryptic ONLY in SNCA (SNCA: ...GLS-K|AKE..., SNCB: ...GLS-M|AKE...). The definitive bug example.
    private const string Synuclein =
        ">sp|P37840|SYUA_HUMAN Alpha-synuclein GN=SNCA\nMDVFMKGLSKAKEGVVAAAEKTKQG\n" +
        ">sp|Q16143|SYUB_HUMAN Beta-synuclein GN=SNCB\nMDVFMKGLSMAKEGVVAAAEKTKQG\n";

    [Fact]
    public void BuildMap_SynucleinPhantom_RemovedByDefault()
    {
        var path = WriteTemp(Synuclein);
        try
        {
            var map = FastaParser.BuildMap(path, new[] { "AKEGVVAAAEK" });
            Assert.Equal(new[] { "P37840" }, Sorted(map.PeptideToProteins["AKEGVVAAAEK"]));
            Assert.False(map.ProteinToPeptides.ContainsKey("Q16143")); // SNCB never claims the phantom
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void BuildMap_SynucleinPhantom_KeptWhenSpecificityNone()
    {
        var path = WriteTemp(Synuclein);
        try
        {
            var map = FastaParser.BuildMap(path, new[] { "AKEGVVAAAEK" }, enzymeSpecificity: "none");
            Assert.Equal(new[] { "P37840", "Q16143" }, Sorted(map.PeptideToProteins["AKEGVVAAAEK"]));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void CleavageBoundaries_TrypsinRulesAndInitiatorMet()
    {
        // MASTVGGK: cleave after K7 (=> 8); index 1 added for initiator-Met excision; termini 0 and 8.
        var b = FastaParser.CleavageBoundaries("MASTVGGK", "trypsin");
        Assert.Contains(0, b);
        Assert.Contains(1, b); // initiator methionine
        Assert.Contains(8, b);

        // K@2 is followed by P: trypsin does NOT cleave; trypsin/p does.
        Assert.DoesNotContain(3, FastaParser.CleavageBoundaries("AAKPEPTIDER", "trypsin"));
        Assert.Contains(3, FastaParser.CleavageBoundaries("AAKPEPTIDER", "trypsin/p"));
    }

    [Fact]
    public void DigestProtein_Trypsin_RespectsKpRuleAndLengthBounds()
    {
        // Cleave after K@7 and R@14 (neither followed by P); trailing "AAA" is too short.
        var peptides = FastaParser.DigestProtein("PEPTIDEKSAMPLERAAA", missedCleavages: 0, minLength: 6);
        Assert.Equal(2, peptides.Count);
        Assert.Contains("PEPTIDEK", peptides);
        Assert.Contains("SAMPLER", peptides);

        // KP is NOT a trypsin site, so the peptide spans the K-P.
        var kp = FastaParser.DigestProtein("PEPKPTIDER", missedCleavages: 0, minLength: 6);
        Assert.Contains("PEPKPTIDER", kp);
    }

    private static string[] Sorted(System.Collections.Generic.IEnumerable<string> s)
    {
        var a = new System.Collections.Generic.List<string>(s);
        a.Sort(StringComparer.Ordinal);
        return a.ToArray();
    }

    private static string WriteTemp(string content)
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_fasta_" + Guid.NewGuid().ToString("N") + ".fasta");
        File.WriteAllText(path, content);
        return path;
    }
}
