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
    public void BuildMap_SubstringMatch_WithIlEquivalence()
    {
        var path = WriteTemp(
            ">sp|P001|PROT1 desc1 GN=G1\nPEPTIDEKSAMPLERIEDINK\n" +
            ">sp|P002|PROT2 desc2 GN=G2\nAAAPEPTIDEKBBB\n");
        try
        {
            // LEDLNK is the I/L-normalized form of IEDINK (in P001 only); NOTHERE matches nothing.
            var map = FastaParser.BuildMap(path, new[] { "PEPTIDEK", "LEDLNK", "NOTHERE" });

            Assert.Equal(new[] { "P001", "P002" },
                Sorted(map.PeptideToProteins["PEPTIDEK"]));
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
