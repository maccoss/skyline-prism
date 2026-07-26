#nullable enable

using System;
using System.Linq;
using System.Xml.Linq;

namespace SkylinePrism.Skyline;

/// <summary>
/// Maps a Skyline document's digestion enzyme (Peptide Settings &gt; Digestion) to the enzyme name
/// PRISM's FASTA-mapping terminus check understands (parsimony.enzyme). Skyline exposes the active
/// enzyme over JSON-RPC as an XML element, e.g.:
/// <code>&lt;enzyme name="Trypsin" cut="KR" no_cut="P" sense="C" /&gt;</code>
/// where <c>cut</c> is the cleavage residues, <c>no_cut</c> the blocking residues, and <c>sense</c>
/// the side (<c>C</c> = cleave C-terminal / after, <c>N</c> = N-terminal / before).
/// </summary>
public static class SkylineDigestion
{
    /// <summary>Cleavage rule parsed from a Skyline enzyme XML element.</summary>
    public readonly record struct EnzymeRule(string Cut, string NoCut, string Sense);

    /// <summary>
    /// Parse a Skyline enzyme XML definition. Returns null when the XML is unparseable or uses a
    /// dual-terminal definition (<c>cut_c</c>/<c>cut_n</c>) that has no single PRISM equivalent.
    /// </summary>
    public static EnzymeRule? ParseEnzymeXml(string? xml)
    {
        if (string.IsNullOrWhiteSpace(xml))
            return null;
        XElement root;
        try
        {
            root = XElement.Parse(xml);
        }
        catch (System.Xml.XmlException)
        {
            return null;
        }

        var enzyme = root.Name.LocalName == "enzyme"
            ? root
            : root.Descendants().FirstOrDefault(e => e.Name.LocalName == "enzyme");
        if (enzyme is null)
            return null;

        var cut = (string?)enzyme.Attribute("cut");
        if (string.IsNullOrEmpty(cut))
            return null; // dual-terminal (cut_c/cut_n) or unspecified - fall back to the config default

        var noCut = (string?)enzyme.Attribute("no_cut") ?? "";
        var sense = (string?)enzyme.Attribute("sense") ?? "C";
        return new EnzymeRule(cut, noCut, sense);
    }

    /// <summary>
    /// Map a Skyline cleavage rule to a PRISM enzyme name (see fasta.py ENZYME_RULES), or null when
    /// there is no equivalent (the caller then keeps the config default). Trypsin vs Trypsin/P is
    /// distinguished by whether P blocks cleavage; other enzymes are matched by cut residues + sense.
    /// </summary>
    public static string? MapToPrismEnzyme(EnzymeRule rule)
    {
        // Normalize the cut residues to a sorted, distinct, uppercase key.
        var cut = new string(rule.Cut.Where(char.IsLetter).Select(char.ToUpperInvariant)
            .Distinct().OrderBy(ch => ch).ToArray());
        var sense = (rule.Sense ?? "C").Trim().ToUpperInvariant();
        var blocksProline = (rule.NoCut ?? "").ToUpperInvariant().Contains('P');

        return (cut, sense) switch
        {
            ("KR", "C") => blocksProline ? "trypsin" : "trypsin/p",
            ("K", "C") => "lysc",
            ("K", "N") => "lysn",
            ("R", "C") => "argc",
            ("D", "N") => "aspn",
            ("E", "C") => "gluc",
            ("FLWY", "C") => "chymotrypsin", // Skyline "FWYL" sorts to FLWY
            ("FWY", "C") => "chymotrypsin",
            _ => null,
        };
    }

    /// <summary>Convenience: parse XML and map in one step (null when either step fails).</summary>
    public static string? PrismEnzymeFromXml(string? xml)
    {
        var rule = ParseEnzymeXml(xml);
        return rule is null ? null : MapToPrismEnzyme(rule.Value);
    }
}
