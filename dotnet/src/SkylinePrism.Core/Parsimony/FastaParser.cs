using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace SkylinePrism.Core.Parsimony;

/// <summary>A protein parsed from a FASTA file (fasta.py:ProteinEntry).</summary>
public sealed record ProteinEntry(
    string Accession, string Name, string? Gene, string Description, string Sequence, string DbType);

/// <summary>
/// FASTA parsing + FASTA-based peptide-&gt;protein mapping, porting fasta.py: parse_fasta,
/// strip_modifications, normalize_for_matching, build_peptide_protein_map_from_fasta. Maps each
/// DETECTED peptide (original modified sequence) to every protein whose sequence contains it as a
/// substring after stripping mods and I/L-normalizing. The resulting <see cref="PeptideProteinMap"/>
/// keys peptides by their original strings, so it drops into the existing parsimony engine.
/// </summary>
public static class FastaParser
{
    private static readonly Regex UniProt = new(@"^([sptr]{2})\|([^|]+)\|(\S+)\s*(.*)", RegexOptions.Compiled);
    private static readonly Regex Gn = new(@"GN=(\S+)", RegexOptions.Compiled);
    private static readonly Regex NcbiGi = new(@"^gi\|(\d+)\|[^|]+\|([^|]+)\|?\s*(.*)", RegexOptions.Compiled);
    private static readonly Regex UniProtDesc = new(@"^(.*?)\s*(?:OS=|OX=|GN=|PE=|SV=|$)", RegexOptions.Compiled);
    private static readonly Regex Brackets = new(@"\[[^\]]*\]", RegexOptions.Compiled);
    private static readonly Regex Parens = new(@"\([^)]*\)", RegexOptions.Compiled);
    private static readonly Regex Braces = new(@"\{[^}]*\}", RegexOptions.Compiled);

    public static Dictionary<string, ProteinEntry> Parse(string path)
    {
        var proteins = new Dictionary<string, ProteinEntry>(StringComparer.Ordinal);
        string? header = null;
        var seq = new StringBuilder();

        void Flush()
        {
            if (header is null)
                return;
            var e = ParseHeader(header, seq.ToString());
            proteins.TryAdd(e.Accession, e); // keep first on duplicate accession
        }

        foreach (var raw in File.ReadLines(path))
        {
            var line = raw.Trim();
            if (line.Length == 0)
                continue;
            if (line[0] == '>')
            {
                Flush();
                header = line[1..];
                seq.Clear();
            }
            else if (header is not null)
            {
                seq.Append(line);
            }
        }
        Flush();
        return proteins;
    }

    private static ProteinEntry ParseHeader(string header, string sequence)
    {
        var um = UniProt.Match(header);
        if (um.Success)
        {
            var db = um.Groups[1].Value;
            var acc = um.Groups[2].Value;
            var name = um.Groups[3].Value;
            var full = um.Groups[4].Value;
            var desc = UniProtDesc.Match(full).Groups[1].Value.Trim();
            if (desc.Length == 0)
                desc = name;
            var gn = Gn.Match(header);
            return new ProteinEntry(acc, name, gn.Success ? gn.Groups[1].Value : null, desc, sequence, db);
        }

        var gi = NcbiGi.Match(header);
        if (gi.Success)
        {
            var acc = gi.Groups[2].Value.Trim('|');
            var desc = gi.Groups[3].Value;
            return new ProteinEntry(acc, acc, null, desc.Length == 0 ? acc : desc, sequence, "ncbi");
        }

        var sp = header.IndexOf(' ');
        var accession = sp < 0 ? header : header[..sp];
        var description = sp < 0 ? accession : header[(sp + 1)..].Trim();
        return new ProteinEntry(accession, accession, null, description, sequence, "");
    }

    private static readonly Regex TrypsinSite = new(@"[KR](?!P)", RegexOptions.Compiled);
    private static readonly Regex TrypsinPSite = new(@"[KR]", RegexOptions.Compiled);

    /// <summary>
    /// In-silico digest (fasta.py:digest_protein). Trypsin (cleave after K/R unless followed by P)
    /// and trypsin/p; up to <paramref name="missedCleavages"/> missed cleavages, length-bounded.
    /// </summary>
    public static HashSet<string> DigestProtein(
        string sequence, string enzyme = "trypsin", int missedCleavages = 0, int minLength = 6, int maxLength = 30)
    {
        var e = enzyme.ToLowerInvariant();
        var site = e switch
        {
            "trypsin" => TrypsinSite,
            "trypsin/p" or "trypsin\\p" or "trypsinp" => TrypsinPSite,
            _ => throw new NotSupportedException($"Enzyme '{enzyme}' not supported (trypsin, trypsin/p)."),
        };

        var sites = new SortedSet<int> { 0, sequence.Length };
        foreach (Match m in site.Matches(sequence))
            sites.Add(m.Index + m.Length); // cleave C-terminal to the match
        var arr = sites.ToArray();

        var peptides = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 0; i < arr.Length - 1; i++)
            for (var j = i + 1; j < Math.Min(i + 2 + missedCleavages, arr.Length); j++)
            {
                var len = arr[j] - arr[i];
                if (len >= minLength && len <= maxLength)
                    peptides.Add(sequence.Substring(arr[i], len));
            }
        return peptides;
    }

    /// <summary>
    /// Theoretical (distinct) peptide count per protein for iBAQ (fasta.py:get_theoretical_peptide_
    /// counts). If <paramref name="accessions"/> is given, only those proteins are counted.
    /// </summary>
    public static Dictionary<string, int> GetTheoreticalPeptideCounts(
        string fastaPath, ISet<string>? accessions, string enzyme = "trypsin",
        int missedCleavages = 0, int minLength = 6, int maxLength = 30)
    {
        var proteins = Parse(fastaPath);
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var (acc, entry) in proteins)
        {
            if (accessions is not null && !accessions.Contains(acc))
                continue;
            counts[acc] = DigestProtein(entry.Sequence, enzyme, missedCleavages, minLength, maxLength).Count;
        }
        return counts;
    }

    public static string StripModifications(string s)
    {
        var r = Brackets.Replace(s, "");
        r = Parens.Replace(r, "");
        r = Braces.Replace(r, "");
        var sb = new StringBuilder(r.Length);
        foreach (var c in r)
            if (char.IsUpper(c))
                sb.Append(c);
        return sb.ToString();
    }

    public static string NormalizeForMatching(string s, bool handleIlAmbiguity = true)
    {
        var n = StripModifications(s).ToUpperInvariant();
        return handleIlAmbiguity ? n.Replace('I', 'L') : n;
    }

    /// <summary>
    /// Build the peptide-&gt;protein map by substring-matching each detected peptide against every
    /// protein sequence (both I/L-normalized). O(peptides × proteins); parallelized over peptides.
    /// </summary>
    public static PeptideProteinMap BuildMap(
        string fastaPath, IEnumerable<string> detectedPeptides, bool handleIlAmbiguity = true)
    {
        var proteins = Parse(fastaPath);
        var accs = proteins.Keys.ToArray();
        var normSeq = new string[accs.Length];
        for (var i = 0; i < accs.Length; i++)
        {
            var s = proteins[accs[i]].Sequence.ToUpperInvariant();
            normSeq[i] = handleIlAmbiguity ? s.Replace('I', 'L') : s;
        }

        var peptides = detectedPeptides.Distinct().ToArray();
        var matchesPerPeptide = new List<string>[peptides.Length]; // accessions matched, per peptide

        Parallel.For(0, peptides.Length, p =>
        {
            var norm = NormalizeForMatching(peptides[p], handleIlAmbiguity);
            var hits = new List<string>();
            if (norm.Length > 0)
                for (var i = 0; i < normSeq.Length; i++)
                    if (normSeq[i].Contains(norm, StringComparison.Ordinal))
                        hits.Add(accs[i]);
            matchesPerPeptide[p] = hits;
        });

        var map = new PeptideProteinMap();
        for (var p = 0; p < peptides.Length; p++)
        {
            var hits = matchesPerPeptide[p];
            if (hits.Count == 0)
                continue;
            var pep = peptides[p];
            map.PeptideToProteins[pep] = new HashSet<string>(hits, StringComparer.Ordinal);
            foreach (var acc in hits)
            {
                if (!map.ProteinToPeptides.TryGetValue(acc, out var set))
                    map.ProteinToPeptides[acc] = set = new HashSet<string>(StringComparer.Ordinal);
                set.Add(pep);
                var e = proteins[acc];
                map.ProteinToName[acc] = e.Name;
                map.ProteinToGene[acc] = e.Gene ?? "";
                map.ProteinToDescription[acc] = e.Description;
            }
        }
        return map;
    }
}
