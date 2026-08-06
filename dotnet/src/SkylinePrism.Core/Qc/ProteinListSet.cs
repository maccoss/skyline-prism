using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// A user-defined set of proteins to highlight on the dynamic-range plot - plasma contaminants, EV
/// markers, endothelial markers, and so on - with a colour and a visibility toggle.
/// </summary>
public sealed class ProteinList
{
    public string Name { get; set; } = "New list";

    /// <summary>Hex colour used for this list's points and labels, e.g. "#d62728".</summary>
    public string ColorHex { get; set; } = "#d62728";

    public bool Visible { get; set; } = true;

    /// <summary>Whether this list's members are labelled on the plot.</summary>
    public bool ShowLabels { get; set; }

    /// <summary>
    /// Members as the user typed them. Matched against accession, gene name AND protein name, so a list
    /// may be written in whichever form the user has it - "P02768", "ALB" and "sp|P02768|ALBU_HUMAN" all
    /// match the same protein.
    /// </summary>
    public List<string> Members { get; set; } = new();

    public ProteinList Clone() => new()
    {
        Name = Name,
        ColorHex = ColorHex,
        Visible = Visible,
        ShowLabels = ShowLabels,
        Members = new List<string>(Members),
    };
}

/// <summary>
/// The user's protein lists, persisted once per user rather than per project so the same curated sets are
/// available in every output directory. Matching is deliberately forgiving: curated lists come from
/// papers, spreadsheets and colleagues, and are keyed on whatever identifier that source happened to use.
/// </summary>
public sealed class ProteinListSet
{
    public const string FileName = "protein-lists.json";

    private static readonly JsonSerializerOptions Json = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNameCaseInsensitive = true,
    };

    public List<ProteinList> Lists { get; set; } = new();

    /// <summary>Default location: per-user, so lists follow the user across projects and output folders.</summary>
    public static string DefaultPath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "SkylinePrism", FileName);

    public static ProteinListSet Load(string? path = null)
    {
        path ??= DefaultPath;
        try
        {
            if (!File.Exists(path))
                return new ProteinListSet();
            return JsonSerializer.Deserialize<ProteinListSet>(File.ReadAllText(path), Json)
                   ?? new ProteinListSet();
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            // A corrupt or unreadable file must not stop the tool from opening; the user simply starts
            // with no lists rather than seeing a crash on a plot tab.
            return new ProteinListSet();
        }
    }

    public void Save(string? path = null)
    {
        path ??= DefaultPath;
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(path, JsonSerializer.Serialize(this, Json));
    }

    /// <summary>
    /// Read members from a plain text/CSV file - one identifier per line, or the first column of a CSV.
    /// Blank lines, comment lines (# or ;) and a leading header line that is not an identifier are
    /// skipped. This is how a list that already exists as a spreadsheet column gets in without retyping.
    /// </summary>
    public static List<string> ReadMembersFile(string path)
    {
        var members = new List<string>();
        foreach (var raw in File.ReadAllLines(path))
        {
            var line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#') || line.StartsWith(';'))
                continue;
            var first = line.Split(',', '\t')[0].Trim().Trim('"');
            if (first.Length == 0)
                continue;
            // Skip a spreadsheet header row like "Accession" / "Gene" / "Protein".
            if (members.Count == 0 && IsHeaderWord(first))
                continue;
            members.Add(first);
        }
        return members.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
    }

    private static bool IsHeaderWord(string token) => token.ToLowerInvariant() is
        "accession" or "protein" or "proteins" or "gene" or "genes" or "gene name" or "uniprot"
        or "uniprot id" or "protein name" or "entry" or "id";

    /// <summary>
    /// Index the visible lists for fast lookup. Returns a matcher from an <see cref="AbundanceEntry"/> to
    /// the first visible list that claims it (list order = priority, so an earlier list wins a protein
    /// that appears in two).
    /// </summary>
    public ProteinListMatcher BuildMatcher() => new(Lists.Where(l => l.Visible).ToList());
}

/// <summary>Resolves which protein list (if any) an entry belongs to.</summary>
public sealed class ProteinListMatcher
{
    private readonly List<(ProteinList List, HashSet<string> Tokens)> _lists;

    internal ProteinListMatcher(IReadOnlyList<ProteinList> lists)
    {
        _lists = lists.Select(l => (
            l,
            new HashSet<string>(
                l.Members.SelectMany(Tokenize).Where(t => t.Length > 0),
                StringComparer.OrdinalIgnoreCase))).ToList();
    }

    public IReadOnlyList<ProteinList> Lists => _lists.Select(l => l.List).ToList();

    /// <summary>The first visible list claiming this entry, or null.</summary>
    public ProteinList? Match(AbundanceEntry entry)
    {
        foreach (var (list, tokens) in _lists)
            foreach (var candidate in Candidates(entry))
                if (tokens.Contains(candidate))
                    return list;
        return null;
    }

    // Every identifier an entry could reasonably be listed under.
    private static IEnumerable<string> Candidates(AbundanceEntry entry)
    {
        if (!string.IsNullOrWhiteSpace(entry.Accession))
        {
            yield return entry.Accession!;
            yield return StripIsoform(entry.Accession!);
        }
        if (!string.IsNullOrWhiteSpace(entry.Gene))
            yield return entry.Gene!;
        if (!string.IsNullOrWhiteSpace(entry.ProteinName))
            foreach (var token in Tokenize(entry.ProteinName!))
                yield return token;
        yield return entry.Key;
    }

    /// <summary>
    /// Split an identifier into everything it could be matched on. A FASTA-style name carries three:
    /// "sp|P02768|ALBU_HUMAN" -> the whole string, "P02768", "ALBU_HUMAN" (and "ALBU", since curated
    /// lists often drop the species suffix).
    /// </summary>
    private static IEnumerable<string> Tokenize(string identifier)
    {
        var trimmed = identifier.Trim();
        if (trimmed.Length == 0)
            yield break;
        yield return trimmed;

        var parts = trimmed.Split('|', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length > 1)
        {
            foreach (var part in parts)
            {
                var p = part.Trim();
                if (p.Length == 0 || p is "sp" or "tr")
                    continue;
                yield return p;
                yield return StripIsoform(p);
                var underscore = p.IndexOf('_');
                if (underscore > 0)
                    yield return p[..underscore]; // ALBU_HUMAN -> ALBU
            }
        }
        else
        {
            yield return StripIsoform(trimmed);
        }
    }

    // P02768-2 -> P02768: isoform suffixes are rarely carried in curated lists.
    private static string StripIsoform(string accession)
    {
        var dash = accession.IndexOf('-');
        return dash > 0 ? accession[..dash] : accession;
    }
}
