using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// A user-defined set of proteins to highlight on the dynamic-range plot - plasma contaminants, EV
/// markers, endothelial markers, and so on - with a color and a visibility toggle.
/// </summary>
public sealed class ProteinList
{
    public string Name { get; set; } = "New list";

    /// <summary>Hex color used for this list's points and labels, e.g. "#d62728".</summary>
    public string ColorHex { get; set; } = "#d62728";

    public bool Visible { get; set; } = true;

    /// <summary>Whether this list's members are labeled on the plot.</summary>
    public bool ShowLabels { get; set; }

    /// <summary>
    /// This list describes something to LOOK at, never a denominator to divide by, and
    /// <c>marker_normalization</c> refuses it.
    ///
    /// <para>Two kinds of list are display-only, for the same underlying reason: their abundance is the
    /// signal, not the scale. A readout like Hemolysis or Tubular contamination measures the problem -
    /// normalizing to it removes the evidence. A pathway measures biology - normalizing to it removes the
    /// finding. Both are perfectly good highlight sets and catastrophic normalizers.</para>
    ///
    /// <para>A flag rather than a note in the docs because the failure is silent: the run succeeds, the
    /// numbers look plausible, and what was regressed out is exactly what was being studied.</para>
    /// </summary>
    public bool DisplayOnly { get; set; }

    /// <summary>
    /// Which heading this panel sits under in the Predefined tab. Empty for a user's own list, which
    /// lives in its own tab and is not categorised - at 65 shipped panels a flat list is unreadable,
    /// while a handful of the user's own is not.
    /// </summary>
    public string Category { get; set; } = "";

    /// <summary>Separator between a member's match token and the label shown for it.</summary>
    public const char LabelSeparator = '=';

    /// <summary>
    /// The part of a member that is MATCHED, with any display label stripped.
    /// <para>
    /// Contaminant panels have to be keyed by accession - a bovine serum albumin entry written as its
    /// gene symbol, or even as its UniProt entry name, matches HUMAN albumin - and an accession tells a
    /// reader nothing. So a member may be written <c>P00761 = Trypsin (porcine)</c>: everything left of
    /// the separator is matched, everything right of it is only ever displayed.
    /// </para>
    /// </summary>
    public static string MatchToken(string member)
    {
        var i = member.IndexOf(LabelSeparator);
        return (i < 0 ? member : member[..i]).Trim();
    }

    /// <summary>
    /// What to show for a member: its label when it has one, otherwise the token itself. A separator with
    /// nothing usable after it falls back to the token rather than displaying a blank row - a member typed
    /// as <c>P02769 =</c> is an unfinished edit, and showing nothing for it hides which protein it is.
    /// </summary>
    public static string DisplayName(string member)
    {
        var i = member.IndexOf(LabelSeparator);
        if (i < 0)
            return member.Trim();

        var label = member[(i + 1)..].Trim();
        return label.Length > 0 ? label : MatchToken(member);
    }

    /// <summary>
    /// Members as the user typed them. Matched against accession, gene name AND protein name, so a list
    /// may be written in whichever form the user has it - "P02768", "ALB" and "sp|P02768|ALBU_HUMAN" all
    /// match the same protein.
    /// </summary>
    public List<string> Members { get; set; } = new();

    /// <summary>
    /// The lists PRISM ships. Available without the user having curated anything, in the plot picker
    /// and as <c>marker_normalization.protein_list</c>, and usable from the CLI on a machine that has
    /// no saved lists at all - which a per-user JSON alone could not be.
    /// <para>
    /// A user list of the same name wins, so shipping one never overrides a curated version of it.
    /// </para>
    /// </summary>
    /// <summary>
    /// The lists PRISM ships - see <see cref="BuiltInProteinPanels"/> for the panels themselves and for
    /// what separates a normalizer from a readout. Available with no curation, in the plot picker and as
    /// <c>marker_normalization.protein_list</c>, and on a machine with no saved lists at all.
    /// <para>
    /// A user list of the same name wins that name; the shipped one remains selectable under its
    /// <see cref="ProteinListSet.ShippedSuffix"/> form.
    /// </para>
    /// </summary>
    public static IReadOnlyList<ProteinList> BuiltIns => BuiltInProteinPanels.All;

    /// <summary>The shipped EV panel built for NORMALIZING - the 18 the method was validated on.</summary>
    public const string EvMarkersName = "EV markers (core)";

    /// <summary>The broad EV-association panel, for highlighting rather than normalizing.</summary>
    public const string EvExtendedName = "EV markers (extended)";

    /// <summary>Structural glomerular markers - see the list's own comment on what is left out and why.</summary>
    public const string GlomerulusName = "Glomerulus";

    /// <summary>Tubular markers, for spotting carry-over on the plot. Never a normalizer.</summary>
    public const string TubularContaminationName = "Tubular contamination";

    public ProteinList Clone() => new()
    {
        Name = Name,
        ColorHex = ColorHex,
        Visible = Visible,
        ShowLabels = ShowLabels,
        DisplayOnly = DisplayOnly,
        Category = Category,
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

    /// <summary>
    /// Per-user state for the SHIPPED panels, keyed by panel name: whether each is shown on the plot and
    /// whether its members are labeled.
    /// <para>
    /// An overlay rather than a copy, deliberately. Ticking a shipped panel must not fork its membership
    /// - the whole value of a shipped panel is that it means the same thing on every machine, which is
    /// what makes it citable. Storing only the two view flags means a panel the user has turned on still
    /// picks up any correction to its member list in a later release.
    /// </para>
    /// </summary>
    public Dictionary<string, ShippedListState> Shipped { get; set; } =
        new(StringComparer.OrdinalIgnoreCase);

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
    /// Index the visible lists for fast lookup - the user's and the shipped ones alike, so a panel PRISM
    /// ships can highlight proteins on the plot without being copied into the user's file first (shipped
    /// lists start invisible, so nothing is colored until it is asked for). Returns a matcher from an
    /// <see cref="AbundanceEntry"/> to the first visible list that claims it (list order = priority, so
    /// an earlier list wins a protein that appears in two, and the user's lists come first).
    /// </summary>
    public ProteinListMatcher BuildMatcher() => new(WithBuiltIns().Where(l => l.Visible).ToList());

    /// <summary>
    /// A matcher over ONE named list, whatever its <see cref="ProteinList.Visible"/> state.
    /// <para>
    /// Visibility means "highlight this on the Dynamic Range plot" and has nothing to do with
    /// normalization. Going through <see cref="BuildMatcher"/> for a marker set made every shipped
    /// panel unusable as a normalizer the moment they were changed to ship unticked: the list resolved
    /// by name, then the visibility filter dropped it and the stage reported zero markers found.
    /// </para>
    /// </summary>
    public static ProteinListMatcher MatcherFor(ProteinList list) => new(new[] { list });

    /// <summary>Suffix a shipped list takes when the user has one of the same name.</summary>
    public const string ShippedSuffix = " (PRISM)";

    /// <summary>
    /// The user's lists followed by every shipped list, which is what a picker should offer and what a
    /// config name resolves against.
    ///
    /// <para>A user list of the same name still WINS that name - curating over a shipped default is the
    /// point, and a config saying <c>EV markers</c> must keep meaning what it meant. But the shipped one
    /// is no longer dropped: it comes back as <c>"EV markers (PRISM)"</c>, so it stays selectable.</para>
    ///
    /// <para>Dropping it was a real trap. A user whose saved <c>EV Markers</c> is a 34-protein set
    /// curated for HIGHLIGHTING on the dynamic-range plot would silently normalize against that instead
    /// of the 18-protein panel the method was validated on - same name, different purpose, different
    /// answer, and no way to reach the shipped one or any sign it existed.</para>
    /// </summary>
    public IReadOnlyList<ProteinList> WithBuiltIns()
    {
        var all = new List<ProteinList>(Lists);
        foreach (var builtIn in ProteinList.BuiltIns)
        {
            if (!all.Any(l => string.Equals(l.Name, builtIn.Name, StringComparison.OrdinalIgnoreCase)))
            {
                all.Add(WithUserState(builtIn, builtIn.Name));
                continue;
            }
            // Shadowed: offer it under a distinct name, unless the user has taken that one too.
            var name = builtIn.Name + ShippedSuffix;
            if (!all.Any(l => string.Equals(l.Name, name, StringComparison.OrdinalIgnoreCase)))
                all.Add(WithUserState(builtIn, name));
        }
        return all;
    }

    /// <summary>
    /// A shipped panel under the name it is being offered as, carrying the user's view flags for it.
    /// Always a clone: the static definition must never be mutated by anything a user does.
    /// </summary>
    private ProteinList WithUserState(ProteinList builtIn, string name)
    {
        var copy = builtIn.Clone();
        copy.Name = name;
        if (Shipped.TryGetValue(name, out var state))
        {
            copy.Visible = state.Visible;
            copy.ShowLabels = state.ShowLabels;
        }
        return copy;
    }

    /// <summary>
    /// Record the view flags for a shipped panel, or drop the record when it is back at the default of
    /// hidden and unlabeled - so the saved file stays a list of deliberate choices.
    /// </summary>
    public void SetShippedState(string name, bool visible, bool showLabels)
    {
        if (!visible && !showLabels)
            Shipped.Remove(name);
        else
            Shipped[name] = new ShippedListState { Visible = visible, ShowLabels = showLabels };
    }

    /// <summary>
    /// Find a list by name among the user's and the shipped ones, or null. Case-insensitive, because
    /// the name is typed into a config file by hand.
    /// </summary>
    public ProteinList? Find(string? name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return null;
        var all = WithBuiltIns();
        return all.FirstOrDefault(l => string.Equals(l.Name, name, StringComparison.OrdinalIgnoreCase))
            ?? (Aliases.TryGetValue(name!.Trim(), out var actual)
                ? all.FirstOrDefault(l => string.Equals(l.Name, actual, StringComparison.OrdinalIgnoreCase))
                : null);
    }

    /// <summary>
    /// Names that used to identify a shipped panel and still have to resolve. A config naming a panel
    /// must not stop working because the panel was renamed - it would abort the run with "not found"
    /// rather than doing anything the user could act on. Only consulted after an exact match fails, so a
    /// user list may take one of these names back.
    /// </summary>
    private static readonly Dictionary<string, string> Aliases = new(StringComparer.OrdinalIgnoreCase)
    {
        // Shipped as plain "EV markers" in dotnet-v26.19.0 and v26.20.0, before the core/extended split.
        ["EV markers"] = ProteinList.EvMarkersName,
    };

    /// <summary>
    /// The list a run should normalize against: an explicit members FILE wins over a name, because a
    /// path is reproducible on another machine and a name depends on that machine's saved lists.
    /// Returns null when neither is given; throws when what was asked for does not exist, rather than
    /// carrying on without the normalization the config asked for.
    /// </summary>
    public static ProteinList? Resolve(string? name, string? membersFile, string? setPath = null)
    {
        if (!string.IsNullOrWhiteSpace(membersFile))
        {
            if (!File.Exists(membersFile))
                throw new FileNotFoundException(
                    $"marker_normalization.protein_list_file not found: {membersFile}", membersFile);
            return new ProteinList
            {
                Name = Path.GetFileNameWithoutExtension(membersFile),
                Members = ReadMembersFile(membersFile!),
            };
        }
        if (string.IsNullOrWhiteSpace(name))
            return null;

        var found = Load(setPath).Find(name);
        if (found is { DisplayOnly: true })
            throw new InvalidOperationException(
                $"marker_normalization.protein_list '{found.Name}' is a display-only panel and cannot "
                + "define a normalization. Its abundance is the thing being looked for - a readout "
                + "measures a problem, a pathway measures biology - so dividing by it removes exactly "
                + "what you are trying to see. Highlight it on a plot instead, and normalize against a "
                + "panel that tracks how much material was captured.");
        if (found is null)
            throw new InvalidOperationException(
                $"marker_normalization.protein_list '{name}' was not found. Available: "
                + string.Join(", ", Load(setPath).WithBuiltIns().Select(l => $"'{l.Name}'"))
                + ". Use marker_normalization.protein_list_file to point at a file of members instead.");
        return found;
    }
}

/// <summary>The user's view flags for one shipped panel. See <see cref="ProteinListSet.Shipped"/>.</summary>
public sealed class ShippedListState
{
    public bool Visible { get; set; }

    public bool ShowLabels { get; set; }
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
                l.Members.Select(ProteinList.MatchToken).SelectMany(Tokenize).Where(t => t.Length > 0),
                StringComparer.OrdinalIgnoreCase))).ToList();
    }

    public IReadOnlyList<ProteinList> Lists => _lists.Select(l => l.List).ToList();

    /// <summary>
    /// Whether any list claims a protein described by its raw identifier columns - what the pipeline
    /// has in hand (leading_protein / leading_gene_name / leading_name), as opposed to the plot's
    /// <see cref="AbundanceEntry"/>. Same matching rules either way, deliberately: a list that
    /// highlights a protein on the plot has to select the same protein when it normalizes.
    /// </summary>
    public ProteinList? Match(string? accession, string? gene, string? proteinName)
        => Match(new AbundanceEntry(
            Key: accession ?? gene ?? proteinName ?? "", Label: "", Accession: accession, Gene: gene,
            ProteinName: proteinName, MeanAbundance: 0, Log10Abundance: 0, Rank: 0, SamplesUsed: 0));

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
            foreach (var token in Tokenize(entry.Gene!))
                yield return token;
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

        // An indistinguishable protein group names all its members, slash-joined:
        // "H2AC11 / H2AC18 / H2AJ / H2AC14". Splitting only on '|' left those matchable by the whole
        // string alone, so a panel naming any one member missed the group entirely - which is how a
        // 158-member histone panel found four proteins in a cohort that plainly had more.
        foreach (var member in trimmed.Split('/', StringSplitOptions.RemoveEmptyEntries))
        {
            var m = member.Trim();
            if (m.Length > 0 && m != trimmed)
            {
                yield return m;
                yield return StripIsoform(m);
                var us = m.IndexOf('_');
                if (us > 0)
                    yield return m[..us];
            }
        }

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
            var underscore = trimmed.IndexOf('_');
            if (underscore > 0)
                yield return trimmed[..underscore]; // H4_HUMAN / H4_MOUSE -> H4
        }
    }

    // P02768-2 -> P02768: isoform suffixes are rarely carried in curated lists.
    private static string StripIsoform(string accession)
    {
        var dash = accession.IndexOf('-');
        return dash > 0 ? accession[..dash] : accession;
    }
}
