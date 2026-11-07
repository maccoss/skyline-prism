using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace SkylinePrism.Core.DifferentialAnalysis.Enrichment;

/// <summary>Transport for a JSON POST; injectable so callers/tests avoid the network.</summary>
public interface IJsonPoster
{
    /// <summary>POST <paramref name="payload"/> as JSON to <paramref name="url"/> and return the parsed response root.</summary>
    JsonElement Post(string url, IReadOnlyDictionary<string, object?> payload);
}

/// <summary>Which direction of change counts as a significant hit for enrichment.</summary>
public enum EnrichmentDirection
{
    Both,
    Up,
    Down,
}

/// <summary>One over-representation term from g:Profiler.</summary>
public sealed record EnrichmentTerm(
    string Source,
    string TermId,
    string TermName,
    double PValue,
    int TermSize,
    int QuerySize,
    int IntersectionSize,
    int DomainSize,
    double FoldEnrichment,
    IReadOnlyList<string> IntersectingGenes)
{
    /// <summary>
    /// Which hit list the term came from: features higher in B (<see cref="EnrichmentDirection.Up"/>),
    /// lower in B (<see cref="EnrichmentDirection.Down"/>), or both pooled.
    /// </summary>
    public EnrichmentDirection Direction { get; init; } = EnrichmentDirection.Both;
}

/// <summary>One g:Profiler query of an <see cref="EnrichmentRun"/>: its direction and how many genes it sent.</summary>
public sealed record EnrichmentQuery(EnrichmentDirection Direction, int SignificantGenes, int Terms);

/// <summary>
/// What <see cref="Enrichment.RunByDirection"/> did: the organism queried, the background size, one
/// entry per query, and every term found, each tagged with its <see cref="EnrichmentTerm.Direction"/>.
/// </summary>
public sealed record EnrichmentRun(
    string Organism,
    string OrganismSource,
    bool Pooled,
    int Background,
    IReadOnlyList<EnrichmentQuery> Queries,
    IReadOnlyList<EnrichmentTerm> Terms)
{
    /// <summary>The genes submitted across every query.</summary>
    public int SignificantGenes => Queries.Sum(q => q.SignificantGenes);
}

/// <summary>One gene's novelty classification against Open Targets association scores.</summary>
public sealed record NoveltyRow(string Gene, double? AssocScore, string Status);

/// <summary>
/// Functional-enrichment and target-novelty helpers, ported from the PRISM Differential Explorer:
/// gene-symbol cleaning, g:Profiler over-representation analysis, Open Targets disease search and
/// associated-target scores, and novelty classification. HTTP is injected via
/// <see cref="IJsonPoster"/>. The enrichment universe is the tested proteome (not the genome), which
/// matters in proteomics where the detectable proteome is a biased slice.
/// </summary>
public static class Enrichment
{
    public const string GProfilerUrl = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/";
    public const string OpenTargetsUrl = "https://api.platform.opentargets.org/api/v4/graphql";

    /// <summary>
    /// The g:Profiler sources queried when none are named - public so a report records the set that
    /// was actually queried rather than a copy of it.
    /// </summary>
    public static IReadOnlyList<string> DefaultSources { get; } = new[] { "GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG" };
    private static readonly Regex SymbolSplit = new(@"[;,/|\s]+", RegexOptions.Compiled);

    /// <summary>
    /// De-duplicate and stringify gene symbols (order preserving), splitting delimited multi-gene
    /// fields on <c>; , / |</c> and whitespace and dropping empty / "nan" / "none". Real HGNC symbols
    /// contain none of those characters, so a delimited value must be split before use.
    /// </summary>
    public static List<string> CleanSymbols(IEnumerable<string?> genes)
    {
        var outList = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var g in genes)
        {
            if (g is null)
                continue;
            foreach (var tok in SymbolSplit.Split(g.Trim()))
            {
                var s = tok.Trim();
                if (s.Length == 0 || s.Equals("nan", StringComparison.OrdinalIgnoreCase)
                    || s.Equals("none", StringComparison.OrdinalIgnoreCase) || seen.Contains(s))
                    continue;
                seen.Add(s);
                outList.Add(s);
            }
        }

        return outList;
    }

    /// <summary>
    /// Significant + background gene symbols from a differential result. Background is the genes of all
    /// tested features (the enrichment universe); significant is the genes of features passing the
    /// adjusted-p and log-fold-change cutoffs, filtered by <paramref name="direction"/>.
    /// <paramref name="geneForFeature"/> maps a feature id to its gene (null when unmapped).
    /// </summary>
    public static (List<string> Significant, List<string> Background) SigAndBackgroundGenes(
        DifferentialResult res, Func<string, string?> geneForFeature, double qCut, double lfcCut,
        EnrichmentDirection direction = EnrichmentDirection.Both)
        => SigAndBackgroundGenes(res, geneForFeature,
            new SignificanceRule { PThreshold = qCut, Log2FcThreshold = lfcCut }, direction);

    /// <summary>
    /// <see cref="SigAndBackgroundGenes(DifferentialResult, Func{string, string}, double, double, EnrichmentDirection)"/>
    /// under an explicit <see cref="SignificanceRule"/>, so the genes submitted are exactly the
    /// points the volcano drew red.
    /// </summary>
    public static (List<string> Significant, List<string> Background) SigAndBackgroundGenes(
        DifferentialResult res, Func<string, string?> geneForFeature, SignificanceRule rule,
        EnrichmentDirection direction = EnrichmentDirection.Both)
        => (SignificantGenes(res, geneForFeature, rule, direction), BackgroundGenes(res, geneForFeature));

    /// <summary>The enrichment universe: the genes of every tested feature.</summary>
    private static List<string> BackgroundGenes(DifferentialResult res, Func<string, string?> geneForFeature) =>
        CleanSymbols(res.Rows.Select(r => geneForFeature(r.FeatureId)));

    /// <summary>The genes of the features passing <paramref name="rule"/> and moving in <paramref name="direction"/>.</summary>
    private static List<string> SignificantGenes(DifferentialResult res, Func<string, string?> geneForFeature,
        SignificanceRule rule, EnrichmentDirection direction)
    {
        bool Passes(DifferentialRow r)
        {
            if (!rule.IsSignificant(r))
                return false;
            return direction switch
            {
                EnrichmentDirection.Up => r.LogFc > 0,
                EnrichmentDirection.Down => r.LogFc < 0,
                _ => true,
            };
        }

        return CleanSymbols(res.Rows.Where(Passes).Select(r => geneForFeature(r.FeatureId)));
    }

    /// <summary>The g:Profiler organism used when none is given and none can be inferred.</summary>
    public const string DefaultOrganism = "hsapiens";

    /// <summary>
    /// UniProt species mnemonics (the part after the underscore in an entry name such as
    /// <c>ICAM1_MOUSE</c>) and the g:Profiler organism id each maps to.
    /// </summary>
    public static IReadOnlyDictionary<string, string> OrganismBySpeciesCode { get; } =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["HUMAN"] = "hsapiens",
            ["MOUSE"] = "mmusculus",
            ["RAT"] = "rnorvegicus",
            ["BOVIN"] = "btaurus",
            ["PIG"] = "sscrofa",
            ["SHEEP"] = "oaries",
            ["HORSE"] = "ecaballus",
            ["CANLF"] = "clfamiliaris",
            ["FELCA"] = "fcatus",
            ["RABIT"] = "ocuniculus",
            ["MACMU"] = "mmulatta",
            ["CHICK"] = "ggallus",
            ["DANRE"] = "drerio",
            ["DROME"] = "dmelanogaster",
            ["CAEEL"] = "celegans",
            ["YEAST"] = "scerevisiae",
            ["SCHPO"] = "spombe",
            ["ARATH"] = "athaliana",
        };

    /// <summary>
    /// The g:Profiler organism the features come from, read off their UniProt entry names
    /// (<c>ICAM1_MOUSE</c> -> <c>mmusculus</c>): the species code most of them carry, or null when
    /// none carries a known code or no code is a clear majority.
    /// </summary>
    /// <remarks>
    /// <para>Up to now every query went to g:Profiler as <c>hsapiens</c>. g:Profiler matches symbols
    /// without regard to case, so mouse symbols mostly landed on their human namesakes and the result
    /// looked plausible - but annotations were the human ones, and mouse-only genes fell out of both
    /// the query and the background without a word.</para>
    /// <para>A majority rather than the first name, because a search database carries contaminants
    /// from other species (human keratins and trypsin in a mouse run, yeast enolase as a spike-in).
    /// Only the first entry of a <c>;</c>-separated group is read, for the same reason: a shared
    /// group's other members say nothing the leading one does not.</para>
    /// <para>The majority is taken over EVERY species code, mapped or not, and only the winner is
    /// mapped. Dropping the codes this table does not know first would let a contaminant decide: a
    /// <c>_XENTR</c> run with one mouse keratin would be queried as mouse and reported as inferred.
    /// An unmapped winner gives null, so the caller falls back and says so.</para>
    /// </remarks>
    public static string? InferOrganism(IEnumerable<string?> entryNames) =>
        MajoritySpeciesCode(entryNames) is { } code ? OrganismBySpeciesCode.GetValueOrDefault(code) : null;

    /// <summary>
    /// The species code (<c>MOUSE</c>, <c>XENTR</c>) carried by more than half of the entry names that
    /// carry one, whether or not <see cref="OrganismBySpeciesCode"/> maps it; null when none does.
    /// </summary>
    private static string? MajoritySpeciesCode(IEnumerable<string?> entryNames)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        var withCode = 0;
        foreach (var raw in entryNames)
        {
            if (string.IsNullOrWhiteSpace(raw))
                continue;
            var name = raw.Split(';')[0].Trim();
            var underscore = name.LastIndexOf('_');
            if (underscore <= 0 || underscore == name.Length - 1)
                continue;
            var code = name[(underscore + 1)..].ToUpperInvariant();
            if (!IsSpeciesCode(code))
                continue;
            withCode++;
            counts[code] = counts.GetValueOrDefault(code) + 1;
        }

        if (withCode == 0)
            return null;
        var (top, n) = counts.OrderByDescending(kv => kv.Value).Select(kv => (kv.Key, kv.Value)).First();
        return n * 2 > withCode ? top : null;
    }

    /// <summary>
    /// Whether a token has the shape of a UniProt species mnemonic: 3 to 5 letters or digits, at least
    /// one a letter (<c>MOUSE</c>, <c>RAT</c>, <c>9ACTN</c>). Keeps an accession behind a contaminant
    /// prefix (<c>CON__P12345</c>) or a free-text suffix from being counted as a species.
    /// </summary>
    private static bool IsSpeciesCode(string code) =>
        code.Length is >= 3 and <= 5 && code.All(char.IsAsciiLetterOrDigit) && code.Any(char.IsAsciiLetter);

    /// <summary>
    /// The organism to query: <paramref name="requested"/> when given, else the one inferred from
    /// <paramref name="entryNames"/>, else <see cref="DefaultOrganism"/>. <paramref name="source"/>
    /// says which, in words a report can print.
    /// </summary>
    public static string ResolveOrganism(string? requested, IEnumerable<string?> entryNames, out string source)
    {
        if (!string.IsNullOrWhiteSpace(requested))
        {
            source = "as requested";
            return requested.Trim();
        }

        var code = MajoritySpeciesCode(entryNames);
        if (code is not null && OrganismBySpeciesCode.TryGetValue(code, out var inferred))
        {
            source = "inferred from the UniProt entry names";
            return inferred;
        }

        // Said without naming --organism: this text reaches the tool's status line and the report it
        // writes, and the tool has no organism control to point at.
        source = code is null
            ? "default - no UniProt species code (such as _MOUSE) in the protein names"
            : $"default - most protein names are _{code}, which has no g:Profiler organism mapped here";
        return DefaultOrganism;
    }

    /// <summary>The token for a direction choice, as <c>--enrichment-direction</c> takes it and a report records it.</summary>
    public static string DirectionToken(bool pooled) => pooled ? "pooled" : "separate";

    /// <summary>
    /// Parse <c>--enrichment-direction</c>: <c>separate</c> (the default - up and down as two
    /// queries) or <c>pooled</c> (one query of every hit). Returns whether it is pooled.
    /// </summary>
    public static bool ParseDirectionToken(string? token) => (token ?? "separate").Trim().ToLowerInvariant() switch
    {
        "separate" or "separately" => false,
        "pooled" or "both" => true,
        var other => throw new ArgumentException(
            $"--enrichment-direction must be separate or pooled, not '{other}'."),
    };

    /// <summary>
    /// The bar color for a direction - red for higher in B, blue for lower, green when pooled - shared
    /// by the pane and the report so the two read the same way.
    /// </summary>
    public static string DirectionColor(EnrichmentDirection direction) => direction switch
    {
        EnrichmentDirection.Up => "#d62728",
        EnrichmentDirection.Down => "#1f77b4",
        _ => "#2ca02c",
    };

    /// <summary>What one direction's hit list is, named after arm B: "higher in KO".</summary>
    public static string DescribeDirection(EnrichmentDirection direction, string bLabel) => direction switch
    {
        EnrichmentDirection.Up => $"higher in {bLabel}",
        EnrichmentDirection.Down => $"lower in {bLabel}",
        _ => "either direction (pooled)",
    };

    /// <summary>
    /// Run enrichment for a contrast's hits: by default as two queries - the features higher in B and
    /// those lower in B - or, with <paramref name="pooled"/>, as one query of every hit.
    /// </summary>
    /// <remarks>
    /// <para>Separate is the default because pooling hides the biology it is meant to find. A pathway
    /// whose members all go up shares a pooled list with an unrelated one whose members all go down,
    /// each diluting the other's over-representation; and a pooled term cannot say which way its
    /// genes moved, which is usually the first question about it.</para>
    /// <para>Both queries share one background - every tested feature - so their p-values are on the
    /// same footing. A direction with no hits is recorded with zero genes and no request is sent.</para>
    /// </remarks>
    public static EnrichmentRun RunByDirection(DifferentialResult res, Func<string, string?> geneForFeature,
        SignificanceRule rule, bool pooled, string organism, string organismSource, IJsonPoster poster)
    {
        var directions = pooled
            ? new[] { EnrichmentDirection.Both }
            : new[] { EnrichmentDirection.Up, EnrichmentDirection.Down };
        var queries = new List<EnrichmentQuery>(directions.Length);
        var terms = new List<EnrichmentTerm>();
        // Built once and shared: it is every tested gene whichever direction is queried.
        var background = BackgroundGenes(res, geneForFeature);
        foreach (var direction in directions)
        {
            var sig = SignificantGenes(res, geneForFeature, rule, direction);
            var found = sig.Count == 0
                ? new List<EnrichmentTerm>()
                : GProfiler(sig, background, poster, organism: organism).Select(t => t with { Direction = direction }).ToList();
            queries.Add(new EnrichmentQuery(direction, sig.Count, found.Count));
            terms.AddRange(found);
        }

        return new EnrichmentRun(organism, organismSource, pooled, background.Count, queries, terms);
    }

    /// <summary>
    /// g:Profiler g:GOSt over-representation analysis. Returns terms sorted by p-value, or empty when
    /// the query is empty or nothing is enriched. A non-empty <paramref name="background"/> is sent as a
    /// custom statistical domain (recommended for proteomics).
    /// </summary>
    public static List<EnrichmentTerm> GProfiler(IEnumerable<string?> genes,
        IEnumerable<string?>? background, IJsonPoster poster, string[]? sources = null,
        string organism = "hsapiens", double userThreshold = 0.05)
    {
        var query = CleanSymbols(genes);
        if (query.Count == 0)
            return new List<EnrichmentTerm>();

        var payload = new Dictionary<string, object?>
        {
            ["organism"] = organism,
            ["query"] = query,
            ["sources"] = (sources ?? DefaultSources).ToList(),
            ["user_threshold"] = userThreshold,
            ["significance_threshold_method"] = "g_SCS",
            // Ask for the per-gene evidence so each term can report WHICH of the query's genes it
            // contains - that is what lets the UI show a term's member proteins. It makes the response
            // larger (one evidence-code list per query gene per term), which is why it was off before.
            ["no_evidences"] = false,
        };

        var bg = CleanSymbols(background ?? Enumerable.Empty<string?>());
        if (bg.Count > 0)
        {
            payload["background"] = bg;
            payload["domain_scope"] = "custom_annotated";
        }

        var data = poster.Post(GProfilerUrl, payload);
        if (data.ValueKind != JsonValueKind.Object || !data.TryGetProperty("result", out var result)
            || result.ValueKind != JsonValueKind.Array)
            return new List<EnrichmentTerm>();

        // Each term's `intersections` is a list aligned to the query's genes (in ensg order); a
        // non-empty evidence list at position j means query gene j is in that term. Recover the input
        // symbols from meta.genes_metadata.query.<the one query>: `ensgs` is that order, `mapping` is
        // {input symbol -> [ensg]}, inverted here to name each hit.
        var (ensgOrder, ensgToSymbol) = ReadQueryGeneMap(data);

        var terms = new List<EnrichmentTerm>();
        foreach (var t in result.EnumerateArray())
        {
            var termSize = GetInt(t, "term_size");
            var querySize = GetInt(t, "query_size");
            var intersection = GetInt(t, "intersection_size");
            var domainSize = GetInt(t, "effective_domain_size");
            var denom = domainSize != 0 ? (double)termSize / domainSize : 0.0;
            var fold = denom != 0.0 ? (double)intersection / querySize / denom : double.NaN;
            terms.Add(new EnrichmentTerm(
                GetString(t, "source"), GetString(t, "native"), GetString(t, "name"),
                GetDouble(t, "p_value"), termSize, querySize, intersection, domainSize, fold,
                IntersectingGenesOf(t, ensgOrder, ensgToSymbol)));
        }

        return terms.OrderBy(x => x.PValue).ToList();
    }

    /// <summary>
    /// The query's gene order and an ensg-&gt;input-symbol map from
    /// <c>meta.genes_metadata.query</c>. Empty when the response carries no evidence (e.g.
    /// <c>no_evidences=true</c>, or a minimal test stub), in which case terms report no member genes.
    /// </summary>
    private static (string[] EnsgOrder, Dictionary<string, string> EnsgToSymbol) ReadQueryGeneMap(
        JsonElement data)
    {
        if (!TryNavigate(data, out var query, "meta", "genes_metadata", "query")
            || query.ValueKind != JsonValueKind.Object)
            return (Array.Empty<string>(), new Dictionary<string, string>(StringComparer.Ordinal));

        // One query was sent (an unnamed list becomes "query_1"); take the first regardless of name.
        foreach (var q in query.EnumerateObject())
        {
            var ensgs = q.Value.TryGetProperty("ensgs", out var e) && e.ValueKind == JsonValueKind.Array
                ? e.EnumerateArray().Select(x => x.GetString() ?? string.Empty).ToArray()
                : Array.Empty<string>();

            var ensgToSymbol = new Dictionary<string, string>(StringComparer.Ordinal);
            if (q.Value.TryGetProperty("mapping", out var mapping) && mapping.ValueKind == JsonValueKind.Object)
                foreach (var sym in mapping.EnumerateObject())
                    if (sym.Value.ValueKind == JsonValueKind.Array)
                        foreach (var g in sym.Value.EnumerateArray())
                            if (g.ValueKind == JsonValueKind.String)
                                ensgToSymbol[g.GetString()!] = sym.Name;

            return (ensgs, ensgToSymbol);
        }

        return (Array.Empty<string>(), new Dictionary<string, string>(StringComparer.Ordinal));
    }

    /// <summary>The input symbols one term contains, from its <c>intersections</c> evidence lists.</summary>
    private static List<string> IntersectingGenesOf(
        JsonElement term, string[] ensgOrder, Dictionary<string, string> ensgToSymbol)
    {
        var genes = new List<string>();
        if (ensgOrder.Length == 0
            || !term.TryGetProperty("intersections", out var inter) || inter.ValueKind != JsonValueKind.Array)
            return genes;

        var j = 0;
        foreach (var evidence in inter.EnumerateArray())
        {
            if (j < ensgOrder.Length && evidence.ValueKind == JsonValueKind.Array
                && evidence.GetArrayLength() > 0
                && ensgToSymbol.TryGetValue(ensgOrder[j], out var symbol))
                genes.Add(symbol);
            j++;
        }

        return genes;
    }

    /// <summary>Search Open Targets for a disease by name; returns (efoId, label) for disease hits.</summary>
    public static List<(string EfoId, string Label)> OpenTargetsSearchDisease(string name, IJsonPoster poster)
    {
        if (string.IsNullOrWhiteSpace(name))
            return new List<(string, string)>();

        const string q = "query($q:String!){search(queryString:$q,entityNames:[\"disease\"]," +
            "page:{index:0,size:10}){hits{id name entity}}}";
        var data = poster.Post(OpenTargetsUrl, new Dictionary<string, object?>
        {
            ["query"] = q,
            ["variables"] = new Dictionary<string, object?> { ["q"] = name },
        });

        var results = new List<(string, string)>();
        if (TryNavigate(data, out var hits, "data", "search", "hits") && hits.ValueKind == JsonValueKind.Array)
        {
            foreach (var h in hits.EnumerateArray())
            {
                if (GetString(h, "entity") != "disease")
                    continue;
                var id = GetString(h, "id");
                var label = h.TryGetProperty("name", out var n) && n.ValueKind == JsonValueKind.String
                    ? n.GetString()! : id;
                results.Add((id, label));
            }
        }

        return results;
    }

    /// <summary>Map approved gene symbol (upper-cased) to Open Targets association score for a disease.</summary>
    public static Dictionary<string, double> OpenTargetsDiseaseTargets(string efoId, IJsonPoster poster,
        int size = 3000)
    {
        const string q = "query($efo:String!,$size:Int!){disease(efoId:$efo){associatedTargets(" +
            "page:{index:0,size:$size}){rows{score target{approvedSymbol}}}}}";
        var data = poster.Post(OpenTargetsUrl, new Dictionary<string, object?>
        {
            ["query"] = q,
            ["variables"] = new Dictionary<string, object?> { ["efo"] = efoId, ["size"] = size },
        });

        var scores = new Dictionary<string, double>(StringComparer.Ordinal);
        if (TryNavigate(data, out var rows, "data", "disease", "associatedTargets", "rows")
            && rows.ValueKind == JsonValueKind.Array)
        {
            foreach (var r in rows.EnumerateArray())
            {
                var sym = string.Empty;
                if (r.TryGetProperty("target", out var target) && target.ValueKind == JsonValueKind.Object)
                    sym = GetString(target, "approvedSymbol").Trim().ToUpperInvariant();
                if (sym.Length == 0)
                    continue;
                scores[sym] = r.TryGetProperty("score", out var sc) && sc.ValueKind == JsonValueKind.Number
                    ? sc.GetDouble() : 0.0;
            }
        }

        return scores;
    }

    /// <summary>
    /// Classify significant genes as "known" (association score at or above
    /// <paramref name="knownThreshold"/>) or "candidate-novel" (below, or absent from Open Targets),
    /// sorted by status then descending score (unscored genes last). Absence reflects database
    /// coverage, not proven biological novelty.
    /// </summary>
    public static List<NoveltyRow> ClassifyNovelty(IEnumerable<string?> sigGenes,
        IReadOnlyDictionary<string, double> assocScores, double knownThreshold = 0.1)
    {
        var rows = new List<NoveltyRow>();
        foreach (var g in CleanSymbols(sigGenes))
        {
            double? score = assocScores.TryGetValue(g.ToUpperInvariant(), out var s) ? s : null;
            var known = score.HasValue && score.Value >= knownThreshold;
            rows.Add(new NoveltyRow(g, score, known ? "known" : "candidate-novel"));
        }

        return rows
            .OrderBy(r => r.Status, StringComparer.Ordinal)
            .ThenBy(r => r.AssocScore.HasValue ? 0 : 1) // NaN/None last
            .ThenByDescending(r => r.AssocScore ?? double.NegativeInfinity)
            .ToList();
    }

    private static bool TryNavigate(JsonElement root, out JsonElement value, params string[] path)
    {
        value = root;
        foreach (var key in path)
        {
            if (value.ValueKind != JsonValueKind.Object || !value.TryGetProperty(key, out value))
            {
                value = default;
                return false;
            }
        }

        return true;
    }

    private static string GetString(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString()! : string.Empty;

    private static double GetDouble(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : double.NaN;

    private static int GetInt(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : 0;
}
