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
    double FoldEnrichment);

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

    private static readonly string[] DefaultSources = { "GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG" };
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
    {
        var background = CleanSymbols(res.Rows.Select(r => geneForFeature(r.FeatureId)));
        bool Passes(DifferentialRow r)
        {
            if (!(r.AdjPValue < qCut && Math.Abs(r.LogFc) >= lfcCut))
                return false;
            return direction switch
            {
                EnrichmentDirection.Up => r.LogFc > 0,
                EnrichmentDirection.Down => r.LogFc < 0,
                _ => true,
            };
        }

        var sig = CleanSymbols(res.Rows.Where(Passes).Select(r => geneForFeature(r.FeatureId)));
        return (sig, background);
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
            ["no_evidences"] = true,
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
                GetDouble(t, "p_value"), termSize, querySize, intersection, domainSize, fold));
        }

        return terms.OrderBy(x => x.PValue).ToList();
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
