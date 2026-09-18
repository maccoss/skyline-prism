using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using SkylinePrism.Core.DifferentialAnalysis.Enrichment;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Parity tests for <see cref="Enrichment"/> against the explorer (payload shape, response parsing,
/// fold-enrichment, novelty), using a fake <see cref="IJsonPoster"/> so nothing touches the network.
/// </summary>
public class EnrichmentTests
{
    private sealed class FakePoster : IJsonPoster
    {
        private readonly string _response;
        public FakePoster(string response) => _response = response;
        public int Calls { get; private set; }
        public string? LastUrl { get; private set; }
        public IReadOnlyDictionary<string, object?>? LastPayload { get; private set; }

        public JsonElement Post(string url, IReadOnlyDictionary<string, object?> payload)
        {
            Calls++;
            LastUrl = url;
            LastPayload = payload;
            return JsonDocument.Parse(_response).RootElement.Clone();
        }
    }

    [Fact]
    public void CleanSymbols_SplitsDelimitedAndDedups()
    {
        Assert.Equal(new[] { "A", "a", "B" },
            Enrichment.CleanSymbols(new[] { "A", "a", "A", "", "nan", " B " }));
        Assert.Equal(new[] { "GENEA", "GENEB" }, Enrichment.CleanSymbols(new[] { "GENEA;GENEB" }));
    }

    [Fact]
    public void GProfiler_ComputesFoldEnrichmentAndSendsPayload()
    {
        const string resp = "{\"result\":[{\"source\":\"GO:BP\",\"native\":\"GO:0006915\"," +
            "\"name\":\"apoptotic process\",\"p_value\":0.001,\"term_size\":200,\"query_size\":10," +
            "\"intersection_size\":6,\"effective_domain_size\":20000}]}";
        var poster = new FakePoster(resp);
        var genes = Enumerable.Range(1, 10).Select(i => $"G{i}").ToArray();

        var terms = Enrichment.GProfiler(genes, null, poster);

        Assert.Single(terms);
        Assert.Equal("GO:BP", terms[0].Source);
        Assert.Equal("GO:0006915", terms[0].TermId);
        Assert.Equal(60.0, terms[0].FoldEnrichment, 12); // (6/10)/(200/20000)

        Assert.Equal(Enrichment.GProfilerUrl, poster.LastUrl);
        Assert.Equal("g_SCS", poster.LastPayload!["significance_threshold_method"]);
        Assert.Equal(true, poster.LastPayload["no_evidences"]);
        Assert.Equal(genes, ((List<string>)poster.LastPayload["query"]!).ToArray());
        Assert.False(poster.LastPayload.ContainsKey("background"));
    }

    [Fact]
    public void GProfiler_WithBackground_SetsCustomDomain()
    {
        var poster = new FakePoster("{\"result\":[]}");
        Enrichment.GProfiler(new[] { "A", "B" }, new[] { "A", "B", "C", "D" }, poster);

        Assert.Equal(new[] { "A", "B", "C", "D" }, ((List<string>)poster.LastPayload!["background"]!).ToArray());
        Assert.Equal("custom_annotated", poster.LastPayload["domain_scope"]);
    }

    [Fact]
    public void GProfiler_EmptyQuery_DoesNotPost()
    {
        var poster = new FakePoster("{\"result\":[]}");
        var terms = Enrichment.GProfiler(new string?[] { "", "nan" }, null, poster);
        Assert.Empty(terms);
        Assert.Equal(0, poster.Calls);
    }

    [Fact]
    public void OpenTargetsSearchDisease_KeepsOnlyDiseaseHits()
    {
        const string resp = "{\"data\":{\"search\":{\"hits\":[" +
            "{\"id\":\"EFO_0000253\",\"name\":\"amyotrophic lateral sclerosis\",\"entity\":\"disease\"}," +
            "{\"id\":\"CHEMBL_1\",\"name\":\"riluzole\",\"entity\":\"drug\"}]}}}";
        var hits = Enrichment.OpenTargetsSearchDisease("ALS", new FakePoster(resp));
        Assert.Single(hits);
        Assert.Equal(("EFO_0000253", "amyotrophic lateral sclerosis"), hits[0]);
    }

    [Fact]
    public void OpenTargetsDiseaseTargets_ParsesAndUppercases()
    {
        const string resp = "{\"data\":{\"disease\":{\"associatedTargets\":{\"rows\":[" +
            "{\"score\":0.9,\"target\":{\"approvedSymbol\":\"lrrk2\"}}," +
            "{\"score\":0.5,\"target\":{\"approvedSymbol\":\"\"}}," +
            "{\"score\":0.3,\"target\":{\"approvedSymbol\":\"SNCA\"}}]}}}}";
        var scores = Enrichment.OpenTargetsDiseaseTargets("EFO_0000253", new FakePoster(resp));
        Assert.Equal(2, scores.Count);
        Assert.Equal(0.9, scores["LRRK2"], 12);
        Assert.Equal(0.3, scores["SNCA"], 12);
        Assert.False(scores.ContainsKey(""));
    }

    [Fact]
    public void ClassifyNovelty_TagsAndSorts()
    {
        var scores = new Dictionary<string, double> { ["A"] = 0.5, ["B"] = 0.05 };
        var rows = Enrichment.ClassifyNovelty(new[] { "A", "B", "C" }, scores);

        // status ascending ("candidate-novel" < "known"), then score desc, unscored last.
        Assert.Equal("B", rows[0].Gene);
        Assert.Equal("candidate-novel", rows[0].Status);
        Assert.Equal(0.05, rows[0].AssocScore!.Value, 12);
        Assert.Equal("C", rows[1].Gene);
        Assert.Null(rows[1].AssocScore);
        Assert.Equal("A", rows[2].Gene);
        Assert.Equal("known", rows[2].Status);
    }
}
