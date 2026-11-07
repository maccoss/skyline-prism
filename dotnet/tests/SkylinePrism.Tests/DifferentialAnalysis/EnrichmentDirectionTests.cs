using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.DifferentialAnalysis.Enrichment;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Enrichment of the hits higher in B and those lower in B as separate lists (the default), or pooled;
/// and the organism queried, inferred from the UniProt entry names instead of always human.
/// </summary>
public class EnrichmentDirectionTests
{
    /// <summary>Records every payload and answers with one term.</summary>
    private sealed class RecordingPoster : IJsonPoster
    {
        public List<IReadOnlyDictionary<string, object?>> Payloads { get; } = new();

        public JsonElement Post(string url, IReadOnlyDictionary<string, object?> payload)
        {
            Payloads.Add(payload);
            return JsonDocument.Parse(
                "{\"result\":[{\"source\":\"GO:BP\",\"native\":\"GO:0000001\",\"name\":\"a term\","
                + "\"p_value\":0.001,\"term_size\":10,\"query_size\":2,\"intersection_size\":2,"
                + "\"effective_domain_size\":100}]}").RootElement.Clone();
        }

        public List<string> Query(int i) => ((IEnumerable<string>)Payloads[i]["query"]!).ToList();
    }

    /// <summary>
    /// Eight features over 4 + 4 samples: UP1/UP2 four-fold higher in B, DN1/DN2 four-fold lower, the
    /// rest unchanged - with a little noise so the variance is not zero.
    /// </summary>
    private static (DifferentialResult Result, Func<string, string?> Gene) Contrast()
    {
        string[] ids = { "UP1", "UP2", "DN1", "DN2", "N1", "N2", "N3", "N4" };
        double[] effect = { 2, 2, -2, -2, 0, 0, 0, 0 };
        double[] noise = { 0.05, -0.04, 0.03, -0.06, 0.02, -0.01, 0.04, -0.03 };
        var expr = new double[ids.Length, 8];
        for (var i = 0; i < ids.Length; i++)
            for (var j = 0; j < 8; j++)
                expr[i, j] = 20 + i * 0.3 + (j >= 4 ? effect[i] : 0) + noise[(i + j) % noise.Length];
        var res = Differential.Run(expr, ids, new[] { 0, 1, 2, 3 }, new[] { 4, 5, 6, 7 },
            new DifferentialOptions { Prior = VariancePrior.Global });
        return (res, id => "G" + id);
    }

    private static readonly SignificanceRule Rule = new() { PThreshold = 0.05, Log2FcThreshold = 1 };

    [Fact]
    public void Separate_SendsTheUpAndDownHitsAsTwoLists_AndTagsEachTerm()
    {
        var (res, gene) = Contrast();
        var poster = new RecordingPoster();

        var run = Enrichment.RunByDirection(res, gene, Rule, pooled: false, "mmusculus", "test", poster);

        Assert.Equal(2, poster.Payloads.Count);
        Assert.Equal(new[] { "GUP1", "GUP2" }, poster.Query(0).OrderBy(g => g));
        Assert.Equal(new[] { "GDN1", "GDN2" }, poster.Query(1).OrderBy(g => g));
        // One background for both, so their p-values are on the same footing.
        Assert.Equal(8, ((IEnumerable<string>)poster.Payloads[0]["background"]!).Count());
        Assert.Equal(8, ((IEnumerable<string>)poster.Payloads[1]["background"]!).Count());
        Assert.All(poster.Payloads, p => Assert.Equal("mmusculus", p["organism"]));

        Assert.Equal(new[] { EnrichmentDirection.Up, EnrichmentDirection.Down }, run.Queries.Select(q => q.Direction));
        Assert.Equal(new[] { EnrichmentDirection.Up, EnrichmentDirection.Down }, run.Terms.Select(t => t.Direction));
        Assert.Equal(4, run.SignificantGenes);
    }

    [Fact]
    public void Pooled_SendsEveryHitAsOneList()
    {
        var (res, gene) = Contrast();
        var poster = new RecordingPoster();

        var run = Enrichment.RunByDirection(res, gene, Rule, pooled: true, "hsapiens", "test", poster);

        Assert.Single(poster.Payloads);
        Assert.Equal(new[] { "GDN1", "GDN2", "GUP1", "GUP2" }, poster.Query(0).OrderBy(g => g));
        Assert.All(run.Terms, t => Assert.Equal(EnrichmentDirection.Both, t.Direction));
    }

    /// <summary>A direction with no hits is recorded with zero genes and nothing is sent for it.</summary>
    [Fact]
    public void ADirectionWithNoHits_SendsNothing()
    {
        var (res, gene) = Contrast();
        var poster = new RecordingPoster();
        var upOnly = new SignificanceRule { PThreshold = 0.05, Log2FcThreshold = 1 };

        // Only the UP features have a gene; the DOWN ones map to nothing.
        var run = Enrichment.RunByDirection(res, id => id.StartsWith("UP", StringComparison.Ordinal) ? "G" + id : null,
            upOnly, pooled: false, "hsapiens", "test", poster);

        Assert.Single(poster.Payloads);
        Assert.Equal(0, run.Queries.Single(q => q.Direction == EnrichmentDirection.Down).SignificantGenes);
    }

    [Fact]
    public void InferOrganism_TakesTheMajorityPastContaminants()
    {
        var names = new[]
        {
            "ICAM1_MOUSE", "ALBU_MOUSE", "APOA1_MOUSE;APOA1_RAT", "K1C10_HUMAN", "TRYP_PIG", "ENO1_YEAST",
            "A0A075B5J9_MOUSE", null, "", "PG0001",
        };

        Assert.Equal("mmusculus", Enrichment.InferOrganism(names));
    }

    [Fact]
    public void InferOrganism_IsNull_WithoutACodeOrAMajority()
    {
        Assert.Null(Enrichment.InferOrganism(new[] { "PG0001", "sp|P12345", null }));
        Assert.Null(Enrichment.InferOrganism(new[] { "A_MOUSE", "B_HUMAN" }));
        Assert.Null(Enrichment.InferOrganism(new[] { "A_UNKNOWNSPECIES" }));
    }

    /// <summary>
    /// The majority is over every species code, mapped or not: a run that is mostly an unmapped species
    /// is not handed to the one mapped contaminant it carries, and the fallback names the species.
    /// </summary>
    [Fact]
    public void InferOrganism_AnUnmappedMajorityIsNotWonByAMappedContaminant()
    {
        var names = Enumerable.Range(0, 99).Select(i => $"P{i}_XENTR").Append("K1C10_MOUSE").ToArray();

        Assert.Null(Enrichment.InferOrganism(names));
        Assert.Equal("hsapiens", Enrichment.ResolveOrganism(null, names, out var source));
        Assert.Contains("_XENTR", source);

        // An accession behind a contaminant prefix is not a species code, so it does not dilute one.
        Assert.Equal("mmusculus", Enrichment.InferOrganism(new[] { "A_MOUSE", "CON__P12345", "CON__Q99999" }));
    }

    [Fact]
    public void ResolveOrganism_RequestedThenInferredThenHuman()
    {
        Assert.Equal("rnorvegicus", Enrichment.ResolveOrganism("rnorvegicus", new[] { "A_MOUSE" }, out var s1));
        Assert.Equal("as requested", s1);
        Assert.Equal("mmusculus", Enrichment.ResolveOrganism(null, new[] { "A_MOUSE" }, out var s2));
        Assert.Contains("inferred", s2);
        Assert.Equal("hsapiens", Enrichment.ResolveOrganism(null, new[] { "PG1" }, out var s3));
        // Says why, without naming a command-line flag: this text also reaches the tool's status line
        // and its report, and the tool has no organism control.
        Assert.Contains("no UniProt species code", s3);
        Assert.DoesNotContain("--organism", s3);
    }

    [Theory]
    [InlineData(null, false)]
    [InlineData("separate", false)]
    [InlineData("Pooled", true)]
    [InlineData("both", true)]
    public void ParseDirectionToken_AcceptsTheTwoChoices(string? token, bool pooled) =>
        Assert.Equal(pooled, Enrichment.ParseDirectionToken(token));

    [Fact]
    public void ParseDirectionToken_RefusesAnythingElse() =>
        Assert.Throws<ArgumentException>(() => Enrichment.ParseDirectionToken("upward"));

    /// <summary>
    /// The recorded command names both choices - the organism as resolved, so a re-run queries the
    /// same species even if inference changes - and names neither when enrichment did not run.
    /// </summary>
    [Fact]
    public void QuantCommand_RecordsDirectionAndOrganism_OnlyWhenEnrichmentRuns()
    {
        var ds = DifferentialDataset.Load(Fixtures.Path2("mini", "e2e-sum", "output"), FeatureLevel.Protein);
        QuantRequest Request(bool enrich, bool pooled) => new()
        {
            OutputDir = "C:\\runs\\out",
            Dataset = ds,
            Options = new DifferentialOptions(),
            Rule = SignificanceRule.Default,
            GroupBy = "sample_type",
            ALevels = new[] { "qc" },
            BLevels = new[] { "experimental" },
            EnrichmentPoster = enrich ? new RecordingPoster() : null,
            EnrichmentPooled = pooled,
        };

        Assert.True(QuantCommand.TryArguments(Request(true, false), out var separate, out _));
        var s = separate.ToList();
        Assert.Equal("separate", s[s.IndexOf("--enrichment-direction") + 1]);
        Assert.Equal("mmusculus", s[s.IndexOf("--organism") + 1]); // the fixture is a mouse run

        Assert.True(QuantCommand.TryArguments(Request(true, true), out var pooled, out _));
        Assert.Equal("pooled", pooled[pooled.ToList().IndexOf("--enrichment-direction") + 1]);

        Assert.True(QuantCommand.TryArguments(Request(false, false), out var none, out _));
        Assert.DoesNotContain("--enrichment-direction", none);
        Assert.DoesNotContain("--organism", none);
    }
}
