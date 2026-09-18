using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Tests for <see cref="DifferentialDataset"/> against the committed mini output fixture, including an
/// end-to-end check (load -> contrast -> moderated-t) matching the Python load_prism + differential.
/// </summary>
public class DifferentialDatasetTests
{
    private static string MiniOutput => Fixtures.Path2("mini", "e2e-sum", "output");

    private static void AssertRel(double expected, double actual, double rtol)
    {
        Assert.True(Math.Abs(actual - expected) / Math.Abs(expected) <= rtol,
            $"expected {expected:R}, actual {actual:R}");
    }

    [Fact]
    public void Load_Protein_MatchesLoadPrism()
    {
        var d = DifferentialDataset.Load(MiniOutput, FeatureLevel.Protein);

        Assert.Equal(3, d.FeatureIds.Length);
        Assert.Equal(166, d.SampleIds.Length);
        Assert.Equal("protein_group", d.IdColumn);
        Assert.Equal("leading_gene_name", d.LabelColumn);
        Assert.Equal("PG0001", d.FeatureIds[0]);
        Assert.Equal("IRType-Plasma-201_078__@__mini_plate2", d.SampleIds[0]);
        Assert.Equal(5.821064533147193, d.ExprLog2[0, 0], 9); // linear -> log2

        Assert.Contains("sample_type", d.MetadataColumns);
        Assert.Contains("batch", d.MetadataColumns);
        Assert.Equal(138, d.MetadataValues("sample_type").Count(v => v == "experimental"));
    }

    [Fact]
    public void LoadThenDifferential_MatchesExplorerEndToEnd()
    {
        var d = DifferentialDataset.Load(MiniOutput, FeatureLevel.Protein);

        // Experimental sample columns in matrix order, split first half vs second half.
        var types = d.MetadataValues("sample_type");
        var experimental = Enumerable.Range(0, d.SampleIds.Length)
            .Where(j => types[j] == "experimental").ToList();
        var half = experimental.Count / 2;
        var groupA = experimental.Take(half).ToList();
        var groupB = experimental.Skip(half).ToList();

        var res = Differential.Run(d.ExprLog2, d.FeatureIds, groupA, groupB);

        Assert.Equal(3, res.NFeaturesTested);
        Assert.Equal(0.2924877138903692, res.DfPrior, 9);

        Assert.Equal("PG0002", res.Rows[0].FeatureId); // smallest p
        var byId = res.Rows.ToDictionary(r => r.FeatureId);
        Assert.Equal(-0.0002532187140976061, byId["PG0002"].LogFc, 9);
        Assert.Equal(-0.9933318101885763, byId["PG0002"].T, 9);
        AssertRel(0.3223085567035089, byId["PG0002"].PValue, 1e-9);
        Assert.Equal(-0.06552109959229187, byId["PG0001"].LogFc, 9);
        AssertRel(0.6155448330123279, byId["PG0001"].PValue, 1e-9);
        AssertRel(0.78184737677963, byId["PG0003"].AdjPValue, 1e-9);
    }

    [Fact]
    public void AttachClinical_JoinsByAutoDetectedKey_AddsColumns()
    {
        var d = DifferentialDataset.Load(MiniOutput, FeatureLevel.Protein);
        var sampleNames = d.MetadataValues("sample");

        // Build a clinical CSV keyed on the sample name, assigning a Diagnosis by matrix position so
        // every sample maps exactly once (a one-to-one key the inference should prefer).
        var path = Path.Combine(Path.GetTempPath(), $"prism-clin-{Guid.NewGuid():N}.csv");
        try
        {
            using (var w = new StreamWriter(path))
            {
                w.WriteLine("PatientName,Diagnosis");
                for (var i = 0; i < sampleNames.Length; i++)
                    w.WriteLine($"{sampleNames[i]},{(i % 2 == 0 ? "AD" : "Control")}");
            }

            var before = d.MetadataColumns.Count;
            var result = d.AttachClinical(path);

            Assert.Equal("PatientName", result.KeyColumn);
            Assert.Equal(1.0, result.MatchRate, 9); // every sample name is present in the CSV
            Assert.Contains("Diagnosis", result.AddedColumns);
            Assert.Equal(before + 1, d.MetadataColumns.Count);

            // Every sample got a non-null Diagnosis and both categories are represented. (Bare sample
            // names collide across merged plates, so exact per-index values are not asserted.)
            var diag = d.MetadataValues("Diagnosis");
            Assert.All(diag, v => Assert.False(string.IsNullOrEmpty(v)));
            Assert.Contains("AD", diag);
            Assert.Contains("Control", diag);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void AlignSampleColumns_ExactSampleIdMatch_Preferred()
    {
        var meta = new System.Collections.Generic.Dictionary<string, string?[]>
        {
            ["R1__@__b"] = new string?[] { "R1", "experimental" },
            ["R2__@__b"] = new string?[] { "R2", "experimental" },
        };
        var cols = new[] { "protein_group", "R1__@__b", "R2__@__b" };

        var aligned = DifferentialDataset.AlignSampleColumns(cols, meta, new[] { "sample", "sample_type" });

        Assert.Equal(new[] { "R1__@__b", "R2__@__b" }, aligned.Select(a => a.Col));
        Assert.Equal("R1", aligned[0].Meta[0]);
    }

    [Fact]
    public void AlignSampleColumns_FallsBackToBareReplicateName_OnStemMismatch()
    {
        // Matrix columns use one document stem, metadata sample_id uses another; the bare replicate
        // name (before "__@__") is identical, matching the metadata "sample" value.
        var meta = new System.Collections.Generic.Dictionary<string, string?[]>
        {
            ["R1__@__merged_data"] = new string?[] { "R1", "experimental" },
            ["R2__@__merged_data"] = new string?[] { "R2", "qc" },
        };
        var cols = new[] { "protein_group", "R1__@__PRISM", "R2__@__PRISM" };

        var aligned = DifferentialDataset.AlignSampleColumns(cols, meta, new[] { "sample", "sample_type" });

        // Columns keep the matrix names; metadata is resolved by bare name.
        Assert.Equal(new[] { "R1__@__PRISM", "R2__@__PRISM" }, aligned.Select(a => a.Col));
        Assert.Equal("experimental", aligned[0].Meta[1]);
        Assert.Equal("qc", aligned[1].Meta[1]);
    }

    [Fact]
    public void AlignSampleColumns_AmbiguousBareName_Skipped()
    {
        // Two metadata rows share the bare name "R1" (batch collision); it cannot be resolved.
        var meta = new System.Collections.Generic.Dictionary<string, string?[]>
        {
            ["R1__@__b1"] = new string?[] { "R1", "experimental" },
            ["R1__@__b2"] = new string?[] { "R1", "qc" },
            ["R2__@__b1"] = new string?[] { "R2", "experimental" },
        };
        var cols = new[] { "R1__@__X", "R2__@__X" };

        var aligned = DifferentialDataset.AlignSampleColumns(cols, meta, new[] { "sample", "sample_type" });

        Assert.Equal(new[] { "R2__@__X" }, aligned.Select(a => a.Col)); // R1 ambiguous, dropped
    }

    [Fact]
    public void AttachClinical_NoMatch_AddsNothing()
    {
        var d = DifferentialDataset.Load(MiniOutput, FeatureLevel.Protein);
        var before = d.MetadataColumns.Count;

        var path = Path.Combine(Path.GetTempPath(), $"prism-clin-{Guid.NewGuid():N}.csv");
        try
        {
            File.WriteAllText(path, "Unrelated,Value\nZZZ-999,foo\nYYY-888,bar\n");
            var result = d.AttachClinical(path);

            Assert.Null(result.KeyColumn);
            Assert.Empty(result.AddedColumns);
            Assert.Equal(before, d.MetadataColumns.Count);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
