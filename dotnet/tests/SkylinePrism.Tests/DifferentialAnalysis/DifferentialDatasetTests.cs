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

    /// <summary>
    /// Genes are their own array, never the display label. At protein level the two agree wherever a
    /// gene is known - the label is the gene, falling back to the accession only where there is none
    /// (<see cref="Load_Protein_PlaceholderGene_FallsBackToAccession_NeverNA"/>) - so this side only
    /// fixes that FeatureGenes is populated.
    /// </summary>
    [Fact]
    public void Load_Protein_CarriesGeneSymbols()
    {
        var d = DifferentialDataset.Load(MiniOutput, FeatureLevel.Protein);

        Assert.Equal(d.FeatureIds.Length, d.FeatureGenes.Length);
        Assert.Contains(d.FeatureGenes, g => !string.IsNullOrEmpty(g));
    }

    /// <summary>
    /// The half that matters: on a peptide matrix the label is the modified SEQUENCE, and a sequence
    /// must never reach FeatureGenes. Enrichment reads FeatureGenes and submits it to g:Profiler as
    /// gene symbols; a sequence passes CleanSymbols untouched, so substituting the label there asked
    /// g:Profiler about peptide sequences and captioned the empty answer as a gene enrichment.
    ///
    /// <para>This fixture predates the leading_gene_name column on corrected_peptides (the C# engine
    /// stamps it on now), which makes it exactly the "no genes known" case: every entry must be
    /// empty rather than fall back to the sequence.</para>
    /// </summary>
    [Fact]
    public void Load_Peptide_NeverPutsTheSequenceInTheGeneArray()
    {
        var d = DifferentialDataset.Load(MiniOutput, FeatureLevel.Peptide);

        Assert.Equal(d.FeatureIds.Length, d.FeatureGenes.Length);
        // The labels really are sequences here - the premise of the test.
        Assert.Equal(d.FeatureIds, d.FeatureLabels);
        Assert.All(d.FeatureGenes, g => Assert.True(string.IsNullOrEmpty(g),
            $"a gene array must not carry the peptide sequence, but held '{g}'"));
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
            // Recorded, because a report on a clinical column needs the file to be reproduced.
            Assert.Equal(new[] { Path.GetFullPath(path) }, d.AttachedClinicalCsvs);

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
            // A join that added nothing is not recorded: nothing in the analysis depends on it.
            Assert.Empty(d.AttachedClinicalCsvs);
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>
    /// The mini output with its identity columns replaced: PG0002 has no gene, PG0003 has neither a
    /// gene nor an accession. "NA" is what parsimony writes for an unknown name
    /// (<c>ParsimonyEngine</c>, <c>ProteinGroup</c>), so that is what a real output carries -
    /// a non-model-organism run has thousands of them. With <paramref name="sharedLastRow"/>, the
    /// last row is instead in two groups, the first unknown and the second ICAM1, as a shared
    /// peptide's identity columns list them.
    /// </summary>
    private static string WriteMiniWithPlaceholderNames(bool sharedLastRow = false)
    {
        var dir = Path.Combine(Path.GetTempPath(), $"prism-na-labels-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        File.Copy(Path.Combine(MiniOutput, "sample_metadata.csv"), Path.Combine(dir, "sample_metadata.csv"));

        var mini = SkylinePrism.Core.IO.ParquetTable.Load(Path.Combine(MiniOutput, "corrected_proteins.parquet"));
        var sampleCols = mini.ColumnNames.Where(c => c.Contains("__@__", StringComparison.Ordinal)).ToList();
        var meta = new[]
        {
            SkylinePrism.Core.IO.ParquetWideWriter.Strings("protein_group",
                new[] { "PG0001", "PG0002", sharedLastRow ? "PG0003;PG0004" : "PG0003" }),
            // PG0002's leading_protein is the whole FASTA id, as when Skyline knew the protein by name
            // only; the clean accession is in leading_uniprot_id.
            SkylinePrism.Core.IO.ParquetWideWriter.Strings("leading_protein",
                new[] { "E9Q0F0", "sp|P13597|ICAM1_MOUSE", sharedLastRow ? "NA;P13597" : "NA" }),
            SkylinePrism.Core.IO.ParquetWideWriter.Strings("leading_name",
                new[] { "E9Q0F0_MOUSE", "ICAM1_MOUSE", sharedLastRow ? "NA;ICAM1_MOUSE" : "NA" }),
            SkylinePrism.Core.IO.ParquetWideWriter.Strings("leading_uniprot_id",
                new[] { "E9Q0F0", "P13597", sharedLastRow ? "NA;P13597" : "NA" }),
            SkylinePrism.Core.IO.ParquetWideWriter.Strings("leading_gene_name",
                new[] { "Krt78", "NA", sharedLastRow ? "NA;Icam1" : "NA" }),
            SkylinePrism.Core.IO.ParquetWideWriter.Longs("n_peptides", new long[] { 3, 3, 3 }),
        };
        var values = sampleCols.Select(c => mini.GetDouble(c).Select(v => v ?? double.NaN).ToArray()).ToList();
        SkylinePrism.Core.IO.ParquetWideWriter.Write(
            Path.Combine(dir, "corrected_proteins.parquet"), meta, sampleCols, values, 3);
        return dir;
    }

    /// <summary>
    /// A volcano, hit table and boxplot title label a protein by its gene. With no gene known the
    /// placeholder "NA" was shown instead - every unannotated protein labeled the same, which says
    /// nothing. The label must fall back to the protein's UniProt accession - the clean one, not a
    /// whole FASTA id - and with neither, to the protein-group id. The gene
    /// array must not carry the placeholder either: enrichment submits it to g:Profiler as a symbol.
    /// </summary>
    [Fact]
    public void Load_Protein_PlaceholderGene_FallsBackToAccession_NeverNA()
    {
        var dir = WriteMiniWithPlaceholderNames();
        try
        {
            var d = DifferentialDataset.Load(dir, FeatureLevel.Protein);

            Assert.Equal(new[] { "Krt78", "P13597", "PG0003" }, d.FeatureLabels);
            Assert.Equal(new[] { "Krt78", "", "" }, d.FeatureGenes);
            var identity = d.IdentityOf("PG0003")!;
            Assert.Empty(identity.Accessions);
            Assert.Empty(identity.ProteinNames);
            Assert.Empty(identity.Genes);
            // The readout and the CSV use the label as is: a blank one printed nothing.
            Assert.StartsWith("PG0003", identity.Describe());

            var row = WriteCsvLines(d, dir).Single(l => l.StartsWith("PG0003,", StringComparison.Ordinal));
            Assert.StartsWith("PG0003,PG0003,,,,", row);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// A row in two groups whose FIRST group has no known gene, name or accession. The identity lists
    /// are positional - member k is group k's - so the unknown member stays as an empty slot, in the
    /// identity and in differential.csv. Dropping it put the second group's gene at index 0, where it
    /// reads as the first group's.
    /// </summary>
    [Fact]
    public void IdentityOf_AnUnknownFirstMember_KeepsEveryListAlignedWithItsGroups()
    {
        var dir = WriteMiniWithPlaceholderNames(sharedLastRow: true);
        try
        {
            var d = DifferentialDataset.Load(dir, FeatureLevel.Protein);
            var identity = d.IdentityOf("PG0003;PG0004")!;

            Assert.Equal(new[] { "PG0003", "PG0004" }, identity.ProteinGroups);
            Assert.Equal(new[] { "", "Icam1" }, identity.Genes);
            Assert.Equal(new[] { "", "ICAM1_MOUSE" }, identity.ProteinNames);
            Assert.Equal(new[] { "", "P13597" }, identity.Accessions);

            var row = WriteCsvLines(d, dir).Single(l => l.StartsWith("PG0003;PG0004,", StringComparison.Ordinal));
            Assert.StartsWith("PG0003;PG0004,Icam1,;Icam1,;ICAM1_MOUSE,;P13597,", row);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>differential.csv for the experimental samples split in half, as the end-to-end test splits them.</summary>
    private static string[] WriteCsvLines(DifferentialDataset d, string dir)
    {
        var types = d.MetadataValues("sample_type");
        var experimental = Enumerable.Range(0, d.SampleIds.Length).Where(j => types[j] == "experimental").ToList();
        var half = experimental.Count / 2;
        var res = Differential.Run(d.ExprLog2, d.FeatureIds, experimental.Take(half).ToList(), experimental.Skip(half).ToList());
        var csv = Path.Combine(dir, "differential.csv");
        DifferentialCsv.Write(csv, res, d, new DifferentialOptions(), SignificanceRule.Default, "g", "A", "B");
        return File.ReadAllLines(csv);
    }

    [Theory]
    [InlineData("NA;ALB", new[] { "", "ALB" })]
    [InlineData("ALB;NA", new[] { "ALB", "" })]
    [InlineData(" ALB ; HBB ", new[] { "ALB", "HBB" })]
    [InlineData("NA;N/A", new string[0])]
    [InlineData("", new string[0])]
    [InlineData(null, new string[0])]
    public void NamePlaceholder_SplitAligned_KeepsEachMemberInItsPlace(string? value, string[] expected)
    {
        Assert.Equal(expected, SkylinePrism.Core.IO.NamePlaceholder.SplitAligned(value));
        Assert.Equal(expected.Length == 0 ? "" : string.Join(";", expected),
            SkylinePrism.Core.IO.NamePlaceholder.JoinAligned(expected));
    }
}
