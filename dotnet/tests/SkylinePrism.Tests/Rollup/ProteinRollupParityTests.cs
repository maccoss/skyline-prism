using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.BatchCorrection;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Normalization;
using SkylinePrism.Core.Parsimony;
using SkylinePrism.Core.Rollup;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Rollup;

/// <summary>
/// Layer 7 parity: peptide-&gt;protein rollup (proteins_raw) and the full protein arm
/// (median normalize + ComBat + log2-&gt;linear = corrected_proteins). Uses the golden
/// protein_groups.csv as the peptide-&gt;group map (parsimony is Layer 6). Exercises the
/// median_polish (RAB6A, 3 peptides) and direct/sum_linear (singletons) branches.
/// </summary>
public class ProteinRollupParityTests
{
    private const string PepCol = "Peptide Modified Sequence Unimod Ids";
    private static string E2eDir => Fixtures.Path2("mini", "e2e-sum", "output");

    [Fact]
    public void ProteinRollup_MatchesProteinsRaw()
    {
        var groups = ProteinGroupsCsv.Read(Path.Combine(E2eDir, "protein_groups.csv"));
        var internalPath = Path.Combine(E2eDir, "peptides_log2_internal.parquet");
        var golden = ParquetTable.Load(Path.Combine(E2eDir, "proteins_raw.parquet"));

        var cfg = new ProteinRollupConfig { Method = ProteinRollupMethod.MedianPolish, MinPeptides = 2 };
        var tempOut = TempPath();
        try
        {
            ProteinRollup.Run(internalPath, groups, cfg, PepCol, tempOut);
            var actual = ParquetTable.Load(tempOut);

            AssertProteinsEqual(golden, actual, linear: false);
        }
        finally { Cleanup(tempOut); }
    }

    [Fact]
    public void ProteinArm_MatchesCorrectedProteins()
    {
        var proteinsRaw = ParquetTable.Load(Path.Combine(E2eDir, "proteins_raw.parquet"));
        var golden = ParquetTable.Load(Path.Combine(E2eDir, "corrected_proteins.parquet"));

        var sampleCols = ProteinSampleCols(proteinsRaw);
        var (matrix, keys) = Fixtures.WideMatrix(proteinsRaw, "protein_group", sampleCols);
        var batchLabels = sampleCols.Select(BatchFromSampleId).ToList();

        var normalized = Normalizer.MedianNormalize(matrix);
        var corrected = ComBat.Run(normalized, batchLabels);
        // Stage 5 converts protein output log2 -> linear.
        var nR = corrected.GetLength(0);
        var nC = corrected.GetLength(1);
        var linear = new double[nR, nC];
        for (var i = 0; i < nR; i++)
            for (var j = 0; j < nC; j++)
                linear[i, j] = Math.Pow(2.0, corrected[i, j]);

        var goldenKeys = golden.GetString("protein_group");
        var goldenIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < goldenKeys.Length; i++)
            goldenIndex[goldenKeys[i]!] = i;
        var goldenCols = sampleCols.ToDictionary(c => c, golden.GetDouble);

        for (var i = 0; i < keys.Length; i++)
        {
            var gi = goldenIndex[keys[i]];
            for (var j = 0; j < sampleCols.Count; j++)
            {
                var e = goldenCols[sampleCols[j]][gi]!.Value;
                var a = linear[i, j];
                var diff = Math.Abs(e - a);
                var tol = 1e-6 + 1e-9 * Math.Abs(e); // linear scale -> looser absolute tol
                Assert.True(diff <= tol,
                    $"corrected_proteins mismatch {keys[i]}/{sampleCols[j]}: {e} vs {a} (|d|={diff})");
            }
        }
    }

    private static void AssertProteinsEqual(ParquetTable golden, ParquetTable actual, bool linear)
    {
        Assert.Equal(golden.RowCount, actual.RowCount);
        var gKeys = golden.GetString("protein_group");
        var aKeys = actual.GetString("protein_group");
        var aIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < aKeys.Length; i++)
            aIndex[aKeys[i]!] = i;

        var sampleCols = ProteinSampleCols(golden);
        var gSamples = sampleCols.ToDictionary(c => c, golden.GetDouble);
        var aSamples = sampleCols.ToDictionary(c => c, actual.GetDouble);

        string[] StrCol(ParquetTable t, string c) => t.GetString(c).Select(s => s ?? "").ToArray();
        double?[] NumCol(ParquetTable t, string c) => t.GetDouble(c);

        var metaStr = new[] { "leading_protein", "leading_name", "leading_uniprot_id", "leading_gene_name", "leading_description" };
        var gStr = metaStr.ToDictionary(c => c, c => StrCol(golden, c));
        var aStr = metaStr.ToDictionary(c => c, c => StrCol(actual, c));
        var gN = NumCol(golden, "n_peptides");
        var aN = NumCol(actual, "n_peptides");
        var gU = NumCol(golden, "n_unique_peptides");
        var aU = NumCol(actual, "n_unique_peptides");

        for (var gi = 0; gi < gKeys.Length; gi++)
        {
            var key = gKeys[gi]!;
            var ai = aIndex[key];
            foreach (var c in metaStr)
                Assert.Equal(gStr[c][gi], aStr[c][ai]);
            Assert.Equal(gN[gi]!.Value, aN[ai]!.Value, 9);
            Assert.Equal(gU[gi]!.Value, aU[ai]!.Value, 9);

            foreach (var col in sampleCols)
            {
                var e = gSamples[col][gi]!.Value;
                var a = aSamples[col][ai]!.Value;
                var diff = Math.Abs(e - a);
                var tol = 1e-9 + 1e-9 * Math.Abs(e);
                Assert.True(diff <= tol, $"proteins_raw mismatch {key}/{col}: {e} vs {a} (|d|={diff})");
            }
        }
    }

    private static List<string> ProteinSampleCols(ParquetTable t) => Fixtures.SampleColumns(
        t, "protein_group", "leading_protein", "leading_name", "leading_uniprot_id",
        "leading_gene_name", "leading_description", "n_peptides", "n_unique_peptides", "low_confidence");

    private static string BatchFromSampleId(string sampleId)
    {
        const string sep = "__@__";
        var idx = sampleId.IndexOf(sep, StringComparison.Ordinal);
        return idx >= 0 ? sampleId[(idx + sep.Length)..] : sampleId;
    }

    private static string TempPath() => Path.Combine(
        Path.GetTempPath(), "prism_prot_" + Guid.NewGuid().ToString("N"), "proteins_raw.parquet");

    private static void Cleanup(string tempOut)
    {
        var dir = Path.GetDirectoryName(tempOut);
        if (dir is not null && Directory.Exists(dir))
            Directory.Delete(dir, recursive: true);
    }
}
