using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// End-to-end coverage for combining SEVERAL Skyline documents in one run - the case the external tool's
/// Inputs tab creates when a study is split one document per batch/plate.
///
/// <para>The hazard these tests pin down: reference and QC injections are normally given the SAME
/// replicate name in every plate's document ("Ref_01" exists in all of them). Everything downstream -
/// distinct samples, per-document sample types, per-document batch labels, and therefore whether ComBat
/// runs at all - depends on those being kept apart via the merged "&lt;replicate&gt;__@__&lt;document&gt;"
/// sample ID and the document-scoped metadata lookup. A regression here is silent: batches collapse into
/// one label and batch correction is skipped without an error.</para>
/// </summary>
public class MultiDocumentBatchTests
{
    // Replicate names deliberately IDENTICAL across documents, as real plates are.
    private static readonly string[] References = { "Ref_01", "Ref_02", "Ref_03" };
    private static readonly string[] Qcs = { "QC_01", "QC_02", "QC_03" };
    private static readonly string[] Studies = { "S_01", "S_02", "S_03", "S_04", "S_05", "S_06" };

    private static IEnumerable<string> AllReplicates => References.Concat(Qcs).Concat(Studies);

    private const int NProteins = 3;
    private const int PeptidesPerProtein = 4;
    private const int TransitionsPerPeptide = 4;

    private sealed class Workspace : IDisposable
    {
        public string Dir { get; } = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "prism_multidoc_" + Guid.NewGuid().ToString("N"));

        public Workspace() => Directory.CreateDirectory(Dir);

        public string Path(string name) => System.IO.Path.Combine(Dir, name);

        public void Dispose()
        {
            try { Directory.Delete(Dir, recursive: true); }
            catch (IOException) { /* best effort */ }
        }
    }

    /// <summary>
    /// Write a synthetic Skyline transition report. <paramref name="batchOffset"/> scales every area, so
    /// the two documents carry a real batch effect for ComBat to remove.
    /// </summary>
    private static string WriteTransitionReport(Workspace ws, string label, double batchOffset, int seed)
    {
        var rng = new Random(seed);
        var sb = new StringBuilder();
        sb.AppendLine(
            "Protein,Protein Accession,Protein Gene,Peptide,Peptide Modified Sequence Unimod Ids,"
            + "Precursor Charge,Precursor Mz,Isotope Dot Product,Detection Q Value,Fragment Ion,"
            + "Product Charge,Product Mz,Area,Retention Time,Start Time,End Time,Fwhm,Shape Correlation,"
            + "Coeluting,Truncated,Replicate Name,File Name,Total Ion Current Area,Acquired Time");

        var acquired = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var replicateIndex = 0;
        foreach (var replicate in AllReplicates)
        {
            var time = acquired.AddMinutes(30 * replicateIndex++);
            for (var p = 0; p < NProteins; p++)
            {
                for (var pep = 0; pep < PeptidesPerProtein; pep++)
                {
                    var sequence = $"PEPTIDE{p}{pep}K";
                    var rt = 10 + p * 2 + pep * 0.5;
                    for (var t = 0; t < TransitionsPerPeptide; t++)
                    {
                        // Deterministic magnitude per (protein, peptide, transition), a per-sample jitter,
                        // and the document-level offset.
                        var baseArea = 1e6 * (p + 1) * (pep + 1) * (1.0 + 0.1 * t);
                        var area = baseArea * batchOffset * (0.9 + 0.2 * rng.NextDouble());
                        sb.Append(CultureInfo.InvariantCulture, $"sp|P0000{p}|PROT{p}_HUMAN,P0000{p},GENE{p},");
                        sb.Append(CultureInfo.InvariantCulture, $"{sequence},{sequence},2,{500.0 + p:F4},");
                        sb.Append(CultureInfo.InvariantCulture, $"0.95,0.001,y{t + 3},1,{600.0 + t:F4},");
                        sb.Append(CultureInfo.InvariantCulture, $"{area:F1},{rt:F2},{rt - 0.2:F2},{rt + 0.2:F2},");
                        sb.Append(CultureInfo.InvariantCulture, $"0.1,0.99,false,false,{replicate},");
                        sb.Append(CultureInfo.InvariantCulture,
                            $"{label}_{replicate}.raw,1000000,{time:yyyy-MM-dd HH:mm:ss}");
                        sb.AppendLine();
                    }
                }
            }
        }

        var path = ws.Path(label + ".csv");
        File.WriteAllText(path, sb.ToString());
        return path;
    }

    /// <summary>Replicates report for one document: Skyline sample types plus a Plate annotation.</summary>
    private static string WriteMetadata(Workspace ws, string label, string plate)
    {
        var sb = new StringBuilder();
        sb.AppendLine("Replicate,SampleType,Plate");
        foreach (var r in References)
            sb.AppendLine($"{r},Standard,{plate}");
        foreach (var r in Qcs)
            sb.AppendLine($"{r},Quality Control,{plate}");
        foreach (var r in Studies)
            sb.AppendLine($"{r},Unknown,{plate}");
        var path = ws.Path(label + ".metadata.csv");
        File.WriteAllText(path, sb.ToString());
        return path;
    }

    private static PrismConfig Config(string? batchColumn = null)
    {
        var c = new PrismConfig();
        c.TransitionRollup.Method = "sum";
        c.TransitionRollup.MinTransitions = TransitionsPerPeptide;
        c.ProteinRollup.MinPeptides = PeptidesPerProtein;
        c.GlobalNormalization.Method = "median"; // rt_lowess needs more RT spread than the fixture has
        c.QcReport.Enabled = false;              // keep the test to the pipeline, not plot rendering
        c.QcReport.SavePlots = false;
        if (batchColumn is not null)
            c.Metadata.BatchColumn = batchColumn;
        return c;
    }

    private static Dictionary<string, (string Type, string Batch)> ReadSampleMetadata(string outputDir)
    {
        var path = Path.Combine(outputDir, "sample_metadata.csv");
        Assert.True(File.Exists(path), "sample_metadata.csv was not written");
        var lines = File.ReadAllLines(path);
        var header = lines[0].Split(',');
        int Col(string name) => Array.FindIndex(header, h => h.Trim().Equals(name, StringComparison.OrdinalIgnoreCase));
        var idIdx = Col("sample_id");
        var typeIdx = Col("sample_type");
        var batchIdx = Col("batch");

        var result = new Dictionary<string, (string, string)>(StringComparer.Ordinal);
        foreach (var line in lines.Skip(1).Where(l => !string.IsNullOrWhiteSpace(l)))
        {
            var f = line.Split(',');
            result[f[idIdx]] = (f[typeIdx], f[batchIdx]);
        }
        return result;
    }

    [Fact]
    public void TwoDocuments_WithIdenticalReplicateNames_StayDistinctSamples()
    {
        using var ws = new Workspace();
        var a = WriteTransitionReport(ws, "PlateA", batchOffset: 1.0, seed: 1);
        var b = WriteTransitionReport(ws, "PlateB", batchOffset: 1.8, seed: 2);
        var outDir = ws.Path("out");

        var result = PrismPipeline.Run(new[] { a, b }, outDir, Config());

        // Every replicate appears in BOTH documents; none may be merged away.
        var expected = AllReplicates.Count() * 2;
        Assert.Equal(expected, result.NSamples);

        var samples = ReadSampleMetadata(outDir);
        Assert.Equal(expected, samples.Count);
        foreach (var replicate in AllReplicates)
        {
            Assert.True(samples.ContainsKey($"{replicate}__@__PlateA"), $"missing {replicate} from PlateA");
            Assert.True(samples.ContainsKey($"{replicate}__@__PlateB"), $"missing {replicate} from PlateB");
        }
    }

    [Fact]
    public void TwoDocuments_WithoutMetadata_TakeBatchFromTheSourceDocument()
    {
        using var ws = new Workspace();
        var a = WriteTransitionReport(ws, "PlateA", 1.0, 1);
        var b = WriteTransitionReport(ws, "PlateB", 1.8, 2);
        var outDir = ws.Path("out");

        var result = PrismPipeline.Run(new[] { a, b }, outDir, Config());

        // No Replicates report: the file stem is the batch label, which is what makes the tool's
        // per-input "Batch label" meaningful.
        Assert.Equal(new[] { "PlateA", "PlateB" }, result.Batches);
        var samples = ReadSampleMetadata(outDir);
        Assert.Equal("PlateA", samples["Ref_01__@__PlateA"].Batch);
        Assert.Equal("PlateB", samples["Ref_01__@__PlateB"].Batch);
    }

    [Fact]
    public void TwoDocuments_WithPerDocumentMetadata_KeepTheirOwnBatchAndSampleType()
    {
        using var ws = new Workspace();
        var a = WriteTransitionReport(ws, "PlateA", 1.0, 1);
        var b = WriteTransitionReport(ws, "PlateB", 1.8, 2);
        var metaA = WriteMetadata(ws, "PlateA", "P1");
        var metaB = WriteMetadata(ws, "PlateB", "P2");
        var outDir = ws.Path("out");

        var result = PrismPipeline.Run(
            new[] { a, b }, outDir, Config(batchColumn: "Plate"), new[] { metaA, metaB });

        // THE REGRESSION GUARD: keyed by bare replicate name the second metadata file would overwrite the
        // first, every sample would resolve to P2, and the run would silently drop to a single batch.
        Assert.Equal(new[] { "P1", "P2" }, result.Batches);

        var samples = ReadSampleMetadata(outDir);
        foreach (var replicate in AllReplicates)
        {
            Assert.Equal("P1", samples[$"{replicate}__@__PlateA"].Batch);
            Assert.Equal("P2", samples[$"{replicate}__@__PlateB"].Batch);
        }

        // Sample types come from each document's own Replicates report.
        Assert.Equal("reference", samples["Ref_01__@__PlateA"].Type);
        Assert.Equal("reference", samples["Ref_01__@__PlateB"].Type);
        Assert.Equal("qc", samples["QC_01__@__PlateA"].Type);
        Assert.Equal("experimental", samples["S_01__@__PlateB"].Type);
    }

    [Fact]
    public void TwoDocuments_ComBatRemovesTheBetweenDocumentOffset()
    {
        using var ws = new Workspace();
        // A 1.8x multiplicative offset on PlateB = a large log2 shift for ComBat to remove.
        var a = WriteTransitionReport(ws, "PlateA", 1.0, 1);
        var b = WriteTransitionReport(ws, "PlateB", 1.8, 2);
        var outDir = ws.Path("out");

        var result = PrismPipeline.Run(new[] { a, b }, outDir, Config());
        Assert.Equal(2, result.Batches.Count); // ComBat only runs with >= 2 batches

        var corrected = ParquetTable.Load(Path.Combine(outDir, "corrected_peptides.parquet"));
        var raw = ParquetTable.Load(Path.Combine(outDir, "peptides_rollup.parquet"));

        // corrected_peptides is LINEAR (see the scale contract); peptides_rollup is LOG2.
        var correctedGap = MeanBatchGap(corrected, log2Input: false);
        var rawGap = MeanBatchGap(raw, log2Input: true);

        Assert.True(rawGap > 0.3,
            $"fixture should carry a batch effect before correction (log2 gap was {rawGap:F3})");
        Assert.True(correctedGap < rawGap / 2,
            $"ComBat should shrink the between-document gap: raw {rawGap:F3} -> corrected {correctedGap:F3}");
    }

    /// <summary>Mean |PlateA - PlateB| per feature, on the log2 scale, over the shared replicates.</summary>
    private static double MeanBatchGap(ParquetTable table, bool log2Input)
    {
        var aCols = table.ColumnNames.Where(c => c.EndsWith("__@__PlateA", StringComparison.Ordinal)).ToList();
        var bCols = table.ColumnNames.Where(c => c.EndsWith("__@__PlateB", StringComparison.Ordinal)).ToList();
        Assert.NotEmpty(aCols);
        Assert.NotEmpty(bCols);

        double Mean(IEnumerable<string> cols, int row)
        {
            var values = cols
                .Select(c => table.GetDouble(c)[row])
                .Where(v => v.HasValue && !double.IsNaN(v.Value))
                .Select(v => log2Input ? v!.Value : Math.Log2(v!.Value))
                .ToList();
            return values.Count == 0 ? double.NaN : values.Average();
        }

        var gaps = new List<double>();
        for (var row = 0; row < table.RowCount; row++)
        {
            var gap = Math.Abs(Mean(aCols, row) - Mean(bCols, row));
            if (!double.IsNaN(gap))
                gaps.Add(gap);
        }
        Assert.NotEmpty(gaps);
        return gaps.Average();
    }

    [Fact]
    public void ThreeDocuments_ProduceThreeBatches()
    {
        using var ws = new Workspace();
        var inputs = new[]
        {
            WriteTransitionReport(ws, "PlateA", 1.0, 1),
            WriteTransitionReport(ws, "PlateB", 1.4, 2),
            WriteTransitionReport(ws, "PlateC", 0.7, 3),
        };
        var outDir = ws.Path("out");

        var result = PrismPipeline.Run(inputs, outDir, Config());

        Assert.Equal(new[] { "PlateA", "PlateB", "PlateC" }, result.Batches);
        Assert.Equal(AllReplicates.Count() * 3, result.NSamples);
    }

    [Fact]
    public void MetadataCountMismatch_FallsBackToBareReplicateKeys_WithoutCrashing()
    {
        using var ws = new Workspace();
        var a = WriteTransitionReport(ws, "PlateA", 1.0, 1);
        var b = WriteTransitionReport(ws, "PlateB", 1.8, 2);
        var metaA = WriteMetadata(ws, "PlateA", "P1"); // only ONE metadata file for two inputs
        var outDir = ws.Path("out");

        // Document scoping needs a 1:1 pairing; with a partial set the pipeline must still run, using the
        // unqualified replicate keys (the tool avoids this by dropping metadata entirely when incomplete).
        var result = PrismPipeline.Run(
            new[] { a, b }, outDir, Config(batchColumn: "Plate"), new[] { metaA });

        Assert.Equal(AllReplicates.Count() * 2, result.NSamples);
        var samples = ReadSampleMetadata(outDir);
        Assert.Equal("reference", samples["Ref_01__@__PlateB"].Type); // bare-name fallback still types it
    }
}
