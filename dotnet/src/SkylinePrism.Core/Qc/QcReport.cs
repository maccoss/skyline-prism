using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Numerics;
using SkylinePrism.Core.Visualization;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// Generates the self-contained qc_report.html from an output directory, porting the
/// structure of validation.py:generate_comprehensive_qc_report: dataset summary, the
/// peptide/protein median-CV tables (LINEAR CVs), and base64-embedded static plots.
/// Regenerable standalone (the `prism qc` command) and called as Stage 5b.
/// </summary>
public static class QcReport
{
    private const string PepMetaN = "n_transitions";
    private const string PepMetaRt = "mean_rt";
    private static readonly string[] ProtMeta =
    {
        "protein_group", "leading_protein", "leading_name", "leading_uniprot_id",
        "leading_gene_name", "leading_description", "n_peptides", "n_unique_peptides", "low_confidence",
    };

    public static string Generate(string outputDir, PrismConfig config, bool savePlots = false)
    {
        var sampleTypes = ReadSampleTypes(Path.Combine(outputDir, "sample_metadata.csv"));

        var peptideCol = DetectPeptideColumn(Path.Combine(outputDir, "peptides_rollup.parquet"));
        var pepMeta = new[] { peptideCol, PepMetaN, PepMetaRt };

        var pepRaw = LoadMatrix(Path.Combine(outputDir, "peptides_rollup.parquet"), pepMeta);
        var pepCorrected = LoadMatrix(Path.Combine(outputDir, "peptides_log2_internal.parquet"), pepMeta);
        var protRaw = LoadMatrix(Path.Combine(outputDir, "proteins_raw.parquet"), ProtMeta);
        var protCorrectedLinear = LoadMatrix(Path.Combine(outputDir, "corrected_proteins.parquet"), ProtMeta);
        var protCorrected = Log2(protCorrectedLinear);

        var sampleCols = pepRaw.SampleCols;
        var refIdx = IndicesOfType(sampleCols, sampleTypes, "reference");
        var qcIdx = IndicesOfType(sampleCols, sampleTypes, "qc");

        var pepRef = CvMetrics.Compute(pepRaw.Values, pepCorrected.Values, refIdx);
        var pepQc = CvMetrics.Compute(pepRaw.Values, pepCorrected.Values, qcIdx);
        var protRef = CvMetrics.Compute(protRaw.Values, protCorrected.Values, refIdx);
        var protQc = CvMetrics.Compute(protRaw.Values, protCorrected.Values, qcIdx);

        var plotsDir = Path.Combine(outputDir, "qc_plots");
        if (savePlots)
            Directory.CreateDirectory(plotsDir);

        var typeLabels = sampleCols.Select(s => sampleTypes.GetValueOrDefault(s, "unknown")).ToList();

        var peptidePlots = RenderLevelPlots("peptide", pepCorrected, sampleCols, typeLabels, refIdx, qcIdx, savePlots, plotsDir);
        var proteinPlots = RenderLevelPlots("protein", protCorrected, sampleCols, typeLabels, refIdx, qcIdx, savePlots, plotsDir);

        var html = BuildHtml(
            outputDir, sampleCols.Count, sampleTypes,
            pepRaw.RowCount, protRaw.RowCount,
            pepRef, pepQc, protRef, protQc, peptidePlots, proteinPlots);

        var htmlPath = Path.Combine(outputDir, "qc_report.html");
        File.WriteAllText(htmlPath, html);
        return htmlPath;
    }

    private static List<(string Title, byte[] Png)> RenderLevelPlots(
        string level, Matrix corrected, IReadOnlyList<string> sampleCols, IReadOnlyList<string> typeLabels,
        IReadOnlyList<int> refIdx, IReadOnlyList<int> qcIdx, bool savePlots, string plotsDir)
    {
        var plots = new List<(string, byte[])>();

        void Add(string title, string fileStem, Func<byte[]> render)
        {
            try
            {
                var png = render();
                plots.Add((title, png));
                if (savePlots)
                    File.WriteAllBytes(Path.Combine(plotsDir, fileStem + ".png"), png);
            }
            catch (Exception ex)
            {
                // Rendering can fail on a headless host missing fontconfig; keep the report.
                plots.Add((title + " (render failed: " + ex.GetType().Name + ")", Array.Empty<byte>()));
            }
        }

        var cap = char.ToUpperInvariant(level[0]) + level[1..];

        if (refIdx.Count >= 2)
        {
            var cvs = CvMetrics.PerFeatureCvs(corrected.Values, refIdx);
            var med = Stats.NanMedian(cvs);
            Add($"{cap} CV (Reference Samples)", $"{level}_cv_reference",
                () => PlotRenderer.CvHistogram(cvs, $"{cap} CV (Reference)", "#d62728", med));
        }
        if (qcIdx.Count >= 2)
        {
            var cvs = CvMetrics.PerFeatureCvs(corrected.Values, qcIdx);
            var med = Stats.NanMedian(cvs);
            Add($"{cap} CV (QC Samples)", $"{level}_cv_qc",
                () => PlotRenderer.CvHistogram(cvs, $"{cap} CV (QC)", "#ff7f0e", med));
        }

        Add($"{cap} PCA Analysis", $"{level}_pca_comparison",
            () => PlotRenderer.PcaScatter(Pca.Fit2D(Transpose(corrected.Values)), typeLabels, $"{cap} PCA"));

        Add($"{cap} Intensity Distribution", $"{level}_intensity_distribution",
            () => PlotRenderer.IntensityDistribution(corrected.Values, typeLabels, $"{cap} Intensity Distribution"));

        return plots;
    }

    private static string BuildHtml(
        string outputDir, int nSamples, IReadOnlyDictionary<string, string> sampleTypes,
        int nPeptides, int nProteins,
        CvMetrics.BeforeAfter? pepRef, CvMetrics.BeforeAfter? pepQc,
        CvMetrics.BeforeAfter? protRef, CvMetrics.BeforeAfter? protQc,
        List<(string Title, byte[] Png)> peptidePlots, List<(string Title, byte[] Png)> proteinPlots)
    {
        var nRef = sampleTypes.Values.Count(v => v == "reference");
        var nQc = sampleTypes.Values.Count(v => v == "qc");
        var nExp = sampleTypes.Values.Count(v => v == "experimental");

        var sb = new StringBuilder();
        sb.Append("""
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PRISM QC Report</title>
<style>
body { font-family: -apple-system, Segoe UI, Arial, sans-serif; color: #222; margin: 0; padding: 24px; }
.container { max-width: 1400px; margin: 0 auto; }
h1 { color: #1a3c6e; }
h2 { color: #1a3c6e; border-bottom: 2px solid #dfe6ef; padding-bottom: 4px; margin-top: 32px; }
.box { background: #f6f8fb; border: 1px solid #dfe6ef; border-radius: 6px; padding: 12px 16px; margin: 12px 0; }
table { border-collapse: collapse; margin: 8px 0; }
th, td { border: 1px solid #cfd8e3; padding: 6px 12px; text-align: right; }
th { background: #eaf0f7; }
td:first-child, th:first-child { text-align: left; }
.improvement-positive { color: #1a7f37; font-weight: 600; }
.improvement-negative { color: #b42318; font-weight: 600; }
.section-header { background: linear-gradient(90deg,#1a3c6e,#3a6ea5); color:#fff; padding:8px 14px; border-radius:6px; margin-top:24px; }
.plot-section { margin: 14px 0; }
.plot-section img { max-width: 100%; border: 1px solid #dfe6ef; border-radius: 4px; }
.footer { color: #888; font-size: 12px; margin-top: 32px; }
</style></head><body><div class="container">
""");
        sb.Append("<h1>PRISM QC Report</h1>");

        sb.Append("<div class=\"box\"><h2>Dataset Summary</h2>");
        sb.Append($"<p>Samples: <strong>{nSamples}</strong> (experimental {nExp}, reference {nRef}, qc {nQc})<br>");
        sb.Append($"Peptides: <strong>{nPeptides}</strong> &nbsp; Proteins: <strong>{nProteins}</strong></p>");
        sb.Append($"<p style=\"color:#666\">Output directory: <code>{HtmlEncode(outputDir)}</code></p>");
        sb.Append("</div>");

        sb.Append("<h2>Summary Metrics (Median CV %)</h2>");
        sb.Append(CvTable("Peptide-Level CV", pepRef, pepQc));
        sb.Append(CvTable("Protein-Level CV", protRef, protQc));

        sb.Append("<div class=\"section-header\">Peptide-Level QC</div>");
        AppendPlots(sb, peptidePlots);
        sb.Append("<div class=\"section-header\">Protein-Level QC</div>");
        AppendPlots(sb, proteinPlots);

        sb.Append($"<p class=\"footer\">Generated by Skyline-PRISM (C#) at {DateTimeStamp()}</p>");
        sb.Append("</div></body></html>");
        return sb.ToString();
    }

    private static string CvTable(string title, CvMetrics.BeforeAfter? refBa, CvMetrics.BeforeAfter? qcBa)
    {
        var sb = new StringBuilder();
        sb.Append($"<h3>{title}</h3><table><tr><th>Sample Type</th><th>Before</th><th>After</th><th>Improvement</th></tr>");
        void Row(string label, CvMetrics.BeforeAfter? ba)
        {
            if (ba is null)
                return;
            var v = ba.Value;
            var cls = v.ImprovementPercent >= 0 ? "improvement-positive" : "improvement-negative";
            sb.Append($"<tr><td>{label}</td><td>{v.Before:0.0}%</td><td>{v.After:0.0}%</td>");
            sb.Append($"<td class=\"{cls}\">{v.ImprovementPercent:+0.0;-0.0}%</td></tr>");
        }
        Row("Reference", refBa);
        Row("QC", qcBa);
        sb.Append("</table>");
        return sb.ToString();
    }

    private static void AppendPlots(StringBuilder sb, List<(string Title, byte[] Png)> plots)
    {
        foreach (var (title, png) in plots)
        {
            sb.Append($"<div class=\"plot-section\"><h3>{title}</h3>");
            if (png.Length > 0)
                sb.Append($"<img src=\"data:image/png;base64,{Convert.ToBase64String(png)}\" alt=\"{title}\" />");
            sb.Append("</div>");
        }
    }

    // -- helpers --

    private sealed record Matrix(double[,] Values, List<string> SampleCols, int RowCount);

    private static Matrix LoadMatrix(string path, IReadOnlyList<string> metaCols)
    {
        var table = ParquetTable.Load(path);
        var meta = new HashSet<string>(metaCols, StringComparer.Ordinal);
        var sampleCols = table.ColumnNames.Where(c => !meta.Contains(c)).ToList();
        var n = table.RowCount;
        var m = new double[n, sampleCols.Count];
        for (var j = 0; j < sampleCols.Count; j++)
        {
            var col = table.GetDouble(sampleCols[j]);
            for (var i = 0; i < n; i++)
                m[i, j] = col[i] ?? double.NaN;
        }
        return new Matrix(m, sampleCols, n);
    }

    private static Matrix Log2(Matrix linear)
    {
        var n = linear.Values.GetLength(0);
        var c = linear.Values.GetLength(1);
        var m = new double[n, c];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < c; j++)
                m[i, j] = Math.Log2(linear.Values[i, j]);
        return new Matrix(m, linear.SampleCols, n);
    }

    private static double[,] Transpose(double[,] a)
    {
        var n = a.GetLength(0);
        var c = a.GetLength(1);
        var t = new double[c, n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < c; j++)
                t[j, i] = a[i, j];
        return t;
    }

    private static List<int> IndicesOfType(
        IReadOnlyList<string> sampleCols, IReadOnlyDictionary<string, string> types, string type)
    {
        var idx = new List<int>();
        for (var i = 0; i < sampleCols.Count; i++)
            if (types.GetValueOrDefault(sampleCols[i], "unknown") == type)
                idx.Add(i);
        return idx;
    }

    private static Dictionary<string, string> ReadSampleTypes(string metadataCsv)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!File.Exists(metadataCsv))
            return map;
        var lines = File.ReadAllLines(metadataCsv);
        if (lines.Length < 2)
            return map;
        var header = lines[0].Split(',');
        var idIdx = Array.IndexOf(header, "sample_id");
        var typeIdx = Array.IndexOf(header, "sample_type");
        if (idIdx < 0 || typeIdx < 0)
            return map;
        for (var i = 1; i < lines.Length; i++)
        {
            var f = lines[i].Split(',');
            if (f.Length > Math.Max(idIdx, typeIdx))
                map[f[idIdx]] = f[typeIdx];
        }
        return map;
    }

    private static string DetectPeptideColumn(string peptidesRollupParquet)
    {
        var cols = ParquetTable.Load(peptidesRollupParquet).ColumnNames;
        return cols.FirstOrDefault(c => c != PepMetaN && c != PepMetaRt && !c.Contains("__@__"))
            ?? "Peptide Modified Sequence";
    }

    private static string HtmlEncode(string s) => s
        .Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");

    private static string DateTimeStamp()
    {
        // Avoid DateTime.Now for determinism-friendliness; UTC file-write time is fine here.
        return DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss 'UTC'", CultureInfo.InvariantCulture);
    }
}
