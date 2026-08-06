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
using SkylinePrism.Core.Rollup;

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
    // From the writer, not repeated here - see ProteinRollup.MetadataColumns.
    private static readonly string[] ProtMeta = ProteinRollup.MetadataColumns;

    // PRISM app icon as a 32x32 PNG data-URI favicon, so the self-contained report shows the
    // prism logo on the browser tab. Regenerate from images/skyline-prism-icon.png (pad to square,
    // resize to 32, PNG-encode, base64) if the icon changes.
    private const string FaviconLinkTag =
        "<link rel=\"icon\" type=\"image/png\" href=\"data:image/png;base64," +
        "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" +
        "\">\n";

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

        // RT diagnostic plots are only meaningful when RT-lowess normalization actually ran (matches
        // Python, which gates them on rt_lowess_result being present).
        var rtLowessRan = string.Equals(config.GlobalNormalization.Method, "rt_lowess", StringComparison.OrdinalIgnoreCase);
        var peptidePlots = RenderLevelSections("peptide", pepRaw, pepCorrected, typeLabels, refIdx, qcIdx, savePlots, plotsDir, rtLowessRan);
        var proteinPlots = RenderLevelSections("protein", protRaw, protCorrected, typeLabels, refIdx, qcIdx, savePlots, plotsDir, rtLowessRan);

        var validation = ValidationStatus.Compute(pepRaw.Values, pepCorrected.Values, refIdx, qcIdx);

        var html = BuildHtml(
            outputDir, sampleCols.Count, sampleTypes,
            pepRaw.RowCount, protRaw.RowCount,
            pepRef, pepQc, protRef, protQc, peptidePlots, proteinPlots, validation);

        var htmlPath = Path.Combine(outputDir, "qc_report.html");
        File.WriteAllText(htmlPath, html);
        return htmlPath;
    }

    private sealed record PlotImage(string Caption, byte[] Png);
    private sealed record PlotSection(string Title, List<PlotImage> Images);

    /// <summary>
    /// Render the before/after comparison sections for one level, mirroring the Python report:
    /// intensity distribution (before vs after), PCA (before vs after), and comparative CV
    /// distributions for reference and QC samples.
    /// </summary>
    private static List<PlotSection> RenderLevelSections(
        string level, Matrix raw, Matrix corrected, IReadOnlyList<string> typeLabels,
        IReadOnlyList<int> refIdx, IReadOnlyList<int> qcIdx, bool savePlots, string plotsDir, bool rtLowessRan)
    {
        var cap = char.ToUpperInvariant(level[0]) + level[1..];
        var sections = new List<PlotSection>();

        PlotImage Img(string caption, string fileStem, Func<byte[]> render)
        {
            try
            {
                var png = render();
                if (savePlots && png.Length > 0)
                    File.WriteAllBytes(Path.Combine(plotsDir, fileStem + ".png"), png);
                return new PlotImage(caption, png);
            }
            catch (Exception ex)
            {
                // Rendering can fail on a headless host missing fontconfig; keep the report.
                return new PlotImage(caption + " (render failed: " + ex.GetType().Name + ")", Array.Empty<byte>());
            }
        }

        // Per-sample intensity density curves; the title reports how much normalization tightened the
        // spread of per-sample medians (the Python "Median range: X -> Y (Z% reduction)" super-title).
        var beforeMedRange = MedianRange(raw.Values);
        var afterMedRange = MedianRange(corrected.Values);
        var medReduction = beforeMedRange > 0 ? (1.0 - afterMedRange / beforeMedRange) * 100.0 : 0.0;
        sections.Add(new PlotSection(
            // Spell out that this spans ALL samples. The CV tables above are computed WITHIN the reference
            // group and WITHIN the QC group, so a large reduction here does not imply a large CV
            // improvement there: normalization removes a per-sample offset, and if the controls were
            // already aligned with each other there is little of it left inside those groups to remove.
            $"{cap} Intensity Distribution (all samples): median range {beforeMedRange:0.00} -> "
            + $"{afterMedRange:0.00} log2 ({medReduction:0.0}% reduction between samples)",
            new List<PlotImage>
            {
                Img("Before normalization", $"{level}_intensity_before",
                    () => PlotRenderer.IntensityDistribution(raw.Values, typeLabels, "Before Normalization")),
                Img("After normalization", $"{level}_intensity_after",
                    () => PlotRenderer.IntensityDistribution(corrected.Values, typeLabels, "After Normalization")),
            }));

        sections.Add(new PlotSection($"{cap} PCA: Before vs After", new List<PlotImage>
        {
            Img("Before (raw rollup)", $"{level}_pca_before",
                () => PlotRenderer.PcaScatter(Pca.Fit2D(Transpose(raw.Values)), typeLabels, "Before")),
            Img("After (normalized + corrected)", $"{level}_pca_after",
                () => PlotRenderer.PcaScatter(Pca.Fit2D(Transpose(corrected.Values)), typeLabels, "After")),
        }));

        if (refIdx.Count >= 2)
            sections.Add(new PlotSection($"{cap} CV Distribution (Reference): Before vs After", new List<PlotImage>
            {
                Img("", $"{level}_cv_reference", () => PlotRenderer.CvComparison(
                    CvMetrics.PerFeatureCvs(raw.Values, refIdx),
                    CvMetrics.PerFeatureCvs(corrected.Values, refIdx),
                    $"{cap} CV (Reference)", "#d62728")),
            }));

        if (qcIdx.Count >= 2)
            sections.Add(new PlotSection($"{cap} CV Distribution (QC): Before vs After", new List<PlotImage>
            {
                Img("", $"{level}_cv_qc", () => PlotRenderer.CvComparison(
                    CvMetrics.PerFeatureCvs(raw.Values, qcIdx),
                    CvMetrics.PerFeatureCvs(corrected.Values, qcIdx),
                    $"{cap} CV (QC)", "#ff7f0e")),
            }));

        // Control-sample (reference + QC) correlation heatmap, before vs after.
        var controlIdx = refIdx.Concat(qcIdx).Distinct().OrderBy(x => x).ToList();
        var controlTypes = controlIdx.Select(i => typeLabels[i]).ToList();
        if (controlIdx.Count >= 2)
            sections.Add(new PlotSection($"{cap} Control-Sample Correlation: Before vs After", new List<PlotImage>
            {
                Img("Before (raw rollup)", $"{level}_control_corr_before",
                    () => PlotRenderer.CorrelationHeatmap(raw.Values, controlIdx, $"{cap} Control Sample Correlation (Before Correction)", controlTypes)),
                Img("After (normalized + corrected)", $"{level}_control_corr_after",
                    () => PlotRenderer.CorrelationHeatmap(corrected.Values, controlIdx, $"{cap} Control Sample Correlation (After Correction)", controlTypes)),
            }));

        // RT-dependent diagnostics (peptide level only - proteins have no RT).
        if (rtLowessRan && raw.MeanRt is not null && corrected.MeanRt is not null)
        {
            sections.Add(new PlotSection($"{cap} RT-Lowess Curves: Before vs After", new List<PlotImage>
            {
                Img("Before (raw rollup)", $"{level}_rt_lowess_before",
                    () => PlotRenderer.RtLowessCurves(raw.Values, raw.MeanRt, typeLabels, "Before")),
                Img("After (normalized + corrected)", $"{level}_rt_lowess_after",
                    () => PlotRenderer.RtLowessCurves(corrected.Values, corrected.MeanRt, typeLabels, "After")),
            }));

            if (refIdx.Count >= 2)
                sections.Add(new PlotSection($"{cap} RT-Binned CV (Reference)", new List<PlotImage>
                {
                    Img("", $"{level}_rt_bin_cv_ref", () => PlotRenderer.RtBinCv(
                        raw.Values, corrected.Values, raw.MeanRt, refIdx, "RT-binned CV (Reference)", "#d62728")),
                }));
            if (qcIdx.Count >= 2)
                sections.Add(new PlotSection($"{cap} RT-Binned CV (QC)", new List<PlotImage>
                {
                    Img("", $"{level}_rt_bin_cv_qc", () => PlotRenderer.RtBinCv(
                        raw.Values, corrected.Values, raw.MeanRt, qcIdx, "RT-binned CV (QC)", "#ff7f0e")),
                }));

            sections.Add(new PlotSection($"{cap} Abundance by RT Bin: Before vs After", new List<PlotImage>
            {
                Img("Before (raw rollup)", $"{level}_rt_bin_box_before",
                    () => PlotRenderer.RtBinBoxplot(raw.Values, raw.MeanRt, "Before", "#1f77b4")),
                Img("After (normalized + corrected)", $"{level}_rt_bin_box_after",
                    () => PlotRenderer.RtBinBoxplot(corrected.Values, corrected.MeanRt, "After", "#1f77b4")),
            }));
        }

        return sections;
    }

    private static string BuildHtml(
        string outputDir, int nSamples, IReadOnlyDictionary<string, string> sampleTypes,
        int nPeptides, int nProteins,
        CvMetrics.BeforeAfter? pepRef, CvMetrics.BeforeAfter? pepQc,
        CvMetrics.BeforeAfter? protRef, CvMetrics.BeforeAfter? protQc,
        List<PlotSection> peptidePlots, List<PlotSection> proteinPlots,
        ValidationStatus? validation)
    {
        var nRef = sampleTypes.Values.Count(v => v == "reference");
        var nQc = sampleTypes.Values.Count(v => v == "qc");
        var nExp = sampleTypes.Values.Count(v => v == "experimental");

        var sb = new StringBuilder();
        sb.Append("<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"><title>PRISM QC Report</title>\n");
        sb.Append(FaviconLinkTag);
        // The plot font, resolved, so the page text matches the axis labels in the images below it.
        sb.Append("<style>\nbody { font-family: ")
          .Append(PlotRenderer.HtmlFontStack)
          .Append("; color: #222; margin: 0; padding: 24px; }\n");
        sb.Append("""
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
.plot-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }
.plot-item { flex: 1 1 0; min-width: 320px; text-align: center; }
.plot-item .cap { color: #555; font-size: 13px; margin-top: 2px; }
.note { color: #555; font-size: 13px; max-width: 900px; margin: 6px 0 12px; line-height: 1.45; }
.footer { color: #888; font-size: 12px; margin-top: 32px; }
.status { padding: 10px 16px; border-radius: 6px; font-weight: 700; font-size: 16px; margin: 12px 0; }
.status-pass { background: #e6f4ea; color: #1a7f37; border: 1px solid #a6d8b6; }
.status-fail { background: #fdeceb; color: #b42318; border: 1px solid #f0b3ad; }
.warnings { background: #fff8e6; border: 1px solid #f2d98a; border-radius: 6px; padding: 8px 14px; margin: 8px 0; }
.warnings li { color: #8a6d00; }
</style></head><body><div class="container">
""");
        sb.Append("<h1>PRISM QC Report</h1>");

        sb.Append("<div class=\"box\"><h2>Dataset Summary</h2>");
        sb.Append($"<p>Samples: <strong>{nSamples}</strong> (experimental {nExp}, reference {nRef}, qc {nQc})<br>");
        sb.Append($"Peptides: <strong>{nPeptides}</strong> &nbsp; Proteins: <strong>{nProteins}</strong></p>");
        sb.Append($"<p style=\"color:#666\">Output directory: <code>{HtmlEncode(outputDir)}</code></p>");
        sb.Append("</div>");

        AppendValidation(sb, validation);

        sb.Append("<h2>Summary Metrics (Median CV %)</h2>");
        // Readers reasonably expect these to track the intensity-distribution reduction below, and they do
        // not: that figure spans every sample, while each CV row is computed only among samples of that
        // one type. Aligning a cohort mostly moves the experimental samples onto the controls.
        sb.Append(
            "<p class=\"note\">Each row is the median CV <em>within</em> that sample type only "
            + "(reference vs reference, QC vs QC), on the linear scale. It is not comparable to the "
            + "between-sample reduction reported for the intensity distribution below, which spans all "
            + "samples - normalization removes a per-sample offset, so it improves these CVs only to the "
            + "extent that samples of the same type were offset from <em>each other</em>.</p>");
        sb.Append(CvTable("Peptide-Level CV", pepRef, pepQc));
        sb.Append(CvTable("Protein-Level CV", protRef, protQc));

        sb.Append("<div class=\"section-header\">Peptide-Level QC</div>");
        AppendSections(sb, peptidePlots);
        sb.Append("<div class=\"section-header\">Protein-Level QC</div>");
        AppendSections(sb, proteinPlots);

        sb.Append($"<p class=\"footer\">Generated by Skyline-PRISM (C#) at {DateTimeStamp()}</p>");
        sb.Append("</div></body></html>");
        return sb.ToString();
    }

    private static void AppendValidation(StringBuilder sb, ValidationStatus? v)
    {
        if (v is null)
        {
            sb.Append("<div class=\"box\" style=\"color:#666\">Validation verdict needs &gt;=2 reference "
                + "and &gt;=2 QC samples (dual-control design); not enough of both were found.</div>");
            return;
        }

        sb.Append($"<div class=\"status {(v.Passed ? "status-pass" : "status-fail")}\">"
            + $"Validation: {(v.Passed ? "PASSED" : "FAILED")}</div>");

        sb.Append("<table><tr><th>Control CV</th><th>Before</th><th>After</th><th>Improvement</th></tr>");
        void Row(string label, double before, double after, double impFrac)
        {
            var cls = impFrac >= 0 ? "improvement-positive" : "improvement-negative";
            sb.Append($"<tr><td>{label}</td><td>{before:0.0}%</td><td>{after:0.0}%</td>");
            sb.Append($"<td class=\"{cls}\">{impFrac * 100:+0.0;-0.0}%</td></tr>");
        }
        Row("Reference", v.ReferenceCvBefore, v.ReferenceCvAfter, v.ReferenceCvImprovement);
        Row("QC", v.QcCvBefore, v.QcCvAfter, v.QcCvImprovement);
        sb.Append("</table>");

        var rvr = double.IsInfinity(v.RelativeVarianceReduction) ? "inf" : v.RelativeVarianceReduction.ToString("0.00");
        var pca = double.IsNaN(v.PcaDistanceRatio) ? "n/a" : v.PcaDistanceRatio.ToString("0.00");
        sb.Append($"<p style=\"color:#555\">Relative variance reduction (QC/reference improvement): "
            + $"<strong>{rvr}</strong> (overfitting if &gt; 2.0) &nbsp;&middot;&nbsp; "
            + $"PCA QC-reference distance ratio (after/before): <strong>{pca}</strong> (collapse if &lt; 0.5)</p>");

        if (v.Warnings.Count > 0)
        {
            sb.Append("<div class=\"warnings\"><strong>WARNINGS</strong><ul>");
            foreach (var w in v.Warnings)
                sb.Append($"<li>{HtmlEncode(w)}</li>");
            sb.Append("</ul></div>");
        }
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

    private static void AppendSections(StringBuilder sb, List<PlotSection> sections)
    {
        foreach (var sec in sections)
        {
            sb.Append($"<div class=\"plot-section\"><h3>{HtmlEncode(sec.Title)}</h3><div class=\"plot-row\">");
            foreach (var img in sec.Images)
            {
                sb.Append("<div class=\"plot-item\">");
                if (img.Png.Length > 0)
                    sb.Append($"<img src=\"data:image/png;base64,{Convert.ToBase64String(img.Png)}\" alt=\"{HtmlEncode(img.Caption)}\" />");
                if (!string.IsNullOrEmpty(img.Caption))
                    sb.Append($"<div class=\"cap\">{HtmlEncode(img.Caption)}</div>");
                sb.Append("</div>");
            }
            sb.Append("</div></div>");
        }
    }

    // -- helpers --

    private sealed record Matrix(double[,] Values, List<string> SampleCols, int RowCount, double[]? MeanRt);

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

        double[]? meanRt = null;
        if (table.HasColumn(PepMetaRt))
        {
            var rt = table.GetDouble(PepMetaRt);
            meanRt = new double[n];
            for (var i = 0; i < n; i++)
                meanRt[i] = rt[i] ?? double.NaN;
        }
        return new Matrix(m, sampleCols, n, meanRt);
    }

    private static Matrix Log2(Matrix linear)
    {
        var n = linear.Values.GetLength(0);
        var c = linear.Values.GetLength(1);
        var m = new double[n, c];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < c; j++)
                m[i, j] = Math.Log2(linear.Values[i, j]);
        return new Matrix(m, linear.SampleCols, n, linear.MeanRt);
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

    /// <summary>Spread (max - min) of the per-sample median LOG2 abundance; 0 when no data.</summary>
    private static double MedianRange(double[,] log2Matrix)
    {
        var nF = log2Matrix.GetLength(0);
        var nS = log2Matrix.GetLength(1);
        var buf = new double[nF];
        double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
        for (var s = 0; s < nS; s++)
        {
            var n = 0;
            for (var f = 0; f < nF; f++)
            {
                var v = log2Matrix[f, s];
                if (!double.IsNaN(v))
                    buf[n++] = v;
            }
            if (n == 0)
                continue;
            var med = Numerics.Stats.NanMedian(buf.AsSpan(0, n));
            if (med < lo) lo = med;
            if (med > hi) hi = med;
        }
        return hi >= lo ? hi - lo : 0.0;
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
        var cols = ParquetTable.ReadColumnNames(peptidesRollupParquet); // schema-only
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
