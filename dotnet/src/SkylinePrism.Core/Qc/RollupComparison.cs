using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Numerics;
using SkylinePrism.Core.Visualization;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// Compare two PRISM runs' corrected_peptides by control-sample CV: which rollup/config gives
/// tighter CVs, with peptides ranked by the CV difference. A general port of `prism compare` (the
/// Python version additionally does library-fit visualization, which requires a library_assist run).
/// </summary>
public static class RollupComparison
{
    /// <summary>Generate a self-contained HTML comparison report; returns the output path.</summary>
    public static string Generate(string run1Dir, string run2Dir, string outputPath, string sampleType, int topN)
    {
        var r1 = LoadRun(run1Dir, sampleType);
        var r2 = LoadRun(run2Dir, sampleType);

        var shared = r1.Cvs.Keys.Where(r2.Cvs.ContainsKey).ToList();
        var diffs = shared
            .Select(p => (Peptide: p, Cv1: r1.Cvs[p], Cv2: r2.Cvs[p], Diff: r1.Cvs[p] - r2.Cvs[p]))
            .Where(x => !double.IsNaN(x.Diff))
            .ToList();

        var med1 = shared.Select(p => r1.Cvs[p]).Where(v => !double.IsNaN(v)).ToArray();
        var med2 = shared.Select(p => r2.Cvs[p]).Where(v => !double.IsNaN(v)).ToArray();
        var medianCv1 = med1.Length > 0 ? Stats.NanMedian(med1) : double.NaN;
        var medianCv2 = med2.Length > 0 ? Stats.NanMedian(med2) : double.NaN;

        var improved = diffs.OrderByDescending(x => x.Diff).Take(topN).ToList(); // run2 lower CV
        var worsened = diffs.OrderBy(x => x.Diff).Take(topN).ToList();           // run2 higher CV

        var html = BuildHtml(
            run1Dir, run2Dir, r1.Method, r2.Method, sampleType,
            shared.Count, medianCv1, medianCv2, improved, worsened);
        File.WriteAllText(outputPath, html);
        return outputPath;
    }

    private sealed record Run(string Method, Dictionary<string, double> Cvs);

    private static Run LoadRun(string dir, string sampleType)
    {
        var pepPath = Path.Combine(dir, "corrected_peptides.parquet");
        if (!File.Exists(pepPath))
            throw new FileNotFoundException($"Missing corrected_peptides.parquet in {dir}", pepPath);

        var table = ParquetTable.Load(pepPath);
        var types = ReadSampleTypes(Path.Combine(dir, "sample_metadata.csv"));

        bool Wanted(string s) => sampleType == "all"
            || string.Equals(types.GetValueOrDefault(s, "experimental"), sampleType, StringComparison.OrdinalIgnoreCase);
        var sampleCols = table.ColumnNames.Where(c => types.ContainsKey(c) && Wanted(c)).ToList();

        var peptideCol = table.ColumnNames.FirstOrDefault(
            c => c != "n_transitions" && c != "mean_rt" && !c.Contains("__@__") && !types.ContainsKey(c))
            ?? "Peptide Modified Sequence";
        var peptides = table.GetString(peptideCol);
        var cols = sampleCols.Select(table.GetDouble).ToList();

        // Per-peptide CV over the selected samples (corrected_peptides is LINEAR).
        var cvs = new Dictionary<string, double>(StringComparer.Ordinal);
        var buf = new double[sampleCols.Count];
        for (var i = 0; i < table.RowCount; i++)
        {
            var cnt = 0;
            for (var j = 0; j < sampleCols.Count; j++)
            {
                var v = cols[j][i];
                if (v.HasValue && !double.IsNaN(v.Value) && v.Value > 0)
                    buf[cnt++] = v.Value;
            }
            var key = peptides[i] ?? "";
            if (cnt < 2 || key.Length == 0)
                continue;
            double mean = 0;
            for (var k = 0; k < cnt; k++) mean += buf[k];
            mean /= cnt;
            double ss = 0;
            for (var k = 0; k < cnt; k++) { var d = buf[k] - mean; ss += d * d; }
            var std = Math.Sqrt(ss / (cnt - 1));
            cvs[key] = mean > 0 ? std / mean * 100.0 : double.NaN;
        }

        return new Run(ReadMethod(dir), cvs);
    }

    private static string ReadMethod(string dir)
    {
        var path = Path.Combine(dir, "parameters.json");
        try
        {
            if (File.Exists(path))
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(path));
                if (doc.RootElement.TryGetProperty("processing_parameters", out var pp)
                    && pp.TryGetProperty("transition_rollup", out var tr)
                    && tr.TryGetProperty("method", out var m))
                    return m.GetString() ?? Path.GetFileName(dir);
            }
        }
        catch { /* fall through to dir name */ }
        return Path.GetFileName(dir.TrimEnd(Path.DirectorySeparatorChar));
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

    private static string BuildHtml(
        string run1Dir, string run2Dir, string method1, string method2, string sampleType,
        int nShared, double medianCv1, double medianCv2,
        List<(string Peptide, double Cv1, double Cv2, double Diff)> improved,
        List<(string Peptide, double Cv1, double Cv2, double Diff)> worsened)
    {
        var sb = new StringBuilder();
        sb.Append("<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>PRISM Rollup Comparison</title><style>");
        // The plot font, resolved, so the page text matches the plots it sits alongside.
        sb.Append($"body{{font-family:{PlotRenderer.HtmlFontStack};color:#222;margin:24px;}}");
        sb.Append("h1,h2{color:#1a3c6e;} table{border-collapse:collapse;margin:8px 0;} ");
        sb.Append("th,td{border:1px solid #cfd8e3;padding:6px 12px;text-align:right;} th{background:#eaf0f7;} ");
        sb.Append("td:first-child,th:first-child{text-align:left;} .better{color:#1a7f37;} .worse{color:#b42318;}");
        sb.Append("</style></head><body>");
        sb.Append("<h1>PRISM Rollup Comparison</h1>");
        sb.Append($"<p>Run 1 (<strong>{Enc(method1)}</strong>): <code>{Enc(run1Dir)}</code><br>");
        sb.Append($"Run 2 (<strong>{Enc(method2)}</strong>): <code>{Enc(run2Dir)}</code><br>");
        sb.Append($"Sample type: <strong>{Enc(sampleType)}</strong> &nbsp; Shared peptides: <strong>{nShared:N0}</strong></p>");

        var delta = medianCv1 - medianCv2;
        var cls = delta >= 0 ? "better" : "worse";
        sb.Append("<h2>Median CV</h2><table><tr><th>Run</th><th>Method</th><th>Median CV %</th></tr>");
        sb.Append($"<tr><td>Run 1</td><td>{Enc(method1)}</td><td>{medianCv1:0.00}</td></tr>");
        sb.Append($"<tr><td>Run 2</td><td>{Enc(method2)}</td><td>{medianCv2:0.00}</td></tr></table>");
        sb.Append($"<p>Run 2 is <span class=\"{cls}\">{Math.Abs(delta):0.00}% CV {(delta >= 0 ? "lower" : "higher")}</span> "
            + "than Run 1 at the median.</p>");

        void Table(string title, List<(string Peptide, double Cv1, double Cv2, double Diff)> rows)
        {
            sb.Append($"<h2>{title}</h2><table><tr><th>Peptide</th><th>Run1 CV%</th><th>Run2 CV%</th><th>&Delta; (Run1-Run2)</th></tr>");
            foreach (var r in rows)
            {
                var c = r.Diff >= 0 ? "better" : "worse";
                sb.Append($"<tr><td>{Enc(r.Peptide)}</td><td>{r.Cv1:0.0}</td><td>{r.Cv2:0.0}</td>"
                    + $"<td class=\"{c}\">{r.Diff:+0.0;-0.0}</td></tr>");
            }
            sb.Append("</table>");
        }
        Table($"Top {improved.Count}: Run 2 improved (lower CV)", improved);
        Table($"Top {worsened.Count}: Run 2 worsened (higher CV)", worsened);

        sb.Append("</body></html>");
        return sb.ToString();
    }

    private static string Enc(string s) => s
        .Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");
}
