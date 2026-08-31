using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Core.Qc;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Layer 8: CV metric computation (validation.py:calc_median_cv) and end-to-end QC report
/// generation. Plots render via ScottPlot/SkiaSharp; on a headless host missing fontconfig
/// they fail gracefully (the report still contains the CV tables), so the plot assertion is
/// tolerant.
/// </summary>
public class QcReportTests
{
    [Fact]
    public void MedianCv_MatchesHandComputed()
    {
        // 1 feature, 2 samples log2=[10,11] -> linear [1024,2048]; mean 1536;
        // std(ddof=1)=724.0773...; cv = 47.1404...%.
        var m = new double[,] { { 10.0, 11.0 } };
        var cv = CvMetrics.MedianCv(m, new[] { 0, 1 });
        Assert.True(Math.Abs(cv - 47.14045208) < 1e-6, $"cv={cv}");
    }

    [Fact]
    public void MedianCv_ZeroForConstantFeature()
    {
        var m = new double[,] { { 12.0, 12.0, 12.0 } };
        Assert.Equal(0.0, CvMetrics.MedianCv(m, new[] { 0, 1, 2 }), 9);
    }

    /// <summary>
    /// The peptide "after" numbers must come from the peptide arm's ACTUAL output. Reading
    /// peptides_log2_internal instead (post-normalization, PRE-ComBat since dotnet-v26.15.0) made
    /// every peptide CV, plot and the validation verdict measure normalization alone while the
    /// panels said "normalized + corrected".
    /// </summary>
    [Fact]
    public void PeptideAfterMatrix_IsTheBatchCorrectedOutput_NotThePreCombatInternalFile()
    {
        var mergeDir = Fixtures.Path2("mini", "merge");
        var inputs = new[]
        {
            Path.Combine(mergeDir, "mini_plate1.csv"),
            Path.Combine(mergeDir, "mini_plate2.csv"),
        };
        var config = PrismConfig.Load(Path.Combine(Fixtures.Path2("mini", "e2e-medpolish"), "config.yaml"));
        config.BatchCorrection.Enabled = true; // two batches -> ComBat actually moves the numbers
        config.QcReport.Enabled = false;

        var dir = Path.Combine(Path.GetTempPath(), "prism_qcafter_" + Guid.NewGuid().ToString("N"));
        try
        {
            PrismPipeline.Run(inputs, dir, config);

            // Sanity: ComBat did move the peptides, so the two candidate files really do differ.
            var internalCv = MedianControlCv(dir, "peptides_log2_internal.parquet", log2Input: true);
            var correctedCv = MedianControlCv(dir, "corrected_peptides.parquet", log2Input: false);
            // A wide margin, so the test cannot quietly stop discriminating: on this fixture the
            // reference median CV is ~117.6% pre-ComBat and ~34.1% after it.
            Assert.True(Math.Abs(internalCv - correctedCv) > 1.0,
                $"fixture no longer exercises ComBat (internal {internalCv} vs corrected {correctedCv})");

            var html = File.ReadAllText(QcReport.Generate(dir, config, savePlots: false));
            var reported = PeptideAfterCvFromHtml(html);
            Assert.True(Math.Abs(reported - correctedCv) < 0.05,
                $"report shows {reported}% for the peptide 'after'; corrected_peptides is {correctedCv}%, "
                + $"peptides_log2_internal is {internalCv}%");
        }
        finally
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>Median LINEAR-scale CV across reference samples, read straight from a matrix file.</summary>
    private static double MedianControlCv(string dir, string file, bool log2Input)
    {
        var types = File.ReadAllLines(Path.Combine(dir, "sample_metadata.csv"));
        var header = types[0].Split(',');
        var idIdx = Array.IndexOf(header, "sample_id");
        var typeIdx = Array.IndexOf(header, "sample_type");
        var refs = types.Skip(1)
            .Select(l => l.Split(','))
            .Where(f => f.Length > Math.Max(idIdx, typeIdx) && f[typeIdx] == "reference")
            .Select(f => f[idIdx])
            .ToHashSet(StringComparer.Ordinal);

        using var reader = ParquetColumnReader.Open(Path.Combine(dir, file));
        var cols = reader.ColumnNames.Where(c => refs.Contains(c)).ToList();
        var values = cols.Select(reader.ReadDoubles).ToList();
        var cvs = new List<double>();
        for (var row = 0; row < reader.RowCount; row++)
        {
            var linear = values
                .Select(v => log2Input ? Math.Pow(2, v[row]) : v[row])
                .Where(x => !double.IsNaN(x))
                .ToList();
            if (linear.Count < 2)
                continue;
            var mean = linear.Average();
            var sd = Math.Sqrt(linear.Sum(x => (x - mean) * (x - mean)) / (linear.Count - 1));
            if (mean > 0)
                cvs.Add(sd / mean * 100.0);
        }
        cvs.Sort();
        return cvs.Count == 0 ? double.NaN
            : cvs.Count % 2 == 1 ? cvs[cvs.Count / 2]
            : (cvs[cvs.Count / 2 - 1] + cvs[cvs.Count / 2]) / 2.0;
    }

    /// <summary>The "After" cell of the Peptide-Level CV table's Reference row.</summary>
    private static double PeptideAfterCvFromHtml(string html)
    {
        var table = html[html.IndexOf("Peptide-Level CV", StringComparison.Ordinal)..];
        var row = table[table.IndexOf("<td>Reference</td>", StringComparison.Ordinal)..];
        var cells = System.Text.RegularExpressions.Regex.Matches(row, @"<td[^>]*>([\d.+-]+)%</td>");
        return double.Parse(cells[1].Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
    }

    [Fact]
    public void Generate_ProducesHtmlWithCvTables()
    {
        var mergeDir = Fixtures.Path2("mini", "merge");
        var inputs = new[]
        {
            Path.Combine(mergeDir, "mini_plate1.csv"),
            Path.Combine(mergeDir, "mini_plate2.csv"),
        };

        // Enable QC in the config used for the run.
        var config = PrismConfig.Load(Path.Combine(Fixtures.Path2("mini", "e2e-sum"), "config.yaml"));
        config.QcReport.Enabled = true;
        config.QcReport.SavePlots = false;

        var tempOut = Path.Combine(Path.GetTempPath(), "prism_qc_" + Guid.NewGuid().ToString("N"));
        try
        {
            PrismPipeline.Run(inputs, tempOut, config);

            var htmlPath = Path.Combine(tempOut, "qc_report.html");
            Assert.True(File.Exists(htmlPath), "qc_report.html not generated");
            var html = File.ReadAllText(htmlPath);

            Assert.Contains("PRISM QC Report", html);

            // A report has to say what produced it and with which settings, or its numbers cannot be
            // traced back to a run. Version and date come from the run's own parameters.json.
            var version = System.Reflection.Assembly.GetAssembly(typeof(QcReport))!
                .GetName().Version!.ToString();
            Assert.Contains("Analysis Information", html);
            Assert.Contains($"PRISM v{version}", html);
            Assert.Contains("Processing date", html);
            Assert.Contains("Computer", html);
            Assert.Contains("Processing Parameters", html);
            Assert.Contains("Transition -&gt; peptide rollup", html);
            Assert.Contains($"method={config.TransitionRollup.Method}", html);
            Assert.Contains("Peptide normalization", html);
            Assert.Contains("Peptide -&gt; protein rollup", html);
            Assert.Contains("Protein normalization", html);
            Assert.Contains("Batch correction", html);
            Assert.Contains("Full configuration (YAML)", html);

            Assert.Contains("Peptide-Level CV", html);
            Assert.Contains("Protein-Level CV", html);
            Assert.Contains("Peptide-Level QC", html);
            Assert.Contains("Protein-Level QC", html);
            Assert.Contains("Reference", html);
            Assert.Matches(@"\d+\.\d%", html); // a CV percentage cell
            // Plots embed as base64 when the renderer is available (fontconfig on Linux CI).
            Assert.True(html.Contains("data:image/png;base64,") || html.Contains("render failed"),
                "expected embedded plots or a graceful render-failed marker");
        }
        finally
        {
            if (Directory.Exists(tempOut))
                Directory.Delete(tempOut, recursive: true);
        }
    }

    /// <summary>
    /// A run whose sample types never arrived must SAY SO WITH THE COUNTS, not merely decline to reach a
    /// verdict.
    ///
    /// <para>This is what a failed headless metadata export does to a cohort. Measured, in the field log:
    /// the metadata export for one of two plates crashed on launch, PRISM fell back to inferring sample
    /// types from replicate names, the inference matched nothing, and the run completed reporting
    /// "0 reference, 0 qc, 192 experimental" - against the 16/16/160 the same cohort produces when the
    /// export succeeds. So 32 control samples silently became experimental, and the QC report's
    /// dual-control verdict had nothing to validate.</para>
    ///
    /// <para>The report said only that it needed "&gt;=2 reference and &gt;=2 QC samples ... not enough of
    /// both were found", which reads as a fact about the study design - indistinguishable from a cohort
    /// that never had controls. It must instead give the counts, so 0-of-N reads as the alarm it is, and
    /// name the metadata as the thing to check.</para>
    /// </summary>
    [Fact]
    public void AReportWithNoControlsNamesTheCountsAndPointsAtTheMetadata()
    {
        var mergeDir = Fixtures.Path2("mini", "merge");
        var inputs = new[]
        {
            Path.Combine(mergeDir, "mini_plate1.csv"),
            Path.Combine(mergeDir, "mini_plate2.csv"),
        };
        var config = PrismConfig.Load(Path.Combine(Fixtures.Path2("mini", "e2e-medpolish"), "config.yaml"));
        config.QcReport.Enabled = false;

        var dir = Path.Combine(Path.GetTempPath(), "prism_qcnoctl_" + Guid.NewGuid().ToString("N"));
        try
        {
            PrismPipeline.Run(inputs, dir, config);

            var metadata = Path.Combine(dir, "sample_metadata.csv");
            var lines = File.ReadAllLines(metadata);
            var header = CsvLine.Split(lines[0]);
            var typeIdx = CsvLine.IndexOf(header, "sample_type");
            Assert.True(typeIdx >= 0, "sample_metadata.csv has no sample_type column");

            // Exactly what losing the metadata does: every sample typed experimental. Asserting the
            // fixture HAD controls first, so this cannot pass by testing a cohort that never had any.
            var hadControls = lines.Skip(1)
                .Where(l => !string.IsNullOrWhiteSpace(l))
                .Select(l => CsvLine.Split(l)[typeIdx])
                .Any(t => t is "reference" or "qc");
            Assert.True(hadControls, "fixture has no controls, so this test would prove nothing");

            for (var i = 1; i < lines.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(lines[i]))
                    continue;
                var f = CsvLine.Split(lines[i]);
                f[typeIdx] = "experimental";
                lines[i] = string.Join(",", f);
            }
            File.WriteAllLines(metadata, lines);

            var html = File.ReadAllText(QcReport.Generate(dir, config, savePlots: false));

            // The counts, which is the whole point - a reader must be able to see 0 of N.
            Assert.Contains("<strong>0</strong> reference", html);
            Assert.Contains("<strong>0</strong> QC", html);
            // And where to look. Only emitted when BOTH are zero, which is the signature of types never
            // arriving rather than of a cohort with few controls.
            Assert.Contains("No controls were identified at all", html);
            Assert.Contains("check the replicate metadata", html);
        }
        finally
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
    }
}
