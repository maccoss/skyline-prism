using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
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
}
