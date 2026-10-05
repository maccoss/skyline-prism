using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.DifferentialAnalysis.Detection;
using SkylinePrism.Core.DifferentialAnalysis.Enrichment;
using SkylinePrism.Core.Qc;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Tests for <see cref="QuantReport"/> against the committed mini output fixture: it must write the
/// self-contained HTML plus its companion CSVs, cap long tables to a preview, and export the raw
/// per-sample abundances in LINEAR scale (2^log2), the same scale as PRISM's corrected parquet.
/// </summary>
public class QuantReportTests
{
    private static string MiniOutput => Fixtures.Path2("mini", "e2e-sum", "output");

    // One options object for the run AND the report inputs, as the pane and the CLI both pass the
    // options the contrast actually ran under.
    private static readonly DifferentialOptions Options = new() { Prior = VariancePrior.Global };

    private static (DifferentialDataset Ds, DifferentialResult Res, List<int> Cols) RunContrast()
    {
        var ds = DifferentialDataset.Load(MiniOutput, FeatureLevel.Protein);
        var types = ds.MetadataValues("sample_type");
        var experimental = Enumerable.Range(0, ds.SampleIds.Length)
            .Where(j => types[j] == "experimental").ToList();
        var half = experimental.Count / 2;
        var groupA = experimental.Take(half).ToList();
        var groupB = experimental.Skip(half).ToList();
        var res = Differential.Run(ds.ExprLog2, ds.FeatureIds, groupA, groupB, Options);
        return (ds, res, groupA.Concat(groupB).ToList());
    }

    private static QuantConfig ConfigFor(DifferentialResult res) => new(
        Level: "protein",
        Contrast: new QuantContrast("sample_type", new[] { "first half" }, new[] { "second half" }, null),
        Design: "unpaired",
        Test: "moderated",
        Prior: "global",
        PriorUsed: res.VariancePrior,
        Correction: "bh",
        Covariates: res.CovariatesUsed,
        HitRule: "adj.P < 0.05, |log2FC| >= 1",
        DetectionEnabled: false,
        DetectionQ: 0.01,
        EnrichmentEnabled: false,
        EnrichmentSources: Array.Empty<string>(),
        EnrichmentDirection: "both",
        MarkerPanels: Array.Empty<string>());

    private static QuantReportInputs InputsFor(DifferentialDataset ds, DifferentialResult res,
        List<int>? cols, SignificanceRule rule, DifferentialOptions? options = null) => new()
    {
        Differential = res,
        Rule = rule,
        Corrected = true,
        Contrast = "experimental split A vs B",
        EffectName = "log2FC",
        LabelFor = id => id,
        Options = options ?? Options,
        GroupBy = "sample_type",
        ALabel = "first half",
        BLabel = "second half",
        Dataset = ds,
        ContrastColumns = cols,
    };

    [Fact]
    public void Write_ProducesReportAndCompanionFiles()
    {
        var (ds, res, cols) = RunContrast();
        var dir = Path.Combine(Path.GetTempPath(), $"prism-quant-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            // No parameters.json in this temp dir: the header falls back gracefully rather than throwing.
            var htmlPath = QuantReport.Write(dir, ConfigFor(res), InputsFor(ds, res, cols, SignificanceRule.Default));

            Assert.EndsWith("quant_report.html", htmlPath);
            var quant = Path.Combine(dir, "quant");
            Assert.True(File.Exists(htmlPath));
            Assert.True(File.Exists(Path.Combine(quant, "quant_parameters.yaml")));
            Assert.True(File.Exists(Path.Combine(quant, "quant_parameters.json")));
            Assert.True(File.Exists(Path.Combine(quant, "differential.csv")));
            Assert.True(File.Exists(Path.Combine(quant, "differential_values.csv")));

            // differential.csv is `prism differential`'s file: four provenance lines, the column header,
            // then one row per tested feature.
            var diffLines = File.ReadAllLines(Path.Combine(quant, "differential.csv"));
            Assert.Equal(res.Rows.Count + 5, diffLines.Length);
            Assert.StartsWith("# contrast: sample_type = second half vs first half", diffLines[0]);
            Assert.StartsWith("# method:", diffLines[1]);
            Assert.Equal("feature_id,label,gene,protein,accession,log2fc,fc,ave_expr,statistic,"
                + "p_value,adj_p_value,mean_a,mean_b", diffLines[4]);

            // The HTML is self-contained: title, contrast, parameters block, and an embedded volcano PNG.
            var html = File.ReadAllText(htmlPath);
            Assert.Contains("PRISM Quantification Report", html);
            Assert.Contains("Quantification Parameters", html);
            Assert.Contains("experimental split A vs B", html);
            Assert.Contains("data:image/png;base64,", html);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Write_IntensityTrendPrior_SaysHowItWasFitted_InTheCsvAndTheReport()
    {
        // One description behind both: the CSV's "# prior:" line straight after "# method:", and the
        // report's parameter row, which names the field itself and so drops the label.
        var ds = DifferentialDataset.Load(MiniOutput, FeatureLevel.Protein);
        var types = ds.MetadataValues("sample_type");
        var experimental = Enumerable.Range(0, ds.SampleIds.Length).Where(j => types[j] == "experimental").ToList();
        var groupA = experimental.Take(experimental.Count / 2).ToList();
        var groupB = experimental.Skip(experimental.Count / 2).ToList();
        var options = new DifferentialOptions
        {
            Prior = VariancePrior.IntensityTrend, PriorGroupColumns = ControlSampleTypes.PriorGroups(types),
        };
        var res = Differential.Run(ds.ExprLog2, ds.FeatureIds, groupA, groupB, options);
        Assert.NotNull(options.PriorGroupColumns);
        Assert.StartsWith("prior: intensity trend from the controls, scaled x", res.DescribePriorFit(), StringComparison.Ordinal);

        var dir = Path.Combine(Path.GetTempPath(), $"prism-quant-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var html = File.ReadAllText(QuantReport.Write(dir, ConfigFor(res),
                InputsFor(ds, res, groupA.Concat(groupB).ToList(), SignificanceRule.Default, options)));

            var lines = File.ReadAllLines(Path.Combine(dir, "quant", "differential.csv"));
            Assert.StartsWith("# method:", lines[1], StringComparison.Ordinal);
            Assert.Equal("# " + res.DescribePriorFit(), lines[2]);
            Assert.Contains("<td>Variance prior</td><td>" + res.DescribePriorFitBody() + "</td>", html, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Write_RawValueMatrix_IsLinearNotLog2()
    {
        var (ds, res, cols) = RunContrast();
        var dir = Path.Combine(Path.GetTempPath(), $"prism-quant-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            QuantReport.Write(dir, ConfigFor(res), InputsFor(ds, res, cols, SignificanceRule.Default));

            var lines = File.ReadAllLines(Path.Combine(dir, "quant", "differential_values.csv"));
            var header = lines[0].Split(',');
            // header: feature_id,label,<sample id of cols[0]>,...
            Assert.Equal("feature_id", header[0]);
            Assert.Equal(ds.SampleIds[cols[0]], header[2]);
            Assert.Equal(cols.Count + 2, header.Length);
            Assert.Equal(res.Rows.Count + 1, lines.Length);

            // Spot-check the first data row against the log2 matrix: the exported cell must be 2^log2,
            // i.e. the linear abundance, not the log2 value on disk in peptides_log2_internal.
            var rowOf = new Dictionary<string, int>(StringComparer.Ordinal);
            for (var i = 0; i < ds.FeatureIds.Length; i++)
                rowOf[ds.FeatureIds[i]] = i;

            var firstId = res.Rows[0].FeatureId;
            var dataLine = lines.Skip(1).First(l => l.StartsWith(firstId + ",", StringComparison.Ordinal));
            var actual = double.Parse(dataLine.Split(',')[2], CultureInfo.InvariantCulture);
            var expected = Math.Pow(2.0, ds.ExprLog2[rowOf[firstId], cols[0]]);
            Assert.Equal(expected, actual, 9);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Write_WithAllSections_RendersEachAndWritesItsCsv()
    {
        var (ds, res, cols) = RunContrast();

        // Detection and enrichment are the already-computed results the report renders, so they can be
        // built directly here without a merged_data read or a g:Profiler call.
        var detection = new DetectionAnalysisResult(DetectionMethod.FisherExact, new List<DetectionRow>
        {
            new("PEPTIDEA", 4, 5, 1, 5, 0.8, 0.2, 0.04, 0.09),
            new("PEPTIDEB", 2, 5, 3, 5, 0.4, 0.6, 0.5, 0.5),
        }, 5, 5, 0, UnpairedReason.None);
        var enrichment = new List<EnrichmentTerm>
        {
            new("GO:BP", "GO:0006915", "apoptotic process", 1e-4, 120, 50, 3, 20000, 11.5,
                new[] { "ALB", "CD9", "APOE" }),
        };

        // A real marker section from the fixture: pick a gene the dataset actually carries so the panel
        // finds at least one member and MarkerFeatureIds are real ids (which the values CSV needs).
        var gene = ds.FeatureGenes.First(g => !string.IsNullOrEmpty(g));
        var panel = new ProteinList { Name = "Test panel" };
        panel.Members.Add(gene);
        var identities = Enumerable.Range(0, ds.FeatureIds.Length).Select(ds.IdentityOf).ToArray();
        var groups = ds.MetadataValues("sample_type");
        var markerResult = MarkerPanel.Evaluate(ds.ExprLog2, identities, groups, ds.SampleIds, panel, false);
        Assert.True(markerResult.Found >= 1);
        var markers = new List<MarkerReportSection>
        {
            new("Test panel", "sample_type", markerResult),
        };

        var inputs = new QuantReportInputs
        {
            Differential = res,
            Rule = SignificanceRule.Default,
            Corrected = true,
            Contrast = "experimental split A vs B",
            EffectName = "log2FC",
            LabelFor = id => id,
            Options = Options,
            GroupBy = "sample_type",
            ALabel = "first half",
            BLabel = "second half",
            Detection = detection,
            Enrichment = enrichment,
            Markers = markers,
            Dataset = ds,
            ContrastColumns = cols,
        };

        var dir = Path.Combine(Path.GetTempPath(), $"prism-quant-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var htmlPath = QuantReport.Write(dir, ConfigFor(res), inputs);
            var quant = Path.Combine(dir, "quant");

            // Detection: a line naming the test, the column header, one row per peptide.
            var detCsv = File.ReadAllLines(Path.Combine(quant, "detection.csv"));
            Assert.StartsWith("# test: Fisher exact", detCsv[0]);
            Assert.Equal("peptide,detected_a,n_a,detected_b,n_b,rate_a,rate_b,p_value,adj_p_value", detCsv[1]);
            Assert.Equal(detection.Rows.Count + 2, detCsv.Length);

            // Enrichment: the full gene list reaches the CSV (the HTML truncates it).
            var enrCsv = File.ReadAllText(Path.Combine(quant, "enrichment_terms.csv"));
            Assert.Contains("apoptotic process", enrCsv);
            Assert.Contains("ALB;CD9;APOE", enrCsv);

            // Markers: both the z-score and the raw LINEAR value matrices, named after the panel.
            Assert.True(File.Exists(Path.Combine(quant, "markers_Test_panel_zscores.csv")));
            var mvals = File.ReadAllLines(Path.Combine(quant, "markers_Test_panel_values.csv"));
            Assert.True(mvals.Length >= 2); // header + at least the one matched member
            Assert.StartsWith("feature_id,label,", mvals[0]);

            var html = File.ReadAllText(htmlPath);
            Assert.Contains("Detection frequency", html);
            Assert.Contains("Fisher exact", html);
            Assert.DoesNotContain("UNPAIRED", html); // an unpaired design ran the unpaired test
            Assert.Contains("Functional enrichment", html);
            Assert.Contains("Marker panels", html);
            Assert.Contains("Test panel", html);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// The report's differential.csv must be the file `prism differential` writes, not a lookalike:
    /// there were two writers once, and the report's dropped the provenance header and the annotation
    /// columns while the docs called the two the same file.
    /// </summary>
    [Fact]
    public void Write_DifferentialCsv_IsByteIdenticalToTheSharedWriter()
    {
        var (ds, res, cols) = RunContrast();
        var dir = Path.Combine(Path.GetTempPath(), $"prism-quant-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var inputs = InputsFor(ds, res, cols, SignificanceRule.Default);
            QuantReport.Write(dir, ConfigFor(res), inputs);

            var direct = Path.Combine(dir, "direct.csv");
            DifferentialCsv.Write(direct, res, ds, Options, SignificanceRule.Default,
                inputs.GroupBy, inputs.ALabel, inputs.BLabel, inputs.EffectName);

            Assert.Equal(File.ReadAllBytes(direct),
                File.ReadAllBytes(Path.Combine(dir, "quant", "differential.csv")));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>Write a report around one detection result and return (html, detection.csv lines or null).</summary>
    private static (string Html, string[]? DetCsv) WriteWithDetection(DetectionAnalysisResult detection)
    {
        var (ds, res, cols) = RunContrast();
        var dir = Path.Combine(Path.GetTempPath(), $"prism-quant-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var inputs = InputsFor(ds, res, cols, SignificanceRule.Default);
            var html = File.ReadAllText(QuantReport.Write(dir, ConfigFor(res), new QuantReportInputs
            {
                Differential = inputs.Differential, Rule = inputs.Rule, Corrected = inputs.Corrected,
                Contrast = inputs.Contrast, EffectName = inputs.EffectName, LabelFor = inputs.LabelFor,
                Options = inputs.Options, GroupBy = inputs.GroupBy, ALabel = inputs.ALabel,
                BLabel = inputs.BLabel, Dataset = inputs.Dataset, ContrastColumns = inputs.ContrastColumns,
                Detection = detection,
            }));
            var csv = Path.Combine(dir, "quant", "detection.csv");
            return (html, File.Exists(csv) ? File.ReadAllLines(csv) : null);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Write_McNemarDetection_NamesTheTestAndKeepsTheDiscordantCounts()
    {
        var (ds, det, a, b) = DetectionAnalysisTests.Setup();
        var paired = DetectionAnalysis.Run(det, ds, a, b, DifferentialDesign.Paired,
            DetectionAnalysisTests.Subjects(ds, a, b), null, MultipleTesting.BenjaminiHochberg);

        var (html, csv) = WriteWithDetection(paired);

        Assert.Contains("McNemar's exact test over 20 matched subjects", html);
        Assert.Contains("discordant pairs", html);
        Assert.StartsWith("# test: McNemar", csv![0]);
        Assert.Equal("peptide,detected_a,n_a,detected_b,n_b,rate_a,rate_b,only_a,only_b,p_value,adj_p_value", csv[1]);
        // The discordant counts written are McNemar's own, row for row.
        var firstPaired = paired.PairedRows![0];
        Assert.StartsWith($"{firstPaired.PeptideId},", csv[2]);
        Assert.Contains($",{firstPaired.OnlyA},{firstPaired.OnlyB},", csv[2]);
    }

    [Fact]
    public void Write_UnidentifiableAdjustedDetection_SaysSoAndWritesNoTable()
    {
        var glm = new DetectionGlmResult(Array.Empty<DetectionGlmRow>(), identifiable: false,
            groupCollinearityR2: 0.97, nParams: 3, nA: 5, nB: 5, covariatesUsed: new[] { "batch" },
            dropped: Array.Empty<string>(), nNonConverged: 0, warning: null);
        var result = new DetectionAnalysisResult(DetectionMethod.FirthGlm, new List<DetectionRow>(),
            5, 5, 0, UnpairedReason.None, glm: glm);

        var (html, csv) = WriteWithDetection(result);

        Assert.Contains("not identifiable", html);
        Assert.Contains("0.97", html);
        Assert.DoesNotContain("Detection volcano", html);
        Assert.Null(csv);
    }

    [Fact]
    public void Write_PairedDesignRunUnpaired_SaysWhy()
    {
        var (ds, det, a, b) = DetectionAnalysisTests.Setup();
        // Paired with no subject column: Fisher, with the reason recorded.
        var fallback = DetectionAnalysis.Run(det, ds, a, b, DifferentialDesign.Paired, null, null,
            MultipleTesting.BenjaminiHochberg);

        var (html, csv) = WriteWithDetection(fallback);

        Assert.Contains("tested UNPAIRED", html);
        Assert.Contains("UNPAIRED", csv![1]);
    }

    [Fact]
    public void Write_NoContrastColumns_NamesTheMissingValuesFile()
    {
        var (ds, res, _) = RunContrast();
        var dir = Path.Combine(Path.GetTempPath(), $"prism-quant-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var html = File.ReadAllText(QuantReport.Write(dir, ConfigFor(res),
                InputsFor(ds, res, cols: null, SignificanceRule.Default)));

            Assert.False(File.Exists(Path.Combine(dir, "quant", "differential_values.csv")));
            Assert.Contains("No <code>differential_values.csv</code>", html);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Write_CapsLongTablesToAPreview()
    {
        var (ds, res, cols) = RunContrast();
        // A permissive rule makes every tested feature a hit, so a cap of 1 must trip the preview note.
        var rule = new SignificanceRule { PThreshold = 1.1, Log2FcThreshold = 0.0, UseAdjusted = true };
        var dir = Path.Combine(Path.GetTempPath(), $"prism-quant-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            QuantReport.Write(dir, ConfigFor(res), InputsFor(ds, res, cols, rule), maxHitRows: 1);
            var html = File.ReadAllText(Path.Combine(dir, "quant", "quant_report.html"));
            Assert.Contains($"Showing the top 1 of {res.Rows.Count}", html);
            Assert.Contains("differential.csv", html);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
