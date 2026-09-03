using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Numerics;
using SkylinePrism.Core.Pipeline;
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

    public static string Generate(
        string outputDir, PrismConfig config, bool savePlots = false, Action<string>? log = null)
    {
        var sampleTypes = ReadSampleTypes(Path.Combine(outputDir, "sample_metadata.csv"));

        // What the report SAYS about the run has to come from the run, not from whatever config this
        // invocation was handed: `prism qc -c qc_only.yaml` supplies report options, and every section
        // it omits would otherwise be filled with defaults and printed as though it had been used.
        var runConfig = ReadRunConfig(outputDir) ?? config;

        var peptideCol = DetectPeptideColumn(Path.Combine(outputDir, "peptides_rollup.parquet"));
        var pepMeta = new[] { peptideCol, PepMetaN, PepMetaRt };

        // "Corrected" is each arm's ACTUAL output - corrected_peptides / corrected_proteins (LINEAR, so
        // log2 here) - not the internal file. peptides_log2_internal is post-normalization and
        // PRE-ComBat: since dotnet-v26.15.0 the protein arm branches from it, so reading it as the
        // peptide "after" left every peptide CV, every peptide plot and the validation verdict
        // measuring normalization alone, while the panels were labelled "normalized + corrected" and
        // the protein half of the same report showed a fully corrected matrix. On the mini fixture
        // with ComBat on, the two files differ in every cell (up to 4.7 log2).
        var pepRaw = LoadMatrix(Path.Combine(outputDir, "peptides_rollup.parquet"), pepMeta);
        var pepCorrected = Log2(LoadMatrix(Path.Combine(outputDir, "corrected_peptides.parquet"), pepMeta));
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
        var rtLowessRan = string.Equals(runConfig.GlobalNormalization.Method, "rt_lowess", StringComparison.OrdinalIgnoreCase);
        var peptidePlots = RenderLevelSections("peptide", pepRaw, pepCorrected, typeLabels, refIdx, qcIdx, savePlots, plotsDir, rtLowessRan);
        var proteinPlots = RenderLevelSections("protein", protRaw, protCorrected, typeLabels, refIdx, qcIdx, savePlots, plotsDir, rtLowessRan);

        var validation = ValidationStatus.Compute(pepRaw.Values, pepCorrected.Values, refIdx, qcIdx);
        var runInfo = Provenance.ReadRunInfo(Path.Combine(outputDir, "parameters.json"));

        var signalPlots = RenderMs2SignalSection(
            outputDir, runConfig, sampleTypes, savePlots, plotsDir, log);

        var html = BuildHtml(
            outputDir, sampleCols.Count, sampleTypes,
            pepRaw.RowCount, protRaw.RowCount,
            pepRef, pepQc, protRef, protQc, peptidePlots, proteinPlots, signalPlots,
            validation, runConfig, runInfo);

        var htmlPath = Path.Combine(outputDir, "qc_report.html");
        File.WriteAllText(htmlPath, html);
        return htmlPath;
    }

    private sealed record PlotImage(string Caption, byte[] Png);
    private sealed record PlotSection(string Title, List<PlotImage> Images);

    /// <summary>
    /// The MS2 signal accounting section, or an empty list when there is nothing to show.
    ///
    /// <para>Reuses the cached <c>ms2_signal_accounting.parquet</c> when it answers the same question,
    /// so <c>prism qc -d</c> replots an existing run for free and keeps working on a directory whose
    /// <c>merged_data/</c> was cleaned up. Computing is a full pass over the merged table, so it
    /// happens only when the run asked for the section and the cache cannot serve it.</para>
    ///
    /// <para><b>The cache is keyed on the settings, not just on the file name.</b> It used to be
    /// preferred unconditionally: a re-run that switched measure, tolerance, isolation scheme or
    /// protein lists then replotted the PREVIOUS run's numbers, captioned with the previous run's
    /// settings, with nothing in the log - and a re-run that turned the section OFF still rendered
    /// it. Both look entirely right on the page, which is what makes keying it properly worth the
    /// resolve work below on every report.</para>
    /// </summary>
    private static List<PlotSection> RenderMs2SignalSection(
        string outputDir, PrismConfig config, IReadOnlyDictionary<string, string> sampleTypes,
        bool savePlots, string plotsDir, Action<string>? log)
    {
        var sections = new List<PlotSection>();
        var settings = config.QcReport.Ms2Signal;
        // Not asked for: a leftover cache from an earlier run into this directory is not a reason for
        // the section to exist. (For `prism qc -d` these settings come from the directory's own
        // parameters.json, so a run that DID ask for it still replots.)
        if (!settings.Enabled)
            return sections;

        var cached = Ms2SignalAccounting.ReadCached(outputDir);

        var tolerance = ProductMassTolerance.ParseSetting(settings.ExtractionTolerance);
        if (tolerance is null)
        {
            log?.Invoke(
                "  MS2 signal accounting: qc_report.ms2_signal.extraction_tolerance "
                + $"'{settings.ExtractionTolerance}' is not a tolerance. Write it as \"10 ppm\" "
                + "or \"0.4 m/z\".");
        }

        // Both of these read files, and neither is needed to REPLOT a cache that already matches - so
        // a failure here is only fatal when there is nothing cached to fall back on.
        var scheme = tolerance is null ? null : ResolveScheme(outputDir, settings.IsolationScheme, log);
        var lists = ResolveMs2Lists(settings.ProteinLists, settings.ProteinListFiles, log);
        var measure = string.Equals(settings.Measure, "ions", StringComparison.OrdinalIgnoreCase)
            ? Ms2SignalMeasure.Ions
            : Ms2SignalMeasure.Signal;

        // A cached result records the scheme it used, so a directory whose isolation_schemes.xml has
        // been cleaned up can still have its cache validated against everything else - which is what
        // keeps `prism qc -d` a free replot there rather than a warning.
        var schemeName = scheme?.Name ?? cached?.IsolationScheme;

        Ms2SignalAccounting.Result? result = null;
        if (tolerance is not null && schemeName is not null)
        {
            var wanted = Ms2SignalAccounting.SettingsKeyFor(
                measure, tolerance.Describe(), schemeName, lists.Select(l => l.Name));
            if (cached is not null
                && cached.MatchesSettings(wanted, measure, tolerance.Describe(), schemeName))
            {
                result = cached;   // the free replot the cache exists for
            }
            else if (scheme is not null)
            {
                if (cached is not null)
                {
                    log?.Invoke(
                        $"  MS2 signal accounting: the cached results are for {cached.SettingsSummary()}; "
                        + "this run asks for "
                        + Ms2SignalAccounting.SummarizeSettings(
                            measure, tolerance.Describe(), scheme.Name, lists.Count)
                        + ". Recomputing.");
                }
                result = Ms2SignalAccounting.Compute(
                    outputDir, scheme, tolerance, lists, sampleTypes, log, measure: measure);
                if (result is not null)
                    Ms2SignalAccounting.Write(outputDir, result);
            }
        }

        if (result is null && cached is not null)
        {
            // The settings cannot be honoured - no tolerance, no scheme, or no merged_data/ to
            // recompute from; whichever it was has been logged above. The cached numbers are
            // self-describing, and the caption below names the settings that produced them, so
            // showing them beats dropping the section - as long as the mismatch is said plainly.
            log?.Invoke(
                "  MS2 signal accounting: showing the CACHED results instead. They are for "
                + $"{cached.SettingsSummary()}, which is not what this run asks for; the caption "
                + "names the settings that produced them.");
            result = cached;
        }

        if (result is null || result.IsEmpty)
            return sections;

        var caption = result.Measure == Ms2SignalMeasure.Ions
            ? $"MS2 IONS per replicate - intensity x injection time, summed per spectrum across each "
              + $"peak - shared signal counted once (extraction {result.Tolerance}, isolation scheme "
              + $"\"{result.IsolationScheme}\", {result.AssignedPeptides:N0} peptides). Not background "
              + "subtracted."
            : $"Integrated MS2 signal per replicate, shared signal counted once "
              + $"(extraction {result.Tolerance}, isolation scheme \"{result.IsolationScheme}\", "
              + $"{result.AssignedPeptides:N0} peptides).";

        var images = new List<PlotImage>();
        try
        {
            var png = PlotRenderer.Ms2AccountingPng(
                result,
                result.Measure == Ms2SignalMeasure.Ions
                    ? "MS2 Ions Assigned to Peptides (shared signal counted once)"
                    : "MS2 Signal Assigned to Peptides (shared signal counted once)");
            if (savePlots && png.Length > 0)
            {
                Directory.CreateDirectory(plotsDir);
                File.WriteAllBytes(Path.Combine(plotsDir, "ms2_signal_accounting.png"), png);
            }
            images.Add(new PlotImage(caption, png));
        }
        catch (Exception ex)
        {
            images.Add(new PlotImage(
                caption + " (render failed: " + ex.GetType().Name + ")", Array.Empty<byte>()));
        }

        sections.Add(new PlotSection(
            // Deliberately not "total MS2": this is what Skyline integrated for the document's targets.
            // The acquired total needs the instrument files, and calling this that would turn unknown
            // coverage into apparently complete coverage.
            "Signal Skyline Integrated for This Document's Targets", images));
        return sections;
    }

    /// <summary>
    /// The protein lists to draw a line for, dropping each one this machine cannot resolve with a
    /// message rather than throwing.
    ///
    /// <para><see cref="ProteinListSet.ResolveForDisplay"/> throws on an unresolvable name, which is
    /// right for marker normalization - a normalization nobody can reproduce should not run - but not
    /// for an optional QC plot: a config written on another machine can name a list saved only there,
    /// and that used to abort the whole run at Stage 5b, after every output had been computed and
    /// with no <c>qc_report.html</c> to show for it. Every other failure in this section logs and
    /// carries on; so does this one now, one list at a time so the resolvable ones survive.</para>
    /// </summary>
    private static IReadOnlyList<ProteinList> ResolveMs2Lists(
        IEnumerable<string>? names, IEnumerable<string>? memberFiles, Action<string>? log)
    {
        var resolved = new List<ProteinList>();
        // Files first, then names - the order ResolveForDisplay itself uses, so the plot's lines come
        // out the same however they were specified.
        foreach (var file in memberFiles ?? Enumerable.Empty<string>())
            Add(() => ProteinListSet.ResolveForDisplay(null, new[] { file }));
        foreach (var name in names ?? Enumerable.Empty<string>())
            Add(() => ProteinListSet.ResolveForDisplay(new[] { name }));
        return resolved;

        void Add(Func<IReadOnlyList<ProteinList>> resolve)
        {
            try
            {
                resolved.AddRange(resolve());
            }
            catch (Exception ex)
            {
                log?.Invoke(
                    $"  MS2 signal accounting: {ex.Message} Continuing without that list - name only "
                    + "lists saved on this machine, or give the members as a file via "
                    + "qc_report.ms2_signal.protein_list_files.");
            }
        }
    }

    /// <summary>
    /// The isolation scheme to account against: the one the config names, or the only usable one in
    /// <c>isolation_schemes.xml</c>. Never guessed - fragments in different isolation windows never
    /// share signal, so the wrong scheme silently changes every number on the plot.
    /// </summary>
    private static IsolationScheme? ResolveScheme(string outputDir, string? named, Action<string>? log)
    {
        var catalog = IsolationSchemeCatalog.Load(
            Path.Combine(outputDir, IsolationSchemeCatalog.FileName));
        var usable = catalog?.UsableSchemes ?? (IReadOnlyList<IsolationScheme>)Array.Empty<IsolationScheme>();

        if (!string.IsNullOrWhiteSpace(named))
        {
            var match = usable.FirstOrDefault(
                s => string.Equals(s.Name, named, StringComparison.OrdinalIgnoreCase));
            if (match is not null)
                return match;
            log?.Invoke(
                $"  MS2 signal accounting skipped: no isolation scheme named '{named}' in "
                + $"{IsolationSchemeCatalog.FileName}.");
            return null;
        }

        if (usable.Count == 1)
            return usable[0];

        log?.Invoke(usable.Count == 0
            ? "  MS2 signal accounting skipped: no isolation windows are known for this cohort. A DIA "
              + "document normally stores none (isolation_scheme name=\"Results only\"), so import them "
              + "from a data file in the Skyline tool, or set qc_report.ms2_signal.isolation_scheme."
            : $"  MS2 signal accounting skipped: {usable.Count} isolation schemes are known for this "
              + "cohort. Name one with qc_report.ms2_signal.isolation_scheme.");
        return null;
    }

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
                Img("Raw", $"{level}_intensity_before",
                    () => PlotRenderer.IntensityDistribution(raw.Values, typeLabels, "Raw")),
                Img("Corrected", $"{level}_intensity_after",
                    () => PlotRenderer.IntensityDistribution(corrected.Values, typeLabels, "Corrected")),
            }));

        sections.Add(new PlotSection($"{cap} PCA: Raw vs Corrected", new List<PlotImage>
        {
            Img("Raw", $"{level}_pca_before",
                () => PlotRenderer.PcaScatter(
                    Pca.Fit2DOfFeaturesBySamples(raw.Values), typeLabels, "Raw")),
            Img("Corrected", $"{level}_pca_after",
                () => PlotRenderer.PcaScatter(
                    Pca.Fit2DOfFeaturesBySamples(corrected.Values), typeLabels, "Corrected")),
        }));

        if (refIdx.Count >= 2)
            sections.Add(new PlotSection($"{cap} CV Distribution (Reference): Raw vs Corrected", new List<PlotImage>
            {
                Img("", $"{level}_cv_reference", () => PlotRenderer.CvComparison(
                    CvMetrics.PerFeatureCvs(raw.Values, refIdx),
                    CvMetrics.PerFeatureCvs(corrected.Values, refIdx),
                    $"{cap} CV (Reference)", "#d62728")),
            }));

        if (qcIdx.Count >= 2)
            sections.Add(new PlotSection($"{cap} CV Distribution (QC): Raw vs Corrected", new List<PlotImage>
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
            sections.Add(new PlotSection($"{cap} Control-Sample Correlation: Raw vs Corrected", new List<PlotImage>
            {
                Img("Raw", $"{level}_control_corr_before",
                    () => PlotRenderer.CorrelationHeatmap(raw.Values, controlIdx, $"{cap} Control Sample Correlation (Raw)", controlTypes)),
                Img("Corrected", $"{level}_control_corr_after",
                    () => PlotRenderer.CorrelationHeatmap(corrected.Values, controlIdx, $"{cap} Control Sample Correlation (Corrected)", controlTypes)),
            }));

        // RT-dependent diagnostics (peptide level only - proteins have no RT).
        if (rtLowessRan && raw.MeanRt is not null && corrected.MeanRt is not null)
        {
            sections.Add(new PlotSection($"{cap} RT-Lowess Curves: Raw vs Corrected", new List<PlotImage>
            {
                Img("Raw", $"{level}_rt_lowess_before",
                    () => PlotRenderer.RtLowessCurves(raw.Values, raw.MeanRt, typeLabels, "Raw")),
                Img("Corrected", $"{level}_rt_lowess_after",
                    () => PlotRenderer.RtLowessCurves(corrected.Values, corrected.MeanRt, typeLabels, "Corrected")),
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

            sections.Add(new PlotSection($"{cap} Abundance by RT Bin: Raw vs Corrected", new List<PlotImage>
            {
                Img("Raw", $"{level}_rt_bin_box_before",
                    () => PlotRenderer.RtBinBoxplot(raw.Values, raw.MeanRt, "Raw", "#1f77b4")),
                Img("Corrected", $"{level}_rt_bin_box_after",
                    () => PlotRenderer.RtBinBoxplot(corrected.Values, corrected.MeanRt, "Corrected", "#1f77b4")),
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
        List<PlotSection> signalPlots,
        ValidationStatus? validation, PrismConfig config, Provenance.RunInfo? runInfo)
    {
        // One reading for the whole page: rendering a large cohort's plots takes seconds, and a header
        // and footer that disagree about when the report was made read as two different reports.
        var generatedAt = DateTimeStamp();
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
.notes { background: #f2f6fc; border: 1px solid #cfd8e3; border-radius: 6px; padding: 8px 14px; margin: 8px 0; }
.notes li { color: #40536e; }
table.kv td:nth-child(2) { text-align: left; }
details { margin: 10px 0; }
summary { cursor: pointer; color: #1a3c6e; font-weight: 600; }
pre { background: #f6f8fb; border: 1px solid #dfe6ef; border-radius: 6px; padding: 10px 14px;
      overflow-x: auto; font-size: 12.5px; line-height: 1.4; }
</style></head><body><div class="container">
""");
        sb.Append("<h1>PRISM QC Report</h1>");

        AppendRunInfo(sb, runInfo, config, generatedAt);

        sb.Append("<div class=\"box\"><h2>Dataset Summary</h2>");
        sb.Append($"<p>Samples: <strong>{nSamples}</strong> (experimental {nExp}, reference {nRef}, qc {nQc})<br>");
        sb.Append($"Peptides: <strong>{nPeptides}</strong> &nbsp; Proteins: <strong>{nProteins}</strong></p>");
        sb.Append($"<p style=\"color:#666\">Output directory: <code>{HtmlEncode(outputDir)}</code></p>");
        sb.Append("</div>");

        AppendValidation(sb, validation, nRef, nQc, nSamples);

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
        if (signalPlots.Count > 0)
        {
            sb.Append("<div class=\"section-header\">MS2 Signal Accounting</div>");
            AppendSections(sb, signalPlots);
        }

        sb.Append($"<p class=\"footer\">Generated by Skyline-PRISM (C#) at {generatedAt}</p>");
        sb.Append("</div></body></html>");
        return sb.ToString();
    }

    /// <summary>
    /// "Analysis Information": which PRISM produced these numbers, when, on what machine, from what
    /// inputs, and with which settings. The provenance facts come from the run's parameters.json - not
    /// from the binary rendering the page, which may be a later `prism qc` - while the settings come
    /// from the config handed to the report.
    /// </summary>
    private static void AppendRunInfo(
        StringBuilder sb, Provenance.RunInfo? info, PrismConfig config, string generatedAt)
    {
        sb.Append("<div class=\"box\"><h2>Analysis Information</h2>");
        sb.Append("<table class=\"kv\">");
        void Row(string label, string valueHtml) =>
            sb.Append($"<tr><td><strong>{HtmlEncode(label)}</strong></td><td>{valueHtml}</td></tr>");

        // Without a parameters.json - a hand-assembled directory, or a run from before provenance was
        // written - the run facts are unknown. Say so rather than passing the rendering process off as
        // the one that produced the outputs.
        Row("Pipeline", info is not null
            ? $"PRISM v{HtmlEncode(info.PipelineVersion)} (C#)"
            : $"PRISM v{HtmlEncode(Provenance.AssemblyVersion)} (C#) <span style=\"color:#888\">- report "
              + "generator only; no parameters.json here, so the version that produced these outputs is "
              + "unrecorded</span>");
        if (info is not null)
        {
            Row("Processing date", HtmlEncode(info.ProcessingDate));
            Row("Computer", HtmlEncode(info.Host));
            Row("Source files", info.SourceFiles.Count == 0
                ? "<em>not recorded</em>"
                : "<ul style=\"margin:0;padding-left:18px\">"
                  + string.Concat(info.SourceFiles.Select(f => $"<li><code>{HtmlEncode(f)}</code></li>"))
                  + "</ul>");
        }
        Row("Report generated", HtmlEncode(generatedAt));
        sb.Append("</table>");

        sb.Append("<h3>Processing Parameters</h3><table class=\"kv\">");
        foreach (var (stage, setting) in ParameterRows(config))
            sb.Append($"<tr><td>{HtmlEncode(stage)}</td><td>{HtmlEncode(setting)}</td></tr>");
        sb.Append("</table>");

        // The table above is the readable summary; this is the exact, re-runnable config behind it
        // (ConfigWriter emits only what applies to the selected methods, so it stays short).
        sb.Append("<details><summary>Full configuration (YAML)</summary><pre>")
          .Append(HtmlEncode(ConfigWriter.ToYaml(config)))
          .Append("</pre></details>");
        sb.Append("</div>");
    }

    /// Display names for the config sections, in the order <see cref="ConfigWriter.Sections"/> emits
    /// them. A section without an entry here is shown under its own key rather than dropped - a new
    /// config section must not be able to vanish from the report by being forgotten in this table.
    private static readonly Dictionary<string, string> StageLabels = new(StringComparer.Ordinal)
    {
        ["data"] = "Input columns",
        ["transition_rollup"] = "Transition -> peptide rollup",
        ["global_normalization"] = "Peptide normalization",
        ["sample_outlier_detection"] = "Sample outlier detection",
        ["batch_correction"] = "Batch correction",
        ["parsimony"] = "Protein parsimony",
        ["protein_rollup"] = "Peptide -> protein rollup",
        ["protein_normalization"] = "Protein normalization",
        ["qc_report"] = "QC report",
        ["output"] = "Output",
        ["processing"] = "Processing",
        ["batch_estimation"] = "Batch estimation",
        ["sample_annotations"] = "Sample-type patterns",
    };

    /// <summary>
    /// One row per config section: the method that ran and the operands that shaped it, rendered from
    /// <see cref="ConfigWriter.Sections"/> - the same builder that produces the YAML below the table,
    /// so the two cannot disagree and a new config key needs no second edit here. Keys that the
    /// selected method does not read are already absent from that builder's output.
    /// </summary>
    private static List<(string Stage, string Setting)> ParameterRows(PrismConfig c)
    {
        var rows = new List<(string, string)>();
        foreach (var (name, values) in ConfigWriter.Sections(c))
        {
            var parts = new List<string>();
            Flatten(values, prefix: "", parts);
            if (parts.Count > 0)
                rows.Add((StageLabels.GetValueOrDefault(name, name), string.Join(", ", parts)));
        }
        return rows;
    }

    /// <summary>Render one config section as `key=value` parts, `sub.key=value` for nested blocks.</summary>
    private static void Flatten(IReadOnlyDictionary<string, object?> values, string prefix, List<string> parts)
    {
        foreach (var (key, value) in values)
        {
            if (value is IReadOnlyDictionary<string, object?> nested)
                Flatten(nested, prefix + key + ".", parts);
            else if (value is IDictionary<string, object?> nestedRw)
                Flatten(new Dictionary<string, object?>(nestedRw), prefix + key + ".", parts);
            else
                parts.Add($"{prefix}{key}={FormatValue(value)}");
        }
    }

    /// <summary>
    /// Config values as they read in a config file, except booleans, which keep the Python report's
    /// True/False. Numbers round-trip ("R"): a rounded operand in the table that disagrees with the
    /// YAML below it would defeat the point of printing the settings at all.
    /// </summary>
    private static string FormatValue(object? value) => value switch
    {
        null => "(none)",
        bool b => b ? "True" : "False",
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        float f => f.ToString("R", CultureInfo.InvariantCulture),
        IFormattable n and (int or long or decimal) => n.ToString(null, CultureInfo.InvariantCulture),
        System.Collections.IEnumerable list and not string =>
            "[" + string.Join(" | ", list.Cast<object?>().Select(FormatValue)) + "]",
        _ => value.ToString() ?? "",
    };

    /// <summary>
    /// The config a run was made with, from its parameters.json. Null when there is none or it cannot
    /// be read - a QC report must still render beside a hand-assembled directory or a truncated
    /// provenance file, so this never throws.
    /// </summary>
    private static PrismConfig? ReadRunConfig(string outputDir)
    {
        try
        {
            var path = Path.Combine(outputDir, "parameters.json");
            return File.Exists(path) ? Provenance.LoadConfig(path) : null;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// The verdict, or - when there are not enough controls to reach one - WHY, with the counts.
    ///
    /// <para>The counts are the point. This used to say only "not enough of both were found", which reads
    /// as a statement about the study design, so a cohort that HAD 16 reference and 16 QC samples and lost
    /// them produced a report indistinguishable from one that never had any. That happened: a headless
    /// metadata export crashed, PRISM fell back to inferring sample types from replicate names, the
    /// inference matched nothing, and the run completed and reported "0 reference, 0 qc, 192
    /// experimental". "0 of 192" is an alarm; "not enough of both" is not.</para>
    /// </summary>
    private static void AppendValidation(
        StringBuilder sb, ValidationStatus? v, int nRef, int nQc, int nSamples)
    {
        if (v is null)
        {
            sb.Append("<div class=\"box\" style=\"color:#666\">Validation verdict needs &gt;=2 reference "
                + $"and &gt;=2 QC samples (dual-control design); this run has <strong>{nRef}</strong> "
                + $"reference and <strong>{nQc}</strong> QC of {nSamples} samples.");
            // Only when BOTH are zero, because that is the signature of sample types never arriving
            // rather than of a cohort with few controls.
            if (nRef == 0 && nQc == 0)
                sb.Append(" No controls were identified at all. If this cohort does have them, their "
                    + "sample types did not reach PRISM - check the replicate metadata, because a failed "
                    + "metadata export leaves every sample typed as experimental.");
            sb.Append("</div>");
            return;
        }

        sb.Append($"<div class=\"status {(v.Passed ? "status-pass" : "status-fail")}\">"
            + $"Validation: {(v.Passed ? "PASSED" : "FAILED")}</div>");
        // Say what the verdict is answering, so a FAILED banner points at something actionable and a
        // PASSED one is not read as a blanket endorsement.
        sb.Append("<p class=\"note\">The verdict asks one question: did the processing damage the "
            + "controls? It fails only if the QC CV got <em>worse</em>, or if the QC and reference "
            + "samples collapsed onto each other in PCA space. The reference and QC groups improving by "
            + "different amounts is not a failure - they are different materials injected at different "
            + "amounts, so one can have more excess variance to remove than the other.</p>");

        sb.Append("<table><tr><th>Control CV</th><th>Raw</th><th>Corrected</th><th>Improvement</th></tr>");
        void Row(string label, double before, double after, double impFrac)
        {
            var cls = impFrac >= 0 ? "improvement-positive" : "improvement-negative";
            sb.Append($"<tr><td>{label}</td><td>{before:0.0}%</td><td>{after:0.0}%</td>");
            sb.Append($"<td class=\"{cls}\">{impFrac * 100:+0.0;-0.0}%</td></tr>");
        }
        Row("Reference", v.ReferenceCvBefore, v.ReferenceCvAfter, v.ReferenceCvImprovement);
        Row("QC", v.QcCvBefore, v.QcCvAfter, v.QcCvImprovement);
        sb.Append("</table>");

        var rvr = double.IsNaN(v.RelativeVarianceReduction) || double.IsInfinity(v.RelativeVarianceReduction)
            ? "n/a"
            : v.RelativeVarianceReduction.ToString("0.00");
        var pca = double.IsNaN(v.PcaDistanceRatio) ? "n/a" : v.PcaDistanceRatio.ToString("0.00");
        sb.Append($"<p style=\"color:#555\">Relative variance reduction (QC/reference improvement): "
            + $"<strong>{rvr}</strong> (reported for information - it does not affect the verdict) "
            + "&nbsp;&middot;&nbsp; "
            + $"PCA QC-reference distance ratio (after/before): <strong>{pca}</strong> (collapse if &lt; 0.5)</p>");

        if (v.Warnings.Count > 0)
        {
            sb.Append("<div class=\"warnings\"><strong>WARNINGS</strong><ul>");
            foreach (var w in v.Warnings)
                sb.Append($"<li>{HtmlEncode(w)}</li>");
            sb.Append("</ul></div>");
        }

        if (v.Notes.Count > 0)
        {
            sb.Append("<div class=\"notes\"><strong>NOTES</strong><ul>");
            foreach (var n in v.Notes)
                sb.Append($"<li>{HtmlEncode(n)}</li>");
            sb.Append("</ul></div>");
        }
    }

    private static string CvTable(string title, CvMetrics.BeforeAfter? refBa, CvMetrics.BeforeAfter? qcBa)
    {
        var sb = new StringBuilder();
        sb.Append($"<h3>{title}</h3><table><tr><th>Sample Type</th><th>Raw</th><th>Corrected</th><th>Improvement</th></tr>");
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

    /// <summary>
    /// The feature x sample matrix, read one column at a time.
    /// <para>
    /// Deliberately NOT <see cref="ParquetTable.Load"/>: that materializes every sample column as a
    /// nullable <c>double?[]</c> - 16 bytes per cell - and the matrix below then copies it into 8 more,
    /// so both are live at ~24 bytes per cell. On a 100-document cohort (75k peptides x ~9,600 runs)
    /// that is ~17 GB for a report. Column-at-a-time costs the matrix plus one column.
    /// </para>
    /// </summary>
    private static Matrix LoadMatrix(string path, IReadOnlyList<string> metaCols)
    {
        using var reader = ParquetColumnReader.Open(path);
        var meta = new HashSet<string>(metaCols, StringComparer.Ordinal);
        // A non-numeric column is metadata whatever it is called: corrected_peptides carries the
        // derived protein_group / leading_* strings that peptides_rollup does not, and reading one as
        // a replicate would throw. (Same rule as DynamicRange.PeptideMetadataColumns, which lists only
        // the NUMERIC metadata for exactly this reason.)
        var sampleCols = reader.ColumnNames
            .Where(c => !meta.Contains(c) && reader.IsNumericColumn(c))
            .ToList();
        var n = reader.RowCount;
        var m = new double[n, sampleCols.Count];
        for (var j = 0; j < sampleCols.Count; j++)
        {
            var col = reader.ReadDoubles(sampleCols[j]);
            for (var i = 0; i < n; i++)
                m[i, j] = col[i];
        }

        double[]? meanRt = null;
        if (reader.ColumnNames.Contains(PepMetaRt, StringComparer.Ordinal))
            meanRt = reader.ReadDoubles(PepMetaRt);
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
        // Quote-aware: this file carries the replicate annotations, whose values can contain commas
        // (so can a batch label or a sample name, which the writer has always quoted). Splitting on
        // every comma shifts the fields and keys a sample by the wrong string - it does not throw, it
        // just drops that sample out of the control groups.
        var header = CsvLine.Split(lines[0]);
        var idIdx = CsvLine.IndexOf(header, "sample_id");
        var typeIdx = CsvLine.IndexOf(header, "sample_type");
        if (idIdx < 0 || typeIdx < 0)
            return map;
        for (var i = 1; i < lines.Length; i++)
        {
            if (string.IsNullOrWhiteSpace(lines[i]))
                continue;
            var f = CsvLine.Split(lines[i]);
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
