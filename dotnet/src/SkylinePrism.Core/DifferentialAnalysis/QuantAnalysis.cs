using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis.Detection;
using SkylinePrism.Core.DifferentialAnalysis.Enrichment;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// One restriction on the samples a trend is fitted over: keep those whose <paramref name="Column"/>
/// holds one of <paramref name="Values"/>.
/// </summary>
public sealed record QuantRestriction(string Column, IReadOnlyList<string> Values)
{
    /// <summary><c>Study = Verapamil</c>; several values listed, since any of them is kept.</summary>
    public string Describe() => $"{Column} = {string.Join(", ", Values)}";

    /// <summary>Several restrictions, which narrow together: <c>Study = Verapamil; Arm = Drug</c>.</summary>
    public static string Describe(IEnumerable<QuantRestriction> restrictions) =>
        string.Join("; ", restrictions.Where(r => r.Values.Count > 0).Select(r => r.Describe()));
}

/// <summary>
/// One quantification analysis to run and report: a resolved contrast (two arms, or a trend under
/// <see cref="DifferentialOptions.TrendColumn"/>) plus which of the other views to include.
/// </summary>
/// <remarks>
/// Built by whichever front end took the choices - the Differential pane from its controls,
/// <c>prism differential --report</c> from its flags - and run by <see cref="QuantAnalysis.Run"/>,
/// so the two cannot compute a report differently.
/// </remarks>
public sealed class QuantRequest
{
    /// <summary>The finished PRISM run; the report is written under its <c>quant/</c> folder.</summary>
    public required string OutputDir { get; init; }

    /// <summary>The loaded dataset the arm columns index.</summary>
    public required DifferentialDataset Dataset { get; init; }

    /// <summary>The contrast's statistical options - design, test, prior, correction, covariates.</summary>
    public required DifferentialOptions Options { get; init; }

    /// <summary>Which features count as hits.</summary>
    public required SignificanceRule Rule { get; init; }

    /// <summary>Two-arm contrasts: the grouping column. Ignored under a trend design.</summary>
    public string? GroupBy { get; init; }

    /// <summary>
    /// The metadata column <see cref="DifferentialOptions.SubjectLabels"/> came from, under a paired or
    /// within-subject design. The labels alone cannot name it, and the reproducing command needs it.
    /// </summary>
    public string? SubjectColumn { get; init; }

    /// <summary>
    /// Trend designs: which samples to fit over, as the metadata values to keep. Empty for all.
    /// </summary>
    /// <remarks>
    /// Carried as COLUMNS AND VALUES rather than as a list of sample indices, because the report has
    /// to write the command that reproduces it and <c>--restrict-to Study=...</c> is the only form a
    /// command line can carry. The indices are derived from these in one place, so the samples a
    /// report describes and the samples its command would select cannot differ.
    ///
    /// <para>A LIST, because restrictions narrow together - a study AND an on-drug window - and one
    /// pair would have quietly kept whichever the caller happened to pass last.</para>
    /// </remarks>
    public IReadOnlyList<QuantRestriction> Restrictions { get; init; } = Array.Empty<QuantRestriction>();

    /// <summary>Two-arm contrasts: arm A's dataset columns, as resolved by <see cref="ContrastArms.Resolve"/>.</summary>
    public IReadOnlyList<int> GroupA { get; init; } = Array.Empty<int>();

    /// <summary>Two-arm contrasts: arm B's dataset columns.</summary>
    public IReadOnlyList<int> GroupB { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Two-arm contrasts: the levels forming arm A, as picked (several make a union arm). The label
    /// every output shows is derived from these, so it cannot be missing or disagree with them.
    /// </summary>
    public IReadOnlyList<string> ALevels { get; init; } = Array.Empty<string>();

    /// <summary>Two-arm contrasts: the levels forming arm B.</summary>
    public IReadOnlyList<string> BLevels { get; init; } = Array.Empty<string>();

    /// <summary>
    /// The contrast's result, when the caller has already computed it for exactly these options and
    /// columns - the CLI has, having just written differential.csv - so it is not run twice. Null runs it.
    /// </summary>
    public DifferentialResult? Differential { get; init; }

    /// <summary>Marker panels to evaluate; empty omits the section (with a note).</summary>
    public IReadOnlyList<ProteinList> MarkerPanels { get; init; } = Array.Empty<ProteinList>();

    /// <summary>The column to group marker panels by; defaults to <see cref="GroupBy"/> for a two-arm contrast.</summary>
    public string? MarkerGroupBy { get; init; }

    /// <summary>Where g:Profiler requests go; null skips enrichment (with a note).</summary>
    public IJsonPoster? EnrichmentPoster { get; init; }

    /// <summary>A detection matrix already loaded for <see cref="OutputDir"/>, to reuse rather than re-read merged_data.</summary>
    public DetectionMatrixData? CachedDetection { get; init; }
}

/// <summary>What <see cref="QuantAnalysis.Run"/> produced.</summary>
/// <param name="HtmlPath">The written <c>quant_report.html</c>.</param>
/// <param name="Differential">The contrast's result.</param>
/// <param name="Detection">The detection test's result, or null when it did not run.</param>
/// <param name="DetectionMatrix">The detection matrix read (or reused), for a caller to cache.</param>
/// <param name="Notes">Each view that was skipped or limited, and why - for the status line or console.</param>
/// <param name="CommandArguments">The recorded reproducing command's arguments (after <c>prism</c>), or
/// null when no command can express the request.</param>
public sealed record QuantAnalysisResult(
    string HtmlPath, DifferentialResult Differential, DetectionAnalysisResult? Detection,
    DetectionMatrixData? DetectionMatrix, IReadOnlyList<string> Notes,
    IReadOnlyList<string>? CommandArguments = null);

/// <summary>
/// Runs a <see cref="QuantRequest"/> across every view and writes the quant report.
/// </summary>
/// <remarks>
/// <para>The one orchestration behind the pane's <b>Quant report...</b> button and
/// <c>prism differential --report</c>. It used to live in the pane, where it could not be tested and
/// the CLI could not reach it.</para>
/// <para>A view that cannot run is skipped and named in <see cref="QuantAnalysisResult.Notes"/>,
/// never allowed to fail the report: no merged_data (detection), no network or no significant genes
/// (enrichment), no panels or no grouping column (markers), a trend (detection, raw values).</para>
/// </remarks>
public static class QuantAnalysis
{
    /// <summary>The DetectionQValue cut a cell must fall below to count as detected.</summary>
    public const double DetectionQ = 0.01;

    /// <summary>
    /// The options the analysis actually runs under: covariates only when the test can take them.
    /// </summary>
    /// <remarks>
    /// Only the moderated t has a design matrix to put a covariate in. The pane greys its Adjust-for
    /// list out for the other tests but leaves the ticks in place, so a Welch contrast can arrive here
    /// still carrying covariates. They are dropped - and said so - rather than passed through, because
    /// the command this report records must be one the CLI accepts, and the CLI refuses
    /// <c>--adjust-for</c> without <c>--test moderated</c>. Dropping them here means the contrast, the
    /// detection test and the recorded command all describe the same unadjusted analysis.
    /// </remarks>
    public static DifferentialOptions EffectiveOptions(DifferentialOptions options, out string? note)
    {
        note = null;
        if (options.Test == DifferentialTest.ModeratedT || options.Covariates is not { Count: > 0 } covariates)
            return options;

        note = $"Not adjusted for {string.Join(", ", covariates.Select(c => c.Name))}: only the moderated t "
            + "takes covariates, so this report (and its detection test) is unadjusted. Switch to the "
            + "moderated t to adjust, or untick them.";
        return options with { Covariates = null };
    }

    /// <summary>
    /// Refuse a restriction that names a column the run does not have, or a value that column never
    /// holds.
    /// </summary>
    /// <remarks>
    /// Here, on the shared path, and not only in the CLI: a request built any other way - the pane, any
    /// caller of this API - would otherwise have an unknown column skipped, fitting over every sample
    /// while its recorded command claimed the restriction, or an absent value keep nothing and report
    /// "nothing to fit", which sends the reader to their data instead of to the restriction.
    /// </remarks>
    /// <param name="what">How the message names the restriction: the CLI's flag, or the pane's control.</param>
    public static void ValidateRestrictions(
        IReadOnlyList<QuantRestriction> restrictions, DifferentialDataset ds, string what = "Restrict to")
    {
        foreach (var r in restrictions)
        {
            if (!ds.MetadataColumns.Contains(r.Column))
                throw new ArgumentException($"{what}: no metadata column '{r.Column}'.");

            var present = ds.MetadataValues(r.Column)
                .Where(v => !string.IsNullOrEmpty(v)).Select(v => v!)
                .Distinct(StringComparer.Ordinal).OrderBy(p => p, StringComparer.Ordinal).ToList();
            foreach (var want in r.Values.Where(w => !present.Contains(w, StringComparer.Ordinal)))
                throw new ArgumentException(
                    $"{what}: '{r.Column}' has no value '{want}'. Present: "
                    + string.Join(", ", present.Select(p => $"'{p}'")) + ".");
        }
    }

    /// <summary>
    /// The sample columns <paramref name="restrictions"/> keep - every sample when there are none -
    /// intersected, so several narrow together rather than the last one winning. Possibly none.
    /// </summary>
    /// <remarks>
    /// The one place a restriction becomes samples: the pane's own trend, the report and the CLI all
    /// come through here, so the samples a view shows and the samples its report and command fit
    /// cannot differ. The restriction drops the COLUMN rather than NaN-ing its x, because NaN reaches
    /// <see cref="TrendSamples.Resolve"/> as "no value in the trend column" - which would report a
    /// deliberate subset as missing data.
    /// </remarks>
    public static int[] KeptColumns(IReadOnlyList<QuantRestriction> restrictions, DifferentialDataset ds)
    {
        var kept = Enumerable.Range(0, ds.SampleIds.Length).ToHashSet();
        foreach (var r in restrictions)
        {
            if (r.Values.Count == 0 || !ds.MetadataColumns.Contains(r.Column))
                continue;
            var values = ds.MetadataValues(r.Column);
            kept.IntersectWith(Enumerable.Range(0, ds.SampleIds.Length)
                .Where(i => i < values.Length && values[i] is { } v
                    && r.Values.Contains(v, StringComparer.Ordinal)));
        }

        return kept.OrderBy(i => i).ToArray();
    }

    /// <summary>
    /// The sample columns a trend runs over: those <see cref="QuantRequest.Restrictions"/> keep, after
    /// refusing a restriction the run cannot honor (<see cref="ValidateRestrictions"/>) and one that
    /// keeps no sample at all.
    /// </summary>
    public static int[] TrendColumnsFor(QuantRequest request, DifferentialDataset ds)
    {
        ValidateRestrictions(request.Restrictions, ds);
        var kept = KeptColumns(request.Restrictions, ds);
        if (kept.Length == 0)
            throw new ArgumentException(
                "Restricting the trend kept no samples: "
                + string.Join("; ", request.Restrictions.Select(r => $"{r.Column}={string.Join(",", r.Values)}")),
                nameof(request));
        return kept;
    }

    /// <summary>Run every view and write the report.</summary>
    public static QuantAnalysisResult Run(QuantRequest request)
    {
        var ds = request.Dataset;
        var notes = new List<string>();
        var options = EffectiveOptions(request.Options, out var dropped);
        if (dropped is not null)
            notes.Add(dropped);
        var rule = request.Rule;

        // Built once, never looked up per feature: enrichment and the CSVs walk every tested feature,
        // and a per-feature scan is quadratic (see DifferentialDataset.IdentityOf). The label falls back
        // to the id; the gene does NOT, because a protein group id or a peptide sequence is not a gene
        // symbol and g:Profiler would be asked about it as one. The same maps the pane builds.
        var labelById = new Dictionary<string, string>(StringComparer.Ordinal);
        var geneById = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var i = 0; i < ds.FeatureIds.Length; i++)
        {
            labelById[ds.FeatureIds[i]] =
                string.IsNullOrEmpty(ds.FeatureLabels[i]) ? ds.FeatureIds[i] : ds.FeatureLabels[i];
            if (!string.IsNullOrEmpty(ds.FeatureGenes[i]))
                geneById[ds.FeatureIds[i]] = ds.FeatureGenes[i];
        }
        var isTrend = options.Design is DifferentialDesign.LinearTrend
            or DifferentialDesign.LinearTrendWithinSubject;

        // --- the contrast, and what differential.csv's header records about it ---
        DifferentialResult res;
        string groupBy, aLabel, bLabel, contrastLabel, effectName;
        QuantContrast contrast;
        int[]? trendColumns = null;
        var restricted = request.Restrictions.Where(r => r.Values.Count > 0).ToList();
        if (isTrend)
        {
            var trendOver = options.TrendColumn
                ?? throw new ArgumentException("A trend design needs Options.TrendColumn.", nameof(request));
            // Through TrendAxis, not NumericValues: TrendColumn carries the axis LABEL, which on a
            // column whose values embed a number ("Longitudinal Draw Description (Week)") is not a
            // column name at all - NumericValues would return every sample NaN and the trend would
            // report that it had nothing to fit.
            var axis = TrendAxis.Find(trendOver, ds.MetadataColumns, ds.MetadataValues)
                ?? throw new ArgumentException(
                    $"No trend axis '{trendOver}' in this run.", nameof(request));
            var x = TrendAxis.Read(ds.MetadataValues(axis.Column), axis);
            trendColumns = TrendColumnsFor(request, ds);
            res = request.Differential
                ?? Differential.RunTrend(ds.ExprLog2, ds.FeatureIds, trendColumns, x, options);
            groupBy = trendOver;
            // Over the samples the fit used: the endpoints become the span the results header says
            // log2fc is the change across, so an excluded sample's x would misdescribe every row.
            (aLabel, bLabel) = DifferentialCsv.TrendEndpoints(
                trendColumns.Select(c => x[c]).ToList());
            effectName = $"log2 change across {trendOver}";
            // The restriction is part of WHAT was compared, so it goes wherever the contrast is
            // recorded - the report's title, quant_parameters, differential.csv's header - and not
            // only into the reproducing command. Without it two reports restricted to different
            // studies described themselves identically.
            contrast = new QuantContrast(null, null, null, trendOver, restricted.Count > 0 ? restricted : null);
            contrastLabel = contrast.Describe();
        }
        else
        {
            groupBy = request.GroupBy
                ?? throw new ArgumentException("A two-arm contrast needs GroupBy.", nameof(request));
            // Refused rather than defaulted: a label invented here would reach differential.csv's
            // provenance header as a level that does not exist.
            if (request.ALevels.Count == 0 || request.BLevels.Count == 0)
                throw new ArgumentException("A two-arm contrast needs the levels of both arms.", nameof(request));
            aLabel = ContrastArms.Describe(request.ALevels);
            bLabel = ContrastArms.Describe(request.BLevels);
            res = request.Differential
                ?? Differential.Run(ds.ExprLog2, ds.FeatureIds, request.GroupA, request.GroupB, options);
            contrastLabel = $"{bLabel} vs {aLabel} by {groupBy}";
            effectName = "log2FC";
            contrast = new QuantContrast(groupBy, request.ALevels, request.BLevels, null);
        }

        // --- raw per-sample values: over the samples the contrast actually ran on ---
        List<int>? valueColumns = null;
        if (isTrend)
        {
            notes.Add("No differential_values.csv: a trend does not record which samples entered the fit.");
        }
        else
        {
            var (usedA, usedB) = PairedSamples.ColumnsUsed(options.Design, options.SubjectLabels,
                request.GroupA, request.GroupB);
            valueColumns = usedA.Concat(usedB).ToList();
        }

        // --- detection: the test the design calls for, two-arm contrasts only ---
        DetectionMatrixData? detMatrix = request.CachedDetection;
        DetectionAnalysisResult? detection = null;
        if (isTrend)
        {
            notes.Add("Detection skipped: it compares two groups, and a trend has none.");
        }
        else
        {
            try
            {
                detMatrix ??= DetectionMatrix.Load(request.OutputDir, DetectionQ, null);
                detection = DetectionAnalysis.Run(detMatrix, ds, request.GroupA, request.GroupB,
                    options.Design, options.SubjectLabels, options.Covariates?.Select(c => c.Name).ToList(),
                    options.Correction, options.Covariates?.ToDictionary(c => c.Name, c => c.Kind));
                if (DetectionAnalysis.UnpairedNote(detection.UnpairedReason) is { } unpaired)
                    notes.Add(unpaired);
                if (!detection.Identifiable)
                    notes.Add("Adjusted detection is not identifiable (group confounded with the covariates); "
                        + "the detection section says so and has no table.");
            }
            catch (DetectionSamplesNotFoundException)
            {
                // merged_data is there; the contrast's samples are not in it. A different fix from a
                // missing merged_data, so a different sentence.
                notes.Add("Detection skipped: none of an arm's samples are in merged_data.");
            }
            catch (Exception ex)
            {
                // Broad on purpose, as the pane's was: a missing or unreadable merged_data surfaces as
                // anything from a DirectoryNotFoundException to a DuckDB error, and none of them is a
                // reason to lose the rest of the report.
                notes.Add("Detection skipped (needs merged_data in the output directory): " + ex.Message);
            }
        }

        // --- enrichment: needs the network ---
        IReadOnlyList<EnrichmentTerm>? enrichment = null;
        if (request.EnrichmentPoster is null)
        {
            notes.Add("Enrichment skipped: not requested.");
        }
        else
        {
            try
            {
                var (sig, background) = Enrichment.Enrichment.SigAndBackgroundGenes(
                    res, id => geneById.GetValueOrDefault(id), rule);
                if (sig.Count == 0)
                    notes.Add("Enrichment skipped: no significant genes to submit.");
                else
                    enrichment = Enrichment.Enrichment.GProfiler(sig, background, request.EnrichmentPoster);
            }
            catch (Exception ex)
            {
                // Any failure here is the network or the service, never a reason to lose the report.
                notes.Add("Enrichment skipped (needs internet access to g:Profiler): " + ex.Message);
            }
        }

        // --- marker panels ---
        var markers = new List<MarkerReportSection>();
        var markerColumn = request.MarkerGroupBy ?? (isTrend ? null : groupBy);
        if (request.MarkerPanels.Count == 0)
        {
            notes.Add("Markers omitted: no marker panels were selected.");
        }
        else if (markerColumn is null || !ds.MetadataColumns.Contains(markerColumn))
        {
            notes.Add(markerColumn is null
                ? "Markers omitted: no column to group the panels by."
                : $"Markers omitted: no metadata column '{markerColumn}' to group the panels by.");
        }
        else
        {
            var groups = ds.MetadataValues(markerColumn);
            // Over the samples the trend was fitted on: a sample with no group is one the panels leave
            // out, so a restriction is applied the same way. Every sample here would put a study the
            // report excluded into its marker plots, with nothing saying so.
            if (trendColumns is not null && restricted.Count > 0)
            {
                var kept = trendColumns.ToHashSet();
                groups = groups.Select((g, i) => kept.Contains(i) ? g : null).ToArray();
            }

            var identities = Enumerable.Range(0, ds.FeatureIds.Length).Select(ds.IdentityOf).ToArray();
            foreach (var panel in request.MarkerPanels)
                markers.Add(new MarkerReportSection(panel.Name, markerColumn,
                    MarkerPanel.Evaluate(ds.ExprLog2, identities, groups, ds.SampleIds, panel, perSample: false)));
        }

        // The command that regenerates this report, from the options it actually ran with - or, where
        // no command can express the request faithfully, the reason, in place of a command that would
        // quietly run something else.
        IReadOnlyList<string>? commandArgs = null;
        string? commandLine = null;
        if (QuantCommand.TryArguments(request, options, out var args, out var noCommand))
        {
            commandArgs = args;
            // Rendered from the arguments just resolved, not by resolving the request again, so the
            // recorded list and the printed line cannot be two different answers.
            commandLine = QuantCommand.Line(args);
        }
        else
        {
            notes.Add("No command-line equivalent: " + noCommand + ".");
        }

        var quant = new QuantConfig(
            Level: DifferentialTokens.Level(ds.Level),
            Contrast: contrast,
            Design: DifferentialTokens.Design(options.Design),
            Test: DifferentialTokens.Test(options.Test),
            Prior: DifferentialTokens.Prior(options.Prior),
            PriorUsed: res.VariancePrior,
            Correction: DifferentialTokens.Correction(options.Correction),
            Covariates: res.CovariatesUsed,
            HitRule: rule.Describe(effectName),
            DetectionEnabled: detection is not null,
            DetectionQ: DetectionQ,
            EnrichmentEnabled: enrichment is not null,
            EnrichmentSources: Enrichment.Enrichment.DefaultSources,
            EnrichmentDirection: "both",
            MarkerPanels: markers.Select(m => m.PanelName).ToList(),
            ClinicalCsvs: ds.AttachedClinicalCsvs.ToList(),
            Command: commandLine,
            CommandUnavailable: noCommand,
            CovariateTypes: options.Covariates?.Select(c => $"{c.Name}={CovariateTyping.Token(c.Kind)}").ToList(),
            DetectionTest: detection is null
                ? null
                : DetectionAnalysis.Describe(detection) + (detection.Identifiable ? string.Empty : " (not identifiable - no result)"));

        var inputs = new QuantReportInputs
        {
            Differential = res,
            Rule = rule,
            Corrected = options.Correction != MultipleTesting.None,
            Contrast = contrastLabel,
            EffectName = effectName,
            LabelFor = id => labelById.GetValueOrDefault(id, id),
            Options = options,
            GroupBy = groupBy,
            ALabel = aLabel,
            BLabel = bLabel,
            Restrictions = restricted,
            Dataset = ds,
            ContrastColumns = valueColumns,
            Detection = detection,
            Enrichment = enrichment,
            Markers = markers,
        };

        var html = QuantReport.Write(request.OutputDir, quant, inputs);
        return new QuantAnalysisResult(html, res, detection, detMatrix, notes, commandArgs);
    }
}
