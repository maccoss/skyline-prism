using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.DifferentialAnalysis.Detection;
using SkylinePrism.Core.DifferentialAnalysis.Enrichment;
using SkylinePrism.Core.Visualization;

namespace SkylinePrism.App;

/// <summary>
/// The Differential visualization pane: load the corrected matrix from the current output directory
/// and offer three views over a two-group contrast - a limma moderated-t Volcano, a sample PCA, and a
/// peptide Detection-frequency test. All statistics live in SkylinePrism.Core; this is presentation.
/// </summary>
public partial class MainWindow
{
    private static readonly string[] DiffPalette =
    {
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    };

    private static readonly HashSet<string> ReservedMetaColumns =
        new(StringComparer.Ordinal) { "sample", "sample_type", "batch" };

    private DifferentialDataset? _diffDataset;
    private Dictionary<string, string> _diffLabelById = new(StringComparer.Ordinal);
    private DetectionMatrixData? _detectionData;
    private string? _detectionDir;
    private string? _clinicalCsvPath;
    private List<QcGroupValue> _diffCovariateValues = new();
    private HttpJsonPoster? _diffPoster;
    private bool _diffSuppress;
    private int _diffRequest;
    private bool _diffLoaded;
    private string? _diffLoadedDir;
    private FeatureLevel _diffLoadedLevel;

    // Volcano click-to-boxplot state: each plotted point's location + feature id, the per-feature row for
    // the title, and the two groups' sample indices so the boxplot can pull that feature's abundances.
    private List<(ScottPlot.Coordinates Loc, string FeatureId)> _volcanoPoints = new();
    private Dictionary<string, DifferentialRow> _volcanoRowById = new(StringComparer.Ordinal);
    private List<int> _volcanoGroupA = new();
    private List<int> _volcanoGroupB = new();
    private string _volcanoAName = "A";
    private string _volcanoBName = "B";
    private FeatureDetailWindow? _featureDetailWindow;

    private enum DiffView
    {
        Volcano,
        Pca,
        Detection,
        Enrichment,
    }

    private sealed record VolcanoRow(string Feature, double Log2FC, double P, double AdjP, string FeatureId);

    private sealed record PcaVarRow(string Component, double VariancePct);

    private sealed record DetRow(string Peptide, double RateA, double RateB, double P, double Q);

    private sealed record DetGlmRow(string Peptide, double RateA, double RateB, double LogOR, double P, double Q);

    private sealed record EnrichRow(string Source, string Term, double PValue, double Fold);

    private HttpJsonPoster DiffPoster => _diffPoster ??= new HttpJsonPoster();

    /// <summary>Forget the loaded matrix so the pane reloads on its next show (new dir / new run).</summary>
    private void InvalidateDifferential()
    {
        _diffLoaded = false;
        _diffLoadedDir = null;
        _diffDataset = null;
        _detectionData = null;
        _detectionDir = null;
    }

    private FeatureLevel DiffSelectedLevel() =>
        (DiffLevelCombo.SelectedItem as ComboBoxItem)?.Content as string == "Peptide"
            ? FeatureLevel.Peptide
            : FeatureLevel.Protein;

    private DiffView DiffSelectedView() =>
        ((DiffViewCombo.SelectedItem as ComboBoxItem)?.Content as string) switch
        {
            "PCA" => DiffView.Pca,
            "Detection" => DiffView.Detection,
            "Enrichment" => DiffView.Enrichment,
            _ => DiffView.Volcano,
        };

    /// <summary>Load the corrected matrix + metadata for the current output directory and level.</summary>
    private async Task LoadDifferentialAsync()
    {
        var dir = OutputDirBox.Text?.Trim();
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
        {
            InvalidateDifferential();
            ClearDiffOutput();
            DiffStatusText.Text = "Set a PRISM output directory above to run a contrast.";
            return;
        }

        _diffSuppress = true;
        try
        {
            if (DiffViewCombo.SelectedItem is null)
                DiffViewCombo.SelectedIndex = 0;
            if (DiffLevelCombo.SelectedItem is null)
                DiffLevelCombo.SelectedIndex = 0;
        }
        finally
        {
            _diffSuppress = false;
        }

        var level = DiffSelectedLevel();
        if (_diffLoaded && _diffLoadedDir == dir && _diffLoadedLevel == level)
            return; // already current for this directory + level

        ClearDiffOutput();
        DiffStatusText.Text = "Loading...";
        var request = ++_diffRequest;

        DifferentialDataset ds;
        try
        {
            ds = await Task.Run(() => DifferentialDataset.Load(dir, level));
        }
        catch (Exception ex)
        {
            if (request == _diffRequest)
            {
                InvalidateDifferential();
                DiffStatusText.Text = "Load failed: " + ex.Message;
            }

            return;
        }

        if (request != _diffRequest)
            return; // a newer load superseded this one

        _diffDataset = ds;
        _diffLoaded = true;
        _diffLoadedDir = dir;
        _diffLoadedLevel = level;
        _diffLabelById = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var i = 0; i < ds.FeatureIds.Length; i++)
            _diffLabelById[ds.FeatureIds[i]] =
                string.IsNullOrEmpty(ds.FeatureLabels[i]) ? ds.FeatureIds[i] : ds.FeatureLabels[i];

        // Re-apply a previously attached clinical CSV to the freshly loaded dataset (best-effort).
        if (_clinicalCsvPath is not null && File.Exists(_clinicalCsvPath))
        {
            try
            {
                ds.AttachClinical(_clinicalCsvPath);
            }
            catch
            {
                // a mismatched clinical file just adds nothing; not fatal
            }
        }

        _diffSuppress = true;
        try
        {
            DiffGroupByCombo.ItemsSource = ds.MetadataColumns;
            DiffGroupByCombo.SelectedItem = DefaultContrastColumn(ds);
        }
        finally
        {
            _diffSuppress = false;
        }

        PopulateDiffGroupValues();
        UpdateDiffCaveat();
        DiffStatusText.Text =
            $"Loaded {ds.FeatureIds.Length} {level.ToString().ToLowerInvariant()} features x " +
            $"{ds.SampleIds.Length} samples. Pick groups and Run.";
    }

    /// <summary>Prefer the first non-reserved metadata column with at least two values; else sample_type.</summary>
    private static string? DefaultContrastColumn(DifferentialDataset ds)
    {
        foreach (var col in ds.MetadataColumns)
        {
            if (ReservedMetaColumns.Contains(col))
                continue;
            var distinct = ds.MetadataValues(col).Where(v => !string.IsNullOrEmpty(v)).Distinct().Count();
            if (distinct >= 2)
                return col;
        }

        return ds.MetadataColumns.FirstOrDefault(c => c == "sample_type")
            ?? ds.MetadataColumns.FirstOrDefault();
    }

    private void PopulateDiffGroupValues()
    {
        if (_diffDataset is null || DiffGroupByCombo.SelectedItem is not string col)
            return;

        var values = _diffDataset.MetadataValues(col)
            .Where(v => !string.IsNullOrEmpty(v))
            .Distinct()
            .OrderBy(v => v, StringComparer.Ordinal)
            .ToList();

        DiffACombo.ItemsSource = values;
        DiffBCombo.ItemsSource = values;
        if (values.Count >= 2)
        {
            DiffACombo.SelectedIndex = 0;
            DiffBCombo.SelectedIndex = 1;
        }
        else
        {
            DiffACombo.SelectedIndex = -1;
            DiffBCombo.SelectedIndex = -1;
        }

        PopulateDiffCovariates(col);
    }

    private void PopulateDiffCovariates(string groupByColumn)
    {
        if (_diffDataset is null)
            return;

        // Any metadata column can be a covariate except the sample id itself and the contrast column.
        _diffCovariateValues = _diffDataset.MetadataColumns
            .Where(c => c != groupByColumn && c != "sample")
            .Select(c => new QcGroupValue { Name = c })
            .ToList();
        DiffCovariatesCombo.ItemsSource = _diffCovariateValues;
    }

    /// <summary>Ticked covariates, with values aligned to <paramref name="targetSampleIds"/>, or null if none.</summary>
    private IReadOnlyList<Covariate>? SelectedCovariatesFor(IReadOnlyList<string> targetSampleIds)
    {
        if (_diffDataset is null)
            return null;
        var chosen = _diffCovariateValues.Where(v => v.IsSelected).Select(v => v.Name).ToList();
        if (chosen.Count == 0)
            return null;

        var indexById = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < _diffDataset.SampleIds.Length; i++)
            indexById[_diffDataset.SampleIds[i]] = i;

        var result = new List<Covariate>(chosen.Count);
        foreach (var col in chosen)
        {
            var colValues = _diffDataset.MetadataValues(col);
            var aligned = new string?[targetSampleIds.Count];
            for (var k = 0; k < targetSampleIds.Count; k++)
                aligned[k] = indexById.TryGetValue(targetSampleIds[k], out var di) ? colValues[di] : null;
            result.Add(Covariate.FromMetadata(col, aligned));
        }

        return result;
    }

    private async void OnDiffLevelChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_diffSuppress || !IsInitialized)
            return;
        try
        {
            await LoadDifferentialAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffLevelChanged), ex);
        }
    }

    private void OnDiffGroupByChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_diffSuppress)
            return;
        PopulateDiffGroupValues();
    }

    /// <summary>
    /// Choose an external clinical metadata CSV (a top-level input, beside the metadata report). The
    /// path is remembered and applied whenever a differential dataset is loaded; if one is already
    /// loaded, it is joined immediately and the Differential pane's selectors refresh.
    /// </summary>
    private void OnBrowseClinical(object sender, RoutedEventArgs e)
    {
        var dlg = new Microsoft.Win32.OpenFileDialog
        {
            Title = "Choose clinical metadata CSV",
            Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*",
            CheckFileExists = true,
        };
        if (dlg.ShowDialog(this) != true)
            return;

        _clinicalCsvPath = dlg.FileName;
        ClinicalCsvBox.Text = dlg.FileName;

        // If a differential dataset is already loaded, apply the join now; otherwise it will be applied
        // the next time one loads (LoadDifferentialAsync re-applies _clinicalCsvPath).
        if (_diffDataset is not null)
            ApplyClinicalToLoadedDataset();
    }

    /// <summary>Join the remembered clinical CSV to the loaded dataset and refresh the selectors.</summary>
    private void ApplyClinicalToLoadedDataset()
    {
        if (_diffDataset is null || _clinicalCsvPath is null || !File.Exists(_clinicalCsvPath))
            return;

        try
        {
            var result = _diffDataset.AttachClinical(_clinicalCsvPath);
            if (result.KeyColumn is null || result.AddedColumns.Count == 0)
            {
                DiffStatusText.Text =
                    $"Clinical CSV: no column matched the samples (best match rate {result.MatchRate:P0}). " +
                    "Nothing was added.";
                return;
            }

            // Refresh the group-by choices so the new clinical columns appear; select the first one.
            _diffSuppress = true;
            try
            {
                DiffGroupByCombo.ItemsSource = null;
                DiffGroupByCombo.ItemsSource = _diffDataset.MetadataColumns;
                DiffGroupByCombo.SelectedItem = result.AddedColumns[0];
            }
            finally
            {
                _diffSuppress = false;
            }

            PopulateDiffGroupValues();
            DiffStatusText.Text =
                $"Attached {result.AddedColumns.Count} clinical column(s) via key '{result.KeyColumn}' " +
                $"(matched {result.MatchRate:P0} of samples): {string.Join(", ", result.AddedColumns)}.";
        }
        catch (Exception ex)
        {
            DiffStatusText.Text = $"Could not attach clinical CSV: {ex.Message}";
        }
    }

    private async void OnDiffViewChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_diffSuppress || _diffDataset is null)
            return;
        try
        {
            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffViewChanged), ex);
        }
    }

    private async void OnRunDifferential(object sender, RoutedEventArgs e)
    {
        try
        {
            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnRunDifferential), ex);
        }
    }

    private async Task RunCurrentViewAsync()
    {
        if (_diffDataset is null)
        {
            DiffStatusText.Text = "Load a run first (point the output directory at a finished PRISM run).";
            return;
        }

        UpdateDiffCaveat();
        switch (DiffSelectedView())
        {
            case DiffView.Pca:
                await RunPcaAsync();
                break;
            case DiffView.Detection:
                await RunDetectionAsync();
                break;
            case DiffView.Enrichment:
                await RunEnrichmentAsync();
                break;
            default:
                await RunVolcanoAsync();
                break;
        }
    }

    /// <summary>
    /// Show the honest interpretation caveats for the current view, carried over from the explorer's
    /// README so the native tool does not lose them. These are the methodology limits a reader must keep
    /// in mind, not error messages.
    /// </summary>
    private void UpdateDiffCaveat()
    {
        DiffCaveatText.Text = DiffSelectedView() switch
        {
            DiffView.Pca =>
                "PCA is on the log2 matrix, complete-case (only features present in every sample), "
                + "feature-mean-centered and unscaled. The axis SIGN is arbitrary - a component and its "
                + "negative are equivalent, so orientation can differ from other tools while distances and "
                + "variance-explained do not. Colour is metadata only; PCA itself is unsupervised.",
            DiffView.Detection =>
                "Detection recovers genuine on/off from transition-level merged_data (a cell counts as "
                + "detected only where DetectionQValue < 0.01), which the dense abundance matrix cannot "
                + "show. Per-peptide Fisher exact test, or Firth-penalized logistic regression when "
                + "covariates are set. Peptides from one protein are correlated, so the BH q-values are "
                + "exploratory ranking, not protein-level significance.",
            DiffView.Enrichment =>
                "Enrichment runs g:Profiler over the significant hits against the tested background and "
                + "needs network access. It only re-describes the hit list - any inflation from the "
                + "caveats on the Volcano or Detection views is carried straight into it. Exploratory.",
            _ =>
                "Volcano is limma moderated-t on the log2 corrected matrix, which is DENSE: Skyline "
                + "integrates a peak boundary for every replicate, so an 'undetected' peptide is imputed "
                + "baseline, not missing - a fold change can reflect baseline noise rather than real "
                + "signal (use the Detection view for on/off). At peptide level BH q-values are "
                + "anti-conservative (peptides from one protein are correlated). When batch and condition "
                + "are confounded, add batch under 'Adjust for'.",
        };
    }

    private bool TryGetGroups(out string col, out List<int> groupA, out List<int> groupB,
        out string aVal, out string bVal)
    {
        col = string.Empty;
        aVal = string.Empty;
        bVal = string.Empty;
        groupA = new List<int>();
        groupB = new List<int>();
        if (_diffDataset is null || DiffGroupByCombo.SelectedItem is not string c
            || DiffACombo.SelectedItem is not string a || DiffBCombo.SelectedItem is not string b || a == b)
            return false;

        col = c;
        aVal = a;
        bVal = b;
        var meta = _diffDataset.MetadataValues(c);
        groupA = Enumerable.Range(0, meta.Length).Where(j => meta[j] == a).ToList();
        groupB = Enumerable.Range(0, meta.Length).Where(j => meta[j] == b).ToList();
        return true;
    }

    private async Task RunVolcanoAsync()
    {
        if (!TryGetGroups(out _, out var a, out var b, out var aVal, out var bVal))
        {
            DiffStatusText.Text = "Pick a group-by column and two different values.";
            return;
        }

        var dataset = _diffDataset!;
        var covariates = SelectedCovariatesFor(dataset.SampleIds);
        var trend = DiffTrendCheck.IsChecked == true;
        DifferentialResult res;
        try
        {
            res = await Task.Run(() =>
                Differential.Run(dataset.ExprLog2, dataset.FeatureIds, a, b, 2, covariates, trend));
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
        {
            DiffStatusText.Text = "Cannot run this contrast: " + ex.Message;
            return;
        }

        _volcanoGroupA = a;
        _volcanoGroupB = b;
        _volcanoAName = aVal;
        _volcanoBName = bVal;
        RenderVolcano(res);
        DiffGrid.ItemsSource = res.Rows.Select(r => new VolcanoRow(
            _diffLabelById.GetValueOrDefault(r.FeatureId, r.FeatureId), r.LogFc, r.PValue, r.AdjPValue,
            r.FeatureId))
            .ToList();

        var nSig = res.Rows.Count(r => r.AdjPValue < 0.05 && Math.Abs(r.LogFc) >= 1.0);
        var adj = res.CovariatesUsed.Count > 0 ? $"; adjusted for {string.Join(", ", res.CovariatesUsed)}" : string.Empty;
        DiffStatusText.Text =
            $"{aVal} (n={a.Count}) vs {bVal} (n={b.Count}) - {res.NFeaturesTested} tested, {nSig} significant{adj}. "
            + "Click a point for its per-sample boxplot.";
    }

    private async Task RunPcaAsync()
    {
        if (DiffGroupByCombo.SelectedItem is not string col)
        {
            DiffStatusText.Text = "Pick a group-by column to color the PCA by.";
            return;
        }

        var dataset = _diffDataset!;
        var all = Enumerable.Range(0, dataset.SampleIds.Length).ToList();
        PcaResult pca;
        try
        {
            pca = await Task.Run(() => DifferentialPca.Compute(dataset.ExprLog2, dataset.SampleIds, all));
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
        {
            DiffStatusText.Text = "Cannot compute PCA: " + ex.Message;
            return;
        }

        RenderPca(pca, col);
        DiffGrid.ItemsSource = pca.VarianceRatio
            .Select((v, i) => new PcaVarRow($"PC{i + 1}", v * 100.0))
            .ToList();
        DiffStatusText.Text =
            $"PCA over {all.Count} samples, {pca.NFeaturesUsed} complete features, colored by {col}.";
    }

    private async Task RunDetectionAsync()
    {
        if (!TryGetGroups(out _, out var a, out var b, out var aVal, out var bVal))
        {
            DiffStatusText.Text = "Pick a group-by column and two different values.";
            return;
        }

        var dir = OutputDirBox.Text?.Trim();
        if (string.IsNullOrEmpty(dir))
        {
            DiffStatusText.Text = "Set a PRISM output directory above.";
            return;
        }

        DetectionMatrixData det;
        if (_detectionData is not null && _detectionDir == dir)
        {
            det = _detectionData;
        }
        else
        {
            if (_isRunning)
            {
                DiffStatusText.Text = "A PRISM run is in progress - wait for it to finish before reading detection.";
                return;
            }

            DiffStatusText.Text = "Loading detection matrix from merged_data...";
            try
            {
                det = await Task.Run(() => DetectionMatrix.Load(dir!, 0.01, null));
            }
            catch (Exception ex)
            {
                DiffStatusText.Text = "Detection load failed (needs merged_data in the output dir): " + ex.Message;
                return;
            }

            _detectionData = det;
            _detectionDir = dir;
        }

        var dataset = _diffDataset!;
        var detIndex = det.SampleIds.Select((s, i) => (s, i)).ToDictionary(x => x.s, x => x.i, StringComparer.Ordinal);
        var aCols = a.Select(j => dataset.SampleIds[j]).Where(detIndex.ContainsKey).Select(s => detIndex[s]).ToList();
        var bCols = b.Select(j => dataset.SampleIds[j]).Where(detIndex.ContainsKey).Select(s => detIndex[s]).ToList();
        if (aCols.Count == 0 || bCols.Count == 0)
        {
            DiffStatusText.Text = "The selected samples were not found in the detection matrix.";
            return;
        }

        var dropped = a.Count - aCols.Count + (b.Count - bCols.Count);
        var droppedNote = dropped > 0 ? $" ({dropped} samples not in merged_data)" : string.Empty;

        // A ticked covariate switches to the Firth-penalized GLM (adjusted detection); otherwise Fisher.
        var covariates = SelectedCovariatesFor(det.SampleIds);
        if (covariates is not null)
        {
            DetectionGlmResult glm;
            try
            {
                glm = await Task.Run(() =>
                    DetectionGlm.Run(det.Matrix, det.PeptideIds, aCols, bCols, covariates));
            }
            catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
            {
                DiffStatusText.Text = "Adjusted detection failed: " + ex.Message;
                return;
            }

            if (!glm.Identifiable)
            {
                ClearDiffOutput();
                DiffStatusText.Text = "Adjusted detection is not identifiable (group confounded with the "
                    + $"covariates, R^2={glm.GroupCollinearityR2:0.00}). Use the unadjusted view.";
                return;
            }

            RenderDetectionGlm(glm.Rows);
            DiffGrid.ItemsSource = glm.Rows.Take(1000)
                .Select(r => new DetGlmRow(r.PeptideId, r.RateA, r.RateB, r.LogOr, r.P, r.Q)).ToList();
            DiffStatusText.Text =
                $"Adjusted detection (Firth GLM): {aVal} (n={aCols.Count}) vs {bVal} (n={bCols.Count}), "
                + $"adjusted for {string.Join(", ", glm.CovariatesUsed)}, {glm.Rows.Count} peptides{droppedNote}.";
            return;
        }

        IReadOnlyList<DetectionRow> rows;
        try
        {
            rows = await Task.Run(() => DetectionTest.Run(det.Matrix, det.PeptideIds, aCols, bCols));
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
        {
            DiffStatusText.Text = "Detection test failed: " + ex.Message;
            return;
        }

        RenderDetection(rows);
        DiffGrid.ItemsSource = rows.Take(1000)
            .Select(r => new DetRow(r.PeptideId, r.RateA, r.RateB, r.P, r.Q)).ToList();
        DiffStatusText.Text =
            $"Detection (peptide-level, DetectionQValue < 0.01): {aVal} (n={aCols.Count}) vs " +
            $"{bVal} (n={bCols.Count}) over {rows.Count} peptides{droppedNote}.";
    }

    private async Task RunEnrichmentAsync()
    {
        if (!TryGetGroups(out _, out var a, out var b, out var aVal, out var bVal))
        {
            DiffStatusText.Text = "Pick a group-by column and two different values.";
            return;
        }

        var dataset = _diffDataset!;
        var covariates = SelectedCovariatesFor(dataset.SampleIds);
        var trend = DiffTrendCheck.IsChecked == true;
        DifferentialResult res;
        try
        {
            res = await Task.Run(() =>
                Differential.Run(dataset.ExprLog2, dataset.FeatureIds, a, b, 2, covariates, trend));
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
        {
            DiffStatusText.Text = "Cannot run this contrast: " + ex.Message;
            return;
        }

        var (sig, background) = Enrichment.SigAndBackgroundGenes(
            res, fid => _diffLabelById.GetValueOrDefault(fid), 0.05, 1.0);
        if (sig.Count == 0)
        {
            ClearDiffOutput();
            DiffStatusText.Text =
                $"No significant genes (adj.P < 0.05, |log2FC| >= 1) for {aVal} vs {bVal} - nothing to enrich.";
            return;
        }

        DiffStatusText.Text = $"Querying g:Profiler for {sig.Count} genes...";
        List<EnrichmentTerm> terms;
        try
        {
            terms = await Task.Run(() => Enrichment.GProfiler(sig, background, DiffPoster));
        }
        catch (Exception ex)
        {
            DiffStatusText.Text = "Enrichment request failed (needs internet access): " + ex.Message;
            return;
        }

        RenderEnrichment(terms);
        DiffGrid.ItemsSource = terms.Take(1000)
            .Select(t => new EnrichRow(t.Source, t.TermName, t.PValue, t.FoldEnrichment)).ToList();
        DiffStatusText.Text = terms.Count == 0
            ? $"No enriched terms for {sig.Count} significant genes (background {background.Count})."
            : $"{terms.Count} enriched terms for {sig.Count} significant genes (background {background.Count}).";
    }

    private void RenderEnrichment(IReadOnlyList<EnrichmentTerm> terms)
    {
        DiffPlot.Reset();
        var plt = DiffPlot.Plot;
        if (terms.Count > 0)
        {
            var ys = terms.Take(15).Select(t => -Math.Log10(Math.Max(t.PValue, 1e-300))).ToArray();
            plt.Add.Bars(ys);
            plt.XLabel("top enriched terms (ranked by p)");
            plt.YLabel("-log10 p (g:SCS)");
        }

        PlotRenderer.StyleQcPlot(plt);
        DiffPlot.Refresh();
    }

    private void RenderDetectionGlm(IReadOnlyList<DetectionGlmRow> rows)
    {
        DiffPlot.Reset();
        var plt = DiffPlot.Plot;

        var bgX = new List<double>();
        var bgY = new List<double>();
        var sigX = new List<double>();
        var sigY = new List<double>();
        foreach (var r in rows)
        {
            if (!double.IsFinite(r.LogOr) || !double.IsFinite(r.P))
                continue;
            var y = -Math.Log10(Math.Max(r.P, 1e-300));
            if (r.Q < 0.05)
            {
                sigX.Add(r.LogOr);
                sigY.Add(y);
            }
            else
            {
                bgX.Add(r.LogOr);
                bgY.Add(y);
            }
        }

        AddMarkers(plt, bgX, bgY, "#b8c4d0", 6, "q >= 0.05");
        AddMarkers(plt, sigX, sigY, "#2ca02c", 7, "q < 0.05");
        plt.Add.VerticalLine(0.0);
        plt.ShowLegend();
        plt.XLabel("log odds ratio (B / A)");
        plt.YLabel("-log10 P");
        PlotRenderer.StyleQcPlot(plt);
        DiffPlot.Refresh();
    }

    private void OnDiffGridAutoGeneratingColumn(object sender, DataGridAutoGeneratingColumnEventArgs e)
    {
        // FeatureId backs click-to-boxplot from the grid; it is not a column the user needs to see.
        if (e.PropertyName == "FeatureId")
        {
            e.Cancel = true;
            return;
        }

        var (header, format) = e.PropertyName switch
        {
            "Log2FC" => ("log2FC", "0.###"),
            "AdjP" => ("adj.P", "0.##e0"),
            "P" => ("P", "0.##e0"),
            "PValue" => ("p", "0.##e0"),
            "Q" => ("q", "0.##e0"),
            "LogOR" => ("logOR", "0.###"),
            "RateA" => ("rate A", "0.00"),
            "RateB" => ("rate B", "0.00"),
            "VariancePct" => ("variance %", "0.0"),
            "Fold" => ("fold", "0.0"),
            "Source" => ("source", null),
            "Term" => ("term", null),
            _ => (e.PropertyName, (string?)null),
        };

        e.Column.Header = header;
        if (format is not null && e.Column is DataGridTextColumn text && text.Binding is Binding binding)
            binding.StringFormat = format;
    }

    private void ClearDiffOutput()
    {
        DiffGrid.ItemsSource = null;
        DiffPlot.Reset();
        DiffPlot.Refresh();
    }

    /// <summary>
    /// On the Volcano view, a click near a point opens a per-feature boxplot of that feature's log2
    /// abundance split by the two contrast groups (the explorer's "Feature detail").
    /// </summary>
    private void OnDiffPlotMouseDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
    {
        if (DiffSelectedView() != DiffView.Volcano || _volcanoPoints.Count == 0 || _diffDataset is null)
            return;

        var plt = DiffPlot.Plot;
        var pos = e.GetPosition(DiffPlot);
        var scale = DiffPlot.DisplayScale;
        var cursor = new ScottPlot.Pixel(pos.X * scale, pos.Y * scale);
        var idx = QcPlotChrome.NearestPoint(
            _volcanoPoints.Select(p => plt.GetPixel(p.Loc)).ToList(), cursor);
        if (idx < 0)
            return;

        ShowFeatureDetail(_volcanoPoints[idx].FeatureId);
    }

    /// <summary>Selecting a hit row in the sidebar opens the same per-feature boxplot as a volcano click.</summary>
    private void OnDiffGridSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (DiffGrid.SelectedItem is VolcanoRow vr)
            ShowFeatureDetail(vr.FeatureId);
    }

    private void ShowFeatureDetail(string featureId)
    {
        if (_diffDataset is null)
            return;
        var row = Array.IndexOf(_diffDataset.FeatureIds, featureId);
        if (row < 0)
            return;

        var aVals = new List<double>();
        foreach (var s in _volcanoGroupA)
        {
            var v = _diffDataset.ExprLog2[row, s];
            if (double.IsFinite(v))
                aVals.Add(v);
        }

        var bVals = new List<double>();
        foreach (var s in _volcanoGroupB)
        {
            var v = _diffDataset.ExprLog2[row, s];
            if (double.IsFinite(v))
                bVals.Add(v);
        }

        var label = _diffLabelById.GetValueOrDefault(featureId, featureId);
        _volcanoRowById.TryGetValue(featureId, out var dr);

        if (_featureDetailWindow is null)
        {
            _featureDetailWindow = new FeatureDetailWindow { Owner = this };
            _featureDetailWindow.Closed += (_, _) => _featureDetailWindow = null;
        }

        _featureDetailWindow.ShowFeature(label, featureId, dr?.LogFc ?? double.NaN,
            dr?.AdjPValue ?? double.NaN, _volcanoAName, aVals, _volcanoBName, bVals);
        _featureDetailWindow.Show();
        _featureDetailWindow.Activate();
    }

    private void RenderVolcano(DifferentialResult res)
    {
        DiffPlot.Reset();
        var plt = DiffPlot.Plot;

        var bgX = new List<double>();
        var bgY = new List<double>();
        var sigX = new List<double>();
        var sigY = new List<double>();
        var pThresh = double.NaN;
        _volcanoPoints = new List<(ScottPlot.Coordinates, string)>(res.Rows.Count);
        _volcanoRowById = new Dictionary<string, DifferentialRow>(StringComparer.Ordinal);
        foreach (var r in res.Rows)
        {
            var y = -Math.Log10(Math.Max(r.PValue, 1e-300));
            if (double.IsFinite(r.LogFc) && double.IsFinite(y))
                _volcanoPoints.Add((new ScottPlot.Coordinates(r.LogFc, y), r.FeatureId));
            _volcanoRowById[r.FeatureId] = r;
            if (r.AdjPValue < 0.05 && Math.Abs(r.LogFc) >= 1.0)
            {
                sigX.Add(r.LogFc);
                sigY.Add(y);
                if (double.IsNaN(pThresh) || r.PValue > pThresh)
                    pThresh = r.PValue;
            }
            else
            {
                bgX.Add(r.LogFc);
                bgY.Add(y);
            }
        }

        AddMarkers(plt, bgX, bgY, "#b8c4d0", 6, "not significant");
        AddMarkers(plt, sigX, sigY, "#d62728", 7, "significant");
        plt.Add.VerticalLine(1.0);
        plt.Add.VerticalLine(-1.0);
        if (!double.IsNaN(pThresh))
            plt.Add.HorizontalLine(-Math.Log10(Math.Max(pThresh, 1e-300)));

        plt.ShowLegend();
        plt.XLabel("log2 fold change (B / A)");
        plt.YLabel("-log10 P");
        PlotRenderer.StyleQcPlot(plt);
        DiffPlot.Refresh();
    }

    private void RenderPca(PcaResult pca, string colorColumn)
    {
        DiffPlot.Reset();
        var plt = DiffPlot.Plot;

        if (pca.Scores.GetLength(1) < 2)
        {
            DiffPlot.Refresh();
            return;
        }

        var byId = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < _diffDataset!.SampleIds.Length; i++)
            byId[_diffDataset.SampleIds[i]] = i;
        var labels = _diffDataset.MetadataValues(colorColumn);

        var groups = new Dictionary<string, (List<double> X, List<double> Y)>();
        for (var i = 0; i < pca.SampleIds.Length; i++)
        {
            var g = byId.TryGetValue(pca.SampleIds[i], out var idx) && !string.IsNullOrEmpty(labels[idx])
                ? labels[idx]!
                : "(none)";
            if (!groups.TryGetValue(g, out var lists))
                groups[g] = lists = (new List<double>(), new List<double>());
            lists.X.Add(pca.Scores[i, 0]);
            lists.Y.Add(pca.Scores[i, 1]);
        }

        var ci = 0;
        foreach (var (g, lists) in groups.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            AddMarkers(plt, lists.X, lists.Y, DiffPalette[ci++ % DiffPalette.Length], 11, g);

        // Dock the legend OUTSIDE the data area (like the Streamlit app): a color column can have many
        // groups, and an in-plot legend covers the main sample cluster.
        plt.ShowLegend(ScottPlot.Edge.Right);
        var inv = CultureInfo.InvariantCulture;
        plt.XLabel($"PC1 ({(pca.VarianceRatio[0] * 100).ToString("0.0", inv)}%)");
        plt.YLabel($"PC2 ({(pca.VarianceRatio[1] * 100).ToString("0.0", inv)}%)");
        PlotRenderer.StyleQcPlot(plt);
        DiffPlot.Refresh();
    }

    private void RenderDetection(IReadOnlyList<DetectionRow> rows)
    {
        DiffPlot.Reset();
        var plt = DiffPlot.Plot;

        var bgX = new List<double>();
        var bgY = new List<double>();
        var sigX = new List<double>();
        var sigY = new List<double>();
        foreach (var r in rows)
        {
            var x = r.RateB - r.RateA;
            var y = -Math.Log10(Math.Max(r.P, 1e-300));
            if (r.Q < 0.05)
            {
                sigX.Add(x);
                sigY.Add(y);
            }
            else
            {
                bgX.Add(x);
                bgY.Add(y);
            }
        }

        AddMarkers(plt, bgX, bgY, "#b8c4d0", 6, "q >= 0.05");
        AddMarkers(plt, sigX, sigY, "#2ca02c", 7, "q < 0.05");
        plt.Add.VerticalLine(0.0);
        plt.ShowLegend();
        plt.XLabel("detection rate difference (B - A)");
        plt.YLabel("-log10 P");
        PlotRenderer.StyleQcPlot(plt);
        DiffPlot.Refresh();
    }

    private static void AddMarkers(ScottPlot.Plot plt, List<double> xs, List<double> ys, string hex,
        float size, string? legend)
    {
        if (xs.Count == 0)
            return;
        var m = plt.Add.Markers(xs.ToArray(), ys.ToArray());
        m.Color = ScottPlot.Color.FromHex(hex);
        m.MarkerSize = size;
        if (legend is not null)
            m.LegendText = legend;
    }
}
