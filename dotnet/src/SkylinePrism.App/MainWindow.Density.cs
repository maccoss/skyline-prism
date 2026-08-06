using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Microsoft.Win32;
using ScottPlot;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.Visualization;

namespace SkylinePrism.App;

/// <summary>
/// The "Spectrum density" tab: how many peptide precursors were detected in each DIA spectrum of a run,
/// as an (isolation window x retention time) map. Ported from the same plot in Skyline-Cadenza, but fed
/// by the merged PRISM report (Precursor Mz + peak Start/End Time) instead of a DIA-NN report.
///
/// Reads merged_data.parquet from the output directory, so it works for a run that just finished AND for
/// any previous run the output box is pointed at - no Skyline connection needed.
/// </summary>
public partial class MainWindow
{
    private const string MergedParquetName = "merged_data.parquet";

    private bool _densityLoaded;               // sample list matches the current output directory
    private bool _suppressDensityRender;       // set while populating combos
    private string? _densityParquetPath;
    private PrecursorDensity.Columns? _densityColumns;
    private List<string> _densitySampleIds = new();
    private List<DetectedPrecursor>? _densityPrecursors; // cache for the selected run (rebinning is free)
    private PrecursorDensityMap? _densityMap;
    private string _densityQValueApplied = "0.01"; // matches DensityQValueBox's initial text
    private int _densityRequest;                   // newest query wins if the user clicks ahead of it
    private IsolationSchemeCatalog? _densitySchemes;
    private List<IsolationScheme?> _densitySchemeChoices = new(); // parallel to DensitySchemeCombo; null = uniform

    /// <summary>Label of the "no real windows available" entry - the only approximate option.</summary>
    private const string UniformSchemeItem = "(uniform bins - approximate)";

    /// <summary>Opens a file dialog for a Thermo inclusion list (a scheduled PRM/MTM method).</summary>
    private const string LoadInclusionItem = "Load inclusion list (PRM/MTM)...";

    /// <summary>Inclusion lists the user has loaded this session, offered alongside the saved schemes.</summary>
    private readonly List<IsolationScheme> _densityLoadedSchemes = new();

    /// <summary>Bin width used by the approximate fallback when the box is empty or unparseable.</summary>
    private const double UniformBinFallbackTh = 8.0;

    /// <summary>Drop the cached sample list so the tab reloads next time it is shown.</summary>
    private void InvalidateDensity()
    {
        _densityLoaded = false;
        _densityPrecursors = null;
        _densityMap = null;
    }

    private async void OnMainTabChanged(object sender, SelectionChangedEventArgs e)
    {
        // A TabControl also receives the SelectionChanged of every ComboBox inside it, so only act on
        // the tab strip's own event.
        if (!ReferenceEquals(e.Source, MainTabs))
            return;
        if (ReferenceEquals(MainTabs.SelectedItem, DensityTab) && !_densityLoaded)
            await LoadDensitySamplesAsync();
        else if (ReferenceEquals(MainTabs.SelectedItem, DynamicRangeTab) && !_rangeLoaded)
            await LoadDynamicRangeAsync();
    }

    private async void OnDensityReload(object sender, RoutedEventArgs e)
    {
        InvalidateDensity();
        await LoadDensitySamplesAsync();
    }

    /// <summary>
    /// Find merged_data.parquet under the current output directory and list its runs. Both the schema
    /// probe and the DISTINCT scan touch the (possibly very large) parquet, so they run off the UI thread.
    /// </summary>
    private async Task LoadDensitySamplesAsync()
    {
        var outputDir = OutputDirBox.Text?.Trim();
        var path = string.IsNullOrWhiteSpace(outputDir)
            ? null
            : Path.Combine(outputDir, MergedParquetName);

        if (path is null || !File.Exists(path))
        {
            _densityLoaded = false;
            SetDensitySamples(new List<string>());
            ShowDensityMessage($"No {MergedParquetName} in the output directory. Run PRISM, or point the "
                + "output directory at a previous run.");
            return;
        }

        DensityStatusText.Text = "Reading " + MergedParquetName + "...";
        try
        {
            var schemePath = Path.Combine(outputDir!, IsolationSchemeCatalog.FileName);
            var (columns, samples, schemes) = await Task.Run(() =>
            {
                var cols = PrecursorDensity.Resolve(ParquetTable.ReadColumnNames(path).ToHashSet());
                var ids = cols is null
                    ? new List<string>()
                    : MergedParquetReader.GetSortedSamples(path, cols.Sample);
                return (cols, ids, IsolationSchemeCatalog.Load(schemePath));
            });

            _densityParquetPath = path;
            _densityColumns = columns;
            _densitySchemes = schemes;
            _densityLoaded = true;

            if (columns is null)
            {
                SetDensitySamples(new List<string>());
                ShowDensityMessage("This report has no precursor m/z / peak boundary columns - re-export it "
                    + "with the PRISM report definition.");
                return;
            }
            DensityQValueBox.IsEnabled = columns.DetectionQValue is not null;

            SetDensitySamples(samples);
            if (samples.Count == 0)
            {
                ShowDensityMessage("No runs found in " + MergedParquetName + ".");
                return;
            }
            PopulateSchemeCombo();
            await RenderDensityAsync();
        }
        catch (Exception ex)
        {
            _densityLoaded = false;
            App.WriteLog("Spectrum density load failed: " + ex);
            ShowDensityMessage("Could not read " + MergedParquetName + ": " + ex.Message);
        }
    }

    // Fills the run combo, showing the bare replicate name when every run shares one batch suffix.
    private void SetDensitySamples(List<string> sampleIds)
    {
        _suppressDensityRender = true;
        try
        {
            _densitySampleIds = sampleIds;
            _densityPrecursors = null;
            _densityMap = null;
            DensitySampleCombo.Items.Clear();
            foreach (var name in StripSharedBatchSuffix(sampleIds, sampleIds))
                DensitySampleCombo.Items.Add(name);
            DensitySampleCombo.SelectedIndex = sampleIds.Count > 0 ? 0 : -1;
        }
        finally
        {
            _suppressDensityRender = false;
        }
    }

    private async void OnDensitySampleChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_suppressDensityRender)
            return;
        _densityPrecursors = null; // different run -> re-query
        PopulateSchemeCombo();     // a different batch may declare a different scheme
        await RenderDensityAsync();
    }

    /// <summary>
    /// Fill the isolation-scheme picker for the selected run's batch. When that batch's document declares
    /// a scheme WITH windows, it is selected and the picker is locked - the document is authoritative and
    /// there is nothing to choose. Otherwise (the usual "Results only" case) the user picks which of their
    /// saved Skyline schemes the data was acquired with, because Skyline stores those windows only inside
    /// the raw files.
    /// </summary>
    private void PopulateSchemeCombo()
    {
        _suppressDensityRender = true;
        try
        {
            var previous = DensitySchemeCombo.SelectedItem as string;
            var batch = BatchOfSelectedRun();
            var documentScheme = batch is null ? null : _densitySchemes?.DocumentSchemeFor(batch);
            var usable = _densitySchemes?.UsableSchemes ?? Array.Empty<IsolationScheme>();

            DensitySchemeCombo.Items.Clear();
            _densitySchemeChoices = new List<IsolationScheme?>();

            if (documentScheme is not null)
            {
                DensitySchemeCombo.Items.Add($"{documentScheme.Name} (from document)");
                _densitySchemeChoices.Add(documentScheme);
            }
            foreach (var scheme in _densityLoadedSchemes.Concat(usable)
                         .Where(s => documentScheme is null
                             || !string.Equals(s.Name, documentScheme.Name, StringComparison.OrdinalIgnoreCase)))
            {
                DensitySchemeCombo.Items.Add(scheme.Name + (scheme.IsScheduled ? " (scheduled)" : ""));
                _densitySchemeChoices.Add(scheme);
            }
            DensitySchemeCombo.Items.Add(UniformSchemeItem);
            _densitySchemeChoices.Add(null);
            // Always offered: a scheduled PRM/MTM method's windows exist only in the inclusion list that
            // was loaded onto the instrument - Skyline cannot import them, because there is no repeating
            // cycle to find.
            DensitySchemeCombo.Items.Add(LoadInclusionItem);
            _densitySchemeChoices.Add(null);

            // The document's scheme wins; otherwise keep the user's previous pick across runs (they are
            // usually all the same acquisition), else fall back to the approximate grid.
            var index = 0;
            if (documentScheme is null)
            {
                var keep = previous is null ? -1 : DensitySchemeCombo.Items.IndexOf(previous);
                index = keep >= 0 ? keep : DensitySchemeCombo.Items.Count - 1;
            }
            DensitySchemeCombo.SelectedIndex = index;
            DensitySchemeCombo.IsEnabled = documentScheme is null && DensitySchemeCombo.Items.Count > 1;
            UpdateSchemeControls();
        }
        finally
        {
            _suppressDensityRender = false;
        }
    }

    /// <summary>The uniform bin width only applies to the approximate fallback, so only show it there.</summary>
    private void UpdateSchemeControls() =>
        DensityMzBinBox.Visibility = SelectedIsolationScheme() is null ? Visibility.Visible : Visibility.Collapsed;

    private IsolationScheme? SelectedIsolationScheme()
    {
        var i = DensitySchemeCombo.SelectedIndex;
        return i >= 0 && i < _densitySchemeChoices.Count ? _densitySchemeChoices[i] : null;
    }

    // Sample IDs are "<replicate>__@__<batch>"; the batch is what a document scheme is recorded against.
    private string? BatchOfSelectedRun()
    {
        var i = DensitySampleCombo.SelectedIndex;
        if (i < 0 || i >= _densitySampleIds.Count)
            return null;
        const string sep = "__@__";
        var id = _densitySampleIds[i];
        var at = id.IndexOf(sep, StringComparison.Ordinal);
        return at >= 0 ? id[(at + sep.Length)..] : null;
    }

    private void OnDensitySchemeChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_suppressDensityRender)
            return;
        if (ComboText(DensitySchemeCombo, "") == LoadInclusionItem)
        {
            LoadInclusionList();
            return;
        }
        UpdateSchemeControls();
        RebinAndDraw(); // the scheme only changes the binning, not the query
    }

    /// <summary>
    /// Load a scheduled PRM/MTM method's inclusion list (the CSV that went to the instrument) and use its
    /// slots as the map's rows. Each row is an m/z window crossed with the interval it fires in, so cells
    /// outside a slot's schedule are drawn as "not acquired" rather than as an empty spectrum.
    /// </summary>
    private void LoadInclusionList()
    {
        var dialog = new OpenFileDialog
        {
            Title = "Open a Thermo inclusion list (scheduled PRM / MTM method)",
            Filter = "Inclusion list (*.csv;*.tsv;*.txt)|*.csv;*.tsv;*.txt|All files (*.*)|*.*",
        };
        if (dialog.ShowDialog(this) != true)
        {
            RestoreSchemeSelection();
            return;
        }

        IsolationScheme scheme;
        try
        {
            scheme = ThermoInclusionList.Load(dialog.FileName);
        }
        catch (Exception ex)
        {
            App.WriteLog("Inclusion list load failed: " + ex);
            MessageBox.Show(this, ex.Message, "Could not read the inclusion list",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            RestoreSchemeSelection();
            return;
        }

        _densityLoadedSchemes.RemoveAll(
            s => string.Equals(s.Name, scheme.Name, StringComparison.OrdinalIgnoreCase));
        _densityLoadedSchemes.Insert(0, scheme);
        Log($"Inclusion list '{scheme.Name}': {scheme.Describe()}.");

        // Keep it for next time: written into the run's catalog so reopening this output directory offers
        // it again without re-browsing.
        PersistLoadedScheme(scheme);

        PopulateSchemeCombo();
        _suppressDensityRender = true;
        DensitySchemeCombo.SelectedIndex = _densitySchemeChoices.FindIndex(s => ReferenceEquals(s, scheme));
        _suppressDensityRender = false;
        UpdateSchemeControls();
        RebinAndDraw();
    }

    // Put the picker back on a real scheme after a cancelled or failed load, so it never sits on the
    // "Load..." action as though that were the current binning.
    private void RestoreSchemeSelection()
    {
        _suppressDensityRender = true;
        var fallback = _densitySchemeChoices.FindIndex(s => s is not null);
        DensitySchemeCombo.SelectedIndex = fallback >= 0
            ? fallback
            : Math.Max(0, DensitySchemeCombo.Items.IndexOf(UniformSchemeItem));
        _suppressDensityRender = false;
        UpdateSchemeControls();
    }

    private void PersistLoadedScheme(IsolationScheme scheme)
    {
        var outputDir = OutputDirBox.Text?.Trim();
        if (string.IsNullOrWhiteSpace(outputDir) || !Directory.Exists(outputDir))
            return;
        try
        {
            var path = Path.Combine(outputDir, IsolationSchemeCatalog.FileName);
            var catalog = IsolationSchemeCatalog.Load(path) ?? new IsolationSchemeCatalog();
            catalog.AddLibraryScheme(scheme);
            catalog.Save(path);
            _densitySchemes = catalog;
        }
        catch (Exception ex)
        {
            App.WriteLog("Could not save the inclusion list to the run catalog: " + ex);
        }
    }

    private async void OnDensityQValueChanged(object sender, RoutedEventArgs e)
    {
        // LostFocus fires on every tab-out, so re-query only when the cutoff actually changed.
        var text = DensityQValueBox.Text?.Trim() ?? "";
        if (_suppressDensityRender || text == _densityQValueApplied)
            return;
        _densityQValueApplied = text;
        _densityPrecursors = null; // the cutoff is applied in the query
        await RenderDensityAsync();
    }

    private void OnDensityQValueKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter)
            OnDensityQValueChanged(sender, e);
    }

    // Bin sizes only re-bin the precursors already in memory, so no re-query.
    private void OnDensityBinChanged(object sender, RoutedEventArgs e)
    {
        if (!_suppressDensityRender)
            RebinAndDraw();
    }

    private void OnDensityBinKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter)
            OnDensityBinChanged(sender, e);
    }

    private void OnDensityColormapChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_suppressDensityRender && _densityMap is not null)
            DrawDensity();
    }

    /// <summary>Query the selected run's precursors (off the UI thread), then bin and draw them.</summary>
    private async Task RenderDensityAsync()
    {
        if (_densityParquetPath is null || _densityColumns is null)
            return;
        var index = DensitySampleCombo.SelectedIndex;
        if (index < 0 || index >= _densitySampleIds.Count)
            return;

        var path = _densityParquetPath;
        var cols = _densityColumns;
        var sample = _densitySampleIds[index];
        var qCutoff = DensityQValue();

        // Scanning a large merged report takes a moment, so a user clicking down the run list can have
        // several queries in flight. Only the newest one may touch the plot.
        var request = ++_densityRequest;
        DensityStatusText.Text = "Loading " + DensitySampleCombo.SelectedItem + "...";
        try
        {
            var precursors = await Task.Run(() => PrecursorDensity.Load(path, cols, sample, qCutoff));
            if (request != _densityRequest)
                return;
            _densityPrecursors = precursors;
            RebinAndDraw();
        }
        catch (Exception ex)
        {
            App.WriteLog("Spectrum density query failed: " + ex);
            if (request == _densityRequest)
                ShowDensityMessage("Could not build the map: " + ex.Message);
        }
    }

    private void RebinAndDraw()
    {
        if (_densityPrecursors is null)
            return;
        if (_densityPrecursors.Count == 0)
        {
            ShowDensityMessage("No detected precursors in this run"
                + (DensityQValue() is { } q ? $" at q <= {q:0.####}." : "."));
            return;
        }
        var rtBin = DensityBin(DensityRtBinBox, PrecursorDensity.DefaultRtBinMin);
        var scheme = SelectedIsolationScheme();
        _densityMap = scheme is not null
            ? PrecursorDensity.Bin(_densityPrecursors, scheme, rtBin)
            : PrecursorDensity.Bin(
                _densityPrecursors, DensityBin(DensityMzBinBox, UniformBinFallbackTh), rtBin);
        DrawDensity();
    }

    private void DrawDensity()
    {
        var map = _densityMap;
        if (map is null || map.IsEmpty)
            return;

        // The ColorBar attaches as an axis panel and survives Plot.Clear(), so start from a fresh Plot
        // on every draw rather than stacking one colour bar per render. No title: the run is named in the
        // drop-down and the colour bar is labelled, so a title would only repeat them.
        DensityPlot.Reset();
        PlotRenderer.DrawPrecursorDensity(
            DensityPlot.Plot, map, DensityColormap(ComboText(DensityColormapCombo, "Viridis")));
        DensityPlot.Refresh();

        // Say which windows the map is drawn on, and - crucially - flag precursors that fell outside them,
        // which is what a wrong scheme looks like.
        var outside = map.PrecursorsOutsideRows;
        var total = _densityPrecursors?.Count ?? 0;
        // "Busiest spectrum" only means co-fragmentation crowding for DIA. A targeted method isolates one
        // (PRM) or a few (multiplexed) precursors per spectrum by design, and its windows are RT-scheduled
        // rather than a repeating cycle, so say what the map is instead of implying the DIA reading.
        var batch = BatchOfSelectedRun();
        var nonDia = batch is not null && _densitySchemes is not null && _densitySchemes.IsNonDia(batch)
            ? _densitySchemes.AcquisitionFor(batch)
            : null;
        DensityStatusText.Text =
            $"{total:N0} precursors; busiest {(nonDia is null ? "spectrum" : "window")} {map.MaxCount:N0}; "
            + $"{map.RowSource}; {map.MzBins:N0} rows x {map.RtBins:N0} RT bins of {map.RtBinMin:0.###} min"
            + (nonDia is not null
                ? $"; NOTE: {nonDia} acquisition - targets are isolated individually and scheduled by RT, "
                  + "so a cell is a target x time bin, not a spectrum's co-fragmentation load"
                : "")
            + (outside > 0 && total > 0
                ? $"; WARNING: {outside:N0} precursors ({100.0 * outside / total:0.#}%) fall outside every "
                  + "window - is this the scheme the data was acquired with?"
                : "");
    }

    /// <summary>
    /// Read out the cell under the cursor. The colour bar gives the scale; the question this plot is
    /// asked ("how many precursors was THAT spectrum carrying") wants the number itself.
    /// </summary>
    private void OnDensityPlotMouseMove(object sender, MouseEventArgs e)
    {
        var map = _densityMap;
        if (map is null || map.IsEmpty)
            return;

        var pos = e.GetPosition(DensityPlot);
        var scale = DensityPlot.DisplayScale;
        var c = DensityPlot.Plot.GetCoordinates(new Pixel(pos.X * scale, pos.Y * scale));
        var row = map.RowAt(c.Y);
        var col = (int)((c.X - map.RtLow) / map.RtBinMin);
        DensityHoverText.Text = row < 0 || col < 0 || col >= map.RtBins
            ? ""
            : $"m/z {map.Rows[row].Start:0.#}-{map.Rows[row].End:0.#} "
              + $"at {map.RtLow + col * map.RtBinMin:0.##} min: {map.Counts[row, col]:N0} precursors";
    }

    // Message in place of a plot (an empty Plot with a title, as the QC tab does for its empty states).
    // Clears the map too, so the hover readout cannot report cells that are no longer drawn.
    private void ShowDensityMessage(string message)
    {
        _densityMap = null;
        DensityStatusText.Text = message;
        DensityHoverText.Text = "";
        DensityPlot.Reset();
        DensityPlot.Plot.Title(message);
        PlotRenderer.StyleQcPlot(DensityPlot.Plot);
        DensityPlot.Refresh();
    }

    /// <summary>Detection q-value cutoff, or null when the box is blank / unparseable (count every peak).</summary>
    private double? DensityQValue()
    {
        if (_densityColumns?.DetectionQValue is null)
            return null;
        var text = DensityQValueBox.Text?.Trim();
        if (string.IsNullOrEmpty(text))
            return null;
        return double.TryParse(text, out var q) && q >= 0 ? q : null;
    }

    // Bin size from a text box, falling back to the default (and rewriting the box) on bad input, so the
    // control never disagrees with the grid actually drawn.
    private double DensityBin(TextBox box, double fallback)
    {
        if (double.TryParse(box.Text?.Trim(), out var v) && v > 0)
            return v;
        box.Text = fallback.ToString("0.###");
        return fallback;
    }

    // Viridis matches Cadenza. Turbo is the high-contrast option for picking out fine structure; Magma,
    // Inferno and Plasma are the other perceptually uniform maps (dark-background friendly); Thermal is
    // the cmocean equivalent; Grayscale is for print.
    private static IColormap DensityColormap(string name) => name switch
    {
        "Turbo" => new ScottPlot.Colormaps.Turbo(),
        "Magma" => new ScottPlot.Colormaps.Magma(),
        "Inferno" => new ScottPlot.Colormaps.Inferno(),
        "Plasma" => new ScottPlot.Colormaps.Plasma(),
        "Thermal" => new ScottPlot.Colormaps.Thermal(),
        "Grayscale" => new ScottPlot.Colormaps.GrayscaleReversed(),
        _ => new ScottPlot.Colormaps.Viridis(),
    };
}
