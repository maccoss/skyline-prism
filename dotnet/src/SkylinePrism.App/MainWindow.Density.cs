using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using ScottPlot;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using SkylinePrism.Core.Visualization;

namespace SkylinePrism.App;

/// <summary>
/// The "Spectrum density" tab: how many peptide precursors were detected in each DIA spectrum of a run,
/// as an (isolation window x retention time) map. Ported from the same plot in Skyline-Cadenza, but fed
/// by the merged PRISM report (Precursor Mz + peak Start/End Time) instead of a DIA-NN report.
///
/// Reads the merged dataset from the output directory, so it works for a run that just finished AND for
/// any previous run the output box is pointed at - no Skyline connection needed. Accepts both the
/// partitioned merged_data/ directory and the single merged_data.parquet older releases wrote.
/// </summary>
public partial class MainWindow
{
    private const string MergedName = "merged_data";
    private const string LegacyMergedName = "merged_data.parquet";

    private bool _densityLoaded;               // sample list matches the current output directory
    private bool _suppressDensityRender;       // set while populating combos
    private MergedDataset? _densityDataset;
    private PrecursorDensity.Columns? _densityColumns;
    private List<string> _densitySampleIds = new();
    private List<DetectedPrecursor>? _densityPrecursors; // cache for the selected run (rebinning is free)
    private PrecursorDensityMap? _densityMap;
    // Both summaries walk the whole grid, and the hover readout wants them on every mouse move, so they
    // are computed once per map rather than per motion event. Always set through SetDensityMap.
    private int[]? _densityHistogram;
    private IReadOnlyList<(double TimeMin, double Mean, double Min, double Max)>? _densityLoad;
    private string _densityQValueApplied = "0.01"; // matches DensityQValueBox's initial text
    private int _densityRequest;                   // newest query wins if the user clicks ahead of it
    private IsolationSchemeCatalog? _densitySchemes;
    private List<IsolationScheme?> _densitySchemeChoices = new(); // parallel to DensitySchemeCombo; null = uniform
    // The picker only ever suggests. Once the user has named a scheme, nothing chooses one for them
    // again - a default that reasserted itself would silently re-bin the map they were reading.
    private bool _densitySchemeChosen;
    private string? _densityMeasuredFor;   // output directory a data-file read has already been tried for

    /// <summary>Label of the "no real windows available" entry - the only approximate option.</summary>
    private const string UniformSchemeItem = "(uniform bins - approximate)";

    /// <summary>Bin width used by the approximate fallback when the box is empty or unparseable.</summary>
    private const double UniformBinFallbackTh = 8.0;

    /// <summary>Drop the cached sample list so the tab reloads next time it is shown.</summary>
    private void InvalidateDensity()
    {
        _densityLoaded = false;
        _densityPrecursors = null;
        // A different output directory is a different acquisition until proved otherwise, so both the
        // user's pick and the "already looked" mark belong to the directory that was showing.
        _densitySchemeChosen = false;
        _densityMeasuredFor = null;
        SetDensityMap(null);
    }

    /// <summary>
    /// The map and its two summaries move together - a summary left over from the previous map would be
    /// read out under the cursor of the new one.
    /// </summary>
    private void SetDensityMap(PrecursorDensityMap? map)
    {
        _densityMap = map;
        _densityHistogram = map?.PrecursorsPerSpectrumHistogram();
        _densityLoad = map?.LoadOverTime();
    }

    private async void OnMainTabChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            // A TabControl receives the SelectionChanged of everything inside it - every ComboBox, and
            // now the Analysis tab strip and the visualization nav rail as well - so only act on the
            // outer strip's own event. The nav rail has its own handler below.
            if (!ReferenceEquals(e.Source, MainTabs))
                return;
            await ShowSelectedVizPaneAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnMainTabChanged), ex);
        }
    }

    private async void OnVizNavChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            // Nothing below the nav rail exists yet during InitializeComponent, and a Selector raises
            // SelectionChanged from EndInit if its selection is set in XAML. The rail deliberately sets
            // none - the constructor does it - but the guard is what makes that safe to change later.
            // See XamlInitializationOrderTests.
            if (!IsInitialized)
                return;
            await ShowSelectedVizPaneAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnVizNavChanged), ex);
        }
    }

    /// <summary>Bring one of the Analysis panes to the front, selecting the Analysis group first.</summary>
    private void ShowAnalysis(AnalysisPane pane)
    {
        MainTabs.SelectedItem = AnalysisTab;
        AnalysisNav.SelectedIndex = (int)pane;
        ShowAnalysisPane(pane);
    }

    /// <summary>
    /// Which Analysis pane is visible. One <see cref="UIElement.Visibility"/> flip per pane rather
    /// than a TabControl, matching the Visualization rail - and like it, each pane keeps its state
    /// (the grid's rows, the settings, the log's scroll position) while another is shown.
    /// </summary>
    private void ShowAnalysisPane(AnalysisPane pane)
    {
        InputsPane.Visibility = pane == AnalysisPane.Inputs ? Visibility.Visible : Visibility.Collapsed;
        SettingsPane.Visibility = pane == AnalysisPane.Settings ? Visibility.Visible : Visibility.Collapsed;
        LogBox.Visibility = pane == AnalysisPane.Log ? Visibility.Visible : Visibility.Collapsed;
    }

    private void OnAnalysisNavChanged(object sender, SelectionChangedEventArgs e)
    {
        // Guarded because Selector raises this from EndInit while the window is still being built,
        // before the panes it wants to touch exist. Same reason the rail's index is set in code.
        if (!IsInitialized || AnalysisNav.SelectedIndex < 0)
            return;
        try
        {
            ShowAnalysisPane((AnalysisPane)AnalysisNav.SelectedIndex);
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnAnalysisNavChanged), ex);
        }
    }

    /// <summary>
    /// Bring one of the visualization panes to the front. The rail is set BEFORE the outer tab so the
    /// last event raised is the one that finds both in their final state and does the pane's load;
    /// the other order works too, but loads against a selection that is about to change.
    /// </summary>
    private void ShowVisualization(VizPane pane)
    {
        VizNav.SelectedIndex = (int)pane;
        MainTabs.SelectedItem = VisualizationTab;
    }

    /// <summary>
    /// Show the pane the nav rail names and start whatever it needs. Driven by both the outer tab and
    /// the rail, because a pane is only really on screen when both agree - see <see cref="VizNavigation"/>.
    /// </summary>
    private async Task ShowSelectedVizPaneAsync()
    {
        var pane = VizNavigation.Current(
            ReferenceEquals(MainTabs.SelectedItem, VisualizationTab), VizNav.SelectedIndex);

        // Switched by visibility rather than by swapping content, so each pane keeps what the user left
        // on it - the zoom on a plot, a ticked replicate set, the matrices already read off disk.
        QcPane.Visibility = pane == VizPane.Qc ? Visibility.Visible : Visibility.Collapsed;
        DensityPane.Visibility = pane == VizPane.Density ? Visibility.Visible : Visibility.Collapsed;
        RangePane.Visibility = pane == VizPane.DynamicRange ? Visibility.Visible : Visibility.Collapsed;
        IonPane.Visibility = pane == VizPane.IonAccounting ? Visibility.Visible : Visibility.Collapsed;

        SetRangeFollowActive(VizNavigation.ShouldFollowSkylineSelection(pane));

        // Re-checked on every pane change rather than once: the user can point the output box at
        // another directory at any time, and whether that one has a measured denominator is what
        // decides if the entry exists at all.
        UpdateIonNavVisibility();

        if (pane == VizPane.IonAccounting)
            await LoadIonAccountingAsync();
        else if (pane == VizPane.Density && !_densityLoaded)
            await LoadDensitySamplesAsync();
        else if (pane == VizPane.DynamicRange)
        {
            // Marked shown before loading, so a load that FAILS still leaves the level combo live -
            // switching level is how a user gets out of an error, and it used to be inert afterwards.
            _rangeTabShown = true;
            if (!_rangeLoaded)
                await LoadDynamicRangeAsync();
        }
    }

    private async void OnDensityReload(object sender, RoutedEventArgs e)
    {
        try
        {
            InvalidateDensity();
            await LoadDensitySamplesAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDensityReload), ex);
        }
    }

    /// <summary>
    /// Find the merged dataset under the current output directory and list its runs.
    /// </summary>
    /// <remarks>
    /// <b>Every file system call is inside the one background read.</b> Locating <c>merged_data</c>
    /// probes for two candidate paths and used to do it on the UI thread, before the read it belongs
    /// in - which against a share that is slow or disconnected is the window stopped for the SMB
    /// timeout with nothing yet on screen to say why. Anything added here that touches the disk goes
    /// inside the <c>Task.Run</c>.
    /// </remarks>
    private async Task LoadDensitySamplesAsync()
    {
        var outputDir = OutputDirBox.Text?.Trim();
        DensityStatusText.Text = "Reading " + MergedName + "...";
        try
        {
            var schemePath = string.IsNullOrWhiteSpace(outputDir)
                ? null
                : Path.Combine(outputDir, IsolationSchemeCatalog.FileName);
            var (dataset, columns, samples, schemes) = await Task.Run(() =>
            {
                // Either layout: the partitioned directory this release writes, or the single file
                // older ones did.
                var root = MergedDataset.Locate(outputDir);
                if (root is null || schemePath is null)
                {
                    return ((MergedDataset?)null, (PrecursorDensity.Columns?)null,
                            new List<string>(), (IsolationSchemeCatalog?)null);
                }
                var ds = MergedDataset.Open(root);
                var cols = PrecursorDensity.Resolve(
                    ParquetTable.ReadColumnNames(ds.RepresentativeFile()).ToHashSet());
                var ids = cols is null
                    ? new List<string>()
                    : MergedParquetReader.GetSortedSamples(ds, cols.Sample);
                return ((MergedDataset?)ds, cols, ids, IsolationSchemeCatalog.Load(schemePath));
            });

            if (dataset is null)
            {
                _densityLoaded = false;
                SetDensitySamples(new List<string>());
                ShowDensityMessage($"No {MergedName} in the output directory. Run PRISM, or point the "
                    + "output directory at a previous run.");
                return;
            }

            _densityDataset = dataset;
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
                ShowDensityMessage("No runs found in " + MergedName + ".");
                return;
            }
            PopulateSchemeCombo();
            await RenderDensityAsync();
            await MeasureIsolationSchemeAsync(outputDir!);
        }
        catch (Exception ex)
        {
            _densityLoaded = false;
            App.WriteLog("Spectrum density load failed: " + ex);
            ShowDensityMessage("Could not read " + MergedName + ": " + ex.Message);
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
            SetDensityMap(null);
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
        try
        {
            if (_suppressDensityRender)
                return;
            _densityPrecursors = null; // different run -> re-query
            PopulateSchemeCombo();     // a different batch may declare a different scheme
            await RenderDensityAsync();
            // The batch just switched to may be one whose document declares no windows, which is the
            // case worth reading the data for. Cheap after the first attempt - it is marked.
            await MeasureIsolationSchemeAsync(OutputDirBox.Text?.Trim() ?? "");
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDensitySampleChanged), ex);
        }
    }

    /// <summary>
    /// Fill the isolation-scheme picker for the selected run's batch. When that batch's document declares
    /// a scheme WITH windows, it is selected and the picker is locked - the document is authoritative and
    /// there is nothing to choose. Otherwise (the usual "Results only" case, where Skyline keeps the
    /// windows only inside the raw files) the user picks from what this run actually learned: another
    /// batch's document scheme, a scheme reloaded from the run's own catalog, a built-in cycle, or
    /// labeled uniform bins.
    /// <para>
    /// Skyline's saved isolation-scheme list is deliberately NOT among them - see
    /// <see cref="IsolationSchemeCatalog.Library"/> for why offering those generic templates invites a
    /// map that looks plausible and is wrong.
    /// </para>
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
            var offered = usable
                .Where(s => documentScheme is null || s.LayoutKey != documentScheme.LayoutKey)
                .ToList();
            // Two plates can name their schemes the same thing and mean different windows, and both are
            // offered (UsableSchemes deduplicates on the layout, not the name). Say which is which -
            // two identical-looking entries would be a coin toss.
            var ambiguous = offered
                .GroupBy(s => s.Name, StringComparer.OrdinalIgnoreCase)
                .Where(g => g.Count() > 1)
                .Select(g => g.Key)
                .ToHashSet(StringComparer.OrdinalIgnoreCase);
            foreach (var scheme in offered)
            {
                DensitySchemeCombo.Items.Add(scheme.Name
                    + (ambiguous.Contains(scheme.Name) ? $" ({scheme.Windows.Count} windows)" : "")
                    + (scheme.IsScheduled ? " (scheduled)" : "")
                    // Where a scheme came from is the whole basis for trusting it: windows read out of
                    // the acquisition are what it really ran, where every other entry is a guess that
                    // happens to be available. Say which is which in the list itself.
                    + (_densitySchemes?.IsMeasured(scheme) == true ? " (from the data files)" : ""));
                _densitySchemeChoices.Add(scheme);
            }
            // The built-in schemes: modern narrow-window cycles. Enumerated from IsolationScheme.BuiltIns
            // rather than named here, so adding one there is the only edit needed. Each is offered unless
            // the document already declares a scheme by that name, so the fallback is a realistic grid
            // rather than uniform bins or a 25 Th SWATH template from a different era.
            //
            // The first is marked "(default)" only when it really is the one preselected. A scheme read
            // from the data files outranks it (see DensitySchemeDefault), and a built-in still calling
            // itself the default while something else is selected reads as a bug in the picker.
            var builtInIsDefault = !offered.Any(s => _densitySchemes?.IsMeasured(s) == true);
            for (var i = 0; i < IsolationScheme.BuiltIns.Count; i++)
            {
                var builtIn = IsolationScheme.BuiltIns[i];
                if (documentScheme is not null && documentScheme.Name.Equals(
                        builtIn.Name, StringComparison.OrdinalIgnoreCase))
                    continue;
                DensitySchemeCombo.Items.Add(
                    builtIn.Name + (i == 0 && builtInIsDefault ? " (default)" : ""));
                _densitySchemeChoices.Add(builtIn);
            }

            DensitySchemeCombo.Items.Add(UniformSchemeItem);
            _densitySchemeChoices.Add(null);

            // Which one to start on is DensitySchemeDefault's rule, which is where the order is
            // documented and tested. The built-in is found by the marker rather than by name, so
            // reordering IsolationScheme.BuiltIns changes the default with no edit here.
            DensitySchemeCombo.SelectedIndex = DensitySchemeDefault.Choose(
                documentIndex: documentScheme is null ? -1 : 0,
                keptIndex: previous is null ? -1 : DensitySchemeCombo.Items.IndexOf(previous),
                measuredIndex: _densitySchemeChoices.FindIndex(
                    s => s is not null && _densitySchemes?.IsMeasured(s) == true),
                builtInIndex: _densitySchemeChoices.FindIndex(
                    s => s is not null && ReferenceEquals(s, IsolationScheme.BuiltIns[0])),
                fallbackIndex: DensitySchemeCombo.Items.Count - 1);
            DensitySchemeCombo.IsEnabled = documentScheme is null && DensitySchemeCombo.Items.Count > 1;
            UpdateSchemeControls();
        }
        finally
        {
            _suppressDensityRender = false;
        }
    }

    /// <summary>
    /// The uniform bin width only applies to the approximate fallback, so only show it there. It stays
    /// available on all three views: the fallback's bins are what a "spectrum" means for this map, so
    /// changing them changes the histogram and the load curve as much as the heatmap.
    /// </summary>
    private void UpdateSchemeControls()
    {
        var scheme = SelectedIsolationScheme();
        DensityMzBinBox.Visibility = scheme is null ? Visibility.Visible : Visibility.Collapsed;

        // The picker shows names, and a name is nominal - "Astral 3 Th, 400-900 m/z" really runs to
        // 901.66, because 167 windows of ~3.0014 Th from a forbidden-zone edge do not end on a round
        // number (Skyline's own "SWATH (25 m/z)" is the same kind of label). Put the true extents where
        // the choice is made, so the m/z axis never disagrees with the name for no visible reason.
        DensitySchemeCombo.ToolTip = scheme is null
            ? "No real isolation windows available - the map is binned on a uniform m/z grid, which is "
              + "approximate: a cell is not one spectrum."
            : scheme.Describe() + WhereFrom(scheme);
    }

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
        _densitySchemeChosen = true;
        UpdateSchemeControls();
        RebinAndDraw(); // the scheme only changes the binning, not the query
    }

    private async void OnDensityQValueChanged(object sender, RoutedEventArgs e)
    {
        try
        {
            // LostFocus fires on every tab-out, so re-query only when the cutoff actually changed.
            var text = DensityQValueBox.Text?.Trim() ?? "";
            if (_suppressDensityRender || text == _densityQValueApplied)
                return;
            _densityQValueApplied = text;
            _densityPrecursors = null; // the cutoff is applied in the query
            await RenderDensityAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDensityQValueChanged), ex);
        }
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
        // IsInitialized: XAML preselects this combo too, so this runs once during InitializeComponent.
        // It survived only because _densityMap happens to be null then - an accident, not a guard.
        if (!IsInitialized || _suppressDensityRender || _densityMap is null)
            return;
        DrawDensity();
    }

    /// <summary>Which of the three readings of the map to draw.</summary>
    private enum DensityView
    {
        /// <summary>Isolation window x retention time, color = precursors per spectrum.</summary>
        Heatmap,

        /// <summary>How many spectra had how many precursors.</summary>
        Histogram,

        /// <summary>Mean precursors per spectrum against retention time, with the min/max band.</summary>
        LoadOverTime,
    }

    /// <summary>
    /// The view the picker is on. Read from each item's Tag rather than its displayed text, so the label
    /// can be reworded without silently falling back to the heatmap.
    /// </summary>
    private DensityView SelectedDensityView() =>
        (DensityViewCombo.SelectedItem as ComboBoxItem)?.Tag is string tag
        && Enum.TryParse<DensityView>(tag, out var view)
            ? view
            : DensityView.Heatmap;

    // The view only changes how the same map is drawn - no re-query, no re-bin.
    private void OnDensityViewChanged(object sender, SelectionChangedEventArgs e)
    {
        // IsInitialized, not just the suppress flag: the XAML preselects this combo, so WPF raises
        // SelectionChanged from the ComboBox's EndInit - part way through InitializeComponent, when the
        // controls declared AFTER it (DensityColormapLabel, DensityHoverText) have not been created and
        // their fields are still null. Touching one there threw an NRE out of the window's constructor,
        // which is a startup crash, not a handler error.
        if (_suppressDensityRender || !IsInitialized)
            return;
        UpdateViewControls();
        if (_densityMap is not null)
            DrawDensity();
    }

    /// <summary>The colormap only applies to the heatmap; the other two views draw one series.</summary>
    private void UpdateViewControls()
    {
        var forHeatmap = SelectedDensityView() == DensityView.Heatmap
            ? Visibility.Visible
            : Visibility.Collapsed;
        DensityColormapLabel.Visibility = forHeatmap;
        DensityColormapCombo.Visibility = forHeatmap;
        DensityHoverText.Text = ""; // the readout belongs to the view that was showing
    }

    /// <summary>Query the selected run's precursors (off the UI thread), then bin and draw them.</summary>
    private async Task RenderDensityAsync()
    {
        if (_densityDataset is null || _densityColumns is null)
            return;
        var index = DensitySampleCombo.SelectedIndex;
        if (index < 0 || index >= _densitySampleIds.Count)
            return;

        var dataset = _densityDataset;
        var cols = _densityColumns;
        var sample = _densitySampleIds[index];
        var qCutoff = DensityQValue();

        // Scanning a large merged report takes a moment, so a user clicking down the run list can have
        // several queries in flight. Only the newest one may touch the plot.
        var request = ++_densityRequest;
        DensityStatusText.Text = "Loading " + DensitySampleCombo.SelectedItem + "...";
        try
        {
            var precursors = await Task.Run(() => PrecursorDensity.Load(dataset, cols, sample, qCutoff));
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
        SetDensityMap(scheme is not null
            ? PrecursorDensity.Bin(_densityPrecursors, scheme, rtBin)
            : PrecursorDensity.Bin(
                _densityPrecursors, DensityBin(DensityMzBinBox, UniformBinFallbackTh), rtBin));
        DrawDensity();
    }

    private void DrawDensity()
    {
        var map = _densityMap;
        if (map is null || map.IsEmpty)
            return;

        // The ColorBar attaches as an axis panel and survives Plot.Clear(), so start from a fresh Plot
        // on every draw rather than stacking one color bar per render - and so a color bar left by the
        // heatmap does not follow the other two views. No title: the run is named in the drop-down and
        // the color bar is labeled, so a title would only repeat them.
        DensityPlot.Reset();
        switch (SelectedDensityView())
        {
            case DensityView.Histogram:
                PlotRenderer.DrawPrecursorLoadHistogram(DensityPlot.Plot, map);
                break;
            case DensityView.LoadOverTime:
                PlotRenderer.DrawPrecursorLoadOverTime(DensityPlot.Plot, map);
                break;
            default:
                PlotRenderer.DrawPrecursorDensity(
                    DensityPlot.Plot, map, DensityColormap(ComboText(DensityColormapCombo, "Viridis")));
                break;
        }
        DensityPlot.Refresh();

        // Say which windows the map is drawn on, and - crucially - flag precursors that fell outside them,
        // which is what a wrong scheme looks like.
        var outside = map.PrecursorsOutsideRows;
        var total = _densityPrecursors?.Count ?? 0;
        // This tab is for DIA: a cell is a spectrum and its count is that spectrum's co-fragmentation
        // load. A targeted method isolates one (PRM) or a few (multiplexed) precursors per spectrum by
        // design, so that reading does not apply - and since PRISM has no way to obtain a targeted
        // method's real windows, the rows below would be a DIA scheme it was not acquired with. Say so
        // rather than letting the map be read as though it meant the same thing.
        var batch = BatchOfSelectedRun();
        var nonDia = batch is not null && _densitySchemes is not null && _densitySchemes.IsNonDia(batch)
            ? _densitySchemes.AcquisitionFor(batch)
            : null;
        // "Spectrum" is only the right noun when a cell IS one: a real window layout, acquired by DIA.
        // On the uniform-bin fallback a cell is a bin (the picker's tooltip says so), and for a targeted
        // acquisition it is a row of a scheme the data was not acquired with. Calling those a spectrum
        // is the misreading this tab's warnings exist to prevent.
        var cellNoun = nonDia is not null ? "row" : map.RowsAreWindows ? "spectrum" : "bin";
        DensityStatusText.Text =
            $"{total:N0} precursors; busiest {cellNoun} {map.MaxCount:N0}; "
            + $"{map.RowSource}; {map.MzBins:N0} rows x {map.RtBins:N0} RT bins of {map.RtBinMin:0.###} min"
            + (nonDia is not null
                ? $"; WARNING: this is a {nonDia} acquisition, and this map assumes DIA - the rows are not "
                  + "the windows it was acquired with, and a cell is not a co-fragmentation load"
                : "")
            + (outside > 0 && total > 0
                ? $"; WARNING: {outside:N0} precursors ({100.0 * outside / total:0.#}%) fall outside every "
                  + "window - is this the scheme the data was acquired with?"
                : "");
    }

    /// <summary>
    /// Read out whatever is under the cursor, in the terms of the view being shown. On the heatmap the
    /// color bar gives the scale, but the question this plot is asked ("how many precursors was THAT
    /// spectrum carrying") wants the number itself; the other two views need their own readout, or the
    /// m/z and RT this one reports would be nonsense coordinates on a different pair of axes.
    /// </summary>
    private void OnDensityPlotMouseMove(object sender, MouseEventArgs e)
    {
        var map = _densityMap;
        if (map is null || map.IsEmpty)
            return;

        var pos = e.GetPosition(DensityPlot);
        var scale = DensityPlot.DisplayScale;
        var c = DensityPlot.Plot.GetCoordinates(new Pixel(pos.X * scale, pos.Y * scale));
        DensityHoverText.Text = SelectedDensityView() switch
        {
            DensityView.Histogram => HistogramReadout(c.X),
            DensityView.LoadOverTime => LoadOverTimeReadout(c.X),
            _ => HeatmapReadout(map, c),
        };
    }

    private static string HeatmapReadout(PrecursorDensityMap map, Coordinates c)
    {
        var row = map.RowAt(c.Y);
        var col = (int)((c.X - map.RtLow) / map.RtBinMin);
        return row < 0 || col < 0 || col >= map.RtBins
            ? ""
            : $"m/z {map.Rows[row].Start:0.#}-{map.Rows[row].End:0.#} "
              + $"at {map.RtLow + col * map.RtBinMin:0.##} min: {map.Counts[row, col]:N0} precursors";
    }

    // The bars are at integer loads, so the bar under the cursor is the nearest whole number.
    private string HistogramReadout(double x)
    {
        var histogram = _densityHistogram;
        var load = (int)Math.Round(x);
        if (histogram is null || load < 0 || load >= histogram.Length)
            return "";
        var acquired = histogram.Sum();
        // The share is the reading the bar heights cannot give: "1,270 spectra" means nothing without
        // knowing how many were acquired, and that total is nowhere on the plot.
        return $"{load:N0} precursors: {histogram[load]:N0} spectra"
             + (acquired > 0 ? $" ({100.0 * histogram[load] / acquired:0.##}%)" : "");
    }

    private string LoadOverTimeReadout(double x)
    {
        var load = _densityLoad;
        var map = _densityMap;
        if (load is null || map is null)
            return "";
        var bin = (int)((x - map.RtLow) / map.RtBinMin);
        if (bin < 0 || bin >= load.Count)
            return "";
        var (time, mean, min, max) = load[bin];
        return double.IsNaN(mean)
            ? $"{time:0.##} min: nothing acquired"
            : $"{time:0.##} min: mean {mean:0.00} precursors per spectrum (min {min:N0}, max {max:N0})";
    }

    // Message in place of a plot (an empty Plot with a title, as the QC tab does for its empty states).
    // Clears the map too, so the hover readout cannot report cells that are no longer drawn.
    private void ShowDensityMessage(string message)
    {
        SetDensityMap(null);
        DensityStatusText.Text = message;
        DensityHoverText.Text = "";
        DensityPlot.Reset();
        PlotRenderer.DrawEmptyState(DensityPlot.Plot, message);
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

    /// <summary>Where a scheme's windows came from, for the picker's tooltip.</summary>
    private string WhereFrom(IsolationScheme scheme)
    {
        if (_densitySchemes?.IsMeasured(scheme) == true)
        {
            var file = _densitySchemes.Measured
                .FirstOrDefault(m => m.Scheme.LayoutKey == scheme.LayoutKey)?.DataFile;
            return string.IsNullOrWhiteSpace(file)
                ? "\nRead from the instrument data files."
                : $"\nRead from {Path.GetFileName(file)}.";
        }
        return IsolationScheme.BuiltIns.Any(b => ReferenceEquals(b, scheme))
            ? "\nA built-in layout, not this data's - check the out-of-window count below."
            : "";
    }

    /// <summary>
    /// Read the acquisition's real isolation windows out of one data file, when the run has not been
    /// asked yet and nothing better is already known.
    /// </summary>
    /// <remarks>
    /// <para><b>Why the tab does this at all.</b> A DIA analysis document stores
    /// <c>isolation_scheme name="Results only"</c> with no windows - Skyline reads them from the data
    /// at import and does not write them down - so without this the map was binned on a built-in
    /// layout that merely looks like a modern DIA cycle. The files are the only place the real answer
    /// lives, and reading it costs one file open: the windows are scan headers in the first two
    /// cycles.</para>
    ///
    /// <para><b>After the first render, never before it.</b> The map is already on screen binned on
    /// the fallback by the time this starts, so a share that takes ten seconds to open a file delays
    /// an improvement rather than the plot. And it runs once per output directory
    /// (<c>_densityMeasuredFor</c>): a cohort's replicates share one acquisition, so a second read
    /// would return the same windows.</para>
    /// </remarks>
    private async Task MeasureIsolationSchemeAsync(string outputDir)
    {
        if (string.IsNullOrWhiteSpace(outputDir))
            return;
        if (string.Equals(_densityMeasuredFor, outputDir, StringComparison.OrdinalIgnoreCase))
            return;
        // Nothing to improve on: the data has already been read, or this batch's own document declares
        // the windows, which is authoritative and locks the picker.
        if (_densitySchemes?.Measured.Count > 0)
            return;
        if (BatchOfSelectedRun() is { } batch && _densitySchemes?.DocumentSchemeFor(batch) is not null)
            return;

        // Snapshotted on the UI thread; everything that touches the disk happens off it. Finding the
        // raw directory PROBES PATHS A DOCUMENT RECORDED, which on a machine that is not the one the
        // data was imported on are exactly the paths that no longer resolve - and a dead UNC path
        // does not fail fast, it blocks for the SMB timeout.
        var named = IonRawDirText.Text?.Trim();
        var inputs = _inputs.ToArray();

        _densityMeasuredFor = outputDir;
        var previousStatus = DensityStatusText.Text;
        DensityStatusText.Text = "Reading the acquisition's isolation windows from the data files...";
        try
        {
            var (scheme, catalog) = await Task.Run(() =>
            {
                if (!Directory.Exists(outputDir))
                    return ((IsolationScheme?)null, (IsolationSchemeCatalog?)null);
                var rawDir = DensityRawDirectory(named, inputs);
                if (rawDir is null)
                    return ((IsolationScheme?)null, (IsolationSchemeCatalog?)null);

                OptionalReaders.Register(Log);
                if (!IsolationWindowProbe.Available)
                    return ((IsolationScheme?)null, (IsolationSchemeCatalog?)null);

                var read = IsolationSchemeResolver.FromData(outputDir, rawDir, Log);
                return (read, read is null
                    ? null
                    : IsolationSchemeCatalog.Load(
                        Path.Combine(outputDir, IsolationSchemeCatalog.FileName)));
            });

            // The user can have moved on while a file opened over a share.
            if (!_densityLoaded
                || !string.Equals(OutputDirBox.Text?.Trim(), outputDir, StringComparison.OrdinalIgnoreCase))
            {
                // Restored like every other exit. Leaving the transient text behind stranded
                // "Reading the acquisition's isolation windows..." above a map from the previous
                // directory, indefinitely.
                DensityStatusText.Text = previousStatus;
                return;
            }
            if (scheme is null)
            {
                DensityStatusText.Text = previousStatus;
                return;
            }

            _densitySchemes = catalog ?? _densitySchemes;
            PopulateSchemeCombo();
            // Only re-bin when the picker actually moved to the measured scheme. If the user had
            // already named one, PopulateSchemeCombo kept it and the map on screen is still theirs.
            if (!_densitySchemeChosen)
                RebinAndDraw();
            else
                DensityStatusText.Text = previousStatus;
        }
        catch (Exception ex)
        {
            App.WriteLog("Reading isolation windows from the data files failed: " + ex);
            DensityStatusText.Text = previousStatus;
        }
    }

    /// <summary>
    /// Where the instrument files are, for the window read: the directory the Settings tab names, else
    /// wherever an input document says it imported from. Null when neither answers.
    /// </summary>
    /// <remarks>
    /// Static, taking its inputs by argument, because it runs on a background thread - it probes the
    /// file system several times and must not do that on the dispatcher. Nothing here may touch a
    /// control.
    /// </remarks>
    private static string? DensityRawDirectory(string? named, IReadOnlyList<PrismInput> inputs)
    {
        if (!string.IsNullOrWhiteSpace(named) && Directory.Exists(named))
            return named;

        foreach (var input in inputs)
        {
            var guess = input.GuessRawDirectory(App.WriteLog);
            if (!string.IsNullOrWhiteSpace(guess) && Directory.Exists(guess))
                return guess;
        }
        return null;
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
