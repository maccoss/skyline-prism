using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.Numerics;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.Visualization;

namespace SkylinePrism.App;

/// <summary>
/// The Markers pane: evaluate a protein list (marker panel) against the corrected matrix. Shows a row
/// z-scored heatmap of the panel's members across groups (or samples) and a per-group boxplot of the
/// panel score. Reuses the Dynamic Range protein lists as the panel options.
/// </summary>
public partial class MainWindow
{
    private void OnMarkersSettingsCollapse(object sender, RoutedEventArgs e) => SetMarkersSettingsVisible(false);

    private void OnMarkersSettingsExpand(object sender, RoutedEventArgs e) => SetMarkersSettingsVisible(true);

    /// <summary>
    /// Show or hide the Markers settings panel, leaving a strip to bring it back - the same behavior
    /// as the Differential pane's, so the two panes are learned once.
    /// </summary>
    private void SetMarkersSettingsVisible(bool visible)
    {
        MarkersSettingsPanel.Visibility = visible ? Visibility.Visible : Visibility.Collapsed;
        MarkersSettingsExpandButton.Visibility = visible ? Visibility.Collapsed : Visibility.Visible;
    }

    private DifferentialDataset? _markersDataset;
    private string? _markersDir;
    private FeatureLevel _markersLevel;
    private bool _markersSuppress;

    /// <summary>The dataset the pane is currently showing - whether loaded here or reused from the
    /// Differential pane. RenderMarkers reads this directly rather than re-deriving which cache to use.</summary>
    private DifferentialDataset? _markersActive;
    private List<SelectablePanel> _markersPanelItems = new();

    /// <summary>A protein list offered in the Markers panel picker, tickable for a multi-panel union.</summary>
    private sealed class SelectablePanel : System.ComponentModel.INotifyPropertyChanged
    {
        private bool _isSelected;

        public required string Display { get; init; }
        public required ProteinList List { get; init; }

        /// <summary>Raised on tick/untick so the pane can re-render and refresh the summary text.</summary>
        public Action? Changed { get; init; }

        public bool IsSelected
        {
            get => _isSelected;
            set
            {
                if (_isSelected == value)
                    return;
                _isSelected = value;
                PropertyChanged?.Invoke(this, new System.ComponentModel.PropertyChangedEventArgs(nameof(IsSelected)));
                Changed?.Invoke();
            }
        }

        public event System.ComponentModel.PropertyChangedEventHandler? PropertyChanged;

        public override string ToString() => Display;
    }

    private async Task LoadMarkersAsync()
    {
        var dir = OutputDirBox.Text?.Trim();
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
        {
            MarkersStatusText.Text = "Set a PRISM output directory above to evaluate a marker panel.";
            return;
        }

        _markersSuppress = true;
        try
        {
            if (MarkersLevelCombo.SelectedItem is null)
                MarkersLevelCombo.SelectedIndex = 0;
            if (MarkersViewCombo.SelectedItem is null)
                MarkersViewCombo.SelectedIndex = 0;
        }
        finally
        {
            _markersSuppress = false;
        }

        var level = MarkersSelectedLevel();
        DifferentialDataset ds;
        try
        {
            ds = await GetMarkersDatasetAsync(dir, level);
        }
        catch (Exception ex)
        {
            MarkersStatusText.Text = "Load failed: " + ex.Message;
            return;
        }

        _markersActive = ds;

        _markersSuppress = true;
        try
        {
            PopulateMarkersPanels();
            MarkersGroupByCombo.ItemsSource = ds.MetadataColumns;
            if (MarkersGroupByCombo.SelectedItem is null)
                MarkersGroupByCombo.SelectedItem = DefaultContrastColumn(ds);
        }
        finally
        {
            _markersSuppress = false;
        }

        RenderMarkers();
    }

    private async Task<DifferentialDataset> GetMarkersDatasetAsync(string dir, FeatureLevel level)
    {
        // Reuse the Differential pane's dataset when it is the same directory + level (it already has any
        // clinical CSV attached); otherwise load our own and attach the clinical CSV the same way.
        if (_diffDataset is not null && _diffLoadedDir == dir && _diffLoadedLevel == level)
            return _diffDataset;
        if (_markersDataset is not null && _markersDir == dir && _markersLevel == level)
            return _markersDataset;

        var ds = await Task.Run(() => DifferentialDataset.Load(dir, level));
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

        _markersDataset = ds;
        _markersDir = dir;
        _markersLevel = level;
        return ds;
    }

    private void PopulateMarkersPanels()
    {
        // Preserve any current selection by list name across a reload.
        var previouslySelected = new HashSet<string>(
            _markersPanelItems.Where(i => i.IsSelected).Select(i => i.List.Name), StringComparer.Ordinal);

        _markersPanelItems = ProteinListSet.Load().WithBuiltIns()
            .Where(l => l.Members.Count > 0)
            .OrderBy(l => l.Category, StringComparer.Ordinal)
            .ThenBy(l => l.Name, StringComparer.Ordinal)
            .Select(l => new SelectablePanel
            {
                Display = string.IsNullOrEmpty(l.Category) ? l.Name : $"{l.Category}: {l.Name}",
                List = l,
                Changed = OnMarkersPanelToggled,
            })
            .ToList();

        // Restore prior ticks; if nothing was selected, tick the first panel so the pane is not empty.
        var restoredAny = false;
        foreach (var item in _markersPanelItems)
            if (previouslySelected.Contains(item.List.Name))
            {
                item.IsSelected = true;
                restoredAny = true;
            }

        if (!restoredAny && _markersPanelItems.Count > 0)
            _markersPanelItems[0].IsSelected = true;

        MarkersPanelCombo.ItemsSource = _markersPanelItems;
        UpdateMarkersPanelSummary();
    }

    private void OnMarkersPanelToggled()
    {
        if (_markersSuppress)
            return;
        UpdateMarkersPanelSummary();
        RenderMarkers();
    }

    private void UpdateMarkersPanelSummary()
    {
        var selected = _markersPanelItems.Where(i => i.IsSelected).ToList();
        MarkersPanelCombo.Text = selected.Count switch
        {
            0 => "(pick panels)",
            1 => selected[0].List.Name,
            _ => $"{selected.Count} panels",
        };
    }

    private FeatureLevel MarkersSelectedLevel() =>
        (MarkersLevelCombo.SelectedItem as ListBoxItem)?.Content as string == "Peptide"
            ? FeatureLevel.Peptide
            : FeatureLevel.Protein;

    private async void OnMarkersLevelChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_markersSuppress || !IsInitialized)
            return;
        try
        {
            await LoadMarkersAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnMarkersLevelChanged), ex);
        }
    }

    private void OnMarkersGroupByChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_markersSuppress)
            return;
        RenderMarkers();
    }

    private void OnMarkersViewChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_markersSuppress)
            return;
        RenderMarkers();
    }

    /// <summary>
    /// Forget the loaded matrix so the pane reloads on its next show (new dir / new run), the
    /// counterpart of <c>InvalidateDifferential</c>.
    /// </summary>
    /// <remarks>
    /// The cache is keyed on (directory, level), so a re-run into the SAME output directory - which
    /// is the ordinary way to iterate on settings - matched the key and kept serving the previous
    /// run's matrix. Nothing looked wrong: the heatmap drew, captioned as the run that had just
    /// finished. <c>_markersActive</c> is cleared as well because RenderMarkers reads it directly
    /// rather than re-deriving which cache to use, so leaving it set would redraw the stale matrix
    /// even after the cache behind it was dropped.
    /// </remarks>
    private void InvalidateMarkers()
    {
        _markersDataset = null;
        _markersDir = null;
        _markersActive = null;
    }

    private async void OnMarkersReload(object sender, System.Windows.RoutedEventArgs e)
    {
        try
        {
            InvalidateMarkers();
            await LoadMarkersAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnMarkersReload), ex);
        }
    }

    private void RenderMarkers()
    {
        var ds = _markersActive;
        if (ds is null)
            return;
        if (MarkersGroupByCombo.SelectedItem is not string groupCol)
            return;

        var selectedPanels = _markersPanelItems.Where(i => i.IsSelected).Select(i => i.List).ToList();
        if (selectedPanels.Count == 0)
        {
            MarkersHeatPlot.Reset();
            MarkersHeatPlot.Refresh();
            MarkersBoxPlot.Reset();
            MarkersBoxPlot.Refresh();
            MarkersStatusText.Text = "Pick one or more panels to evaluate.";
            MarkersNoteText.Text = "Tick panels in the Panels list. Several tick together into one heatmap.";
            return;
        }

        var combined = CombinePanels(selectedPanels);

        string?[] groups;
        try
        {
            groups = ds.MetadataValues(groupCol);
        }
        catch (ArgumentException)
        {
            return;
        }

        var perSample = (MarkersViewCombo.SelectedItem as ListBoxItem)?.Content as string == "Per sample";
        var identities = Enumerable.Range(0, ds.FeatureIds.Length).Select(ds.IdentityOf).ToList();
        var result = MarkerPanel.Evaluate(ds.ExprLog2, identities, groups,
            ds.SampleIds, combined, perSample);

        if (result.MarkerLabels.Length == 0)
        {
            MarkersHeatPlot.Reset();
            MarkersHeatPlot.Refresh();
            MarkersBoxPlot.Reset();
            MarkersBoxPlot.Refresh();
            MarkersStatusText.Text =
                $"None of {combined.Name}'s {result.Total} members matched a {_markersLevel.ToString().ToLowerInvariant()} feature.";
            MarkersNoteText.Text = result.NotDetected.Count == 0
                ? "No members detected in this run."
                : "Not detected: " + NamePreview.Of(result.NotDetected, NotDetectedShown);
            return;
        }

        MarkersHeatPlot.Reset();
        var heat = MarkersHeatPlot.Plot;
        PlotRenderer.DrawValueHeatmap(heat, result.Heatmap, result.ColumnLabels, result.MarkerLabels,
            result.SymmetricMax, "row z-score", annotate: false,
            columnGroups: result.ColumnGroups, groupNames: result.GroupNames, fontScale: 0.7);
        heat.Title($"{combined.Name} (row z-scored log2) - "
            + (perSample ? "per sample" : $"group means by {groupCol}"));
        MarkersHeatPlot.Refresh();

        DrawMarkerBoxplot(result, groupCol);

        MarkersStatusText.Text =
            $"{combined.Name}: found {result.Found}/{result.Total} members across "
            + $"{result.GroupNames.Length} groups.";
        MarkersNoteText.Text = result.NotDetected.Count == 0
            ? "All panel members detected. Boxplot below is each sample's mean marker z-score per group."
            : $"Not detected ({result.NotDetected.Count}): "
              + NamePreview.Of(result.NotDetected, NotDetectedShown);
    }

    /// <summary>
    /// How many not-detected members the note names before falling back to a count.
    /// </summary>
    /// <remarks>
    /// Enough to scan - the point of the list is to spot a member you expected - but bounded,
    /// because this note sits above the plots in a wrapping block. Ticking all 65 shipped panels
    /// gives 1,273 not-detected members, which filled the pane and pushed the heatmap out of view.
    /// The full count is always stated, so nothing is hidden, only unlisted.
    /// </remarks>
    private const int NotDetectedShown = 12;

    /// <summary>Union several panels' members (dedup by match token) into one list for a combined heatmap.</summary>
    private static ProteinList CombinePanels(IReadOnlyList<ProteinList> panels)
    {
        if (panels.Count == 1)
            return panels[0];

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var members = new List<string>();
        foreach (var panel in panels)
            foreach (var member in panel.Members)
            {
                var token = ProteinList.MatchToken(member);
                if (token.Length > 0 && seen.Add(token))
                    members.Add(member);
            }

        return new ProteinList { Name = $"{panels.Count} panels", Members = members };
    }

    private void DrawMarkerBoxplot(MarkerPanelResult result, string groupCol)
    {
        MarkersBoxPlot.Reset();
        var plt = MarkersBoxPlot.Plot;

        var boxes = new List<ScottPlot.Box>();
        for (var g = 0; g < result.GroupNames.Length; g++)
        {
            var vals = result.PanelScoreByGroup[g];
            var hex = DiffPalette[g % DiffPalette.Length];
            if (vals.Length > 0)
            {
                var q1 = Stats.PercentileLinear(vals, 25);
                var med = Stats.PercentileLinear(vals, 50);
                var q3 = Stats.PercentileLinear(vals, 75);
                var iqr = q3 - q1;
                double dataMin = double.PositiveInfinity, dataMax = double.NegativeInfinity;
                foreach (var v in vals)
                {
                    if (v < dataMin) dataMin = v;
                    if (v > dataMax) dataMax = v;
                }

                boxes.Add(new ScottPlot.Box
                {
                    Position = g,
                    Width = 0.6,
                    BoxMin = q1,
                    BoxMiddle = med,
                    BoxMax = q3,
                    WhiskerMin = Math.Max(dataMin, q1 - 1.5 * iqr),
                    WhiskerMax = Math.Min(dataMax, q3 + 1.5 * iqr),
                    FillColor = ScottPlot.Color.FromHex(hex).WithAlpha((byte)90),
                    LineColor = ScottPlot.Color.FromHex(hex),
                });
            }

            // Jittered per-sample points, seeded for a stable layout.
            var rng = new Random(g * 7919 + result.MarkerLabels.Length);
            var xs = new double[vals.Length];
            for (var i = 0; i < vals.Length; i++)
                xs[i] = g - 0.22 + rng.NextDouble() * 0.44;
            if (vals.Length > 0)
            {
                var m = plt.Add.Markers(xs, vals);
                m.Color = ScottPlot.Color.FromHex(hex);
                m.MarkerSize = 5;
            }
        }

        if (boxes.Count > 0)
            plt.Add.Boxes(boxes);

        var pos = new double[result.GroupNames.Length];
        for (var g = 0; g < pos.Length; g++)
            pos[g] = g;
        plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(pos, result.GroupNames);
        plt.XLabel(groupCol);
        plt.YLabel("mean marker z-score");
        // Scaled down as a whole, the way StyleQcPlot asks to be. Its sizes are for a figure copied
        // into a paper; this box plot is the lower two-fifths of a pane, and at full size its axis
        // text took most of that height - the boxes were squeezed into a sliver and the y-axis title
        // was cut to "mean marker z-". At this scale it matches the heatmap above it.
        PlotRenderer.StyleQcPlot(plt, fontScale: 0.55);
        // Reserve left-axis room so the rotated y-axis title is not clipped at the pane edge.
        plt.Axes.Left.MinimumSize = 60;

        // Rotate the group labels only when they would actually crowd horizontally - a proxy on total
        // label width, so many groups OR long names rotate but a handful of short labels (e.g. F01..F10)
        // stay straight and centered under their boxes. When we do rotate, anchor UpperRight so the text
        // hangs cleanly below the axis instead of tipping up across the axis line (MiddleRight put the
        // top half of each label above the tick).
        var maxLabelLen = result.GroupNames.Length == 0 ? 0 : result.GroupNames.Max(n => n?.Length ?? 0);
        var rotateLabels = result.GroupNames.Length * (maxLabelLen + 1) > 60;
        plt.Axes.Bottom.TickLabelStyle.Rotation = rotateLabels ? 45 : 0;
        plt.Axes.Bottom.TickLabelStyle.Alignment =
            rotateLabels ? ScottPlot.Alignment.UpperRight : ScottPlot.Alignment.UpperCenter;
        plt.Axes.Bottom.MinimumSize = rotateLabels ? 90 : 34;

        // Pin both axes: X to the group slots, Y to the actual score range (auto-scale over-pads to
        // a round +/-10 when the panel scores sit near zero).
        var allScores = result.PanelScoreByGroup.SelectMany(v => v).Where(double.IsFinite).ToList();
        double yMin = -1, yMax = 1;
        if (allScores.Count > 0)
        {
            yMin = allScores.Min();
            yMax = allScores.Max();
            var pad = Math.Max(0.2, (yMax - yMin) * 0.1);
            yMin -= pad;
            yMax += pad;
        }

        plt.Axes.SetLimits(-0.6, result.GroupNames.Length - 0.4, yMin, yMax);
        MarkersBoxPlot.Refresh();
    }
}
