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
using SkylinePrism.Core.Visualization;
using SkylinePrism.Skyline;

namespace SkylinePrism.App;

/// <summary>
/// The "Dynamic Range" tab: log10 abundance against abundance rank - Skyline's Relative Abundance shape -
/// over the CORRECTED PRISM matrices. Clicking a point selects that protein/peptide in the Skyline
/// document tree; user-defined protein lists highlight sets of interest in their own colours.
/// </summary>
public partial class MainWindow
{
    private bool _rangeLoaded;
    private bool _suppressRangeRender;
    private string? _rangeOutputDir;
    private List<AbundanceEntry> _rangeEntries = new();
    private List<string> _rangeSampleColumns = new();
    private ProteinListSet _proteinLists = ProteinListSet.Load();

    // Points as plotted, for the click hit-test, plus the current label mode.
    private List<AbundanceEntry> _rangePlotted = new();
    private RangeLabelMode _rangeLabels = RangeLabelMode.None;
    private AbundanceEntry? _rangeSelected;

    /// <summary>What the plot labels, toggled from the right-click menu (as Skyline does).</summary>
    private enum RangeLabelMode
    {
        None,
        Selected,
        Lists,
    }

    private AbundanceLevel RangeLevel =>
        ComboText(RangeLevelCombo, "Protein").StartsWith("Pep", StringComparison.OrdinalIgnoreCase)
            ? AbundanceLevel.Peptide
            : AbundanceLevel.Protein;

    private void InvalidateDynamicRange()
    {
        _rangeLoaded = false;
        _rangeEntries = new List<AbundanceEntry>();
        _rangePlotted = new List<AbundanceEntry>();
        _rangeSelected = null;
    }

    private async void OnRangeReload(object sender, RoutedEventArgs e)
    {
        InvalidateDynamicRange();
        await LoadDynamicRangeAsync();
    }

    private async void OnRangeLevelChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_suppressRangeRender || !_rangeLoaded)
            return;
        await LoadDynamicRangeAsync(force: true);
    }

    /// <summary>
    /// Read the corrected matrix for the current level and rank it. Parquet I/O runs off the UI thread;
    /// these files are small compared with the merged report but can still sit on a slow share.
    /// </summary>
    private async Task LoadDynamicRangeAsync(bool force = false)
    {
        var outputDir = OutputDirBox.Text?.Trim();
        if (string.IsNullOrWhiteSpace(outputDir))
        {
            ShowRangeMessage("Set an output directory to plot its corrected abundances.");
            return;
        }

        var level = RangeLevel;
        var fileName = level == AbundanceLevel.Protein ? "corrected_proteins" : "corrected_peptides";
        var path = CorrectedMatrixPath(outputDir, fileName);
        if (path is null)
        {
            _rangeLoaded = false;
            // Distinguish "nothing here" from "here, but written as CSV" - the pipeline honours
            // output.format, and this tab reads parquet.
            var textForm = new[] { ".csv", ".tsv" }
                .Select(ext => Path.Combine(outputDir, fileName + ext))
                .FirstOrDefault(File.Exists);
            ShowRangeMessage(textForm is not null
                ? $"{Path.GetFileName(textForm)} is not parquet. This plot reads the parquet matrices - "
                  + "re-run with output.format: parquet."
                : $"No {fileName}.parquet in the output directory. Run PRISM, or point the output "
                  + "directory at a previous run.");
            return;
        }
        if (_rangeLoaded && !force && string.Equals(_rangeOutputDir, outputDir, StringComparison.OrdinalIgnoreCase))
            return;

        RangeStatusText.Text = "Reading " + Path.GetFileName(path) + "...";
        try
        {
            var (entries, samples) = await Task.Run(() =>
            {
                var table = ParquetTable.Load(path);
                var sampleCols = DynamicRange.SampleColumns(table, level);
                return (DynamicRange.Compute(table, level, sampleCols), sampleCols);
            });

            _rangeOutputDir = outputDir;
            _rangeSampleColumns = samples;
            _rangeEntries = entries;
            _rangeLoaded = true;
            PopulateRangeSamples(samples);
            RenderDynamicRange();
        }
        catch (Exception ex)
        {
            _rangeLoaded = false;
            App.WriteLog("Dynamic range load failed: " + ex);
            ShowRangeMessage("Could not read the corrected matrix: " + ex.Message);
        }
    }

    // Replicate picker: same checkbox-combo pattern as the QC tab's group filter; none ticked = All.
    private List<QcGroupValue> _rangeSampleValues = new();

    private void PopulateRangeSamples(IReadOnlyList<string> samples)
    {
        _suppressRangeRender = true;
        try
        {
            var display = StripSharedBatchSuffix(samples.ToList(), samples.ToList());
            _rangeSampleValues = samples
                .Select((s, i) => new QcGroupValue { Name = display[i], Changed = OnRangeSampleToggled })
                .ToList();
            RangeSampleCombo.ItemsSource = _rangeSampleValues;
            UpdateRangeSampleSummary();
        }
        finally
        {
            _suppressRangeRender = false;
        }
    }

    private void OnRangeSampleToggled()
    {
        UpdateRangeSampleSummary();
        if (_suppressRangeRender)
            return;
        // Averaging over a different replicate set changes every abundance, so recompute rather than
        // re-draw: the ranking itself moves.
        RecomputeForSelectedSamples();
    }

    private void UpdateRangeSampleSummary()
    {
        if (RangeSampleCombo is null)
            return;
        var selected = _rangeSampleValues.Where(v => v.IsSelected).Select(v => v.Name).ToList();
        RangeSampleCombo.Text = selected.Count == 0
            ? "All replicates"
            : QcGroupFilter.Summarize(selected, _rangeSampleValues.Count);
    }

    private async void RecomputeForSelectedSamples()
    {
        if (!_rangeLoaded || _rangeOutputDir is null)
            return;
        var level = RangeLevel;
        var chosen = Enumerable.Range(0, _rangeSampleValues.Count)
            .Where(i => _rangeSampleValues[i].IsSelected)
            .Select(i => _rangeSampleColumns[i])
            .ToList();
        var fileName = level == AbundanceLevel.Protein ? "corrected_proteins" : "corrected_peptides";
        var path = CorrectedMatrixPath(_rangeOutputDir, fileName);
        if (path is null)
            return;

        try
        {
            _rangeEntries = await Task.Run(() =>
                DynamicRange.Compute(ParquetTable.Load(path), level, chosen));
            RenderDynamicRange();
        }
        catch (Exception ex)
        {
            App.WriteLog("Dynamic range recompute failed: " + ex);
            ShowRangeMessage("Could not recompute: " + ex.Message);
        }
    }

    private void RenderDynamicRange()
    {
        if (_rangeEntries.Count == 0)
        {
            ShowRangeMessage("No abundances to plot.");
            return;
        }

        RangePlot.Reset();
        var plt = RangePlot.Plot;

        // Split into the grey background and one group per visible protein list.
        var matcher = _proteinLists.BuildMatcher();
        var byList = new Dictionary<ProteinList, List<AbundanceEntry>>();
        var background = new List<AbundanceEntry>();
        foreach (var entry in _rangeEntries)
        {
            var list = matcher.Match(entry);
            if (list is null)
            {
                background.Add(entry);
                continue;
            }
            if (!byList.TryGetValue(list, out var bucket))
                byList[list] = bucket = new List<AbundanceEntry>();
            bucket.Add(entry);
        }

        var highlights = matcher.Lists
            .Where(byList.ContainsKey)
            .Select(l => (l.Name, l.ColorHex, (IReadOnlyList<AbundanceEntry>)byList[l]))
            .ToList();

        PlotRenderer.DrawDynamicRange(
            plt, background, highlights,
            yLabel: "Log10 abundance",
            xLabel: RangeLevel == AbundanceLevel.Protein ? "Protein rank" : "Peptide rank");

        AddRangeLabels(plt, matcher, byList);
        BuildRangeMenu();
        RangePlot.Refresh();
        _rangePlotted = _rangeEntries;

        var matched = byList.Sum(kv => kv.Value.Count);
        RangeStatusText.Text =
            $"{_rangeEntries.Count:N0} {(RangeLevel == AbundanceLevel.Protein ? "protein groups" : "peptides")}; "
            + $"{_rangeEntries[^1].Log10Abundance:0.#}-{_rangeEntries[0].Log10Abundance:0.#} log10 "
            + $"({_rangeEntries[0].Log10Abundance - _rangeEntries[^1].Log10Abundance:0.#} orders of magnitude); "
            + $"averaged over {SelectedReplicateCount():N0} replicate(s)"
            + (matched > 0 ? $"; {matched:N0} in {highlights.Count} list(s)" : "");
    }

    private int SelectedReplicateCount()
    {
        var ticked = _rangeSampleValues.Count(v => v.IsSelected);
        return ticked > 0 ? ticked : _rangeSampleColumns.Count;
    }

    // Labels: nothing, just the element selected in Skyline, or every member of the visible lists.
    private void AddRangeLabels(
        Plot plt, ProteinListMatcher matcher, Dictionary<ProteinList, List<AbundanceEntry>> byList)
    {
        // Offsets in data units, from the plotted extents, so a leader line is the same visual length
        // whatever the abundance range happens to be.
        var xSpan = Math.Max(1, _rangeEntries.Count);
        var ySpan = Math.Max(0.5, _rangeEntries[0].Log10Abundance - _rangeEntries[^1].Log10Abundance);
        var dx = xSpan * 0.035;
        var dy = ySpan * 0.045;

        switch (_rangeLabels)
        {
            case RangeLabelMode.Selected when _rangeSelected is not null:
                AddRangeLabel(plt, _rangeSelected, Colors.Black, xSpan, dx, dy);
                break;

            case RangeLabelMode.Lists:
                // If any list opted into labels explicitly, label only those; otherwise label them all.
                var opted = matcher.Lists.Any(l => l.ShowLabels);
                foreach (var (list, entries) in byList)
                {
                    if (opted && !list.ShowLabels)
                        continue;
                    foreach (var entry in entries)
                        AddRangeLabel(plt, entry, Color.FromHex(list.ColorHex), xSpan, dx, dy);
                }
                break;
        }
    }

    /// <summary>
    /// A label offset from its point with a leader line back to it. Offsetting matters because the points
    /// sit on a dense curve - a label centred on its point buries the very point it names.
    /// </summary>
    private static void AddRangeLabel(
        Plot plt, AbundanceEntry entry, Color color, double xSpan, double dx, double dy)
    {
        // Labels go up-right, except near the right edge where they would run off the canvas.
        var toLeft = entry.Rank > xSpan * 0.8;
        var labelX = entry.Rank + (toLeft ? -dx : dx);
        var labelY = entry.Log10Abundance + dy;

        var leader = plt.Add.Line(entry.Rank, entry.Log10Abundance, labelX, labelY);
        leader.LineColor = color.WithAlpha(0.75);
        leader.LineWidth = 1.5f;
        leader.MarkerSize = 0;

        var text = plt.Add.Text(entry.Label, labelX, labelY);
        text.LabelFontSize = 16;
        text.LabelBold = true;
        text.LabelFontColor = color;
        text.LabelBackgroundColor = Colors.White.WithAlpha(0.75);
        text.LabelBorderColor = color.WithAlpha(0.4);
        text.LabelBorderWidth = 1;
        text.LabelAlignment = toLeft ? Alignment.LowerRight : Alignment.LowerLeft;
    }

    /// <summary>
    /// Click a point -> select that protein/peptide in Skyline's document tree. The locator map is read
    /// from Skyline once and cached; in standalone mode there is nothing to drive, so the click just
    /// reports what was hit.
    /// </summary>
    private void OnRangePlotClick(object sender, MouseButtonEventArgs e)
    {
        if (_rangePlotted.Count == 0)
            return;

        var pos = e.GetPosition(RangePlot);
        var scale = RangePlot.DisplayScale;
        var plt = RangePlot.Plot;
        var mouse = new Pixel(pos.X * scale, pos.Y * scale);

        AbundanceEntry? best = null;
        var bestDistance = double.MaxValue;
        foreach (var entry in _rangePlotted)
        {
            var px = plt.GetPixel(new Coordinates(entry.Rank, entry.Log10Abundance));
            double dx = px.X - mouse.X, dy = px.Y - mouse.Y;
            var d2 = dx * dx + dy * dy;
            if (d2 < bestDistance)
            {
                bestDistance = d2;
                best = entry;
            }
        }

        const double thresholdPx = 14;
        if (best is null || bestDistance > thresholdPx * thresholdPx)
            return;

        _rangeSelected = best;
        SelectInSkyline(best);
        RenderDynamicRange(); // redraw so a "label the selection" mode follows the click
    }

    private void SelectInSkyline(AbundanceEntry entry)
    {
        // Always report every protein the entry belongs to. PRISM runs its OWN parsimony, which can group
        // proteins differently from the Skyline document, so "which proteins is this peptide in" is a
        // question about the PRISM result and is worth stating outright rather than implying the single
        // protein we happened to navigate to.
        var where = DescribeProteins(entry);

        if (_session is null)
        {
            RangeStatusText.Text = $"{entry.Label} (rank {entry.Rank:N0}, log10 {entry.Log10Abundance:0.00})"
                + where + " - not connected to Skyline, so nothing to select.";
            return;
        }

        var locator = ResolveLocator(entry, out var viaFallback);
        if (locator is null)
        {
            RangeStatusText.Text = $"{entry.Label}{where} - no matching element in the Skyline document "
                + "(PRISM's protein grouping can differ from the document's).";
            return;
        }

        var driver = new SkylineReportDriver(_session, Log);
        if (!driver.SelectElement(locator))
        {
            RangeStatusText.Text = $"Could not select {entry.Label} in Skyline.";
            return;
        }

        RangeStatusText.Text =
            $"Selected {entry.Label} in Skyline (rank {entry.Rank:N0}, log10 {entry.Log10Abundance:0.00}){where}"
            + (viaFallback
                ? " - selected under the first protein in the document tree, since PRISM's grouping did "
                  + "not match a protein node."
                : "");
    }

    /// <summary>", in PG0002 (sp|P68871|HBB_HUMAN) and PG0007 (sp|Q9Y6K9|NEMO_HUMAN)" - every group.</summary>
    private static string DescribeProteins(AbundanceEntry entry)
    {
        if (entry.ProteinNames.Count == 0)
            return "";
        var parts = entry.ProteinNames
            .Select((name, i) => i < entry.ProteinGroups.Count ? $"{entry.ProteinGroups[i]} ({name})" : name)
            .ToList();
        return entry.IsShared
            ? $" - shared across {parts.Count} protein groups: {string.Join(", ", parts)}"
            : $" - in {parts[0]}";
    }

    private Dictionary<string, string>? _rangeLocatorMap;
    private AbundanceLevel _rangeLocatorLevel = AbundanceLevel.Protein;

    /// <summary>
    /// The Skyline element to select for a plotted point. At peptide level this selects the PEPTIDE node
    /// (which is what shows its chromatograms), not its protein.
    /// <para>Precedence, because PRISM's parsimony can group proteins differently from the document: try
    /// the peptide under each of its PRISM groups in order, then fall back to the first occurrence of that
    /// sequence in the document tree. <paramref name="viaFallback"/> reports which happened, so the user
    /// is told when the tree's grouping decided it.</para>
    /// </summary>
    private string? ResolveLocator(AbundanceEntry entry, out bool viaFallback)
    {
        viaFallback = false;
        if (_session is null)
            return null;
        var level = RangeLevel;
        if (_rangeLocatorMap is null || _rangeLocatorLevel != level)
        {
            var driver = new SkylineReportDriver(_session, Log);
            _rangeLocatorMap = driver.GetLocatorMap(level == AbundanceLevel.Protein ? "group" : "molecule");
            _rangeLocatorLevel = level;
        }

        if (level == AbundanceLevel.Peptide)
        {
            // "<protein>/<peptide>" per group, in PRISM's group order.
            foreach (var protein in entry.ProteinNames)
                if (_rangeLocatorMap.TryGetValue($"{protein}/{entry.Key}", out var byProtein))
                    return byProtein;

            // Nothing matched a PRISM group, so take the sequence's first occurrence in the tree - the
            // map keeps the FIRST element for an ambiguous bare key, and GetLocations returns them in
            // document order.
            if (_rangeLocatorMap.TryGetValue(entry.Key, out var firstInTree))
            {
                viaFallback = entry.ProteinNames.Count > 0;
                return firstInTree;
            }
            return null;
        }

        foreach (var key in new[] { entry.Key, entry.ProteinName, entry.Accession, entry.Gene })
            if (!string.IsNullOrWhiteSpace(key) && _rangeLocatorMap.TryGetValue(key!, out var locator))
                return locator;
        return null;
    }

    private static string? CorrectedMatrixPath(string outputDir, string stem)
    {
        var parquet = Path.Combine(outputDir, stem + ".parquet");
        return File.Exists(parquet) ? parquet : null;
    }

    /// <summary>
    /// Right-click menu for the label modes, alongside ScottPlot's own Save/Copy items - the same place
    /// Skyline puts its plot options. Rebuilt on each render so the ticks reflect the current mode.
    /// </summary>
    private void BuildRangeMenu()
    {
        var menu = RangePlot.Menu;
        if (menu is null)
            return;
        menu.Reset(); // start from ScottPlot's defaults (Save Image, Copy, Autoscale)
        menu.AddSeparator();
        menu.Add(Tick("No labels", RangeLabelMode.None), _ => SetRangeLabels(RangeLabelMode.None));
        menu.Add(Tick("Label the Skyline selection", RangeLabelMode.Selected),
            _ => SetRangeLabels(RangeLabelMode.Selected));
        menu.Add(Tick("Label protein lists", RangeLabelMode.Lists), _ => SetRangeLabels(RangeLabelMode.Lists));
        menu.AddSeparator();
        menu.Add("Protein lists...", _ => OnManageProteinLists(this, new RoutedEventArgs()));

        string Tick(string label, RangeLabelMode mode) => (_rangeLabels == mode ? "✓ " : "    ") + label;
    }

    private void SetRangeLabels(RangeLabelMode mode)
    {
        _rangeLabels = mode;
        if (_rangeLoaded)
            RenderDynamicRange();
    }

    private void ShowRangeMessage(string message)
    {
        RangeStatusText.Text = message;
        RangePlot.Reset();
        RangePlot.Plot.Title(message);
        PlotRenderer.StyleQcPlot(RangePlot.Plot);
        RangePlot.Refresh();
    }

    private void OnManageProteinLists(object sender, RoutedEventArgs e)
    {
        var dialog = new ProteinListWindow(_proteinLists) { Owner = this };
        if (dialog.ShowDialog() != true)
            return;
        _proteinLists = dialog.Result;
        try
        {
            _proteinLists.Save();
        }
        catch (Exception ex)
        {
            App.WriteLog("Could not save protein lists: " + ex);
        }
        if (_rangeLoaded)
            RenderDynamicRange();
    }
}
