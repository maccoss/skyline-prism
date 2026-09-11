using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using SkylinePrism.Core.Visualization;

namespace SkylinePrism.App;

/// <summary>
/// The "MS2 signal" pane: how much of the MS2 an analysis assigns to a peptide, for the whole cohort
/// and across the gradient for one replicate.
/// </summary>
/// <remarks>
/// <para>Reads an output directory rather than the run that just finished, like the other two
/// visualization panes, so it works on any previous run the output box is pointed at. Everything it
/// needs is in there: the cached accounting names the tolerance and the isolation scheme it used, so
/// a profile can be rebuilt on exactly the settings the bars were computed with rather than on
/// whatever the Settings tab currently says.</para>
///
/// <para>The acquired half is optional throughout. Without it both plots still draw - the bar plot
/// without its background bar, the profile without its band - and the status line says so rather
/// than leaving a reader to assume the assigned signal is all there was.</para>
/// </remarks>
public partial class MainWindow
{
    private bool _ms2Loaded;                     // accounting matches the current output directory
    private bool _suppressMs2Render;             // set while populating combos
    private string? _ms2OutputDir;
    private Ms2SignalAccounting.Result? _ms2Accounting;
    private IsolationScheme? _ms2Scheme;
    private ProductMassTolerance? _ms2Tolerance;
    private int _ms2Request;                     // newest query wins if the user clicks ahead of it

    /// <summary>
    /// Profiles already built, by sample. Re-binning is free but the SCAN is not, so flipping back to
    /// a replicate already looked at is instant.
    /// </summary>
    private readonly Dictionary<string, Ms2SignalProfile> _ms2Profiles = new(StringComparer.Ordinal);

    /// <summary>Drop the cached accounting so the pane reloads next time it is shown.</summary>
    private void InvalidateMs2Signal()
    {
        _ms2Loaded = false;
        _ms2Accounting = null;
        _ms2Profiles.Clear();
    }

    private bool Ms2ProfileSelected =>
        string.Equals(ComboTag(Ms2ViewCombo), "Profile", StringComparison.Ordinal);

    private async void OnMs2ViewChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            // Fires from EndInit if a SelectedIndex is ever set in XAML, before the controls below
            // exist. See XamlInitializationOrderTests.
            if (!IsInitialized || _suppressMs2Render)
                return;
            UpdateMs2Controls();
            await RenderMs2Async();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnMs2ViewChanged), ex);
        }
    }

    private async void OnMs2ReplicateChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _suppressMs2Render)
                return;
            await RenderMs2Async();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnMs2ReplicateChanged), ex);
        }
    }

    private async void OnMs2BinChanged(object sender, RoutedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _suppressMs2Render)
                return;
            // A bin change re-bins the same regions, so the scan is not repeated - but the profile
            // objects hold their bins, so they are dropped and rebuilt.
            _ms2Profiles.Clear();
            await RenderMs2Async();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnMs2BinChanged), ex);
        }
    }

    private void OnMs2BinKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter)
            Ms2Plot.Focus();   // moves focus, which raises LostFocus and re-renders
    }

    private async void OnMs2Reload(object sender, RoutedEventArgs e)
    {
        try
        {
            InvalidateMs2Signal();
            await LoadMs2SignalAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnMs2Reload), ex);
        }
    }

    /// <summary>Show or hide the controls that only apply to one of the two views.</summary>
    private void UpdateMs2Controls()
    {
        if (Ms2ReplicateLabel is null || Ms2ReplicateCombo is null
            || Ms2BinLabel is null || Ms2BinBox is null)
            return;
        var profile = Ms2ProfileSelected;
        var vis = profile ? Visibility.Visible : Visibility.Collapsed;
        Ms2ReplicateLabel.Visibility = vis;
        Ms2ReplicateCombo.Visibility = vis;
        Ms2BinLabel.Visibility = vis;
        Ms2BinBox.Visibility = vis;
    }

    /// <summary>
    /// Read the accounting for the current output directory and fill the replicate picker. The
    /// parquet reads go off the UI thread - an output directory on a share or scanned by Defender
    /// can block for seconds.
    /// </summary>
    private async Task LoadMs2SignalAsync()
    {
        var outputDir = OutputDirBox.Text?.Trim();
        if (string.IsNullOrWhiteSpace(outputDir) || !Directory.Exists(outputDir))
        {
            ShowMs2Message("Set the output directory above to a finished PRISM run.");
            return;
        }
        if (_ms2Loaded && string.Equals(_ms2OutputDir, outputDir, StringComparison.OrdinalIgnoreCase))
            return;

        ShowMs2Message("Loading the MS2 signal accounting...");
        var dir = outputDir;

        var loaded = await Task.Run(() =>
        {
            var accounting = Ms2SignalAccounting.ReadCached(dir);
            if (accounting is null || accounting.IsEmpty)
                return (Accounting: (Ms2SignalAccounting.Result?)null, Scheme: (IsolationScheme?)null,
                        Tolerance: (ProductMassTolerance?)null);

            // The denominator, when a raw read has been done. Joined here so both plots and the
            // status line see the same thing.
            var totals = Ms2AcquiredSignal.ReadTotals(dir);
            if (totals.Count > 0)
                accounting = accounting.WithAcquired(totals);

            // The settings the BARS were computed with, so a profile is built on the same ones.
            var catalog = IsolationSchemeCatalog.Load(Path.Combine(dir, IsolationSchemeCatalog.FileName));
            var scheme = catalog?.Library.FirstOrDefault(
                sch => string.Equals(sch.Name, accounting.IsolationScheme, StringComparison.Ordinal));
            var tolerance = ProductMassTolerance.ParseSetting(accounting.Tolerance);
            return (Accounting: accounting, Scheme: scheme, Tolerance: tolerance);
        });

        _ms2Accounting = loaded.Accounting;
        _ms2Scheme = loaded.Scheme;
        _ms2Tolerance = loaded.Tolerance;
        _ms2OutputDir = dir;
        _ms2Loaded = true;
        _ms2Profiles.Clear();

        if (_ms2Accounting is null)
        {
            ShowMs2Message(
                "No MS2 signal accounting in this directory. Tick \"MS2 signal accounting\" on the "
                + "Settings tab and run PRISM, or point the output directory at a run that used it.");
            return;
        }

        _suppressMs2Render = true;
        try
        {
            Ms2ReplicateCombo.Items.Clear();
            foreach (var row in _ms2Accounting.Rows)
                Ms2ReplicateCombo.Items.Add(ReplicateDataFiles.ReplicateOf(row.Sample));
            if (Ms2ReplicateCombo.Items.Count > 0)
                Ms2ReplicateCombo.SelectedIndex = 0;
            if (Ms2ViewCombo.SelectedIndex < 0)
                Ms2ViewCombo.SelectedIndex = 0;
        }
        finally
        {
            _suppressMs2Render = false;
        }

        UpdateMs2Controls();
        await RenderMs2Async();
    }

    /// <summary>Draw whichever view is selected.</summary>
    private async Task RenderMs2Async()
    {
        if (_ms2Accounting is null)
            return;

        if (!Ms2ProfileSelected)
        {
            RenderMs2Accounting(_ms2Accounting);
            return;
        }
        await RenderMs2ProfileAsync(_ms2Accounting);
    }

    private void RenderMs2Accounting(Ms2SignalAccounting.Result result)
    {
        Ms2Plot.Reset();
        PlotRenderer.DrawMs2Accounting(
            Ms2Plot.Plot, result,
            result.Measure == Ms2SignalMeasure.Ions
                ? "MS2 ions assigned to peptides (shared signal counted once)"
                : "MS2 signal assigned to peptides (shared signal counted once)");
        Ms2Plot.Refresh();
        Ms2StatusText.Text = DescribeMs2(result);
    }

    /// <summary>
    /// Build and draw the profile for the selected replicate. The build is a scan of that
    /// replicate's slice of the merged data, so it goes off the UI thread, and a newer request
    /// supersedes one still running - a user clicking down the replicate list must not be shown
    /// whichever scan happens to finish last.
    /// </summary>
    private async Task RenderMs2ProfileAsync(Ms2SignalAccounting.Result result)
    {
        var index = Ms2ReplicateCombo.SelectedIndex;
        if (index < 0 || index >= result.Rows.Count)
        {
            ShowMs2Message("Pick a replicate to profile.");
            return;
        }
        var sample = result.Rows[index].Sample;
        var replicate = ReplicateDataFiles.ReplicateOf(sample);

        if (_ms2Scheme is null || _ms2Tolerance is null)
        {
            ShowMs2Message(
                $"The profile needs the isolation scheme (\"{result.IsolationScheme}\") and the "
                + $"extraction tolerance ({result.Tolerance}) the accounting used. This directory no "
                + "longer has isolation_schemes.xml, so only the per-replicate view is available.");
            return;
        }

        var binWidth = Ms2BinWidth();
        if (_ms2Profiles.TryGetValue(sample, out var cached))
        {
            DrawMs2Profile(cached, result, sample, replicate);
            return;
        }

        var request = ++_ms2Request;
        Ms2StatusText.Text = $"Building the profile for {replicate}...";
        var dir = _ms2OutputDir!;
        var scheme = _ms2Scheme;
        var tolerance = _ms2Tolerance;
        var lists = _proteinLists.Lists.Where(l => l.Visible).ToList();
        var measure = result.Measure;

        var profile = await Task.Run(() => Ms2SignalProfiler.ForReplicate(
            dir, sample, scheme, tolerance, lists, binWidth, log: null, measure: measure));

        if (request != _ms2Request)
            return;   // the user moved on; a later request owns the plot

        if (profile is null)
        {
            ShowMs2Message(
                $"No MS2 regions for {replicate} - the merged data for this run may have been "
                + "cleaned up. The per-replicate view still works from the cached accounting.");
            return;
        }

        _ms2Profiles[sample] = profile;
        DrawMs2Profile(profile, result, sample, replicate);
    }

    private void DrawMs2Profile(
        Ms2SignalProfile profile, Ms2SignalAccounting.Result result, string sample, string replicate)
    {
        Ms2Plot.Reset();
        PlotRenderer.DrawMs2RtProfile(Ms2Plot.Plot, profile, replicate);
        Ms2Plot.Refresh();

        var row = result.Rows.FirstOrDefault(r => r.Sample == sample);
        Ms2StatusText.Text = row is not null && double.IsFinite(row.AcquiredFraction)
            ? $"{replicate}: {row.AcquiredFraction:P1} of acquired MS2 assigned to a peptide."
            : $"{replicate}: assigned signal only - no instrument file has been read for it, so there "
              + "is no acquired trace to compare against.";
    }

    /// <summary>The status line for the cohort view, which has to say whether a denominator exists.</summary>
    private static string DescribeMs2(Ms2SignalAccounting.Result result)
    {
        var what = result.Measure == Ms2SignalMeasure.Ions ? "ion counts" : "integrated signal";
        if (!result.HasAcquired)
            return $"{result.Rows.Count:N0} replicates, {what}, {result.AssignedPeptides:N0} peptides. "
                + "No instrument files have been read, so these are assigned totals - not a fraction "
                + "of what was acquired. Tick \"read the instrument files\" on the Settings tab.";

        var withDenominator = result.Rows.Count(r => double.IsFinite(r.AcquiredFraction));
        return $"{result.Rows.Count:N0} replicates, {what}, {result.AssignedPeptides:N0} peptides. "
            + $"Median {result.MedianAcquiredFraction():P1} of acquired MS2 assigned"
            + (withDenominator == result.Rows.Count
                ? "."
                : $", over the {withDenominator:N0} replicate(s) whose data file was read.");
    }

    /// <summary>Bin width from the box, falling back to the default (and rewriting the box) on bad input.</summary>
    private double Ms2BinWidth()
    {
        var text = Ms2BinBox.Text?.Trim();
        if (double.TryParse(text, NumberStyles.Float, CultureInfo.CurrentCulture, out var value)
            && value > 0)
            return value;
        Ms2BinBox.Text = Ms2SignalProfile.DefaultBinWidthMin.ToString(CultureInfo.CurrentCulture);
        return Ms2SignalProfile.DefaultBinWidthMin;
    }

    // Message in place of a plot, through the shared empty state so this pane looks like the others.
    private void ShowMs2Message(string message)
    {
        Ms2StatusText.Text = message;
        Ms2Plot.Reset();
        PlotRenderer.DrawEmptyState(Ms2Plot.Plot, message);
        Ms2Plot.Refresh();
    }

    /// <summary>The Tag of a ComboBox's selected item, or null.</summary>
    private static string? ComboTag(ComboBox combo) =>
        (combo?.SelectedItem as ComboBoxItem)?.Tag as string;
}
