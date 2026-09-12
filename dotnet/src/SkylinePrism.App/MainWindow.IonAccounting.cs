using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using ScottPlot;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.Visualization;

namespace SkylinePrism.App;

/// <summary>
/// The "Ion accounting" pane: how many ions reached the detector in each replicate, and what share
/// of them a peptide sequence explains - for the whole cohort, and across the gradient.
/// </summary>
/// <remarks>
/// <para><b>A pure file reader, and that is the point.</b> Measuring these numbers reads every
/// instrument file in the cohort - a terabyte on a network share for the cohort this was built for,
/// at about three and a half minutes a file. So the measurement is a separate step
/// (<c>prism ion-accounting</c>, or the button on the Settings tab) and this pane only reads its two
/// parquet files. Switching replicate, level, view or bin width is therefore instant, which is what
/// makes it worth making interactive at all.</para>
///
/// <para><b>Absent is not zero.</b> Without <c>ion_accounting.parquet</c> there is no denominator,
/// and every plot here needs one - so the nav entry is hidden rather than offering a pane that
/// cannot draw anything. That is deliberate: a fraction with a guessed denominator reads as coverage
/// and is not.</para>
/// </remarks>
public partial class MainWindow
{
    private bool _ionLoaded;                  // the cache matches the current output directory
    private bool _suppressIonRender;          // set while populating combos
    private string? _ionOutputDir;
    private IonAccountingResult? _ionResult;
    private int _ionRequest;                  // newest query wins if the user clicks ahead of it
    private string? _ionNavProbedDir;         // directory whose measured-or-not is already known
    private int _ionNavProbe;                 // newest nav probe wins

    /// <summary>
    /// Cycle traces already read, by sample. The file holds the whole cohort's cycles in one table,
    /// so a second replicate is a filter over an open reader rather than a new measurement - but
    /// keeping them means flipping back is free.
    /// </summary>
    private readonly Dictionary<string, IReadOnlyList<IonCycleRow>> _ionCycles =
        new(StringComparer.Ordinal);

    private const double DefaultIonBinMinutes = 1.0;

    /// <summary>
    /// The rows in the order the bars were DRAWN, which is what the hover readout indexes into. Held
    /// rather than re-derived: the readout has to name the bar actually under the cursor, and a
    /// second sort that disagreed with the drawn one by a single position would attribute every
    /// replicate to its neighbour - wrong in a way that looks entirely plausible.
    /// </summary>
    private IReadOnlyList<IonAccountingRow> _ionDrawn = Array.Empty<IonAccountingRow>();

    /// <summary>Drop the cached result so the pane reloads next time it is shown.</summary>
    private void InvalidateIonAccounting()
    {
        _ionLoaded = false;
        _ionResult = null;
        _ionCycles.Clear();
        _ionDrawn = Array.Empty<IonAccountingRow>();
        // The nav entry's answer is cached against the directory, so a run that has just WRITTEN
        // the accounting has to be able to make it appear.
        _ionNavProbedDir = null;
    }

    /// <summary>
    /// The <c>Tag</c> of a combo's selected item, which is how these combos carry a stable key
    /// independent of the text shown. Lived beside the MS2 signal pane until that was removed.
    /// </summary>
    private static string? ComboTag(ComboBox combo) =>
        (combo?.SelectedItem as ComboBoxItem)?.Tag as string;

    private string IonView => ComboTag(IonViewCombo) ?? "Accounting";

    private bool IonProfileSelected =>
        IonView is "Profile" or "Fraction";

    private PlotRenderer.IonQuantity IonQuantity =>
        string.Equals(ComboTag(IonQuantityCombo), "Signal", StringComparison.Ordinal)
            ? PlotRenderer.IonQuantity.Signal
            : PlotRenderer.IonQuantity.Ions;

    private IonRowOrder.By IonSort =>
        Enum.TryParse<IonRowOrder.By>(ComboTag(IonSortCombo), out var by)
            ? by
            : IonRowOrder.By.RunOrder;

    private PlotRenderer.IonLevel IonLevel =>
        string.Equals(ComboTag(IonLevelCombo), "Ms1", StringComparison.Ordinal)
            ? PlotRenderer.IonLevel.Ms1
            : PlotRenderer.IonLevel.Ms2;

    // ------------------------------------------------------------------ event handlers

    private async void OnIonViewChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            // Fires from EndInit if a SelectedIndex is ever set in XAML, before the controls below
            // exist. See XamlInitializationOrderTests.
            if (!IsInitialized || _suppressIonRender)
                return;
            UpdateIonControls();
            await RenderIonAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnIonViewChanged), ex);
        }
    }

    private async void OnIonQuantityChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _suppressIonRender)
                return;
            await RenderIonAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnIonQuantityChanged), ex);
        }
    }

    private async void OnIonSortChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _suppressIonRender)
                return;
            await RenderIonAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnIonSortChanged), ex);
        }
    }

    private async void OnIonLevelChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _suppressIonRender)
                return;
            await RenderIonAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnIonLevelChanged), ex);
        }
    }

    private async void OnIonReplicateChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _suppressIonRender)
                return;
            await RenderIonAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnIonReplicateChanged), ex);
        }
    }

    private async void OnIonBinChanged(object sender, RoutedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _suppressIonRender)
                return;
            await RenderIonAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnIonBinChanged), ex);
        }
    }

    private void OnIonBinKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key != Key.Enter)
            return;
        // Move focus off the box so LostFocus does the work - one path to the re-render rather than
        // two that can disagree.
        Keyboard.ClearFocus();
        IonPlot.Focus();
    }

    private async void OnIonReload(object sender, RoutedEventArgs e)
    {
        try
        {
            InvalidateIonAccounting();
            await LoadIonAccountingAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnIonReload), ex);
        }
    }

    // ------------------------------------------------------------------ loading

    /// <summary>
    /// Whether the output directory carries measured ion accounting. Drives the nav entry: every plot
    /// on the pane needs the denominator, so with no cache there is nothing to show.
    /// </summary>
    /// <remarks>
    /// <para><b>The probe runs off the UI thread, and its answer is cached against the directory.</b>
    /// It is a file system call, the output directory is normally a network share, and this is
    /// called from the output box's <c>TextChanged</c> - so it used to be one blocking SMB round
    /// trip per keystroke, plus one per pane change. Against a healthy share that is a millisecond
    /// and invisible; against one that is slow, disconnected, or holding a stale credential, every
    /// one of those blocks the dispatcher for the SMB timeout and the whole window stops responding
    /// with nothing on screen to say why.</para>
    /// </remarks>
    private async void UpdateIonNavVisibility()
    {
        try
        {
            var dir = OutputDirBox.Text?.Trim();
            if (string.Equals(_ionNavProbedDir, dir, StringComparison.OrdinalIgnoreCase))
                return;

            var request = ++_ionNavProbe;
            var available = !string.IsNullOrWhiteSpace(dir)
                && await Task.Run(() => File.Exists(Path.Combine(dir!, IonAccountingStore.FileName)));
            // The box moved on while the share was thinking.
            if (request != _ionNavProbe)
                return;

            _ionNavProbedDir = dir;
            IonNavItem.Visibility = available ? Visibility.Visible : Visibility.Collapsed;
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(UpdateIonNavVisibility), ex);
        }
    }

    /// <remarks>
    /// <b>Every file system call this pane makes to open is in the one background read below.</b>
    /// The replicate list used to be read on the UI thread, from <c>PopulateIonReplicates</c>, AFTER
    /// this method had already awaited - so the window froze for the length of a second read of a
    /// file that is normally on a network share, showing "Reading the ion accounting..." and
    /// responding to nothing. Anything added here that touches the disk belongs inside the
    /// <c>Task.Run</c>, not after it.
    /// </remarks>
    private async Task LoadIonAccountingAsync()
    {
        var outputDir = OutputDirBox.Text?.Trim();
        if (string.IsNullOrWhiteSpace(outputDir))
        {
            ShowIonMessage("Set the output directory above to a finished PRISM run.");
            return;
        }
        if (_ionLoaded && string.Equals(_ionOutputDir, outputDir, StringComparison.OrdinalIgnoreCase))
            return;

        ShowIonMessage("Reading the ion accounting...");
        var dir = outputDir;

        var probe = await Task.Run(() =>
        {
            if (!Directory.Exists(dir))
            {
                return (Exists: false, Result: (IonAccountingResult?)null,
                        Samples: (IReadOnlyList<string>)Array.Empty<string>());
            }
            var read = IonAccountingStore.Read(dir);
            // The replicate list is a second trip to the same share, and only the profile views
            // need it - so it is skipped entirely when there is nothing to plot.
            var samples = read is null || read.Rows.Count == 0
                ? (IReadOnlyList<string>)Array.Empty<string>()
                : IonAccountingStore.SamplesWithCycles(dir);
            return (Exists: true, Result: read, Samples: samples);
        });

        // The output directory can have moved on while a slow share was read.
        if (!string.Equals(OutputDirBox.Text?.Trim(), dir, StringComparison.OrdinalIgnoreCase))
            return;

        if (!probe.Exists)
        {
            ShowIonMessage("Set the output directory above to a finished PRISM run.");
            return;
        }

        _ionResult = probe.Result;
        _ionOutputDir = dir;
        _ionLoaded = true;
        _ionCycles.Clear();

        if (_ionResult is null || _ionResult.Rows.Count == 0)
        {
            // Names the ONLY way to produce it. An earlier version of this string offered "the
            // Settings tab", which does not have a control for it - sending a user to look for
            // something that is not there is worse than saying plainly that this is a command.
            ShowIonMessage(
                "No ion accounting in this directory yet. Measure it with:  prism ion-accounting "
                + "-d <this directory> -r <raw file directory> --product-tolerance \"10 ppm\" "
                + "--precursor-tolerance \"10 ppm\"  (use your document's own Full-Scan values). "
                + "It reads every instrument file once, a few minutes per file, and the plots "
                + "appear here once it has.");
            return;
        }

        // Set here rather than in XAML: a Selector raises SelectionChanged from EndInit when its
        // selection is set in markup, which runs while the controls below it are still null. That
        // shipped as a startup crash once - see XamlInitializationOrderTests.
        _suppressIonRender = true;
        try
        {
            if (IonViewCombo.SelectedIndex < 0)
                IonViewCombo.SelectedIndex = 0;
            if (IonLevelCombo.SelectedIndex < 0)
                IonLevelCombo.SelectedIndex = 0;
            if (IonSortCombo.SelectedIndex < 0)
                IonSortCombo.SelectedIndex = 0;
            if (IonQuantityCombo.SelectedIndex < 0)
                IonQuantityCombo.SelectedIndex = 0;
        }
        finally
        {
            _suppressIonRender = false;
        }

        PopulateIonReplicates(probe.Samples);
        UpdateIonControls();
        await RenderIonAsync();
    }

    /// <summary>
    /// The replicate picker, holding only replicates that HAVE cycles cached. A replicate whose file
    /// could not be read has a row in the summary and no trace, so offering it would be an empty plot.
    /// </summary>
    /// <param name="withCycles">
    /// Read by the caller's background pass. Passed in rather than read here: this runs on the UI
    /// thread, and reading it here blocked the dispatcher on a network file.
    /// </param>
    private void PopulateIonReplicates(IReadOnlyList<string> withCycles)
    {
        _suppressIonRender = true;
        try
        {
            var previous = IonReplicateCombo.SelectedItem as string;
            IonReplicateCombo.Items.Clear();
            foreach (var sample in withCycles.OrderBy(s => s, StringComparer.Ordinal))
                IonReplicateCombo.Items.Add(sample);

            if (IonReplicateCombo.Items.Count == 0)
                return;

            // Keep the user's replicate across a reload where it survived.
            var index = previous is null ? -1 : IonReplicateCombo.Items.IndexOf(previous);
            IonReplicateCombo.SelectedIndex = index >= 0 ? index : RepresentativeIndex(withCycles);
        }
        finally
        {
            _suppressIonRender = false;
        }
    }

    /// <summary>
    /// Which replicate to open on: the MEDIAN by assigned share, not the first alphabetically.
    /// </summary>
    /// <remarks>
    /// The first replicate of a cohort is an arbitrary choice that happens to be the one a user reads
    /// as typical. The median actually is typical, and the best and worst are one click away.
    /// </remarks>
    private int RepresentativeIndex(IReadOnlyList<string> withCycles)
    {
        var representatives = _ionResult?.Representatives();
        if (representatives is null || representatives.Count == 0)
            return 0;

        // Representatives() returns best, median, worst; the median is the middle one when there are
        // three and the only sensible pick when there are fewer.
        var median = representatives.Count >= 3 ? representatives[1] : representatives[0];
        var index = IonReplicateCombo.Items.IndexOf(median.Sample);
        return index >= 0 ? index : 0;
    }

    /// <summary>Only the controls the selected view uses are live, so none of them lies.</summary>
    private void UpdateIonControls()
    {
        var profile = IonProfileSelected;
        var visibility = profile ? Visibility.Visible : Visibility.Collapsed;
        IonReplicateLabel.Visibility = visibility;
        IonReplicateCombo.Visibility = visibility;
        IonBinLabel.Visibility = visibility;
        IonBinBox.Visibility = visibility;

        // The order only means anything where there is a bar per replicate; the profile views have
        // one replicate and a retention-time axis.
        var forBars = profile ? Visibility.Collapsed : Visibility.Visible;
        IonSortLabel.Visibility = forBars;
        IonSortCombo.Visibility = forBars;
    }

    /// <summary>
    /// Says so when run order was ASKED FOR and could not be given. Silently falling back to file
    /// name would present an arbitrary order as an acquisition one, which is the reading this plot
    /// is most likely to be used for.
    /// </summary>
    private string IonOrderNote(IonAccountingResult result) =>
        IonSort == IonRowOrder.By.RunOrder && !IonRowOrder.CanOrderByRun(result.Rows)
            ? "; ordered by file name - this cache does not record when each replicate was acquired, "
              + "which a re-measure would add"
            : "";

    /// <summary>
    /// Name the bar under the cursor. The plot has one bar per replicate and no room to label them,
    /// so without this there is no way at all to tell which replicate is which.
    /// </summary>
    private void OnIonPlotMouseMove(object sender, MouseEventArgs e)
    {
        try
        {
            if (_ionDrawn.Count == 0)
                return;

            var position = e.GetPosition(IonPlot);
            var scale = IonPlot.DisplayScale;
            var coordinates = IonPlot.Plot.GetCoordinates(
                new Pixel(position.X * scale, position.Y * scale));

            // Bars sit at integer positions 0..n-1, so the nearest whole number is the one under
            // the cursor - and only when the cursor is actually within its half-width.
            var index = (int)Math.Round(coordinates.X);
            if (index < 0 || index >= _ionDrawn.Count || Math.Abs(coordinates.X - index) > 0.5)
            {
                IonHoverText.Text = "";
                return;
            }
            IonHoverText.Text = DescribeIonBar(_ionDrawn[index], IonLevel, IonQuantity);
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnIonPlotMouseMove), ex);
        }
    }

    /// <summary>
    /// One replicate, in the terms of the level AND the quantity being shown.
    /// </summary>
    /// <remarks>
    /// The quantity matters here as much as on the axis: the ion count and the summed TIC are
    /// different numbers with different fractions, so a readout that always reported ions would
    /// quietly disagree with the bar it is naming.
    /// </remarks>
    private static string DescribeIonBar(
        IonAccountingRow row, PlotRenderer.IonLevel level, PlotRenderer.IonQuantity quantity)
    {
        var name = row.Sample;
        if (!row.IsUsable)
        {
            return $"{name}: not measured ({row.Status})"
                + (string.IsNullOrWhiteSpace(row.DataFile) ? "" : $" - {row.DataFile}");
        }

        var ms1 = level == PlotRenderer.IonLevel.Ms1;
        var signal = quantity == PlotRenderer.IonQuantity.Signal;
        if (signal && !row.HasSignal)
            return $"{name}: no summed TIC in this cache - re-measure it, or switch back to Ions";

        var acquired = (ms1, signal) switch
        {
            (true, false) => row.Ms1Acquired,
            (true, true) => row.Ms1Signal,
            (false, false) => row.Ms2Acquired,
            _ => row.Ms2Signal,
        };
        var assigned = (ms1, signal) switch
        {
            (true, false) => row.Ms1Assigned,
            (true, true) => row.Ms1SignalAssigned,
            (false, false) => row.Ms2Assigned,
            _ => row.Ms2SignalAssigned,
        };
        var fraction = (ms1, signal) switch
        {
            (true, false) => row.Ms1Fraction,
            (true, true) => row.Ms1SignalFraction,
            (false, false) => row.Ms2Fraction,
            _ => row.Ms2SignalFraction,
        };
        var explained = signal ? row.Ms2SignalExplained : row.Ms2Explained;
        var explainedFraction =
            signal ? row.Ms2SignalExplainedFraction : row.Ms2ExplainedFraction;

        var text = $"{name}"
            + (string.IsNullOrWhiteSpace(row.SampleType) ? "" : $" ({row.SampleType})")
            + $": acquired {acquired:0.###e+0}, "
            + (level == PlotRenderer.IonLevel.Ms2 && row.HasExplained ? "quantified " : "assigned ")
            + $"{assigned:0.###e+0}";
        // Exceeded means a defect, and no fraction is shown anywhere else either - see
        // IonAccountingRow.Exceeded.
        text += row.Exceeded
            ? " (assigned exceeds acquired - no fraction shown)"
            : double.IsFinite(fraction) ? $" = {fraction * 100:0.##}%" : "";

        if (level == PlotRenderer.IonLevel.Ms2 && row.HasExplained
            && double.IsFinite(explainedFraction) && !row.Exceeded)
        {
            text += $"; explained {explained:0.###e+0} = {explainedFraction * 100:0.##}%";
        }

        if (row.AcquiredUtc is { } when)
            text += $"; acquired {when.ToLocalTime():yyyy-MM-dd HH:mm}";
        if (!string.IsNullOrWhiteSpace(row.DataFile))
            text += $"; {row.DataFile}";
        return text;
    }

    // ------------------------------------------------------------------ rendering

    private async Task RenderIonAsync()
    {
        if (_ionResult is null || _ionResult.Rows.Count == 0)
            return;

        var level = IonLevel;
        var result = _ionResult;

        if (!IonProfileSelected)
        {
            // Sorted HERE and not in the renderer: the pane has to keep the order it drew to
            // answer the hover, and a renderer that sorted privately would leave it guessing.
            var ordered = IonRowOrder.Sort(result.Rows, IonSort);
            _ionDrawn = ordered;
            PlotRenderer.DrawIonAccounting(
                IonPlot.Plot, result with { Rows = ordered }, level,
                IonBarTitle(result, level), 1.0, IonQuantity);
            IonPlot.Refresh();
            IonStatusText.Text =
                DescribeIon(result, level) + IonOrderNote(result) + IonQuantityNote(result);
            IonHoverText.Text = "";
            return;
        }

        _ionDrawn = Array.Empty<IonAccountingRow>();
        IonHoverText.Text = "";

        if (IonReplicateCombo.SelectedItem is not string sample)
        {
            ShowIonMessage("No replicate has cached cycles to profile.");
            return;
        }

        // Newest request wins: a user clicking through replicates must not have an earlier read
        // overwrite the plot after the later one finished.
        var request = ++_ionRequest;
        var dir = _ionOutputDir!;

        if (!_ionCycles.TryGetValue(sample, out var cycles))
        {
            IonStatusText.Text = $"Reading {sample}...";
            cycles = await Task.Run(() => IonAccountingStore.ReadCycles(dir, sample));
            if (request != _ionRequest)
                return;
            _ionCycles[sample] = cycles;
        }

        var bin = IonBinMinutes();
        var quantity = IonQuantity;
        var noun = quantity == PlotRenderer.IonQuantity.Signal ? "signal (TIC)" : "ions";
        if (string.Equals(IonView, "Fraction", StringComparison.Ordinal))
        {
            PlotRenderer.DrawIonFractionProfile(
                IonPlot.Plot, cycles, level, bin,
                ShareTitle(sample, level, result, noun), 1.0, quantity);
        }
        else
        {
            PlotRenderer.DrawIonProfile(
                IonPlot.Plot, cycles, level, bin,
                $"{sample}: {level.ToString().ToUpperInvariant()} {noun} across the gradient",
                1.0, quantity);
        }
        IonPlot.Refresh();
        IonStatusText.Text = DescribeIonReplicate(result, sample, level, cycles.Count)
            + IonQuantityNote(result);
    }

    private double IonBinMinutes()
    {
        var text = IonBinBox.Text?.Trim();
        if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
            && value > 0)
        {
            return value;
        }
        // A bad value is put right in the box rather than silently reinterpreted, so the plot and
        // the control agree about what was drawn.
        IonBinBox.Text = DefaultIonBinMinutes.ToString(CultureInfo.InvariantCulture);
        return DefaultIonBinMinutes;
    }

    /// <summary>
    /// The status line for the cohort view. Names the settings the numbers were computed with,
    /// because the cache is keyed on them and a reader has no other way to know.
    /// </summary>
    /// <summary>
    /// The bar plot's title. The renderer appends the medians; this is the noun phrase above them,
    /// and it has to stop saying "assigned" the moment there are two numerators to tell apart.
    /// </summary>
    private string IonBarTitle(IonAccountingResult result, PlotRenderer.IonLevel level)
    {
        var noun = IonQuantity == PlotRenderer.IonQuantity.Signal ? "Signal (TIC)" : "Ions";
        return level == PlotRenderer.IonLevel.Ms2 && result.Rows.Any(r => r.HasExplained)
            ? $"{noun} acquired, quantified and explained, per replicate"
            : $"{noun} acquired and assigned, per replicate";
    }

    /// <summary>
    /// Says so when the summed TIC was asked for and this cache does not carry it. Drawing zeros
    /// would read as a run that acquired nothing, which is the one thing the plot must not say.
    /// </summary>
    private string IonQuantityNote(IonAccountingResult result) =>
        IonQuantity == PlotRenderer.IonQuantity.Signal && !result.Rows.Any(r => r.HasSignal)
            ? "; WARNING: this cache carries no summed TIC - it was measured before signal was "
              + "recorded, so re-measure it or switch back to Ions"
            : "";

    /// <inheritdoc cref="IonBarTitle"/>
    private static string ShareTitle(
        string sample, PlotRenderer.IonLevel level, IonAccountingResult result, string noun)
    {
        var name = level.ToString().ToUpperInvariant();
        return level == PlotRenderer.IonLevel.Ms2 && result.Rows.Any(r => r.HasExplained)
            ? $"{sample}: share of acquired {name} {noun} quantified and explained"
            : $"{sample}: share of acquired {name} {noun} assigned";
    }

    private static string DescribeIon(IonAccountingResult result, PlotRenderer.IonLevel level)
    {
        var usable = result.Rows.Where(r => r.IsUsable).ToArray();
        var parts = new List<string>
        {
            $"{usable.Length:N0} of {result.Rows.Count:N0} replicate(s) measured",
            $"product {result.ProductTolerance}",
            $"precursor {result.PrecursorTolerance}",
            $"scheme {result.IsolationScheme}",
            $"{result.AssignedPeptides:N0} peptides claiming signal",
        };

        var offScale = usable.Count(r => r.IonScaleImplausible);
        if (offScale > 0)
        {
            // The fraction is still right when the unit is wrong, so nothing else on the pane will
            // look amiss. See IonAccountingRecord.IonScaleImplausible.
            parts.Add(
                $"WARNING: {offScale:N0} replicate(s) report an impossible number of ions per scan, "
                + "so the totals are in the wrong unit (the fractions are unaffected)");
        }

        var exceeded = usable.Count(r => r.Exceeded);
        if (exceeded > 0)
        {
            parts.Add(
                $"WARNING: {exceeded:N0} replicate(s) assigned more than was acquired, which is "
                + "impossible - no fraction is shown");
        }
        else
        {
            var fractions = usable
                .Select(r => level == PlotRenderer.IonLevel.Ms1 ? r.Ms1Fraction : r.Ms2Fraction)
                .Where(double.IsFinite)
                .ToArray();
            if (fractions.Length > 0)
            {
                var name = level.ToString().ToUpperInvariant();
                var explained = usable
                    .Where(r => r.HasExplained)
                    .Select(r => r.Ms2ExplainedFraction)
                    .Where(double.IsFinite)
                    .ToArray();

                // "quantified" only once there is a second number to tell it apart from; on its own
                // the old wording says the same thing and is what every earlier report used.
                parts.Add(
                    (level == PlotRenderer.IonLevel.Ms2 && explained.Length > 0
                        ? $"{name} quantified share "
                        : $"{name} assigned share ")
                    + $"{IonAccountingStore.Percent(fractions.Min())} to "
                    + $"{IonAccountingStore.Percent(fractions.Max())}");

                if (level == PlotRenderer.IonLevel.Ms2 && explained.Length > 0)
                {
                    parts.Add(
                        $"explained share {IonAccountingStore.Percent(explained.Min())} to "
                        + $"{IonAccountingStore.Percent(explained.Max())}");
                }
            }
        }

        var noFile = result.Rows.Count(r => string.IsNullOrEmpty(r.DataFile));
        if (noFile > 0)
            parts.Add($"{noFile:N0} replicate(s) had no data file of their own");

        return string.Join(" | ", parts);
    }

    private static string DescribeIonReplicate(
        IonAccountingResult result, string sample, PlotRenderer.IonLevel level, int cycles)
    {
        var row = result.Rows.FirstOrDefault(
            r => string.Equals(r.Sample, sample, StringComparison.Ordinal));
        if (row is null)
            return $"{cycles:N0} cycles";

        var fraction = level == PlotRenderer.IonLevel.Ms1 ? row.Ms1Fraction : row.Ms2Fraction;
        var parts = new List<string>
        {
            $"{cycles:N0} cycles",
            $"{row.Ms1Count:N0} MS1 and {row.Ms2Count:N0} MS2 spectra",
            $"{row.Claims:N0} claimed regions",
        };
        var levelName = level.ToString().ToUpperInvariant();
        if (row.Exceeded)
        {
            parts.Add("WARNING: assigned exceeds acquired, so no share is shown");
        }
        else if (level == PlotRenderer.IonLevel.Ms2 && row.HasExplained)
        {
            parts.Add(
                $"{levelName} quantified {IonAccountingStore.Percent(fraction)}, "
                + $"explained {IonAccountingStore.Percent(row.Ms2ExplainedFraction)}");
        }
        else
        {
            parts.Add($"{levelName} share {IonAccountingStore.Percent(fraction)}");
        }
        if (row.ScansOutsideScheme > 0)
        {
            parts.Add(
                $"{row.ScansOutsideScheme:N0} scans fell in no isolation window of the scheme");
        }
        if (row.SpectraMissingInjectionTime > 0)
        {
            parts.Add(
                $"{row.SpectraMissingInjectionTime:N0} scans reported no ion injection time");
        }
        return string.Join(" | ", parts);
    }

    /// <summary>
    /// Clear the plot to an explicit message. Goes through <see cref="PlotRenderer.DrawEmptyState"/>
    /// so an empty pane looks like the other panes' empty state rather than like a broken plot.
    /// </summary>
    private void ShowIonMessage(string message)
    {
        _ionDrawn = Array.Empty<IonAccountingRow>();
        PlotRenderer.DrawEmptyState(IonPlot.Plot, message);
        IonPlot.Refresh();
        IonStatusText.Text = message;
        IonHoverText.Text = "";
    }
}
