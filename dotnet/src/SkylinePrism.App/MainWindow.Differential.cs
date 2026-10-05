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
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.Visualization;
using SkylinePrism.Skyline;

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

    /// <summary>
    /// Marker sizes for the scatter views (Volcano and the two Detection plots), the significant
    /// series a little larger so it still reads as the emphasized one.
    /// </summary>
    /// <remarks>
    /// Raised from 6/7. At that size a point on a full-screen Volcano was a two-pixel speck: hard to
    /// see at all on a sparse plot, and much smaller than the 18 px
    /// <see cref="QcPlotChrome.HoverRadiusPx"/> that decides what a click lands on - so the target
    /// was far bigger than the thing it was aiming at, which reads as a plot that does not respond.
    /// Named rather than repeated at each call site, because the three views must agree for the
    /// hover radius to mean the same thing on each.
    /// </remarks>
    private const float DiffPointSize = 9;

    /// <summary><see cref="DiffPointSize"/> for the significant series.</summary>
    private const float DiffSigPointSize = 11;

    private DifferentialDataset? _diffDataset;
    private Dictionary<string, string> _diffLabelById = new(StringComparer.Ordinal);

    /// <summary>Feature id -> gene symbol(s), for enrichment only. No entry where none is known.</summary>
    private Dictionary<string, string> _diffGeneById = new(StringComparer.Ordinal);
    private DetectionMatrixData? _detectionData;
    private string? _detectionDir;
    private string? _clinicalCsvPath;
    private List<QcGroupValue> _diffCovariateValues = new();

    // The two contrast arms, as tick lists. An arm is a SET of metadata values whose samples are
    // pooled, not a single value - which is what lets two control classes be contrasted against the
    // rest as one arm.
    private List<QcGroupValue> _diffAValues = new();
    private List<QcGroupValue> _diffBValues = new();
    private HttpJsonPoster? _diffPoster;
    private bool _diffSuppress;
    private int _diffRequest;

    /// <summary>
    /// Generation counter for the VIEW computations, separate from the loader's.
    /// </summary>
    /// <remarks>
    /// The loader has always had one; the views did not, and every selector in this pane now
    /// re-runs on change - design, test, prior, correction, level, view - so two contrasts are
    /// routinely in flight at once. Without a token the SLOWER one wins whenever it happens to
    /// finish last, painting a plot and a hit table that the controls no longer describe. Checked
    /// after every await, because that is where another run can have started.
    /// </remarks>
    private int _diffViewRequest;
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

    // Hover readout and selection ring, both created hidden by RenderVolcano and then only moved.
    // Moving a plottable and refreshing is far cheaper than re-rendering, and it means the highlight
    // does not depend on keeping the DifferentialResult alive to redraw from.
    private ScottPlot.Plottables.Marker? _volcanoHoverMarker;
    private ScottPlot.Plottables.Text? _volcanoHoverText;
    private ScottPlot.Plottables.Marker? _volcanoSelMarker;

    /// <summary>The feature the plot ring and the grid row are both pointing at, or null.</summary>
    private string? _volcanoSelectedId;

    /// <summary>
    /// Set while one of the two selections is being driven from the other. The grid raises
    /// SelectionChanged when its SelectedItem is set in code, so without this a plot click would
    /// select the row, which would re-enter and select the point, and each hop would re-issue the
    /// Skyline selection and reopen the detail window.
    /// </summary>
    private bool _volcanoSyncing;

    private enum DiffView
    {
        Volcano,
        Detection,
        Enrichment,
    }

    private sealed record VolcanoRow(string Feature, double Log2FC, double P, double AdjP, string FeatureId);

    private sealed record DetRow(string Peptide, double RateA, double RateB, double P, double Q);

    /// <summary>
    /// A paired detection row. OnlyA and OnlyB are the DISCORDANT pair counts - the only observations
    /// McNemar's test uses - so a result resting on three pairs cannot look like one resting on
    /// thirty.
    /// </summary>
    private sealed record DetPairedRow(
        string Peptide, double RateA, double RateB, int OnlyA, int OnlyB, double P, double Q);

    private sealed record DetGlmRow(string Peptide, double RateA, double RateB, double LogOR, double P, double Q);

    private sealed record EnrichRow(string Source, string Term, double PValue, double Fold, string TermId);

    /// <summary>A significant protein that a clicked enrichment term contains.</summary>
    private sealed record TermProtein(string Protein, string Gene, double Log2FC, double AdjP);

    // Enrichment click-to-members state: the terms just shown, keyed by id, and the significant
    // features of the run behind them, keyed by gene - so a clicked term names its member proteins.
    private Dictionary<string, EnrichmentTerm> _enrichTermsById = new(StringComparer.Ordinal);
    private Dictionary<string, List<TermProtein>> _enrichFeaturesByGene =
        new(StringComparer.OrdinalIgnoreCase);
    private TermProteinsWindow? _termProteinsWindow;

    private HttpJsonPoster DiffPoster => _diffPoster ??= new HttpJsonPoster();

    /// <summary>Forget the loaded matrix so the pane reloads on its next show (new dir / new run).</summary>
    private void InvalidateDifferential()
    {
        _diffLoaded = false;
        _diffLoadedDir = null;
        _diffDataset = null;
        // A new directory is a new run with its own controls, so the prior-source default applies
        // again. Without this, opening a run WITHOUT controls and then one WITH them would leave
        // the second fitting its prior on design groups with nothing said.
        _diffPriorSourceDefaulted = false;
        _detectionData = null;
        _detectionDir = null;
    }

    private FeatureLevel DiffSelectedLevel() =>
        (DiffLevelCombo.SelectedItem as ListBoxItem)?.Content as string == "Peptide"
            ? FeatureLevel.Peptide
            : FeatureLevel.Protein;

    /// <summary>
    /// The variance prior the Prior combo is pointing at, defaulting to the lab's choice before the
    /// combo has been populated (the first render happens during window construction).
    /// </summary>
    private VariancePrior DiffSelectedPrior() =>
        ((DiffPriorCombo.SelectedItem as ComboBoxItem)?.Tag as string) switch
        {
            "Global" => VariancePrior.Global,
            "LimmaTrend" => VariancePrior.LimmaTrend,
            "PeptideCount" => VariancePrior.PeptideCount,
            _ => VariancePrior.IntensityTrend,
        };

    /// <summary>The design the Design combo is pointing at.</summary>
    private DifferentialDesign DiffSelectedDesign() =>
        ((DiffDesignCombo.SelectedItem as ListBoxItem)?.Tag as string) switch
        {
            "Paired" => DifferentialDesign.Paired,
            "LinearTrend" => DifferentialDesign.LinearTrend,
            "LinearTrendWithinSubject" => DifferentialDesign.LinearTrendWithinSubject,
            _ => DifferentialDesign.Unpaired,
        };

    /// <summary>
    /// The subject key per sample, aligned to the dataset's sample order, or null when no subject
    /// column is chosen.
    /// </summary>
    private string?[]? DiffSubjectLabels()
    {
        if (_diffDataset is null || DiffPairByCombo.SelectedItem is not string col)
            return null;
        return _diffDataset.MetadataColumns.Contains(col) ? _diffDataset.MetadataValues(col) : null;
    }

    /// <summary>The estimator the Test combo is pointing at.</summary>
    private DifferentialTest DiffSelectedTest() =>
        ((DiffTestCombo.SelectedItem as ComboBoxItem)?.Tag as string) switch
        {
            "WelchT" => DifferentialTest.WelchT,
            "StudentT" => DifferentialTest.StudentT,
            "MannWhitney" => DifferentialTest.MannWhitney,
            "PairedT" => DifferentialTest.PairedT,
            "Wilcoxon" => DifferentialTest.Wilcoxon,
            _ => DifferentialTest.ModeratedT,
        };

    /// <summary>
    /// Keep the trend-column picker in step with the run's numeric columns.
    /// </summary>
    /// <remarks>
    /// Rewritten only when the list actually changes, because this is called from
    /// <c>UpdateDiffControls</c> - which runs on every control change - and reassigning ItemsSource
    /// clears the selection. Attaching a clinical CSV is what makes the list grow mid-session, and
    /// losing the user's pick every time they touched an unrelated combo would be worse than the
    /// comparison costs.
    /// </remarks>
    private void PopulateTrendColumns(IReadOnlyList<TrendAxisOption> axes)
    {
        // Compared on everything the option READS with and shows, not the label alone: another output
        // directory can offer the same label at a different position ("Visit (Week)" out of
        // "V1_Week 4" and out of "Week 4"), and keeping the old option read every sample as NaN while
        // the preview went on showing the previous run's values.
        static string Key(TrendAxisOption a) =>
            string.Join("|", a.Label, a.Column, a.Position, a.Distinct, a.Covered, TrendAxis.DescribePreview(a));
        if (DiffTrendOverCombo.ItemsSource is IEnumerable<TrendAxisOption> current
            && current.Select(Key).SequenceEqual(axes.Select(Key), StringComparer.Ordinal))
            return;

        // Kept by LABEL, not by reference: the options are rebuilt from the metadata each time, so
        // the equal option is a different object and holding the old one would drop the selection.
        var keep = (DiffTrendOverCombo.SelectedItem as TrendAxisOption)?.Label;
        using (SuppressDiff())
        {
            DiffTrendOverCombo.ItemsSource = axes;
            // Defaults only to an axis that needed no interpreting (Position 0 - the column IS a
            // number). An axis read OUT of text is a judgement with a wrong answer available:
            // "V2_Week 8" yields the visit as readily as the week, and a pane that opened on one of
            // them would have the user fitting slopes against patient numbers without ever choosing
            // to. With none to default to, the picker opens empty and the status line asks.
            DiffTrendOverCombo.SelectedItem =
                axes.FirstOrDefault(a => string.Equals(a.Label, keep, StringComparison.Ordinal))
                ?? axes.FirstOrDefault(a => a.Position == 0);
        }
    }

    /// <summary>
    /// Show what the chosen trend axis parsed, or hide the line when there is nothing to check.
    /// </summary>
    /// <remarks>
    /// Shown rather than tucked into a tooltip because it is the only guard against the axis being
    /// the wrong number. "V2_Week 8" reads as 2 or as 8 and both are legitimate columns; a slope
    /// fitted against the visit index when the week was meant is wrong in a way no other part of the
    /// output reveals.
    /// </remarks>
    private void UpdateTrendPreview()
    {
        var axis = DiffTrendOverCombo.SelectedItem as TrendAxisOption;
        var line = axis is null ? string.Empty : TrendAxis.DescribePreview(axis);
        if (!DiffIsTrend() || line.Length == 0)
        {
            DiffTrendPreviewText.Visibility = Visibility.Collapsed;
            return;
        }

        DiffTrendPreviewText.Text = "Reading " + line;
        DiffTrendPreviewText.Visibility = Visibility.Visible;
    }

    private async void OnDiffTrendOverChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _diffSuppress || _diffDataset is null)
                return;
            if (!DiffIsTrend())
                return; // the column is only consulted by a trend design
            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffTrendOverChanged), ex);
        }
    }

    /// <summary>
    /// What fitting the prior on controls does, and why it is the default.
    /// </summary>
    /// <remarks>
    /// Held here rather than in the XAML because the disabled case prefixes it with a reason, and
    /// two copies of a paragraph this long would drift.
    /// </remarks>
    private const string PriorFromControlsHelp =
        "Fit the variance prior on the run's QC and reference replicates - the default wherever the "
        + "run has them. A design group's within-group spread contains the inter-subject biology "
        + "the analysis is looking for, so a prior fitted on it describes measurement noise plus "
        + "that biology and over-shrinks real signal; dedicated control injections are nominal "
        + "replicates, so their spread is the measurement variance the prior is meant to describe. "
        + "Only the per-feature scale comes from them - the prior degrees of freedom stay global, "
        + "estimated from the study samples, so the amount of shrinkage still matches the data "
        + "being analyzed. Usually the controls take no part in the contrast itself - they carry no "
        + "study condition and no timepoint - and the status line says so when they do. Untick to fit on the "
        + "contrast groups instead, which is only worth doing to reproduce an older result.";

    /// <summary>
    /// Suppress the pane's selector handlers until the returned scope is disposed.
    /// </summary>
    /// <remarks>
    /// Saves and RESTORES the previous value rather than setting false on the way out, because these
    /// nest: assigning a combo's SelectedIndex raises SelectionChanged synchronously, whose handler
    /// calls UpdateDiffControls, which suppresses around its own assignments. A plain
    /// <c>= false</c> at the end of the inner one ended the OUTER window too, so the remaining
    /// assignments in a load ran their handlers unsuppressed - each re-running a contrast against a
    /// half-initialized pane. Disposal also makes it exception-safe: a throw inside used to leave
    /// the whole pane suppressed for the rest of the session, with every selector silently inert.
    /// </remarks>
    private SuppressScope SuppressDiff() => new(this);

    private readonly struct SuppressScope : IDisposable
    {
        private readonly MainWindow _owner;
        private readonly bool _previous;

        public SuppressScope(MainWindow owner)
        {
            _owner = owner;
            _previous = owner._diffSuppress;
            owner._diffSuppress = true;
        }

        public void Dispose() => _owner._diffSuppress = _previous;
    }

    /// <summary>Whether the current design fits a slope rather than contrasting two arms.</summary>
    private bool DiffIsTrend() => DiffSelectedDesign()
        is DifferentialDesign.LinearTrend or DifferentialDesign.LinearTrendWithinSubject;

    /// <summary>
    /// The trend axis's name, or null when none is picked. The LABEL, not the bare column: on a
    /// column offering two readings it is the only form that says which one ran, and it is what a
    /// recorded command carries back to the CLI.
    /// </summary>
    private string? DiffTrendColumn() => DiffTrendAxis()?.Label;

    /// <summary>The picked trend axis, or null; its <see cref="TrendAxisOption.Column"/> is the metadata column tested.</summary>
    private TrendAxisOption? DiffTrendAxis() => DiffTrendOverCombo.SelectedItem as TrendAxisOption;

    /// <summary>
    /// The trend value per sample, read through the picked axis: NaN where the sample has no value on
    /// it. The Restrict-to picker does not appear here - it removes samples from
    /// <see cref="DiffTrendColumns"/> instead, so a deliberate subset is never counted as missing data.
    /// </summary>
    private double[]? DiffTrendValues() =>
        _diffDataset is null || DiffTrendOverCombo.SelectedItem is not TrendAxisOption axis
            || !_diffDataset.MetadataColumns.Contains(axis.Column)
            ? null
            : TrendAxis.Read(_diffDataset.MetadataValues(axis.Column), axis);

    /// <summary>
    /// The sample columns a trend runs over: every sample, less those the Restrict-to picker excludes.
    /// </summary>
    /// <remarks>
    /// The restriction leaves the COLUMN LIST rather than NaN-ing the x values, though both would fit
    /// the same samples. NaN reaches <see cref="TrendSamples.Resolve"/> as "no value in the trend
    /// column", so a deliberate subset was counted and reported as missing data - 46 left out, of
    /// which 30 were the user's own choice. Removing the column instead means the only samples that
    /// message ever counts are ones that genuinely have no value on the axis.
    /// </remarks>
    private int[] DiffTrendColumns() =>
        _diffDataset is null ? Array.Empty<int>() : QuantAnalysis.KeptColumns(DiffRestrictions(), _diffDataset);

    /// <summary>
    /// The Restrict-to choice as restrictions - empty for all samples - the one form both this pane's
    /// trend and its Quant report take, so the report fits the samples the pane shows and its recorded
    /// command carries the same <c>--restrict-to</c>.
    /// </summary>
    private IReadOnlyList<QuantRestriction> DiffRestrictions() =>
        DiffRestrictColumn() is { } col && DiffRestrictValues() is { Count: > 0 } keep
            ? new[] { new QuantRestriction(col, keep) }
            : Array.Empty<QuantRestriction>();

    /// <summary>The Restrict-to column, or null for "(all samples)".</summary>
    private string? DiffRestrictColumn() =>
        DiffRestrictColumnCombo.SelectedItem as string is { } s && s != RestrictNone ? s : null;

    /// <summary>The ticked Restrict-to values, or null when none are ticked.</summary>
    private List<string>? DiffRestrictValues() =>
        (DiffRestrictValuesCombo.ItemsSource as IEnumerable<QcGroupValue>)
            ?.Where(v => v.IsSelected).Select(v => v.Name).ToList() is { Count: > 0 } picked
            ? picked
            : null;

    /// <summary>The "no restriction" entry, first in the column picker.</summary>
    private const string RestrictNone = "(all samples)";

    /// <summary>
    /// Every axis a trend can be fitted against: each column whose values are numbers, and each
    /// readable position of a number inside a text column ("V2_Week 8" offers the visit and the week).
    /// </summary>
    /// <remarks>
    /// Built by <see cref="TrendAxis.AllFor"/>, the list the CLI's <c>--trend-over</c> is checked
    /// against, so a label offered here is one the command line accepts. An axis needs two DISTINCT
    /// values to carry a slope, so a constant column offers none.
    /// </remarks>
    private IReadOnlyList<TrendAxisOption> DiffTrendAxes() =>
        _diffDataset is null
            ? Array.Empty<TrendAxisOption>()
            : TrendAxis.AllFor(_diffDataset.MetadataColumns, c => _diffDataset.MetadataValues(c));

    /// <summary>
    /// Fill the Restrict-to pickers: every metadata column, and the ticked values of the chosen one.
    /// </summary>
    /// <remarks>
    /// Only under a trend, where the Group by / A / B triple is hidden - so the row does not grow,
    /// and the control appears exactly where a reader is already looking for "which samples".
    /// </remarks>
    private void UpdateRestrictControls(bool trend)
    {
        var visible = trend ? Visibility.Visible : Visibility.Collapsed;
        DiffRestrictLabel.Visibility = visible;
        DiffRestrictColumnCombo.Visibility = visible;

        if (_diffDataset is not null)
        {
            var columns = new List<string> { RestrictNone };
            columns.AddRange(_diffDataset.MetadataColumns);
            if (DiffRestrictColumnCombo.ItemsSource is not IEnumerable<string> have
                || !have.SequenceEqual(columns, StringComparer.Ordinal))
            {
                var keep = DiffRestrictColumnCombo.SelectedItem as string;
                using (SuppressDiff())
                {
                    DiffRestrictColumnCombo.ItemsSource = columns;
                    DiffRestrictColumnCombo.SelectedItem =
                        keep is not null && columns.Contains(keep, StringComparer.Ordinal) ? keep : RestrictNone;
                }
            }
        }

        var col = DiffRestrictColumn();
        DiffRestrictValuesCombo.Visibility = trend && col is not null ? Visibility.Visible : Visibility.Collapsed;
        if (col is null || _diffDataset is null)
            return;

        var distinct = _diffDataset.MetadataValues(col)
            .Where(v => !string.IsNullOrEmpty(v)).Select(v => v!)
            .Distinct(StringComparer.Ordinal).OrderBy(v => v, StringComparer.Ordinal).ToList();
        if (DiffRestrictValuesCombo.ItemsSource is IEnumerable<QcGroupValue> existing
            && existing.Select(v => v.Name).SequenceEqual(distinct, StringComparer.Ordinal))
            return;

        using (SuppressDiff())
        {
            // Changed refreshes the closed-state text as boxes are ticked, the way the arm lists do.
            DiffRestrictValuesCombo.ItemsSource = distinct
                .Select(v => new QcGroupValue { Name = v, Changed = UpdateDiffArmSummaries })
                .ToList();
            UpdateDiffArmSummaries();
        }
    }

    private async void OnDiffRestrictColumnChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _diffSuppress || _diffDataset is null)
                return;
            UpdateDiffControls();
            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffRestrictColumnChanged), ex);
        }
    }

    /// <summary>The multiple-testing correction the Correct combo is pointing at.</summary>
    private MultipleTesting DiffSelectedCorrection() =>
        ((DiffCorrectionCombo.SelectedItem as ComboBoxItem)?.Tag as string) switch
        {
            "BenjaminiYekutieli" => MultipleTesting.BenjaminiYekutieli,
            "Holm" => MultipleTesting.Holm,
            "Bonferroni" => MultipleTesting.Bonferroni,
            "None" => MultipleTesting.None,
            _ => MultipleTesting.BenjaminiHochberg,
        };

    /// <summary>
    /// The run's QC and reference sample columns, for fitting the variance prior on controls rather
    /// than on the contrast groups. Null when there are too few to fit anything.
    /// </summary>
    /// <remarks>Grouping and the two-sample floor are <see cref="ControlSampleTypes.PriorGroups"/>.</remarks>
    private IReadOnlyList<IReadOnlyList<int>>? DiffControlColumns()
    {
        if (_diffDataset is null || !_diffDataset.MetadataColumns.Contains("sample_type"))
            return null;

        return ControlSampleTypes.PriorGroups(_diffDataset.MetadataValues("sample_type"));
    }

    /// <summary>
    /// The hit rule the pane is currently showing: which p, what cut, and how big an effect.
    /// </summary>
    /// <remarks>
    /// Read on the UI thread and passed down, never re-read inside a Task.Run - touching a WPF
    /// control from a worker thread throws, and this pane has already shipped that bug once.
    /// </remarks>
    private SignificanceRule DiffRule() => new()
    {
        UseAdjusted = (DiffPKindCombo.SelectedItem as ListBoxItem)?.Tag as string != "Raw",
        // The floor is not 0: -log10(0) is +infinity, and the volcano draws its guide line at
        // exactly that, which leaves the plot with a non-finite axis limit and nothing on it. A cut
        // of zero admits no feature either, so nothing is lost by refusing to go below the
        // smallest p a double can carry.
        PThreshold = ComboNumber(DiffPCutCombo, fallback: 0.05, min: 1e-300, max: 1.0),
        Log2FcThreshold = ComboNumber(DiffEffectCutCombo, fallback: 1.0, min: 0.0, max: double.MaxValue),
    };

    /// <summary>
    /// The number an editable combo is showing, or <paramref name="fallback"/> when it is not a
    /// usable one.
    /// </summary>
    /// <remarks>
    /// Silently falling back rather than refusing, because this runs on every keystroke-completed
    /// edit and a half-typed "0.0" is a normal intermediate state, not an error worth a dialog. The
    /// status line always prints the rule actually applied, so a rejected entry is visible there
    /// rather than being swallowed. Out-of-range is clamped for the same reason: a p-value cut of 2
    /// is a typo, and admitting every feature is a worse answer than admitting the usual ones.
    /// </remarks>
    private static double ComboNumber(ComboBox combo, double fallback, double min, double max)
    {
        var text = (combo.SelectedItem as ComboBoxItem)?.Content as string ?? combo.Text;
        return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)
               && double.IsFinite(v)
            ? Math.Clamp(v, min, max)
            : fallback;
    }

    /// <summary>
    /// A threshold changed. Only the hit RULE moved, not the model, so nothing needs refitting -
    /// but re-running is what keeps every view agreeing, and a contrast is milliseconds once the
    /// matrix is loaded.
    /// </summary>
    private async void OnDiffThresholdChanged(object sender, EventArgs e)
    {
        try
        {
            if (_diffSuppress || _diffDataset is null)
                return;
            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffThresholdChanged), ex);
        }
    }

    // WPF gives these three events three different delegate types, and a handler has to match its
    // own exactly - so they are thin wrappers over the one body above rather than three copies that
    // could drift.
    private void OnDiffThresholdSelected(object sender, SelectionChangedEventArgs e)
        => OnDiffThresholdChanged(sender, e);

    private void OnDiffThresholdLostFocus(object sender, System.Windows.RoutedEventArgs e)
        => OnDiffThresholdChanged(sender, e);

    /// <summary>Enter in an editable threshold box commits it, the way tabbing away does.</summary>
    private void OnDiffThresholdKeyDown(object sender, System.Windows.Input.KeyEventArgs e)
    {
        if (e.Key != System.Windows.Input.Key.Enter)
            return;
        // Moving focus raises LostFocus, which is the one handler that runs the contrast - so the
        // two routes commit through exactly the same path rather than two that can drift.
        if (sender is System.Windows.UIElement el)
            el.MoveFocus(new System.Windows.Input.TraversalRequest(
                System.Windows.Input.FocusNavigationDirection.Next));
    }

    /// <summary>
    /// What the effect on the volcano's x-axis is called, for the status line and the hit rule.
    /// </summary>
    /// <remarks>
    /// A trend does not report a fold change between arms - it reports the modeled change across
    /// the trend column's observed range - so calling it log2FC would name the wrong quantity in
    /// the one place a reader checks what a threshold meant.
    /// </remarks>
    private string DiffEffectName() => DiffIsTrend()
        ? $"log2 change across {DiffTrendColumn() ?? "the trend"}"
        : "log2FC";

    /// <summary>The volcano's x-axis name.</summary>
    private string DiffXAxisLabel()
    {
        if (!DiffIsTrend())
            return "log2 fold change (B / A)";
        // The SHORT name ("Week"), not the label: the label is what tells two readings of one column
        // apart in the picker, but as an axis title it ran past both ends of the plot.
        var col = (DiffTrendOverCombo.SelectedItem as TrendAxisOption)?.Short is { Length: > 0 } shortName
            ? shortName
            : DiffTrendColumn() ?? "trend";
        // The observed span is in the label, because the axis is a change ACROSS it and the number
        // is meaningless without knowing across what.
        return _diffTrendRange is { } r
            ? $"log2 change across {col} ({r.Min:0.###} to {r.Max:0.###})"
            : $"log2 change across {col}";
    }

    /// <summary>The trend column's observed span in the last run, for the axis label.</summary>
    private (double Min, double Max)? _diffTrendRange;

    /// <summary>
    /// Whether the "fit prior on controls" default has been applied to the loaded dataset yet.
    /// </summary>
    /// <remarks>
    /// A default, not a lock: it is applied once when a dataset with controls is loaded, and never
    /// re-applied, so unticking it stays unticked.
    /// </remarks>
    private bool _diffPriorSourceDefaulted;

    /// <summary>What the contrast views should run: the current selections, as Core sees them.</summary>
    private DifferentialOptions DiffOptions(IReadOnlyList<Covariate>? covariates) =>
        new()
        {
            Covariates = covariates,
            Design = DiffSelectedDesign(),
            Test = DiffSelectedTest(),
            Prior = DiffSelectedPrior(),
            Correction = DiffSelectedCorrection(),
            SubjectLabels = DiffSubjectLabels(),
            TrendColumn = DiffIsTrend() ? DiffTrendColumn() : null,
            TimeValues = DiffIsTrend() ? DiffTrendValues() : null,
            PeptideCounts = _diffDataset?.PeptideCounts,
            PriorGroupColumns = DiffPriorFromControlsCheck.IsChecked == true
                ? DiffControlColumns()
                : null,
            MinPerGroup = 2,
        };

    /// <summary>
    /// Show only the controls the selected test actually uses.
    /// </summary>
    /// <remarks>
    /// The same two rules the QC pane's <c>UpdateQcControls</c> documents, for the same reasons.
    /// The variance prior is <b>hidden</b> outside the moderated t, because it means nothing there
    /// and this row is already crowded - a disabled control still invites a click. "Adjust for" is
    /// <b>grayed</b> rather than hidden, because it is a real and common setting that simply cannot
    /// be honored by a test with no design matrix; hiding it would make a ticked covariate vanish
    /// with the control, and leaving it live would imply the contrast had been adjusted when it had
    /// not.
    /// </remarks>
    private void UpdateDiffControls()
    {
        var design = DiffSelectedDesign();
        var paired = design == DifferentialDesign.Paired;
        var trend = design is DifferentialDesign.LinearTrend
            or DifferentialDesign.LinearTrendWithinSubject;
        var withinSubject = design == DifferentialDesign.LinearTrendWithinSubject;

        // A trend design needs an axis to fit against. With none in the run, both trend entries are
        // collapsed AND disabled - and if one was already selected (a clinical CSV was detached, say)
        // the design falls back rather than leaving an invisible selection.
        var axes = DiffTrendAxes();
        PopulateTrendColumns(axes);
        ShowTest(DiffDesignTrendItem, axes.Count > 0);
        ShowTest(DiffDesignTrendSubjectItem, axes.Count > 0);
        if (DiffDesignCombo.SelectedItem is ListBoxItem { IsEnabled: false })
        {
            using (SuppressDiff())
            {
                DiffDesignCombo.SelectedIndex = 0; // Unpaired
            }
            UpdateDiffControls();
            return;
        }

        // The subject column is meaningful under a paired design and a within-subject trend, and
        // nowhere else.
        var pairVisibility = paired || withinSubject ? Visibility.Visible : Visibility.Collapsed;
        DiffPairByLabel.Visibility = pairVisibility;
        DiffPairByCombo.Visibility = pairVisibility;

        // "Trend over" replaces the whole Group by / A / B triple: a trend has no arms, so leaving
        // them on screen would invite a selection that takes no part in the result.
        var trendVisibility = trend ? Visibility.Visible : Visibility.Collapsed;
        var armVisibility = trend ? Visibility.Collapsed : Visibility.Visible;
        DiffTrendOverLabel.Visibility = trendVisibility;
        DiffTrendOverCombo.Visibility = trendVisibility;
        DiffGroupByLabel.Visibility = armVisibility;
        DiffGroupByCombo.Visibility = armVisibility;
        DiffALabel.Visibility = armVisibility;
        DiffACombo.Visibility = armVisibility;
        DiffBLabel.Visibility = armVisibility;
        DiffBCombo.Visibility = armVisibility;
        UpdateRestrictControls(trend);
        UpdateTrendPreview();

        // The effect on a trend is a change across a range, not a fold change between arms.
        DiffEffectLabel.Text = trend ? "|log2 change| >=" : "|log2FC| >=";

        // A test belongs to one design or the other. Collapsed AND disabled, because WPF's
        // arrow-key and type-ahead selection skip only disabled items, so hiding alone would leave
        // an inapplicable test one keypress away. Every two-sample test is meaningless on a trend:
        // there are no two samples to compare, only a slope.
        ShowTest(DiffTestWelchItem, !paired && !trend);
        ShowTest(DiffTestStudentItem, !paired && !trend);
        ShowTest(DiffTestMannWhitneyItem, !paired && !trend);
        ShowTest(DiffTestPairedTItem, paired);
        ShowTest(DiffTestWilcoxonItem, paired);

        // Detection tests observed-versus-not between two GROUPS (Fisher, or McNemar when paired).
        // Its trend equivalent is a logistic regression of detection on x, which is not
        // implemented - so the view goes rather than running the wrong test under its name.
        ShowTest(DiffViewDetectionItem, !trend);
        // Enrichment takes its hit list from a two-arm contrast; under a trend it would run
        // TryGetGroups against the hidden arm pickers and surface Core's "call RunTrend instead"
        // message, which is written for a caller and not for a reader.
        ShowTest(DiffViewEnrichmentItem, !trend);
        if (DiffViewCombo.SelectedItem is ListBoxItem { IsEnabled: false })
        {
            using (SuppressDiff())
            {
                DiffViewCombo.SelectedIndex = 0; // Volcano
            }
        }

        // Changing the design can strand the selection on a test that no longer applies. Fall back
        // to the moderated t, which is valid under both, rather than leaving an invisible selection.
        if (DiffTestCombo.SelectedItem is ComboBoxItem { IsEnabled: false })
        {
            using (SuppressDiff())
            {
                DiffTestCombo.SelectedIndex = 0;
            }
        }

        var moderated = DiffSelectedTest() == DifferentialTest.ModeratedT;
        var priorVisibility = moderated ? Visibility.Visible : Visibility.Collapsed;
        DiffPriorLabel.Visibility = priorVisibility;
        DiffPriorCombo.Visibility = priorVisibility;

        // The count-based priors need n_peptides, which only the protein matrix carries.
        var hasCounts = _diffDataset?.PeptideCounts is not null;
        ShowTest(DiffPriorPeptideCountItem, hasCounts);
        if (DiffPriorCombo.SelectedItem is ComboBoxItem { IsEnabled: false })
        {
            using (SuppressDiff())
            {
                DiffPriorCombo.SelectedIndex = 0; // Intensity trend, valid at either level
            }
        }

        // Fitting the prior on controls only means something for a prior that HAS a per-feature
        // scale, and only where the run actually has control replicates to fit it on.
        var controls = DiffControlColumns();
        var scaled = DiffSelectedPrior() != VariancePrior.Global;
        DiffPriorFromControlsCheck.Visibility = moderated && scaled
            ? Visibility.Visible
            : Visibility.Collapsed;
        DiffPriorFromControlsCheck.IsEnabled = controls is not null;
        // GREYED, not hidden, and the tooltip says what to do about it. Hiding would be the usual
        // treatment for an option that needs action elsewhere, but it teaches nothing: a reader who
        // never sees the control never learns that setting sample types in Skyline would improve
        // the analysis. The tooltip is rewritten rather than appended to, so the disabled case
        // leads with the reason instead of burying it under a paragraph about what the option does.
        DiffPriorFromControlsCheck.ToolTip = controls is not null
            ? PriorFromControlsHelp
            : "Unavailable: this run has no sample type with two or more replicates. Set replicates "
              + "to Quality Control or Standard in the Skyline document and re-export, and the "
              + "variance prior will be fitted on them."
              + Environment.NewLine + Environment.NewLine + PriorFromControlsHelp;
        if (controls is null && DiffPriorFromControlsCheck.IsChecked == true)
            DiffPriorFromControlsCheck.IsChecked = false;

        // DEFAULT ON wherever the run has controls to fit on, because that is what the lab does and
        // the reason is statistical, not habit: the design groups of a real study contain the
        // biological variation the analysis exists to find, so a prior fitted on them describes
        // measurement noise PLUS that biology and shrinks genuine effects toward nothing. QC and
        // reference injections are nominal replicates, so their spread is the measurement variance
        // the prior is supposed to describe. Only the per-feature scale comes from them; the prior
        // degrees of freedom stay global, estimated from the study samples, so the AMOUNT of
        // shrinkage is still calibrated to the data being analyzed - which is what makes this
        // defensible rather than simply looser. Matches proteomics-toolkit's
        // variance_prior_group_column, which passes fit["d0"] through the same way.
        //
        // Set once per loaded dataset, not on every control change, or unticking it would be
        // undone by the next keystroke elsewhere in the toolbar.
        if (!_diffPriorSourceDefaulted && controls is not null)
        {
            _diffPriorSourceDefaulted = true;
            DiffPriorFromControlsCheck.IsChecked = true;
        }

        DiffCovariatesLabel.IsEnabled = moderated;
        DiffCovariatesCombo.IsEnabled = moderated;
    }

    private static void ShowTest(UIElement item, bool applies)
    {
        item.Visibility = applies ? Visibility.Visible : Visibility.Collapsed;
        item.IsEnabled = applies;
    }

    private void OnDiffSettingsCollapse(object sender, RoutedEventArgs e) => SetDiffSettingsVisible(false);

    private void OnDiffSettingsExpand(object sender, RoutedEventArgs e) => SetDiffSettingsVisible(true);

    /// <summary>
    /// Show or hide the settings panel. Hidden, it leaves a thin strip to bring it back, because a
    /// control that vanishes without a way home is a control nobody hides twice.
    /// </summary>
    private void SetDiffSettingsVisible(bool visible)
    {
        DiffSettingsPanel.Visibility = visible ? Visibility.Visible : Visibility.Collapsed;
        DiffSettingsExpandButton.Visibility = visible ? Visibility.Collapsed : Visibility.Visible;
    }

    private void OnDiffMethodExpanderToggled(object sender, RoutedEventArgs e) => UpdateDiffMethodSummary();

    /// <summary>
    /// The Method section in one line - "Moderated t; intensity trend from controls; BH" - shown
    /// while the section is folded away.
    /// </summary>
    /// <remarks>
    /// The method is folded by default because its defaults are the right ones and most analyses
    /// never touch it. Folding it must not HIDE it, though: a p-value means something different
    /// under a different prior or correction, so what ran has to be readable without opening
    /// anything. Refreshed from <see cref="RunCurrentViewAsync"/>, which every control change
    /// funnels through, so no change to the method can leave the line stale.
    /// </remarks>
    private void UpdateDiffMethodSummary()
    {
        if (DiffMethodExpander.IsExpanded)
        {
            DiffMethodSummary.Visibility = Visibility.Collapsed;
            return;
        }

        static string? Shown(ComboBox c) =>
            (c.SelectedItem as ComboBoxItem)?.Content as string;

        var parts = new List<string>();
        if (Shown(DiffTestCombo) is { } test)
            parts.Add(test);
        if (DiffPriorCombo.Visibility == Visibility.Visible && Shown(DiffPriorCombo) is { } prior)
        {
            var fromControls = DiffPriorFromControlsCheck.Visibility == Visibility.Visible
                && DiffPriorFromControlsCheck.IsChecked == true;
            parts.Add(prior.ToLowerInvariant() + (fromControls ? " from controls" : string.Empty));
        }

        var covariates = (DiffCovariatesCombo.ItemsSource as IEnumerable<QcGroupValue>)
            ?.Where(v => v.IsSelected).Select(v => v.Name).ToList();
        if (covariates is { Count: > 0 })
            parts.Add("adjusted for " + string.Join(", ", covariates));
        if (Shown(DiffCorrectionCombo) is { } correction)
            parts.Add(correction);

        DiffMethodSummary.Text = string.Join("; ", parts);
        DiffMethodSummary.Visibility = parts.Count > 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    private DiffView DiffSelectedView() =>
        ((DiffViewCombo.SelectedItem as ListBoxItem)?.Content as string) switch
        {
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

        using (SuppressDiff())
        {
            if (DiffViewCombo.SelectedItem is null)
                DiffViewCombo.SelectedIndex = 0;
            if (DiffLevelCombo.SelectedItem is null)
                DiffLevelCombo.SelectedIndex = 0;
            // Index 0 is Intensity trend - the lab's default, and deliberately NOT set in the XAML.
            if (DiffPriorCombo.SelectedItem is null)
                DiffPriorCombo.SelectedIndex = 0;
            if (DiffTestCombo.SelectedItem is null)
                DiffTestCombo.SelectedIndex = 0; // Moderated t
            if (DiffDesignCombo.SelectedItem is null)
                DiffDesignCombo.SelectedIndex = 0; // Unpaired
            // The hit rule's three controls. Index 0 is the conventional cut in each - adjusted p,
            // 0.05, two-fold - and none of them is preselected in the XAML, because a Selector that
            // raises SelectionChanged from EndInit runs the handler part way through
            // InitializeComponent. See XamlInitializationOrderTests.
            if (DiffPKindCombo.SelectedItem is null)
                DiffPKindCombo.SelectedIndex = 0;
            if (DiffPCutCombo.SelectedItem is null)
                DiffPCutCombo.SelectedIndex = 0;
            if (DiffEffectCutCombo.SelectedItem is null)
                DiffEffectCutCombo.SelectedIndex = 0;
            if (DiffCorrectionCombo.SelectedItem is null)
                DiffCorrectionCombo.SelectedIndex = 0; // Benjamini-Hochberg
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
        // Genes are kept in their own map: unlike the label there is NO falling back to the feature
        // id, because a protein group id or a peptide sequence is not a gene symbol and enrichment
        // would submit it as one.
        _diffGeneById = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var i = 0; i < ds.FeatureIds.Length; i++)
        {
            _diffLabelById[ds.FeatureIds[i]] =
                string.IsNullOrEmpty(ds.FeatureLabels[i]) ? ds.FeatureIds[i] : ds.FeatureLabels[i];
            if (!string.IsNullOrEmpty(ds.FeatureGenes[i]))
                _diffGeneById[ds.FeatureIds[i]] = ds.FeatureGenes[i];
        }

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

        using (SuppressDiff())
        {
            DiffGroupByCombo.ItemsSource = ds.MetadataColumns;
            DiffGroupByCombo.SelectedItem = DefaultContrastColumn(ds);
        }

        PopulateDiffGroupValues();
        UpdateDiffControls();
        DiffStatusText.Text =
            $"Loaded {ds.FeatureIds.Length} {level.ToString().ToLowerInvariant()} features x " +
            $"{ds.SampleIds.Length} samples. Pick the arms to contrast; it runs as you choose.";
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

        // Default to the first two values, which is what the single pickers did; anything more is
        // an explicit choice by the user.
        _diffAValues = values
            .Select((v, i) => new QcGroupValue
            {
                Name = v!, IsSelected = i == 0, Changed = UpdateDiffArmSummaries,
            })
            .ToList();
        _diffBValues = values
            .Select((v, i) => new QcGroupValue
            {
                Name = v!, IsSelected = i == 1, Changed = UpdateDiffArmSummaries,
            })
            .ToList();

        DiffACombo.ItemsSource = _diffAValues;
        DiffBCombo.ItemsSource = _diffBValues;
        UpdateDiffArmSummaries();

        // Any metadata column can identify a subject except the one being contrasted - pairing by
        // the contrast column itself would put every subject in one arm.
        var pairCandidates = _diffDataset.MetadataColumns
            .Where(m => !string.Equals(m, col, StringComparison.Ordinal))
            .ToList();
        var keepPair = DiffPairByCombo.SelectedItem as string;
        using (SuppressDiff())
        {
            DiffPairByCombo.ItemsSource = pairCandidates;
            DiffPairByCombo.SelectedItem = keepPair is not null && pairCandidates.Contains(keepPair)
                ? keepPair
                : null;
        }

        PopulateDiffCovariates(col);
    }

    /// <summary>
    /// The closed-state text of the two arm pickers, and of the covariates picker beside them.
    /// </summary>
    /// <remarks>
    /// A tick-list ComboBox has no SelectedItem, so WPF has nothing to display when it is closed and
    /// the text stays at whatever the XAML set. Every other tick list in this window writes its own
    /// summary; the covariates one did not, so it read "(none)" however many covariates were ticked -
    /// fixed here rather than left as the odd one out.
    /// </remarks>
    private void UpdateDiffArmSummaries()
    {
        DiffACombo.Text = SummarizeArm(_diffAValues);
        DiffBCombo.Text = SummarizeArm(_diffBValues);
        var covariates = _diffCovariateValues.Where(v => v.IsSelected).Select(v => v.Name).ToList();
        DiffCovariatesCombo.Text = covariates.Count == 0 ? "(none)" : string.Join(", ", covariates);
        // The Restrict-to values are a tick list too, and were added without this - so a ticked study
        // closed to a blank box, reading as though nothing had been chosen.
        DiffRestrictValuesCombo.Text = DiffRestrictValues() is { Count: > 0 } kept
            ? string.Join(", ", kept)
            : "(tick values to keep)";
    }

    /// <summary>
    /// " + " rather than ", ": the values are POOLED into one arm, and a comma reads like a list of
    /// separate things to compare.
    /// </summary>
    private static string SummarizeArm(IReadOnlyList<QcGroupValue> values)
    {
        var on = values.Where(v => v.IsSelected).Select(v => v.Name).ToList();
        return on.Count == 0 ? "(pick one or more)" : string.Join(" + ", on);
    }

    private void PopulateDiffCovariates(string groupByColumn)
    {
        if (_diffDataset is null)
            return;

        // Any metadata column can be a covariate except the sample id itself and the contrast column.
        _diffCovariateValues = _diffDataset.MetadataColumns
            .Where(c => c != groupByColumn && c != "sample")
            .Select(c => new QcGroupValue { Name = c, Changed = UpdateDiffArmSummaries })
            .ToList();
        DiffCovariatesCombo.ItemsSource = _diffCovariateValues;
    }

    /// <summary>Ticked covariates, with values aligned to <paramref name="targetSampleIds"/>, or null if none.</summary>
    /// <summary>
    /// Drop the term being tested from a covariate list, whichever term that is.
    /// </summary>
    /// <remarks>
    /// The covariate picker excludes the Group-by column, but a trend is tested on the Trend-over
    /// column instead - and adjusting a slope for the very column it is a slope in gives the design
    /// [1, x, x], which is exactly singular. Dropped here rather than filtered out of the picker so
    /// that switching design back to a two-arm contrast does not silently lose a tick the user made.
    /// </remarks>
    private IReadOnlyList<Covariate>? WithoutTestedTerm(IReadOnlyList<Covariate>? covariates)
    {
        // The axis's COLUMN, not its label: a covariate is a metadata column, and the label of a
        // reading ("Longitudinal Draw Description (Week)") names none, so comparing with it kept the
        // column's own categories in the design - collinear with the very timepoints tested.
        if (covariates is null || !DiffIsTrend() || DiffTrendAxis() is not { } axis)
            return covariates;

        var kept = covariates
            .Where(c => !string.Equals(c.Name, axis.Column, StringComparison.Ordinal))
            .ToList();
        return kept.Count == covariates.Count ? covariates : kept;
    }

    private IReadOnlyList<Covariate>? SelectedCovariatesFor(IReadOnlyList<string> targetSampleIds) =>
        _diffDataset is null
            ? null
            : DetectionAnalysis.CovariatesFor(_diffDataset, SelectedCovariateNames(), targetSampleIds);

    /// <summary>The metadata columns ticked in Adjust for.</summary>
    private List<string> SelectedCovariateNames() =>
        _diffCovariateValues.Where(v => v.IsSelected).Select(v => v.Name).ToList();

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

    /// <summary>
    /// A new grouping column: rebuild the arm pickers for it, then run, like every other selector.
    /// </summary>
    /// <remarks>
    /// PopulateDiffGroupValues reseeds both arms with the new column's first two levels and rewrites
    /// the closed-state text of both pickers. Without the run, the toolbar then reads "Control vs
    /// Disease" over a volcano and a status line still describing the PREVIOUS column's contrast -
    /// the one state this pane promises cannot happen, since the load line says "it runs as you
    /// choose" and the Run button is only a manual refresh.
    /// </remarks>
    private async void OnDiffGroupByChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _diffSuppress || _diffDataset is null)
                return;
            PopulateDiffGroupValues();
            // A trend ignores the arms entirely, so changing the grouping column cannot move it.
            if (DiffIsTrend())
                return;
            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffGroupByChanged), ex);
        }
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

    /// <summary>
    /// Fold the clinical columns just joined to the differential dataset into the QC pane's
    /// grouping sources, so every plot in the window can be grouped by them.
    /// </summary>
    /// <remarks>
    /// Values are taken from the dataset rather than re-read from the CSV on purpose: the join
    /// picked a key column by best match rate, and re-deriving it here could pick a different one
    /// and color the plot by a slightly different assignment than the one the user is reading the
    /// Volcano against.
    /// </remarks>
    private void PublishClinicalToQcPane(IReadOnlyList<string> addedColumns)
    {
        if (_diffDataset is null || addedColumns.Count == 0)
            return;

        var values = new Dictionary<string, string?[]>(StringComparer.Ordinal);
        foreach (var c in addedColumns)
            values[c] = _diffDataset.MetadataValues(c);

        var byId = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < _diffDataset.SampleIds.Length; i++)
            byId[_diffDataset.SampleIds[i]] = i;

        var clinical = SampleAnnotationTable.FromValues(
            _diffDataset.SampleIds, addedColumns,
            (id, col) => byId.TryGetValue(id, out var idx) ? values[col][idx] : null);

        // MergedWith keeps what is already there on a name clash, so a clinical column sharing a
        // name with one Skyline exported does not quietly replace it.
        _qcExtraAnnotations = _qcExtraAnnotations.MergedWith(clinical);
        PopulateGroupCombos();
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

            // The QC pane draws the only sample PCA now, so its Group-by list has to learn about
            // these columns too - otherwise attaching a clinical CSV would enrich the Volcano's
            // covariates and leave the PCA unable to color by any of them, which is the split
            // that put a second PCA in this pane in the first place.
            PublishClinicalToQcPane(result.AddedColumns);

            // The Markers pane keeps its own cache for the OTHER feature level, and that copy was
            // loaded before this join existed. GetMarkersDatasetAsync reuses the Differential
            // dataset when the directory and level both match - so the stale case is specifically
            // the two panes sitting on different levels, where the marker cache is returned as-is
            // and the clinical columns promised here never appear in it.
            InvalidateMarkers();

            // Refresh the group-by choices so the new clinical columns appear; select the first one.
            using (SuppressDiff())
            {
                DiffGroupByCombo.ItemsSource = null;
                DiffGroupByCombo.ItemsSource = _diffDataset.MetadataColumns;
                DiffGroupByCombo.SelectedItem = result.AddedColumns[0];
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


    private async Task RunCurrentViewAsync()
    {
        UpdateDiffMethodSummary();
        if (_diffDataset is null)
        {
            DiffStatusText.Text = "Load a run first (point the output directory at a finished PRISM run).";
            return;
        }

        // Anything already in flight is now stale, whatever order it finishes in.
        var request = ++_diffViewRequest;
        switch (DiffSelectedView())
        {
            case DiffView.Detection:
                await RunDetectionAsync(request);
                break;
            case DiffView.Enrichment:
                await RunEnrichmentAsync(request);
                break;
            default:
                await RunVolcanoAsync(request);
                break;
        }
    }

    /// <summary>Whether this run is still the current one. False means drop everything and paint nothing.</summary>
    private bool StillCurrent(int request) => request == _diffViewRequest;

    /// <summary>The levels ticked for each arm - what <see cref="TryGetGroups"/> resolves into columns.</summary>
    private (List<string> A, List<string> B) SelectedArmLevels() => (
        _diffAValues.Where(v => v.IsSelected).Select(v => v.Name).ToList(),
        _diffBValues.Where(v => v.IsSelected).Select(v => v.Name).ToList());

    private bool TryGetGroups(out string col, out List<int> groupA, out List<int> groupB,
        out string aVal, out string bVal)
    {
        col = string.Empty;
        aVal = string.Empty;
        bVal = string.Empty;
        groupA = new List<int>();
        groupB = new List<int>();
        if (_diffDataset is null || DiffGroupByCombo.SelectedItem is not string c)
            return false;

        var (aSet, bSet) = SelectedArmLevels();
        var arms = ContrastArms.Resolve(_diffDataset.MetadataValues(c), aSet, bSet);
        if (!arms.Ok)
            return false;

        col = c;
        aVal = ContrastArms.Describe(aSet);
        bVal = ContrastArms.Describe(bSet);
        groupA = arms.A.ToList();
        groupB = arms.B.ToList();
        return true;
    }

    /// <summary>
    /// The sample columns the contrast actually ran over, which under a paired design is the matched
    /// subset rather than everything ticked.
    /// </summary>
    /// <remarks>
    /// The per-feature boxplot reads these. Showing the picked columns there would draw subjects that
    /// took no part in the contrast, and contradict the status line, which reports the arms the test
    /// used. Resolving the pairs twice - here and in Core - is cheap and keeps the two honest; the
    /// alternative is Core returning its column sets, which widens its result type for a display
    /// concern.
    /// </remarks>
    private (List<int> A, List<int> B) ContrastColumns(List<int> pickedA, List<int> pickedB) =>
        PairedSamples.ColumnsUsed(DiffSelectedDesign(), DiffSubjectLabels(), pickedA, pickedB);

    /// <summary>
    /// The Volcano under a trend design: a slope against a numeric column, plotted as the modeled
    /// change across that column's observed range.
    /// </summary>
    /// <remarks>
    /// Its own method rather than a branch threaded through the two-arm one, because almost nothing
    /// is shared past the inputs - there are no arms to resolve, no arm names to print, and the
    /// per-feature view is a scatter rather than two boxes. What IS shared is everything after the
    /// result: the same rule, the same renderer, the same grid.
    /// </remarks>
    private async Task RunTrendVolcanoAsync(int request)
    {
        if (_diffDataset is null)
            return;
        if (DiffTrendColumn() is not { } trendColumn || DiffTrendValues() is not { } xValues)
        {
            // Named rather than "pick a numeric column": with an interpreted axis available the
            // picker deliberately opens empty, and a reader needs to know that is a choice waiting
            // on them rather than a run with nothing to offer.
            DiffStatusText.Text = DiffTrendOverCombo.HasItems
                ? "Pick what to fit the trend against. A column whose values embed a number is "
                  + "offered once per number it holds - check the line underneath says the one you mean."
                : "This run has no column a trend can be fitted against.";
            ClearDiffOutput();
            return;
        }

        // Each early return below clears the plot. Left up, the previous run's volcano - another design,
        // another axis label - sat under a status line saying nothing had run, and read as this result.
        //
        // An empty Subject picker under a within-subject trend is a choice not yet made, not a failure:
        // asked for here rather than letting the run start and come back as "Cannot run this trend".
        if (DiffSelectedDesign() == DifferentialDesign.LinearTrendWithinSubject
            && DiffPairByCombo.SelectedItem is not string)
        {
            DiffStatusText.Text = "Pick the Subject column - the one that identifies each person or "
                + "donor - so the slope is estimated within each subject.";
            ClearDiffOutput();
            return;
        }

        var dataset = _diffDataset;
        var covariates = WithoutTestedTerm(SelectedCovariatesFor(dataset.SampleIds));
        var options = DiffOptions(covariates);
        var rule = DiffRule();
        var columns = DiffTrendColumns();
        var restricted = dataset.SampleIds.Length - columns.Length;
        if (columns.Length == 0)
        {
            DiffStatusText.Text = "Restrict to: no sample matches the ticked values.";
            ClearDiffOutput();
            return;
        }

        DifferentialResult res;
        try
        {
            res = await Task.Run(() =>
                Differential.RunTrend(dataset.ExprLog2, dataset.FeatureIds, columns, xValues, options));
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
        {
            if (StillCurrent(request))
            {
                DiffStatusText.Text = "Cannot run this trend: " + ex.Message;
                ClearDiffOutput();
            }

            return;
        }

        if (!StillCurrent(request))
            return;

        // The span the effect is measured across, for the axis label - over the samples the fit
        // USED. This read every finite x in the column while its comment said otherwise, so with a
        // restriction a Verapamil-only fit was labeled "-2 to 12", -2 being a Liraglutide timepoint:
        // a range the model never saw, on the axis whose numbers are a change across it.
        var used = columns.Select(c => xValues[c]).Where(double.IsFinite).ToList();
        _diffTrendRange = used.Count > 0 ? (used.Min(), used.Max()) : null;

        // A trend has no arms, so the per-feature view has no two groups to box. The scatter reads
        // the trend column instead; these are what it plots against.
        _volcanoGroupA = columns.ToList();
        _volcanoGroupB = new List<int>();
        _volcanoAName = trendColumn;
        _volcanoBName = string.Empty;
        _volcanoTrendX = xValues;

        RenderVolcano(res, rule);
        DiffGrid.ItemsSource = res.Rows.Select(r => new VolcanoRow(
            _diffLabelById.GetValueOrDefault(r.FeatureId, r.FeatureId), r.LogFc, r.PValue, r.AdjPValue,
            r.FeatureId))
            .ToList();

        var nSig = res.Rows.Count(rule.IsSignificant);
        var adj = res.CovariatesUsed.Count > 0
            ? $"; adjusted for {string.Join(", ", res.CovariatesUsed)}"
            : string.Empty;
        var note = res.Messages.Count > 0 ? " " + string.Join(" ", res.Messages) : string.Empty;
        // A restriction removes samples from the column list (DiffTrendColumns), so TrendSamples never
        // counts them as having no value on the axis; how many it removed is said here, separately
        // and FIRST, so a deliberate subset never reads as missing data.
        if (restricted > 0 && DiffRestrictColumn() is { } restrictCol)
            note = $" Restricted to {restrictCol}: {restricted} sample(s) outside the"
                + " ticked values were left out." + note;
        // Samples AND subjects, because under a within-subject design the second is what the test
        // has to work with and the first alone would overstate it.
        var n = res.NSubjects > 0
            ? $"n={res.NA} samples in {res.NSubjects} subjects"
            : $"n={res.NA} samples";
        DiffStatusText.Text =
            $"{options.Describe(res.VariancePrior)}: {n}, {trendColumn} {FormatRange(_diffTrendRange)} - "
            + $"{res.NFeaturesTested} tested, {nSig} significant ({rule.Describe(DiffEffectName())})"
            + $"{adj}.{note} "
            + "Click a point (or a row) for its trajectory; it also selects in Skyline. Hover for the gene.";
    }

    private static string FormatRange((double Min, double Max)? range) =>
        range is { } r
            ? $"{r.Min.ToString("0.###", CultureInfo.InvariantCulture)} to "
              + r.Max.ToString("0.###", CultureInfo.InvariantCulture)
            : "(no range)";

    /// <summary>
    /// The trend column's value per sample from the last trend run, or null when the current view
    /// is a two-arm contrast. The per-feature window plots against it.
    /// </summary>
    private double[]? _volcanoTrendX;

    private async Task RunVolcanoAsync(int request)
    {
        if (DiffIsTrend())
        {
            await RunTrendVolcanoAsync(request);
            return;
        }

        if (!TryGetGroups(out _, out var a, out var b, out var aVal, out var bVal))
        {
            DiffStatusText.Text = "Pick a group-by column, then tick at least one value for each "
                + "arm. A value cannot be in both.";
            ClearDiffOutput();
            return;
        }

        var dataset = _diffDataset!;
        var covariates = WithoutTestedTerm(SelectedCovariatesFor(dataset.SampleIds));
        var options = DiffOptions(covariates);
        // Read BEFORE the worker starts. Reading it after the await would also work today, but the
        // pane has already shipped one cross-thread control read from inside a Task.Run lambda, and
        // a local captured up here cannot become one.
        var rule = DiffRule();
        DifferentialResult res;
        try
        {
            res = await Task.Run(() =>
                Differential.Run(dataset.ExprLog2, dataset.FeatureIds, a, b, options));
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
        {
            if (StillCurrent(request))
            {
                DiffStatusText.Text = "Cannot run this contrast: " + ex.Message;
                ClearDiffOutput();
            }

            return;
        }

        if (!StillCurrent(request))
            return;

        (_volcanoGroupA, _volcanoGroupB) = ContrastColumns(a, b);
        _volcanoAName = aVal;
        _volcanoBName = bVal;
        // Cleared, not merely unset: left over from a previous trend run it would send the next
        // per-feature click to the trajectory view against a column this contrast never used.
        _volcanoTrendX = null;
        _diffTrendRange = null;
        RenderVolcano(res, rule);
        DiffGrid.ItemsSource = res.Rows.Select(r => new VolcanoRow(
            _diffLabelById.GetValueOrDefault(r.FeatureId, r.FeatureId), r.LogFc, r.PValue, r.AdjPValue,
            r.FeatureId))
            .ToList();

        var nSig = res.Rows.Count(rule.IsSignificant);
        var adj = res.CovariatesUsed.Count > 0 ? $"; adjusted for {string.Join(", ", res.CovariatesUsed)}" : string.Empty;
        // Name the method. With a menu this size the status line is the only record of what
        // produced a hit list, and any message Core raised (an unhonored covariate, a prior that
        // could not be fitted) belongs beside it rather than nowhere.
        var note = res.Messages.Count > 0 ? " " + string.Join(" ", res.Messages) : string.Empty;
        // res.NA/NB, not a.Count/b.Count: a paired design drops unmatched subjects, so the arms the
        // test actually used can be smaller than the arms that were picked. Reporting the picked
        // sizes would credit the result with samples that took no part in it.
        DiffStatusText.Text =
            $"{options.Describe(res.VariancePrior)}: {aVal} (n={res.NA}) vs {bVal} (n={res.NB}) - "
            + $"{res.NFeaturesTested} tested, {nSig} significant ({rule.Describe(DiffEffectName())})"
            + $"{adj}.{note} "
            + "Click a point (or a row) for its boxplot; it also selects in Skyline. Hover for the gene.";
    }

    /// <summary>
    /// A paired design's "ran unpaired" note (<see cref="DetectionAnalysis.UnpairedNote"/>), led by a
    /// space for appending to a status line, or empty when the design was honored.
    /// </summary>
    private static string UnpairedSuffix(UnpairedReason reason) =>
        DetectionAnalysis.UnpairedNote(reason) is { } note ? " " + note : string.Empty;

    private async Task RunDetectionAsync(int request)
    {
        if (!TryGetGroups(out _, out var a, out var b, out var aVal, out var bVal))
        {
            DiffStatusText.Text = "Pick a group-by column, then tick at least one value for each "
                + "arm. A value cannot be in both.";
            ClearDiffOutput();
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
                if (!StillCurrent(request))
                    return;
            }
            catch (Exception ex)
            {
                DiffStatusText.Text = "Detection load failed (needs merged_data in the output dir): " + ex.Message;
                return;
            }

            _detectionData = det;
            _detectionDir = dir;
        }

        // Everything read off a WPF control is read here, on the UI thread, never inside the Task.Run:
        // touching a control from a worker throws "The calling thread cannot access this object".
        var dataset = _diffDataset!;
        var design = DiffSelectedDesign();
        var subjects = DiffSubjectLabels();
        var covariateNames = SelectedCovariateNames();
        var correction = DiffSelectedCorrection();
        var corrected = correction != MultipleTesting.None;

        // Which test runs is decided in Core (DetectionAnalysis), so the quant report and the CLI make
        // the same choice this pane does: Firth GLM when covariates are ticked, McNemar when paired,
        // Fisher otherwise.
        DetectionAnalysisResult result;
        try
        {
            result = await Task.Run(() => DetectionAnalysis.Run(det, dataset, a, b, design, subjects,
                covariateNames, correction));
        }
        catch (DetectionSamplesNotFoundException ex)
        {
            DiffStatusText.Text = ex.Message;
            return;
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
        {
            DiffStatusText.Text = "Detection test failed: " + ex.Message;
            return;
        }

        // AFTER the await, not before it: the Firth GLM is the slowest path in the pane, so it is the
        // one most likely to be superseded while it runs.
        if (!StillCurrent(request))
            return;

        var droppedNote = result.DroppedSamples > 0
            ? $" ({result.DroppedSamples} samples not in merged_data)"
            : string.Empty;

        switch (result.Method)
        {
            case DetectionMethod.McNemarPaired:
            {
                // Only the discordant pairs carry information - a subject that agreed with itself is
                // its own control - so the counts that drove the test are in the table beside the
                // rates rather than left implied. The plot takes the unpaired shape, with the pair
                // count standing in for each arm's n.
                var pairedRows = result.PairedRows!;
                RenderDetection(result.Rows, DiffRule(), corrected);
                DiffGrid.ItemsSource = pairedRows.Take(1000)
                    .Select(r => new DetPairedRow(r.PeptideId, r.RateA, r.RateB, r.OnlyA, r.OnlyB, r.P, r.Q))
                    .ToList();

                var lost = result.PairMessages.Count > 0
                    ? " " + string.Join(" ", result.PairMessages)
                    : string.Empty;
                var missing = result.PairsNotInMerged > 0
                    ? $" {result.PairsNotInMerged} matched pair(s) are not in merged_data."
                    : string.Empty;
                DiffStatusText.Text =
                    $"Paired detection (McNemar exact): {aVal} vs {bVal} over {result.NA} matched "
                    + $"subject(s), {pairedRows.Count} peptides.{lost}{missing} Only discordant pairs "
                    + "carry information - the two counts are in the table.";
                return;
            }

            case DetectionMethod.FirthGlm:
            {
                var glm = result.Glm!;
                if (!glm.Identifiable)
                {
                    ClearDiffOutput();
                    DiffStatusText.Text = "Adjusted detection is not identifiable (group confounded with the "
                        + $"covariates, R^2={glm.GroupCollinearityR2:0.00}). Use the unadjusted view.";
                    return;
                }

                RenderDetectionGlm(glm.Rows, DiffRule(), corrected);
                DiffGrid.ItemsSource = glm.Rows.Take(1000)
                    .Select(r => new DetGlmRow(r.PeptideId, r.RateA, r.RateB, r.LogOr, r.P, r.Q)).ToList();
                DiffStatusText.Text =
                    $"Adjusted detection (Firth GLM): {aVal} (n={result.NA}) vs {bVal} (n={result.NB}), "
                    + $"adjusted for {string.Join(", ", glm.CovariatesUsed)}, {glm.Rows.Count} peptides{droppedNote}."
                    + UnpairedSuffix(result.UnpairedReason);
                return;
            }

            default:
                RenderDetection(result.Rows, DiffRule(), corrected);
                DiffGrid.ItemsSource = result.Rows.Take(1000)
                    .Select(r => new DetRow(r.PeptideId, r.RateA, r.RateB, r.P, r.Q)).ToList();
                DiffStatusText.Text =
                    $"Detection (peptide-level, DetectionQValue < 0.01): {aVal} (n={result.NA}) vs " +
                    $"{bVal} (n={result.NB}) over {result.Rows.Count} peptides{droppedNote}."
                    + UnpairedSuffix(result.UnpairedReason);
                return;
        }
    }

    private async Task RunEnrichmentAsync(int request)
    {
        if (!TryGetGroups(out _, out var a, out var b, out var aVal, out var bVal))
        {
            DiffStatusText.Text = "Pick a group-by column, then tick at least one value for each "
                + "arm. A value cannot be in both.";
            ClearDiffOutput();
            return;
        }

        var dataset = _diffDataset!;

        // Enrichment is about GENES, so it reads FeatureGenes and never the display label. At
        // peptide level the label is the modified sequence, and a sequence passes CleanSymbols
        // untouched - it only drops empty/"nan" and splits delimiters - so using the label asked
        // g:Profiler about a list of peptide sequences and captioned the empty answer as a gene
        // enrichment. leading_gene_name is stamped onto corrected_peptides for exactly this, so
        // peptide-level enrichment works; it is only impossible on a peptide file written before
        // that column existed, and then we say so rather than guess.
        if (dataset.FeatureGenes.All(string.IsNullOrEmpty))
        {
            ClearDiffOutput();
            DiffStatusText.Text =
                "Enrichment needs gene symbols and this run carries none - its "
                + (dataset.Level == FeatureLevel.Protein ? "protein" : "peptide")
                + " matrix has no leading_gene_name column. Re-run the pipeline to add it.";
            return;
        }

        var covariates = WithoutTestedTerm(SelectedCovariatesFor(dataset.SampleIds));
        var options = DiffOptions(covariates);
        // Same reason as the volcano path: read on the UI thread, before the worker exists.
        var rule = DiffRule();
        DifferentialResult res;
        try
        {
            res = await Task.Run(() =>
                Differential.Run(dataset.ExprLog2, dataset.FeatureIds, a, b, options));
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
        {
            if (StillCurrent(request))
            {
                DiffStatusText.Text = "Cannot run this contrast: " + ex.Message;
                ClearDiffOutput();
            }

            return;
        }

        if (!StillCurrent(request))
            return;

        var (sig, background) = Enrichment.SigAndBackgroundGenes(
            res, fid => _diffGeneById.GetValueOrDefault(fid), rule);
        if (sig.Count == 0)
        {
            ClearDiffOutput();
            DiffStatusText.Text =
                $"No significant genes ({rule.Describe(DiffEffectName())}) for {aVal} vs {bVal} - "
                + "nothing to enrich.";
            return;
        }

        DiffStatusText.Text = $"Querying g:Profiler for {sig.Count} genes...";
        List<EnrichmentTerm> terms;
        try
        {
            terms = await Task.Run(() => Enrichment.GProfiler(sig, background, DiffPoster));
            if (!StillCurrent(request))
                return;
        }
        // A timeout is named apart from a connectivity failure. HttpClient surfaces its own timeout as
        // a cancellation, so the blanket message sent someone to check their network when the request
        // had in fact been answered too slowly - and with per-gene evidence in the response, a
        // proteome-scale query is exactly the one that takes longest.
        catch (TaskCanceledException)
        {
            DiffStatusText.Text =
                $"Enrichment timed out after {HttpJsonPoster.DefaultTimeout.TotalSeconds:0}s with "
                + $"{sig.Count} genes against a background of {background.Count}. The server is "
                + "reachable but slow - try a stricter hit rule to send fewer genes.";
            return;
        }
        catch (Exception ex)
        {
            DiffStatusText.Text = "Enrichment request failed (needs internet access): " + ex.Message;
            return;
        }

        // Index the terms and map each significant gene to its protein(s), so clicking a term row can
        // list the proteins behind it. Genes come out of the same rule the volcano/enrichment used.
        _enrichTermsById = terms.ToDictionary(t => t.TermId, t => t, StringComparer.Ordinal);
        _enrichFeaturesByGene = BuildEnrichFeaturesByGene(res, rule);

        RenderEnrichment(terms);
        DiffGrid.ItemsSource = terms.Take(1000)
            .Select(t => new EnrichRow(t.Source, t.TermName, t.PValue, t.FoldEnrichment, t.TermId)).ToList();
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
            // Horizontal bars with the term on the y-axis: a vertical bar chart cannot carry a term
            // name at all, so the plot said nothing the table did not. Terms arrive sorted by p; the
            // most significant sits at the TOP (the highest bar position).
            var top = terms.Take(15).ToList();
            var bars = new List<ScottPlot.Bar>(top.Count);
            var positions = new double[top.Count];
            var labels = new string[top.Count];
            for (var i = 0; i < top.Count; i++)
            {
                var pos = top.Count - 1 - i; // i = 0 (smallest p) -> top of the plot
                bars.Add(new ScottPlot.Bar
                {
                    Position = pos,
                    Value = -Math.Log10(Math.Max(top[i].PValue, 1e-300)),
                    Orientation = ScottPlot.Orientation.Horizontal,
                    FillColor = ScottPlot.Color.FromHex("#2ca02c"),
                });
                positions[pos] = pos;
                labels[pos] = TruncateTerm(top[i].TermName);
            }

            plt.Add.Bars(bars);
            plt.Axes.Left.TickGenerator = new ScottPlot.TickGenerators.NumericManual(positions, labels);
            plt.Axes.Left.TickLabelStyle.FontSize = 11;
            plt.Axes.SetLimitsY(-0.7, top.Count - 0.3);
            plt.XLabel("-log10 p (g:SCS)");
        }

        PlotRenderer.StyleQcPlot(plt);
        DiffPlot.Refresh();
    }

    /// <summary>Keep a long GO term readable on an axis; the full name is in the table beside the plot.</summary>
    private static string TruncateTerm(string name) =>
        string.IsNullOrEmpty(name) || name.Length <= 45 ? name ?? string.Empty : name[..44] + "...";

    private void RenderDetectionGlm(IReadOnlyList<DetectionGlmRow> rows, SignificanceRule rule, bool corrected)
    {
        DiffPlot.Reset();
        var plt = DiffPlot.Plot;

        var bgX = new List<double>();
        var bgY = new List<double>();
        var sigX = new List<double>();
        var sigY = new List<double>();
        foreach (var r in rows)
        {
            var p = rule.UseAdjusted ? r.Q : r.P;
            if (!double.IsFinite(r.LogOr) || !double.IsFinite(p))
                continue;
            var y = -Math.Log10(Math.Max(p, 1e-300));
            if (p < rule.PThreshold)
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

        AddMarkers(plt, bgX, bgY, "#b8c4d0", DiffPointSize, "not significant");
        AddMarkers(plt, sigX, sigY, "#2ca02c", DiffSigPointSize, "significant");
        plt.Add.VerticalLine(0.0);
        plt.Add.HorizontalLine(-Math.Log10(rule.PThreshold));
        plt.HideLegend(); // as on the Volcano: the lines and the status line say what the colors mean
        plt.XLabel("log odds ratio (B / A)");
        plt.YLabel(rule.YAxisLabel(corrected));
        PlotRenderer.StyleQcPlot(plt);
        DiffPlot.Refresh();
    }

    private void OnDiffGridAutoGeneratingColumn(object sender, DataGridAutoGeneratingColumnEventArgs e)
    {
        // FeatureId backs click-to-boxplot and TermId backs click-to-members; neither is a column the
        // user needs to see.
        if (e.PropertyName is "FeatureId" or "TermId")
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
    /// On the Volcano view, a click near a point makes that feature the selection: it rings the
    /// point, selects its row in the hit table, opens the per-feature boxplot, and selects the
    /// protein or peptide in Skyline.
    /// </summary>
    private void OnDiffPlotMouseDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
    {
        try
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

            SelectVolcanoFeature(_volcanoPoints[idx].FeatureId);
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffPlotMouseDown), ex);
        }
    }

    /// <summary>
    /// Changing the design changes which tests apply and whether a pairing column is needed, so the
    /// controls are updated before anything is re-run.
    /// </summary>
    private async void OnDiffDesignChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (!IsInitialized)
                return;
            UpdateDiffControls();
            if (_diffSuppress || _diffDataset is null)
                return;
            // A paired design with no pairing column yet is an incomplete request, not an error:
            // say what is missing and wait rather than throwing from the run.
            if (DiffSelectedDesign() == DifferentialDesign.Paired && DiffSubjectLabels() is null)
            {
                DiffStatusText.Text = "Paired: pick the Subject column - the one that identifies each "
                    + "person or donor - so their two samples can be matched.";
                ClearDiffOutput();
                return;
            }

            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffDesignChanged), ex);
        }
    }

    /// <summary>Choosing the pairing column completes a paired request, so run it.</summary>
    private async void OnDiffPairByChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _diffSuppress || _diffDataset is null)
                return;
            if (DiffSelectedDesign() is not (DifferentialDesign.Paired
                or DifferentialDesign.LinearTrendWithinSubject))
                return; // only those two designs consult the subject column
            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffPairByChanged), ex);
        }
    }

    /// <summary>
    /// Changing the estimator re-runs the current view, and shows or hides the controls that only
    /// some estimators use.
    /// </summary>
    private async void OnDiffTestChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (!IsInitialized)
                return;
            // Outside the suppress guard: the controls must follow the combo even while a load is
            // populating the pane, or they would be left describing the previous test.
            UpdateDiffControls();
            if (_diffSuppress || _diffDataset is null)
                return;
            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffTestChanged), ex);
        }
    }

    /// <summary>
    /// Fitting the prior on controls rather than on the contrast groups changes every moderated
    /// p-value, so it re-runs.
    /// </summary>
    private async void OnDiffPriorSourceChanged(object sender, RoutedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _diffSuppress || _diffDataset is null)
                return;
            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffPriorSourceChanged), ex);
        }
    }

    /// <summary>
    /// A tick list re-runs when its dropdown CLOSES, not on every tick.
    /// </summary>
    /// <remarks>
    /// <para>Closing the dropdown is the moment the choice is finished. Running on each tick instead
    /// would fire once per click while an arm is being assembled from several values, and would show
    /// "tick at least one value for each arm" in passing every time the last value was cleared before
    /// the next was set - a complaint about a state the user was moving through, not one they
    /// chose.</para>
    /// <para>It also matters for the Enrichment view, which posts to g:Profiler: once per
    /// interaction is reasonable, once per tick is not.</para>
    /// </remarks>
    private async void OnDiffTickListClosed(object sender, EventArgs e)
    {
        try
        {
            if (!IsInitialized || _diffSuppress || _diffDataset is null)
                return;
            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffTickListClosed), ex);
        }
    }

    /// <summary>Changing the correction re-runs: it changes every adjusted p in the table.</summary>
    private async void OnDiffCorrectionChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (!IsInitialized || _diffSuppress || _diffDataset is null)
                return;
            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffCorrectionChanged), ex);
        }
    }

    /// <summary>
    /// Changing the variance prior re-runs the current view, the way changing the View already does -
    /// it is a different estimator, so the plot on screen is no longer the one the controls describe.
    /// </summary>
    private async void OnDiffPriorChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            // IsInitialized as well as the suppress flag: this combo has a SelectionChanged handler,
            // and if it ever gains a XAML-side default it would fire part way through
            // InitializeComponent. See XamlInitializationOrderTests.
            if (!IsInitialized)
                return;
            UpdateDiffControls(); // the control-prior checkbox does not apply to the global prior
            if (_diffSuppress || _diffDataset is null)
                return;
            await RunCurrentViewAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffPriorChanged), ex);
        }
    }

    /// <summary>
    /// Selecting a hit row does everything a volcano click does - the same selection, reached from
    /// the other side - so the ring moves to that feature's point and Skyline follows too.
    /// </summary>
    private void OnDiffGridSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        try
        {
            if (DiffGrid.SelectedItem is VolcanoRow vr)
                SelectVolcanoFeature(vr.FeatureId);
            else if (DiffGrid.SelectedItem is EnrichRow er)
                ShowTermProteins(er);
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffGridSelectionChanged), ex);
        }
    }

    /// <summary>
    /// Map each significant feature to its gene(s), so a clicked enrichment term can name the proteins
    /// behind it. Only the features the rule counts as hits are included - they are the query
    /// g:Profiler was given, so a term can only contain these.
    /// </summary>
    private Dictionary<string, List<TermProtein>> BuildEnrichFeaturesByGene(
        DifferentialResult res, SignificanceRule rule)
    {
        var byGene = new Dictionary<string, List<TermProtein>>(StringComparer.OrdinalIgnoreCase);
        foreach (var r in res.Rows)
        {
            if (!rule.IsSignificant(r))
                continue;
            var label = _diffLabelById.GetValueOrDefault(r.FeatureId, r.FeatureId);
            foreach (var gene in Enrichment.CleanSymbols(new[] { _diffGeneById.GetValueOrDefault(r.FeatureId) }))
            {
                if (!byGene.TryGetValue(gene, out var list))
                    byGene[gene] = list = new List<TermProtein>();
                list.Add(new TermProtein(label, gene, r.LogFc, r.AdjPValue));
            }
        }

        return byGene;
    }

    /// <summary>Open (or refresh) the popup listing the significant proteins in a clicked term.</summary>
    private void ShowTermProteins(EnrichRow row)
    {
        if (!_enrichTermsById.TryGetValue(row.TermId, out var term))
            return;

        var proteins = term.IntersectingGenes
            .SelectMany(g => _enrichFeaturesByGene.GetValueOrDefault(g) ?? Enumerable.Empty<TermProtein>())
            .GroupBy(p => p.Protein)
            .Select(g => g.First())
            .OrderByDescending(p => Math.Abs(p.Log2FC))
            .Select(p => new TermProteinsWindow.TermProteinRow(p.Protein, p.Gene, p.Log2FC, p.AdjP))
            .ToList();

        if (_termProteinsWindow is null)
        {
            _termProteinsWindow = new TermProteinsWindow { Owner = this };
            _termProteinsWindow.Closed += (_, _) => _termProteinsWindow = null;
        }

        _termProteinsWindow.ShowTerm($"{term.TermId}  {term.TermName}", proteins.Count, term.PValue, proteins);
        _termProteinsWindow.Show();
        _termProteinsWindow.Activate();
    }

    /// <summary>
    /// Select the feature's protein or peptide in the running Skyline document, and say what
    /// happened in the status line.
    /// </summary>
    /// <remarks>
    /// <para>Shares the Dynamic Range pane's locator machinery rather than repeating it: the same
    /// cached document tree, the same "try each of PRISM's groups then fall back to the sequence's
    /// first occurrence" precedence, and the same distinction between "could not read the tree" and
    /// "this one is not in it" - which look identical to a user but need opposite responses.</para>
    /// <para>Nothing happens at all when PRISM is running standalone; that is reported once, in the
    /// status line, rather than being silent.</para>
    /// </remarks>
    private void SelectFeatureInSkyline(string featureId)
    {
        var identity = _diffDataset?.IdentityOf(featureId);
        if (identity is null)
            return;

        var label = identity.Describe();
        if (_session is null)
        {
            // Said rather than skipped: the click asked for something specific and nothing at all
            // happened, which is indistinguishable from a click that missed the point.
            DiffStatusText.Text = $"{label} - not attached to a running Skyline, so nothing to select.";
            return;
        }

        // The DATASET's level, not _diffLoadedLevel: the identity just came out of this dataset, and
        // resolving it against the other level's document tree would look up a peptide among
        // proteins.
        var level = _diffDataset!.Level == FeatureLevel.Protein
            ? AbundanceLevel.Protein
            : AbundanceLevel.Peptide;

        // ResolveLocator speaks AbundanceEntry, so the identity is presented as one. Only the
        // identity fields matter here - the abundance/rank fields are never read on this path.
        var entry = new AbundanceEntry(
            Key: identity.FeatureId,
            Label: identity.Label,
            Accession: identity.Accessions.FirstOrDefault(),
            Gene: identity.Genes.FirstOrDefault(),
            ProteinName: identity.ProteinNames.FirstOrDefault(),
            MeanAbundance: 0, Log10Abundance: 0, Rank: 0, SamplesUsed: 0)
        {
            ProteinGroups = identity.ProteinGroups,
            ProteinNames = identity.ProteinNames,
        };

        var locator = ResolveLocatorLocked(entry, level, out var viaFallback, out var treeUnavailable);
        if (locator is null)
        {
            DiffStatusText.Text = treeUnavailable
                ? $"{label} - could not read the document tree from Skyline (it may be busy; see the "
                  + "Log tab). Click again to retry."
                : $"{label} - no matching element in the Skyline document (PRISM's protein grouping "
                  + "can differ from the document's).";
            return;
        }

        var driver = new SkylineReportDriver(_session, Log);
        if (!driver.SelectElement(locator))
        {
            DiffStatusText.Text = $"Could not select {label} in Skyline.";
            return;
        }

        DiffStatusText.Text = $"Selected {label} in Skyline"
            + (viaFallback
                ? " - under the first protein in the document tree, since PRISM's grouping did not "
                  + "match a protein node."
                : ".");
    }

    private void ShowFeatureDetail(string featureId)
    {
        if (_diffDataset is null)
            return;
        var row = Array.IndexOf(_diffDataset.FeatureIds, featureId);
        if (row < 0)
            return;

        // Values and replicate names are collected together so they stay index-aligned: the
        // non-finite cells are skipped, and a name list built separately would silently shift.
        var aVals = new List<double>();
        var aReps = new List<string>();
        foreach (var s in _volcanoGroupA)
        {
            var v = _diffDataset.ExprLog2[row, s];
            if (!double.IsFinite(v))
                continue;
            aVals.Add(v);
            aReps.Add(_diffDataset.SampleIds[s]);
        }

        var bVals = new List<double>();
        var bReps = new List<string>();
        foreach (var s in _volcanoGroupB)
        {
            var v = _diffDataset.ExprLog2[row, s];
            if (!double.IsFinite(v))
                continue;
            bVals.Add(v);
            bReps.Add(_diffDataset.SampleIds[s]);
        }

        var label = _diffLabelById.GetValueOrDefault(featureId, featureId);
        _volcanoRowById.TryGetValue(featureId, out var dr);

        if (_featureDetailWindow is null)
        {
            _featureDetailWindow = new FeatureDetailWindow { Owner = this };
            _featureDetailWindow.Closed += (_, _) => _featureDetailWindow = null;
        }

        // A trend has no arms to box, so it gets the trajectory instead: abundance against the
        // trend column, the fitted line, and one faint line per subject where there are subjects.
        if (_volcanoTrendX is { } trendX)
        {
            var xs = new List<double>();
            var ys = new List<double>();
            var reps = new List<string>();
            var subs = new List<string>();
            var subjectLabels = DiffSubjectLabels();
            foreach (var s in _volcanoGroupA)
            {
                var v = _diffDataset.ExprLog2[row, s];
                if (!double.IsFinite(v) || s >= trendX.Length || !double.IsFinite(trendX[s]))
                    continue;
                xs.Add(trendX[s]);
                ys.Add(v);
                reps.Add(_diffDataset.SampleIds[s]);
                subs.Add(subjectLabels is not null && s < subjectLabels.Length
                    ? subjectLabels[s] ?? string.Empty
                    : string.Empty);
            }

            _featureDetailWindow.ShowTrendFeature(label, featureId, dr?.LogFc ?? double.NaN,
                dr?.AdjPValue ?? double.NaN, _volcanoAName, xs, ys, reps,
                DiffSelectedDesign() == DifferentialDesign.LinearTrendWithinSubject ? subs : null);
        }
        else
        {
            _featureDetailWindow.ShowFeature(label, featureId, dr?.LogFc ?? double.NaN,
                dr?.AdjPValue ?? double.NaN,
                _volcanoAName, aVals, aReps, _volcanoBName, bVals, bReps);
        }
        _featureDetailWindow.Show();
        _featureDetailWindow.Activate();
    }

    /// <summary>
    /// Draw the volcano. <paramref name="rule"/> is passed in rather than read here, so the points'
    /// coloring, the guide lines and the caller's hit count are one decision made once - and so
    /// this method never touches a WPF control that a worker thread might own.
    /// </summary>
    private void RenderVolcano(DifferentialResult res, SignificanceRule rule)
    {
        DiffPlot.Reset();
        var plt = DiffPlot.Plot;

        var bgX = new List<double>();
        var bgY = new List<double>();
        var sigX = new List<double>();
        var sigY = new List<double>();
        _volcanoPoints = new List<(ScottPlot.Coordinates, string)>(res.Rows.Count);
        _volcanoRowById = new Dictionary<string, DifferentialRow>(StringComparer.Ordinal);
        foreach (var r in res.Rows)
        {
            // The value the rule judges by, which is also what the axis is labeled with - so
            // the cut-off line always sits where the coloring changes. Plotting one p while
            // deciding on the other used to put the line at whatever raw p the weakest surviving
            // hit happened to have: a number that moved with the data and matched nothing the
            // reader could see.
            var y = -Math.Log10(Math.Max(rule.PValueOf(r), 1e-300));
            if (double.IsFinite(r.LogFc) && double.IsFinite(y))
                _volcanoPoints.Add((new ScottPlot.Coordinates(r.LogFc, y), r.FeatureId));
            _volcanoRowById[r.FeatureId] = r;
            if (rule.IsSignificant(r))
            {
                sigX.Add(r.LogFc);
                sigY.Add(y);
            }
            else
            {
                bgX.Add(r.LogFc);
                bgY.Add(y);
            }
        }

        AddMarkers(plt, bgX, bgY, "#b8c4d0", DiffPointSize, "not significant");
        AddMarkers(plt, sigX, sigY, "#d62728", DiffSigPointSize, "significant");
        // The guides ARE the rule, read from the same object that colored the points, so the red
        // region and the lines cannot disagree. A zero effect cut draws no vertical guides, because
        // a line at zero would read as a threshold rather than as its absence.
        if (rule.Log2FcThreshold > 0)
        {
            plt.Add.VerticalLine(rule.Log2FcThreshold);
            plt.Add.VerticalLine(-rule.Log2FcThreshold);
        }

        plt.Add.HorizontalLine(-Math.Log10(rule.PThreshold));

        // Both overlays belong to this Plot instance, so they are recreated with it and must be
        // re-seeded rather than carried over from the previous contrast.
        AddVolcanoOverlays(plt);

        // No legend. The colors mean only "past the lines drawn on the plot", and the status line
        // gives the count and the rule; the legend box was the one element that sat on the data.
        // Hidden explicitly: ScottPlot draws one by default for any series with LegendText.
        plt.HideLegend();
        plt.XLabel(DiffXAxisLabel());
        var corrected = DiffSelectedCorrection() != MultipleTesting.None;
        // Name what is actually on the axis - the rule decides, because the reader can now ask for
        // the raw p explicitly. Ties in an adjusted value show up as horizontal bands, which is a
        // property of the step-up transform and not a rendering fault. With Correct = None the
        // adjusted column simply holds the raw p, and calling it adjusted either way would be the
        // plainest kind of mislabeling.
        plt.YLabel(rule.YAxisLabel(corrected));
        PlotRenderer.StyleQcPlot(plt);
        DiffPlot.Refresh();
    }

    /// <summary>
    /// The Volcano's hidden hover readout (ring + text) and its selection ring, seeded anywhere -
    /// they are positioned when something is hovered or selected.
    /// </summary>
    private void AddVolcanoOverlays(ScottPlot.Plot plt)
    {
        var hover = plt.Add.Marker(
            0, 0, ScottPlot.MarkerShape.OpenCircle, DiffPointSize + 11, ScottPlot.Colors.Black);
        hover.IsVisible = false;
        _volcanoHoverMarker = hover;

        var text = plt.Add.Text(" ", 0, 0);
        // 20 to match the QC pane's readout: StyleQcPlot draws tick labels at 24, so a smaller size
        // makes the one piece of text a user leans in to read the smallest on the plot.
        PlotRenderer.StyleTextLabel(text, 20, bold: true);
        text.LabelFontColor = ScottPlot.Colors.Black;
        text.LabelBackgroundColor = ScottPlot.Colors.White.WithAlpha(0.85);
        text.LabelAlignment = ScottPlot.Alignment.LowerLeft;
        text.IsVisible = false;
        _volcanoHoverText = text;

        // Distinguishable from the hover ring at a glance: bigger, thicker, and in the significant
        // color rather than black, because the two can be on screen at the same time.
        var sel = plt.Add.Marker(
            0, 0, ScottPlot.MarkerShape.OpenCircle, DiffSigPointSize + 15,
            ScottPlot.Color.FromHex("#d62728"));
        sel.MarkerLineWidth = 3;
        sel.IsVisible = false;
        _volcanoSelMarker = sel;

        // A re-render is a new contrast, so nothing is selected until the user picks again - and the
        // old id would point into a hit list that no longer contains it.
        _volcanoSelectedId = null;
    }

    /// <summary>
    /// Show what the cursor is over: the feature's label plus the gene and protein it belongs to,
    /// which for a peptide is the only place that context appears on this plot.
    /// </summary>
    private void OnDiffPlotMouseMove(object sender, System.Windows.Input.MouseEventArgs e)
    {
        try
        {
            if (_volcanoHoverMarker is null || _volcanoHoverText is null)
                return;
            if (DiffSelectedView() != DiffView.Volcano || _volcanoPoints.Count == 0 || _diffDataset is null)
                return;

            var plt = DiffPlot.Plot;
            var pos = e.GetPosition(DiffPlot);
            var scale = DiffPlot.DisplayScale;
            var idx = QcPlotChrome.NearestPoint(
                _volcanoPoints.Select(pt => plt.GetPixel(pt.Loc)).ToList(),
                new ScottPlot.Pixel(pos.X * scale, pos.Y * scale));

            if (idx >= 0)
            {
                var point = _volcanoPoints[idx];
                _volcanoHoverMarker.Location = point.Loc;
                _volcanoHoverMarker.IsVisible = true;
                _volcanoHoverText.Location = point.Loc;
                _volcanoHoverText.LabelText = DescribeFeature(point.FeatureId);
                _volcanoHoverText.IsVisible = true;
                DiffPlot.Refresh();
            }
            else if (_volcanoHoverMarker.IsVisible)
            {
                _volcanoHoverMarker.IsVisible = false;
                _volcanoHoverText.IsVisible = false;
                DiffPlot.Refresh();
            }
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnDiffPlotMouseMove), ex);
        }
    }

    /// <summary>
    /// One line for a feature: its label, and the gene / protein / sharing behind it. Falls back to
    /// the id when the matrix carried no identity columns, rather than inventing any.
    /// </summary>
    private string DescribeFeature(string featureId)
    {
        var identity = _diffDataset?.IdentityOf(featureId);
        if (identity is not null)
            return identity.Describe();
        return _diffLabelById.GetValueOrDefault(featureId, featureId);
    }

    /// <summary>
    /// Make <paramref name="featureId"/> the selected feature everywhere: the ring on the plot, the
    /// row in the hit table, the per-feature boxplot, and the selection in Skyline.
    /// </summary>
    /// <remarks>
    /// One entry point for both directions deliberately. The grid and the plot each raise an event
    /// when the other drives it, so routing both through here with <see cref="_volcanoSyncing"/> set
    /// is what stops a click echoing back and forth - each echo would otherwise re-issue the Skyline
    /// RPC and reopen the detail window.
    /// </remarks>
    private void SelectVolcanoFeature(string featureId)
    {
        if (_volcanoSyncing)
            return;

        _volcanoSyncing = true;
        try
        {
            _volcanoSelectedId = featureId;
            HighlightVolcanoPoint(featureId);
            HighlightGridRow(featureId);
            ShowFeatureDetail(featureId);
            SelectFeatureInSkyline(featureId);
        }
        finally
        {
            _volcanoSyncing = false;
        }
    }

    /// <summary>Move the selection ring onto a feature's point, or hide it if it has none plotted.</summary>
    private void HighlightVolcanoPoint(string featureId)
    {
        if (_volcanoSelMarker is null)
            return;

        foreach (var (loc, id) in _volcanoPoints)
        {
            if (!string.Equals(id, featureId, StringComparison.Ordinal))
                continue;
            _volcanoSelMarker.Location = loc;
            _volcanoSelMarker.IsVisible = true;
            BringIntoView(loc);
            DiffPlot.Refresh();
            return;
        }

        // A feature with a non-finite fold change or p-value is in the table but not on the plot.
        _volcanoSelMarker.IsVisible = false;
        DiffPlot.Refresh();
    }

    /// <summary>
    /// Pan the Volcano so a selected point is actually on screen, keeping the current zoom.
    /// </summary>
    /// <remarks>
    /// Selecting from the hit table can name a point that is outside the view - the table is ranked
    /// by significance and is not affected by zooming, so after zooming into one corner most rows
    /// refer to points that are not visible. Ringing one of those rings nothing the user can see,
    /// which looks identical to the selection having failed.
    /// <para>A minimal pan rather than a re-fit or a re-center: the zoom the user chose is
    /// deliberate, and re-centering on every off-screen pick makes the plot jump further than it
    /// needs to. The point is brought just inside the edge it was past, with a margin so it does not
    /// sit under the axis.</para>
    /// </remarks>
    private void BringIntoView(ScottPlot.Coordinates loc)
    {
        var plt = DiffPlot.Plot;
        var limits = plt.Axes.GetLimits();
        var xSpan = limits.Right - limits.Left;
        var ySpan = limits.Top - limits.Bottom;
        if (!(xSpan > 0) || !(ySpan > 0))
            return; // an unrendered plot has no meaningful limits to preserve

        const double marginFraction = 0.08;
        var xMargin = xSpan * marginFraction;
        var yMargin = ySpan * marginFraction;

        var dx = 0.0;
        if (loc.X < limits.Left + xMargin)
            dx = loc.X - (limits.Left + xMargin);
        else if (loc.X > limits.Right - xMargin)
            dx = loc.X - (limits.Right - xMargin);

        var dy = 0.0;
        if (loc.Y < limits.Bottom + yMargin)
            dy = loc.Y - (limits.Bottom + yMargin);
        else if (loc.Y > limits.Top - yMargin)
            dy = loc.Y - (limits.Top - yMargin);

        if (dx == 0 && dy == 0)
            return; // already comfortably in view: leave the axes exactly as the user set them

        plt.Axes.SetLimits(
            limits.Left + dx, limits.Right + dx,
            limits.Bottom + dy, limits.Top + dy);
    }

    /// <summary>Select a feature's row in the hit table and scroll it into view.</summary>
    private void HighlightGridRow(string featureId)
    {
        if (DiffGrid.ItemsSource is not IEnumerable<VolcanoRow> rows)
            return;

        var row = rows.FirstOrDefault(r => string.Equals(r.FeatureId, featureId, StringComparison.Ordinal));
        if (row is null)
            return;

        DiffGrid.SelectedItem = row;
        DiffGrid.ScrollIntoView(row);
    }

    private void RenderDetection(IReadOnlyList<DetectionRow> rows, SignificanceRule rule, bool corrected)
    {
        DiffPlot.Reset();
        var plt = DiffPlot.Plot;

        var bgX = new List<double>();
        var bgY = new List<double>();
        var sigX = new List<double>();
        var sigY = new List<double>();
        foreach (var r in rows)
        {
            // Judge and plot the SAME p the rule uses (raw or adjusted), so the axis label, the
            // threshold line and the coloring all agree - matching the Volcano. The effect-size cut
            // does not apply here: a detection effect is a rate difference, not a log2 fold change.
            var p = rule.UseAdjusted ? r.Q : r.P;
            if (!double.IsFinite(p))
                continue;
            var x = r.RateB - r.RateA;
            var y = -Math.Log10(Math.Max(p, 1e-300));
            if (p < rule.PThreshold)
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

        AddMarkers(plt, bgX, bgY, "#b8c4d0", DiffPointSize, "not significant");
        AddMarkers(plt, sigX, sigY, "#2ca02c", DiffSigPointSize, "significant");
        plt.Add.VerticalLine(0.0);
        plt.Add.HorizontalLine(-Math.Log10(rule.PThreshold));
        plt.HideLegend(); // as on the Volcano: the lines and the status line say what the colors mean
        plt.XLabel("detection rate difference (B - A)");
        plt.YLabel(rule.YAxisLabel(corrected));
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
