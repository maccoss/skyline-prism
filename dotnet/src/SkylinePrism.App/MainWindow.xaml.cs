using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.Win32;
using ScottPlot;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Numerics;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.Visualization;
using SkylinePrism.Skyline;

namespace SkylinePrism.App;

/// <summary>
/// The Skyline external-tool window: connects to the launching Skyline instance over
/// JSON-RPC, drives it to export the PRISM + Replicates reports, runs the PRISM pipeline,
/// generates the QC report, and shows an interactive PCA of the corrected peptides.
/// </summary>
public partial class MainWindow : Window
{
    private SkylineSession? _session;
    private string? _lastReportPath;

    public MainWindow()
    {
        InitializeComponent();

        OutputDirBox.Text = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "PRISM-output");

        try
        {
            _session = SkylineSession.FromArguments(App.LaunchArgs);
            StatusText.Text = $"Connected to Skyline (pipe: {_session.PipeName}).";
        }
        catch (Exception ex)
        {
            _session = null;
            StatusText.Text =
                "Not connected to a running Skyline instance. Launch this tool from Skyline's Tools menu. "
                + $"({ex.Message})";
        }

        Log("Skyline-PRISM tool ready.");
        Log("Diagnostic log: " + App.LogFilePath);
        Log(_session is not null
            ? "Connected. Set an output directory and click Run PRISM."
            : "Not connected to Skyline. Launch this tool from Skyline's Tools menu.");

        MetadataReportCombo.Items.Add(DefaultMetadataItem);
        MetadataReportCombo.SelectedIndex = 0;

        TransitionRollupCombo.SelectedIndex = 0;
        PeptideNormCombo.SelectedIndex = 0;
        ProteinRollupCombo.SelectedIndex = 0;
        ProteinNormCombo.SelectedIndex = 0;
        QcViewCombo.SelectedIndex = 0;
        QcLevelCombo.SelectedIndex = 0;
        QcPlotCombo.SelectedIndex = 0;

        if (_session is not null)
        {
            _ = LoadReportsAsync();
            _ = LoadLibrariesAsync();
        }
    }

    private async Task LoadLibrariesAsync()
    {
        if (_session is null)
            return;
        try
        {
            var session = _session;
            var libs = await Task.Run(() => new SkylineReportDriver(session).ListDocumentLibraries());
            LibraryCombo.Items.Clear();
            foreach (var l in libs)
                LibraryCombo.Items.Add(l);
            if (LibraryCombo.Items.Count > 0)
                LibraryCombo.SelectedIndex = 0;
            Log($"Found {libs.Count} spectral library file(s) next to the document.");
        }
        catch (Exception ex)
        {
            Log("(could not scan for spectral libraries: " + ex.Message + ")");
        }
    }

    private void OnTransitionRollupChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (LibraryRow is not null)
            LibraryRow.IsEnabled = ComboText(TransitionRollupCombo, "sum") == "library_assist";
    }

    private void OnBrowseLibrary(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "Select spectral library",
            Filter = "BiblioSpec library (*.blib)|*.blib|All files (*.*)|*.*",
        };
        if (dlg.ShowDialog() == true)
        {
            if (!LibraryCombo.Items.Contains(dlg.FileName))
                LibraryCombo.Items.Add(dlg.FileName);
            LibraryCombo.Text = dlg.FileName;
        }
    }

    private void OnOpenProvenance(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "Open provenance (parameters.json)",
            Filter = "PRISM provenance|parameters.json;metadata.json|JSON (*.json)|*.json|All files (*.*)|*.*",
        };
        if (dlg.ShowDialog() != true)
            return;
        try
        {
            var config = Provenance.LoadConfig(dlg.FileName);
            ApplyConfigToUi(config);
            Log($"Loaded settings from provenance: {dlg.FileName}");
        }
        catch (Exception ex)
        {
            Log("Could not load provenance: " + ex.Message);
            MessageBox.Show("Could not load provenance: " + ex.Message, "Open provenance",
                MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    /// <summary>Populate the Settings controls from a loaded config (provenance reproduction).</summary>
    private void ApplyConfigToUi(PrismConfig c)
    {
        SelectCombo(TransitionRollupCombo, c.TransitionRollup.Method);
        MinTransitionsBox.Text = c.TransitionRollup.MinTransitions.ToString();
        UseMs1Check.IsChecked = c.TransitionRollup.UseMs1;
        if (!string.IsNullOrWhiteSpace(c.TransitionRollup.LibraryPath))
        {
            if (!LibraryCombo.Items.Contains(c.TransitionRollup.LibraryPath))
                LibraryCombo.Items.Add(c.TransitionRollup.LibraryPath);
            LibraryCombo.Text = c.TransitionRollup.LibraryPath;
        }
        LibraryRow.IsEnabled = c.TransitionRollup.Method == "library_assist";

        SelectCombo(PeptideNormCombo, c.GlobalNormalization.Method);
        ExcludeOutliersCheck.IsChecked = c.SampleOutlierDetection.Action == "exclude";
        PeptideBatchCheck.IsChecked = c.BatchCorrection.Enabled && c.BatchCorrection.PeptideLevel;
        ProteinBatchCheck.IsChecked = c.BatchCorrection.Enabled && c.BatchCorrection.ProteinLevel;
        ParsimonyCheck.IsChecked = c.Parsimony.Enabled;
        SelectCombo(ProteinRollupCombo, c.ProteinRollup.Method);
        MinPeptidesBox.Text = c.ProteinRollup.MinPeptides.ToString();
        SelectCombo(ProteinNormCombo, c.ProteinNormalization.Method);
    }

    private static void SelectCombo(System.Windows.Controls.ComboBox cb, string value)
    {
        foreach (var item in cb.Items)
        {
            if (item is System.Windows.Controls.ComboBoxItem ci
                && string.Equals(ci.Content?.ToString(), value, StringComparison.OrdinalIgnoreCase))
            {
                cb.SelectedItem = ci;
                return;
            }
        }
        if (cb.IsEditable)
            cb.Text = value;
    }

    /// <summary>Build the pipeline config from the Settings tab controls (call on the UI thread).</summary>
    private PrismConfig BuildConfigFromUi()
    {
        var c = new PrismConfig();
        c.TransitionRollup.Method = ComboText(TransitionRollupCombo, "sum");
        c.TransitionRollup.MinTransitions = ParseInt(MinTransitionsBox.Text, 3);
        c.TransitionRollup.UseMs1 = UseMs1Check.IsChecked == true;
        if (c.TransitionRollup.Method == "library_assist")
        {
            var lib = LibraryCombo.Text?.Trim();
            c.TransitionRollup.LibraryPath = string.IsNullOrWhiteSpace(lib) ? null : lib;
        }

        c.GlobalNormalization.Method = ComboText(PeptideNormCombo, "rt_lowess");

        c.SampleOutlierDetection.Enabled = true;
        c.SampleOutlierDetection.Action = ExcludeOutliersCheck.IsChecked == true ? "exclude" : "report";

        var pepBatch = PeptideBatchCheck.IsChecked == true;
        var protBatch = ProteinBatchCheck.IsChecked == true;
        c.BatchCorrection.Enabled = pepBatch || protBatch;
        c.BatchCorrection.PeptideLevel = pepBatch;
        c.BatchCorrection.ProteinLevel = protBatch;

        c.Parsimony.Enabled = ParsimonyCheck.IsChecked == true;

        c.ProteinRollup.Method = ComboText(ProteinRollupCombo, "median_polish");
        c.ProteinRollup.MinPeptides = ParseInt(MinPeptidesBox.Text, 3);
        c.ProteinNormalization.Method = ComboText(ProteinNormCombo, "median");

        c.QcReport.Enabled = true;
        c.QcReport.SavePlots = false;
        return c;
    }

    private static string ComboText(System.Windows.Controls.ComboBox cb, string fallback)
        => (cb.SelectedItem as System.Windows.Controls.ComboBoxItem)?.Content?.ToString()
           ?? (string.IsNullOrWhiteSpace(cb.Text) ? fallback : cb.Text);

    private static int ParseInt(string? s, int fallback)
        => int.TryParse(s, out var v) && v > 0 ? v : fallback;

    private const string DefaultMetadataItem = "(default) PRISM-Replicates";

    private async Task LoadReportsAsync()
    {
        if (_session is null)
            return;
        try
        {
            var session = _session;
            var reports = await Task.Run(() => new SkylineReportDriver(session).ListAvailableReports());
            MetadataReportCombo.Items.Clear();
            MetadataReportCombo.Items.Add(DefaultMetadataItem);
            foreach (var r in reports.OrderBy(x => x, StringComparer.OrdinalIgnoreCase))
                MetadataReportCombo.Items.Add(r);
            MetadataReportCombo.SelectedIndex = 0;
            Log($"Loaded {reports.Count} report(s) from the document into the Metadata report list.");
        }
        catch (Exception ex)
        {
            Log("(could not load the document's report list: " + ex.Message + ")");
        }
    }

    private void OnBrowse(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog { Title = "Select output directory" };
        if (dlg.ShowDialog() == true)
            OutputDirBox.Text = dlg.FolderName;
    }

    private async void OnRun(object sender, RoutedEventArgs e)
    {
        if (_session is null)
        {
            Log("No Skyline connection. Start this tool from within Skyline.");
            return;
        }

        RunButton.IsEnabled = false;
        OpenReportButton.IsEnabled = false;
        LogBox.Clear();

        var outputDir = OutputDirBox.Text;
        var batchColumn = BatchColumnBox.Text?.Trim();
        var metadataReport = MetadataReportCombo.Text?.Trim();
        if (string.IsNullOrWhiteSpace(metadataReport) || metadataReport == DefaultMetadataItem)
            metadataReport = null;
        var config = BuildConfigFromUi();
        if (!string.IsNullOrWhiteSpace(batchColumn))
            config.Metadata.BatchColumn = batchColumn;
        var session = _session;

        try
        {
            await Task.Run(() => RunPipeline(session, outputDir, metadataReport, config));
            _lastReportPath = Path.Combine(outputDir, "qc_report.html");
            OpenReportButton.IsEnabled = File.Exists(_lastReportPath);
            LoadQcData(outputDir);
            Log("Done.");
        }
        catch (Exception ex)
        {
            Log("ERROR: " + ex.Message);
            App.WriteLog("Run failed: " + ex);
            MessageBox.Show(
                ex.ToString() + Environment.NewLine + Environment.NewLine
                + "See the full log at:" + Environment.NewLine + App.LogFilePath,
                "PRISM run failed", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            RunButton.IsEnabled = true;
        }
    }

    private void RunPipeline(SkylineSession session, string outputDir, string? metadataReport, PrismConfig config)
    {
        Directory.CreateDirectory(outputDir);
        var reportsDir = Path.Combine(outputDir, "skyline-reports");

        Log("Exporting reports from Skyline...");
        var driver = new SkylineReportDriver(session, Log);
        // When no explicit metadata report is chosen, build our PRISM-Replicates report to
        // include the user's batch annotation column (annotation_<name>).
        var batchAnnotation = metadataReport is null ? config.Metadata.BatchColumn : null;
        var reports = driver.Export(reportsDir, metadataReport, batchAnnotation);

        Log($"Running PRISM pipeline on the {(reports.InputIsParquet ? "parquet" : "CSV")} report "
            + $"({Path.GetFileName(reports.InputPath)})...");

        var inputs = new List<string> { reports.InputPath };
        var metadataPaths = reports.ReplicatesCsv is null ? null : new[] { reports.ReplicatesCsv };
        var result = PrismPipeline.Run(inputs, outputDir, config, metadataPaths, Log);
        Log($"Pipeline complete: {result.NPeptides} peptides, {result.NProteins} proteins, "
            + $"{result.NSamples} samples, {result.Batches.Count} batch(es).");
    }

    // Cached QC matrices ([features, samples], LOG2) keyed by "view|level" (view in raw/corrected).
    private readonly Dictionary<string, (double[,] FeaturesBySamples, List<string> Samples)> _qcData = new();
    private Dictionary<string, string> _qcTypes = new();

    private void LoadQcData(string outputDir)
    {
        try
        {
            _qcTypes = ReadSampleTypes(Path.Combine(outputDir, "sample_metadata.csv"));
            _qcData.Clear();
            LoadQcMatrix("raw|peptide", Path.Combine(outputDir, "peptides_rollup.parquet"), isLinear: false);
            LoadQcMatrix("corrected|peptide", Path.Combine(outputDir, "corrected_peptides.parquet"), isLinear: true);
            LoadQcMatrix("raw|protein", Path.Combine(outputDir, "proteins_raw.parquet"), isLinear: false);
            LoadQcMatrix("corrected|protein", Path.Combine(outputDir, "corrected_proteins.parquet"), isLinear: true);
        }
        catch (Exception ex)
        {
            Log("(QC data load skipped: " + ex.Message + ")");
        }
        RenderQc();
    }

    private void LoadQcMatrix(string key, string path, bool isLinear)
    {
        if (!File.Exists(path))
            return;
        var table = ParquetTable.Load(path);
        var sampleCols = table.ColumnNames.Where(_qcTypes.ContainsKey).ToList();
        if (sampleCols.Count == 0)
            return;
        var n = table.RowCount;
        var m = new double[n, sampleCols.Count];
        for (var j = 0; j < sampleCols.Count; j++)
        {
            var col = table.GetDouble(sampleCols[j]);
            for (var i = 0; i < n; i++)
            {
                var v = col[i];
                m[i, j] = !v.HasValue ? double.NaN : (isLinear ? Math.Log2(v.Value) : v.Value);
            }
        }
        _qcData[key] = (m, sampleCols);
    }

    private void OnQcChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e) => RenderQc();

    private void RenderQc()
    {
        if (QcViewCombo is null || _qcData.Count == 0)
            return;
        var view = ComboText(QcViewCombo, "Corrected").ToLowerInvariant();
        var level = ComboText(QcLevelCombo, "Peptide").ToLowerInvariant();
        var kind = ComboText(QcPlotCombo, "PCA");
        var plt = QcPlot.Plot;
        plt.Clear();
        if (!_qcData.TryGetValue($"{view}|{level}", out var d) || d.Samples.Count < 2)
        {
            plt.Title($"No {view} {level} data yet - run PRISM first.");
            QcPlot.Refresh();
            return;
        }
        var types = d.Samples.Select(s => _qcTypes.GetValueOrDefault(s, "unknown")).ToList();
        try
        {
            switch (kind)
            {
                case "CV distribution": DrawCv(plt, d.FeaturesBySamples, types, level, view); break;
                case "Intensity": DrawIntensity(plt, d.FeaturesBySamples, types, level, view); break;
                default: DrawPca(plt, d.FeaturesBySamples, types, level, view); break;
            }
        }
        catch (Exception ex)
        {
            plt.Title("render failed: " + ex.Message);
        }
        QcPlot.Refresh();
    }

    private static void DrawPca(Plot plt, double[,] featuresBySamples, List<string> types, string level, string view)
    {
        var nF = featuresBySamples.GetLength(0);
        var nS = featuresBySamples.GetLength(1);
        var samplesByFeatures = new double[nS, nF];
        for (var f = 0; f < nF; f++)
            for (var s = 0; s < nS; s++)
                samplesByFeatures[s, f] = featuresBySamples[f, s];
        var scores = Pca.Fit2D(samplesByFeatures);
        var groups = new Dictionary<string, (List<double> X, List<double> Y)>();
        for (var i = 0; i < nS; i++)
        {
            var t = types[i];
            if (!groups.TryGetValue(t, out var g))
                groups[t] = g = (new List<double>(), new List<double>());
            g.X.Add(scores[i, 0]);
            g.Y.Add(scores[i, 1]);
        }
        foreach (var (type, g) in groups.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            var markers = plt.Add.Markers(g.X.ToArray(), g.Y.ToArray());
            markers.Color = Color.FromHex(PlotRenderer.TypeColors.GetValueOrDefault(type, "#7f7f7f"));
            markers.MarkerSize = 8;
            markers.LegendText = type;
        }
        plt.ShowLegend();
        plt.Title($"{Cap(level)} PCA ({view})");
        plt.XLabel("PC1");
        plt.YLabel("PC2");
    }

    private static void DrawCv(Plot plt, double[,] featuresBySamples, List<string> types, string level, string view)
    {
        var refIdx = IndicesOf(types, "reference");
        var qcIdx = IndicesOf(types, "qc");
        var drew = false;
        if (refIdx.Count >= 2)
        {
            AddCvHist(plt, CvMetrics.PerFeatureCvs(featuresBySamples, refIdx), "#d62728", "Reference");
            drew = true;
        }
        if (qcIdx.Count >= 2)
        {
            AddCvHist(plt, CvMetrics.PerFeatureCvs(featuresBySamples, qcIdx), "#ff7f0e", "QC");
            drew = true;
        }
        if (!drew)
            AddCvHist(plt, CvMetrics.PerFeatureCvs(featuresBySamples, Enumerable.Range(0, types.Count).ToList()),
                "#1f77b4", "All samples");
        plt.ShowLegend();
        plt.Title($"{Cap(level)} CV distribution ({view})");
        plt.XLabel("CV (%)");
        plt.YLabel("Count");
    }

    private static void AddCvHist(Plot plt, double[] cvs, string colorHex, string label)
    {
        var valid = cvs.Where(c => !double.IsNaN(c)).ToArray();
        if (valid.Length == 0)
            return;
        var maxCv = Math.Min(valid.Max(), 100.0);
        const int bins = 30;
        var binWidth = Math.Max(maxCv / bins, 1e-6);
        var counts = new double[bins];
        var centers = new double[bins];
        for (var b = 0; b < bins; b++)
            centers[b] = (b + 0.5) * binWidth;
        foreach (var cv in valid)
        {
            var b = (int)Math.Min(cv / binWidth, bins - 1);
            if (b >= 0)
                counts[b]++;
        }
        var bars = plt.Add.Bars(centers, counts);
        bars.Color = Color.FromHex(colorHex).WithAlpha((byte)140);
        var med = Stats.NanMedian(cvs);
        var line = plt.Add.VerticalLine(med);
        line.Color = Color.FromHex(colorHex);
        line.LineWidth = 2;
        line.LinePattern = LinePattern.Dashed;
        line.LegendText = $"{label} median {med:0.0}%";
    }

    private static void DrawIntensity(Plot plt, double[,] featuresBySamples, List<string> types, string level, string view)
    {
        var nF = featuresBySamples.GetLength(0);
        var nS = featuresBySamples.GetLength(1);
        var xs = new List<double>();
        var ys = new List<double>();
        var buf = new double[nF];
        for (var s = 0; s < nS; s++)
        {
            var n = 0;
            for (var f = 0; f < nF; f++)
            {
                var v = featuresBySamples[f, s];
                if (!double.IsNaN(v))
                    buf[n++] = v;
            }
            if (n == 0)
                continue;
            xs.Add(s);
            ys.Add(Stats.NanMedian(buf.AsSpan(0, n)));
        }
        var markers = plt.Add.Markers(xs.ToArray(), ys.ToArray());
        markers.MarkerSize = 5;
        plt.Title($"{Cap(level)} median log2 intensity ({view})");
        plt.XLabel("Sample index");
        plt.YLabel("Median log2 intensity");
    }

    private static List<int> IndicesOf(List<string> types, string type)
    {
        var idx = new List<int>();
        for (var i = 0; i < types.Count; i++)
            if (types[i] == type)
                idx.Add(i);
        return idx;
    }

    private static string Cap(string s) => s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s[1..];

    private void OnOpenReport(object sender, RoutedEventArgs e)
    {
        if (_lastReportPath is not null && File.Exists(_lastReportPath))
            Process.Start(new ProcessStartInfo(_lastReportPath) { UseShellExecute = true });
    }

    private static Dictionary<string, string> ReadSampleTypes(string metadataCsv)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!File.Exists(metadataCsv))
            return map;
        var lines = File.ReadAllLines(metadataCsv);
        if (lines.Length < 2)
            return map;
        var header = lines[0].Split(',');
        var idIdx = Array.IndexOf(header, "sample_id");
        var typeIdx = Array.IndexOf(header, "sample_type");
        if (idIdx < 0 || typeIdx < 0)
            return map;
        for (var i = 1; i < lines.Length; i++)
        {
            var f = lines[i].Split(',');
            if (f.Length > Math.Max(idIdx, typeIdx))
                map[f[idIdx]] = f[typeIdx];
        }
        return map;
    }

    private void Log(string message)
    {
        // Persist to the diagnostic file first so a hard crash still leaves a trail.
        App.WriteLog(message);

        if (!Dispatcher.CheckAccess())
        {
            // BeginInvoke (async) so a UI-thread stall never deadlocks the worker.
            Dispatcher.BeginInvoke(() => AppendLine(message));
            return;
        }
        AppendLine(message);
    }

    private void AppendLine(string message)
    {
        LogBox.AppendText(message + Environment.NewLine);
        LogBox.ScrollToEnd();
    }
}
