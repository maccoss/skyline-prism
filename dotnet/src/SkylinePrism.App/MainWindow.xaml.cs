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
            Title = "Open provenance (metadata.json)",
            Filter = "PRISM provenance (metadata.json)|metadata.json|JSON (*.json)|*.json|All files (*.*)|*.*",
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
            LoadPcaPlot(outputDir);
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
        var result = PrismPipeline.Run(inputs, outputDir, config, reports.ReplicatesCsv, Log);
        Log($"Pipeline complete: {result.NPeptides} peptides, {result.NProteins} proteins, "
            + $"{result.NSamples} samples, {result.Batches.Count} batch(es).");
    }

    private void LoadPcaPlot(string outputDir)
    {
        try
        {
            var correctedPeptides = Path.Combine(outputDir, "corrected_peptides.parquet");
            if (!File.Exists(correctedPeptides))
                return;

            var types = ReadSampleTypes(Path.Combine(outputDir, "sample_metadata.csv"));
            var table = ParquetTable.Load(correctedPeptides);
            var sampleCols = table.ColumnNames.Where(types.ContainsKey).ToList();
            if (sampleCols.Count < 3)
                return;

            // Build [nSamples, nFeatures] LOG2 matrix.
            var nSamples = sampleCols.Count;
            var nFeatures = table.RowCount;
            var byFeature = sampleCols.Select(table.GetDouble).ToList();
            var samplesByFeatures = new double[nSamples, nFeatures];
            for (var s = 0; s < nSamples; s++)
                for (var f = 0; f < nFeatures; f++)
                {
                    var v = byFeature[s][f];
                    samplesByFeatures[s, f] = v.HasValue ? Math.Log2(v.Value) : double.NaN;
                }

            var scores = Pca.Fit2D(samplesByFeatures);
            var typeLabels = sampleCols.Select(c => types.GetValueOrDefault(c, "unknown")).ToList();

            var plt = PcaPlot.Plot;
            plt.Clear();
            var groups = new Dictionary<string, (List<double> X, List<double> Y)>();
            for (var i = 0; i < nSamples; i++)
            {
                var t = typeLabels[i];
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
            plt.Title("Peptide PCA (corrected)");
            plt.XLabel("PC1");
            plt.YLabel("PC2");
            PcaPlot.Refresh();
        }
        catch (Exception ex)
        {
            Log("(PCA plot skipped: " + ex.Message + ")");
        }
    }

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
