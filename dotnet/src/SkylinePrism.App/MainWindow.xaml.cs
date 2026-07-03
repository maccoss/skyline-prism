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
    private bool _isRunning;

    public MainWindow()
    {
        InitializeComponent();

        // Run stays disabled until an output directory is set. When connected to a saved document,
        // SetDefaultOutputDirAsync pre-fills "<document folder>/PRISM-Output"; otherwise the box stays
        // empty so the user must choose (no more defaulting into OneDrive-synced Documents).
        RunButton.IsEnabled = false;

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
        // QcGroupByCombo / QcGroupCombo are populated from the Replicates report after a run.

        if (_session is not null)
        {
            _ = SetDefaultOutputDirAsync();
            _ = LoadReportsAsync();
            _ = LoadLibrariesAsync();
        }
    }

    // Suggest "<document folder>/PRISM-Output" as the output directory. The document path comes from
    // Skyline over RPC, so fetch it off the UI thread. Leaves the box empty (Run stays disabled) for
    // an unsaved document or if the user has already typed a directory.
    private async Task SetDefaultOutputDirAsync()
    {
        if (_session is null)
            return;
        try
        {
            var session = _session;
            var docPath = await Task.Run(() =>
            {
                try { return session.Execute(c => c.GetDocumentPath()); }
                catch { return null; }
            });
            var dir = string.IsNullOrWhiteSpace(docPath) ? null : Path.GetDirectoryName(docPath);
            if (!string.IsNullOrEmpty(dir) && string.IsNullOrWhiteSpace(OutputDirBox.Text))
                OutputDirBox.Text = Path.Combine(dir, "PRISM-Output");
        }
        catch (Exception ex)
        {
            Log("(could not derive a default output directory: " + ex.Message + ")");
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
        // SelectedItem is current during SelectionChanged; cb.Text lags a tick for string-item combos
        // (which would populate the value list for the previously selected Group-by column), so prefer it.
        => (cb.SelectedItem as System.Windows.Controls.ComboBoxItem)?.Content?.ToString()
           ?? (cb.SelectedItem as string)
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

    // Show the prism CLI command + config YAML that reproduces the current GUI settings, in a window
    // with buttons to copy the command or the config to the clipboard.
    private void OnShowCommandLine(object sender, RoutedEventArgs e)
    {
        var config = BuildConfigFromUi();
        var batchColumn = BatchColumnBox.Text?.Trim();
        if (!string.IsNullOrWhiteSpace(batchColumn))
            config.Metadata.BatchColumn = batchColumn;

        var outputDir = string.IsNullOrWhiteSpace(OutputDirBox.Text) ? "<output-dir>" : OutputDirBox.Text.Trim();
        var report = Path.Combine(outputDir, "skyline-reports", "PRISM.parquet");
        var metadata = Path.Combine(outputDir, "skyline-reports", "Metadata.csv");
        var configPath = Path.Combine(outputDir, "prism-config.yaml");
        var yaml = SerializeConfig(config);

        var command = $"prism run -i \"{report}\" -o \"{outputDir}\" -c \"{configPath}\" -m \"{metadata}\"";
        var body =
            "Command-line equivalent of the current GUI settings.\r\n\r\n" +
            "The tool exports the Skyline report into <output>\\skyline-reports\\ and runs the pipeline. To\r\n" +
            "reproduce it with the prism CLI: save the config below as prism-config.yaml, then run:\r\n\r\n" +
            command + "\r\n\r\n" +
            "# ---------- prism-config.yaml ----------\r\n" + yaml;

        ShowCopyableTextWindow("PRISM command line", body, command, yaml);
    }

    private static string SerializeConfig(PrismConfig config) =>
        new YamlDotNet.Serialization.SerializerBuilder()
            .WithNamingConvention(YamlDotNet.Serialization.NamingConventions.UnderscoredNamingConvention.Instance)
            .Build()
            .Serialize(config);

    private void ShowCopyableTextWindow(string title, string body, string command, string yaml)
    {
        var box = new System.Windows.Controls.TextBox
        {
            Text = body,
            IsReadOnly = true,
            FontFamily = new System.Windows.Media.FontFamily("Consolas"),
            FontSize = 12,
            TextWrapping = TextWrapping.NoWrap,
            AcceptsReturn = true,
            VerticalScrollBarVisibility = System.Windows.Controls.ScrollBarVisibility.Auto,
            HorizontalScrollBarVisibility = System.Windows.Controls.ScrollBarVisibility.Auto,
            Margin = new Thickness(0, 0, 0, 8),
        };

        var copyCmd = new System.Windows.Controls.Button { Content = "Copy command", Padding = new Thickness(12, 4, 12, 4), Margin = new Thickness(0, 0, 8, 0) };
        var copyYaml = new System.Windows.Controls.Button { Content = "Copy config (YAML)", Padding = new Thickness(12, 4, 12, 4), Margin = new Thickness(0, 0, 8, 0) };
        var close = new System.Windows.Controls.Button { Content = "Close", Padding = new Thickness(12, 4, 12, 4) };
        copyCmd.Click += (_, _) => TrySetClipboard(command);
        copyYaml.Click += (_, _) => TrySetClipboard(yaml);

        var buttons = new System.Windows.Controls.StackPanel
        {
            Orientation = System.Windows.Controls.Orientation.Horizontal,
            HorizontalAlignment = System.Windows.HorizontalAlignment.Right,
        };
        buttons.Children.Add(copyCmd);
        buttons.Children.Add(copyYaml);
        buttons.Children.Add(close);

        var grid = new System.Windows.Controls.Grid { Margin = new Thickness(12) };
        grid.RowDefinitions.Add(new System.Windows.Controls.RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        grid.RowDefinitions.Add(new System.Windows.Controls.RowDefinition { Height = GridLength.Auto });
        System.Windows.Controls.Grid.SetRow(box, 0);
        System.Windows.Controls.Grid.SetRow(buttons, 1);
        grid.Children.Add(box);
        grid.Children.Add(buttons);

        var win = new Window
        {
            Title = title,
            Width = 860,
            Height = 580,
            Content = grid,
            Owner = this,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
        };
        close.Click += (_, _) => win.Close();
        win.ShowDialog();
    }

    private static void TrySetClipboard(string text)
    {
        try { System.Windows.Clipboard.SetText(text); }
        catch { /* the clipboard may be briefly locked by another app; ignore */ }
    }

    private async void OnRun(object sender, RoutedEventArgs e)
    {
        if (_session is null)
        {
            Log("No Skyline connection. Start this tool from within Skyline.");
            return;
        }

        _isRunning = true;
        RunButton.IsEnabled = false;
        OpenReportButton.IsEnabled = false;
        LogBox.Clear();
        MainTabs.SelectedItem = LogTab; // show progress as it runs

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
            // Load the QC matrices (parquet I/O) OFF the UI thread - reading the just-written outputs
            // can block for a long time when the output dir is on OneDrive / scanned by Defender, and
            // doing it on the UI thread would freeze the window.
            Log("Loading QC data for the plots...");
            var reportPath = Path.Combine(outputDir, "qc_report.html");
            var reportExists = await Task.Run(() =>
            {
                LoadQcMatrices(outputDir);
                return File.Exists(reportPath);
            });
            _lastReportPath = reportPath;
            OpenReportButton.IsEnabled = reportExists;
            PopulateGroupCombos(); // fill Group-by / value from the Replicates report
            RenderQc(); // draws on the UI thread (cheap; the ScottPlot control requires it)
            Log("Done.");
            MainTabs.SelectedItem = QcTab; // land on the plots when the run finishes
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
            _isRunning = false;
            RunButton.IsEnabled = !string.IsNullOrWhiteSpace(OutputDirBox.Text);
        }
    }

    // Enable Run only once an output directory is set (and no run is in progress).
    private void OnOutputDirChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
    {
        if (RunButton is not null)
            RunButton.IsEnabled = !_isRunning && !string.IsNullOrWhiteSpace(OutputDirBox.Text);
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
    private readonly Dictionary<string, (double[,] FeaturesBySamples, List<string> Samples, double[]? MeanRt)> _qcData = new();
    private Dictionary<string, string> _qcTypes = new();
    private string? _qcOutputDir;
    // Replicate annotations from the exported Replicates report: replicate -> (column -> value).
    private readonly Dictionary<string, Dictionary<string, string>> _replicateAnn = new(StringComparer.Ordinal);
    private List<string> _groupColumns = new();
    private bool _suppressQcRender;

    // Loads the QC parquet matrices into _qcData. Safe to call on a background thread (no UI access
    // except Log, which marshals to the dispatcher). RenderQc() is invoked separately on the UI thread.
    private void LoadQcMatrices(string outputDir)
    {
        try
        {
            _qcOutputDir = outputDir;
            _qcTypes = ReadSampleTypes(Path.Combine(outputDir, "sample_metadata.csv"));
            _qcData.Clear();
            LoadQcMatrix("raw|peptide", Path.Combine(outputDir, "peptides_rollup.parquet"), isLinear: false);
            LoadQcMatrix("corrected|peptide", Path.Combine(outputDir, "corrected_peptides.parquet"), isLinear: true);
            LoadQcMatrix("raw|protein", Path.Combine(outputDir, "proteins_raw.parquet"), isLinear: false);
            LoadQcMatrix("corrected|protein", Path.Combine(outputDir, "corrected_proteins.parquet"), isLinear: true);
            LoadReplicatesReport(Path.Combine(outputDir, "skyline-reports", "Metadata.csv"));
        }
        catch (Exception ex)
        {
            Log("(QC data load skipped: " + ex.Message + ")");
        }
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
        double[]? meanRt = null;
        if (table.HasColumn("mean_rt"))
        {
            var rt = table.GetDouble("mean_rt");
            meanRt = new double[n];
            for (var i = 0; i < n; i++)
                meanRt[i] = rt[i] ?? double.NaN;
        }
        _qcData[key] = (m, sampleCols, meanRt);
    }

    // Parse the exported Replicates report (dynamic annotation columns) into replicate -> column -> value.
    private void LoadReplicatesReport(string path)
    {
        _replicateAnn.Clear();
        _groupColumns = new List<string>();
        if (!File.Exists(path))
            return;
        var lines = File.ReadAllLines(path);
        if (lines.Length < 2)
            return;
        var header = SplitCsvLine(lines[0]);
        var repIdx = -1;
        foreach (var cand in new[] { "Replicate", "Replicate Name", "ReplicateName", "ReplicateLocator" })
        {
            repIdx = Array.FindIndex(header, h => h.Trim().Equals(cand, StringComparison.OrdinalIgnoreCase));
            if (repIdx >= 0)
                break;
        }
        if (repIdx < 0)
            return;

        var cols = new List<(string Name, int Idx)>();
        for (var i = 0; i < header.Length; i++)
            if (i != repIdx && !string.IsNullOrWhiteSpace(header[i]))
                cols.Add((header[i].Trim(), i));
        _groupColumns = cols.Select(c => c.Name).ToList();

        for (var r = 1; r < lines.Length; r++)
        {
            if (string.IsNullOrWhiteSpace(lines[r]))
                continue;
            var f = SplitCsvLine(lines[r]);
            if (f.Length <= repIdx)
                continue;
            var rep = f[repIdx].Trim();
            if (rep.Length == 0)
                continue;
            var map = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var (name, idx) in cols)
                map[name] = idx < f.Length ? f[idx].Trim() : "";
            _replicateAnn[rep] = map;
        }
    }

    // Sample IDs are "<replicate>__@__<batch>"; the Replicates report is keyed by replicate.
    private static string ReplicateOf(string sampleId)
    {
        const string sep = "__@__";
        var i = sampleId.IndexOf(sep, StringComparison.Ordinal);
        return i >= 0 ? sampleId[..i] : sampleId;
    }

    private string SampleAnnotation(string sampleId, string column)
    {
        if (_replicateAnn.TryGetValue(ReplicateOf(sampleId), out var m) && m.TryGetValue(column, out var v))
            return v;
        // Fallback for the synthetic Sample Type column when no Replicates report is available.
        if (column.Replace(" ", "").Equals("SampleType", StringComparison.OrdinalIgnoreCase))
            return _qcTypes.GetValueOrDefault(sampleId, "");
        return "";
    }

    private static string[] SplitCsvLine(string line)
    {
        var fields = new List<string>();
        var sb = new System.Text.StringBuilder();
        var inQuotes = false;
        for (var i = 0; i < line.Length; i++)
        {
            var c = line[i];
            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i++; }
                    else inQuotes = false;
                }
                else sb.Append(c);
            }
            else if (c == '"') inQuotes = true;
            else if (c == ',') { fields.Add(sb.ToString()); sb.Clear(); }
            else sb.Append(c);
        }
        fields.Add(sb.ToString());
        return fields.ToArray();
    }

    // Fill the Group-by column combo from the Replicates report (default Sample Type) and its values.
    private void PopulateGroupCombos()
    {
        _suppressQcRender = true;
        var columns = _groupColumns.Count > 0 ? _groupColumns : new List<string> { "Sample Type" };
        QcGroupByCombo.Items.Clear();
        foreach (var c in columns)
            QcGroupByCombo.Items.Add(c);
        var def = columns.FindIndex(c => c.Replace(" ", "").Equals("SampleType", StringComparison.OrdinalIgnoreCase));
        QcGroupByCombo.SelectedIndex = def >= 0 ? def : 0;
        PopulateValueCombo();
        _suppressQcRender = false;
    }

    private void PopulateValueCombo()
    {
        var column = ComboText(QcGroupByCombo, "");
        QcGroupCombo.Items.Clear();
        QcGroupCombo.Items.Add("All");
        if (!string.IsNullOrEmpty(column) && _qcData.Count > 0)
        {
            var samples = _qcData.Values.First().Samples;
            var values = new SortedSet<string>(StringComparer.Ordinal);
            foreach (var s in samples)
            {
                var v = SampleAnnotation(s, column);
                if (!string.IsNullOrEmpty(v))
                    values.Add(v);
            }
            foreach (var v in values)
                QcGroupCombo.Items.Add(v);
        }
        QcGroupCombo.SelectedIndex = 0; // All
    }

    private void OnGroupByChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (_suppressQcRender)
            return;
        _suppressQcRender = true;
        PopulateValueCombo();
        _suppressQcRender = false;
        RenderQc();
    }

    private void OnQcChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (!_suppressQcRender)
            RenderQc();
    }

    // PCA / CV / Intensity are live ScottPlot; the rest are the qc_report.html plots rendered as
    // static images by the same PlotRenderer.
    private static readonly HashSet<string> InteractivePlots = new() { "PCA", "CV distribution", "Intensity" };

    private void RenderQc()
    {
        if (QcViewCombo is null || _qcData.Count == 0)
            return;
        var view = ComboText(QcViewCombo, "Corrected").ToLowerInvariant();
        var level = ComboText(QcLevelCombo, "Peptide").ToLowerInvariant();
        var kind = ComboText(QcPlotCombo, "PCA");
        if (InteractivePlots.Contains(kind))
            RenderInteractiveQc(kind, view, level);
        else
            RenderStaticQc(kind, view, level);
    }

    private void RenderInteractiveQc(string kind, string view, string level)
    {
        QcImage.Visibility = Visibility.Collapsed;
        QcPlot.Visibility = Visibility.Visible;
        var plt = QcPlot.Plot;
        plt.Clear();
        if (!_qcData.TryGetValue($"{view}|{level}", out var d) || d.Samples.Count < 2)
        {
            plt.Title($"No {view} {level} data yet - run PRISM first.");
            QcPlot.Refresh();
            return;
        }
        var types = d.Samples.Select(s => _qcTypes.GetValueOrDefault(s, "unknown")).ToList();

        // Optional group filter driven by a Replicates-report column (e.g. "Sample Type" = "Standard").
        var column = ComboText(QcGroupByCombo, "Sample Type");
        var value = ComboText(QcGroupCombo, "All");
        var matrix = d.FeaturesBySamples;
        var groupLabel = "all";
        if (!string.Equals(value, "All", StringComparison.OrdinalIgnoreCase))
        {
            var cols = Enumerable.Range(0, d.Samples.Count)
                .Where(i => string.Equals(SampleAnnotation(d.Samples[i], column), value, StringComparison.OrdinalIgnoreCase))
                .ToList();
            if (cols.Count < 2)
            {
                plt.Title($"Fewer than 2 samples with {column} = {value}.");
                QcPlot.Refresh();
                return;
            }
            matrix = SelectColumns(d.FeaturesBySamples, cols);
            types = cols.Select(i => types[i]).ToList();
            groupLabel = value;
        }

        try
        {
            switch (kind)
            {
                case "CV distribution": DrawCv(plt, matrix, types, level, view, groupLabel); break;
                case "Intensity": DrawIntensity(plt, matrix, types, level, view, groupLabel); break;
                default: DrawPca(plt, matrix, types, level, view, groupLabel); break;
            }
        }
        catch (Exception ex)
        {
            plt.Title("render failed: " + ex.Message);
        }
        // The Plot is reused across view/level/type switches, so fit the axes to the current data.
        plt.Axes.AutoScale();
        QcPlot.Refresh();
    }

    private static double[,] SelectColumns(double[,] m, IReadOnlyList<int> cols)
    {
        var nF = m.GetLength(0);
        var sub = new double[nF, cols.Count];
        for (var j = 0; j < cols.Count; j++)
            for (var f = 0; f < nF; f++)
                sub[f, j] = m[f, cols[j]];
        return sub;
    }

    // Control-correlation heatmap + RT plots: the same plots qc_report.html generates, rendered as a
    // static image (they don't need zoom/pan). Reuses PlotRenderer so they match the report exactly.
    // Maps an interactive selection to the PNG stem the QC report writes under qc_plots/.
    private byte[]? LoadReportPng(string kind, string view, string level)
    {
        if (string.IsNullOrEmpty(_qcOutputDir))
            return null;
        var ba = view == "raw" ? "before" : "after";
        string? stem = kind switch
        {
            "Control correlation" => $"{level}_control_corr_{ba}",
            "RT-lowess" => $"{level}_rt_lowess_{ba}",
            "RT-bin boxplot" => $"{level}_rt_bin_box_{ba}",
            // RT-binned CV is a single before/after plot; the report writes _qc or _ref depending on controls.
            "RT-binned CV" => File.Exists(Path.Combine(_qcOutputDir, "qc_plots", $"{level}_rt_bin_cv_qc.png"))
                ? $"{level}_rt_bin_cv_qc"
                : $"{level}_rt_bin_cv_ref",
            _ => null,
        };
        if (stem is null)
            return null;
        var path = Path.Combine(_qcOutputDir, "qc_plots", stem + ".png");
        return File.Exists(path) ? File.ReadAllBytes(path) : null;
    }

    private void RenderStaticQc(string kind, string view, string level)
    {
        // Prefer the PNG the QC report already rendered (instant) instead of recomputing (RT-lowess in
        // particular re-fits a curve per sample). Fall back to on-the-fly rendering only if it's missing.
        var reportPng = LoadReportPng(kind, view, level);
        if (reportPng is not null)
        {
            ShowStaticImage(reportPng);
            return;
        }
        try
        {
            if (!_qcData.TryGetValue($"{view}|{level}", out var d))
            {
                ShowStaticMessage($"No {view} {level} data yet - run PRISM first.");
                return;
            }
            var types = d.Samples.Select(s => _qcTypes.GetValueOrDefault(s, "unknown")).ToList();
            var refIdx = IndicesOf(types, "reference");
            var qcIdx = IndicesOf(types, "qc");
            var controlIdx = refIdx.Concat(qcIdx).Distinct().OrderBy(x => x).ToList();
            var cap = view == "raw" ? "Raw" : "Corrected";

            byte[]? png = null;
            switch (kind)
            {
                case "Control correlation":
                    if (controlIdx.Count < 2) { ShowStaticMessage("Control correlation needs >= 2 reference/QC samples."); return; }
                    png = PlotRenderer.CorrelationHeatmap(d.FeaturesBySamples, controlIdx, $"Control-sample correlation ({cap})");
                    break;
                case "RT-lowess":
                    if (d.MeanRt is null) { ShowStaticMessage("RT plots are peptide-level only."); return; }
                    png = PlotRenderer.RtLowessCurves(d.FeaturesBySamples, d.MeanRt, types, $"RT-lowess of log2 abundance ({cap})");
                    break;
                case "RT-bin boxplot":
                    if (d.MeanRt is null) { ShowStaticMessage("RT plots are peptide-level only."); return; }
                    png = PlotRenderer.RtBinBoxplot(d.FeaturesBySamples, d.MeanRt, $"Abundance by RT bin ({cap})", "#1f77b4");
                    break;
                case "RT-binned CV":
                    // A before/after comparison, so it always uses both raw and corrected (ignores View).
                    if (!_qcData.TryGetValue($"raw|{level}", out var rawD) ||
                        !_qcData.TryGetValue($"corrected|{level}", out var corrD) || rawD.MeanRt is null)
                    {
                        ShowStaticMessage("RT-binned CV is peptide-level and needs both raw and corrected data.");
                        return;
                    }
                    var idx = qcIdx.Count >= 2 ? qcIdx : refIdx;
                    if (idx.Count < 2) { ShowStaticMessage("RT-binned CV needs >= 2 reference or QC samples."); return; }
                    var label = qcIdx.Count >= 2 ? "QC" : "Reference";
                    var color = qcIdx.Count >= 2 ? "#ff7f0e" : "#d62728";
                    png = PlotRenderer.RtBinCv(rawD.FeaturesBySamples, corrD.FeaturesBySamples, rawD.MeanRt, idx,
                        $"RT-binned CV ({label}, before vs after)", color);
                    break;
            }
            if (png is not null)
                ShowStaticImage(png);
            else
                ShowStaticMessage("(no plot)");
        }
        catch (Exception ex)
        {
            ShowStaticMessage("render failed: " + ex.Message);
        }
    }

    private void ShowStaticImage(byte[] png)
    {
        var bmp = new System.Windows.Media.Imaging.BitmapImage();
        using (var ms = new MemoryStream(png))
        {
            bmp.BeginInit();
            bmp.CacheOption = System.Windows.Media.Imaging.BitmapCacheOption.OnLoad;
            bmp.StreamSource = ms;
            bmp.EndInit();
        }
        bmp.Freeze();
        QcImage.Source = bmp;
        QcPlot.Visibility = Visibility.Collapsed;
        QcImage.Visibility = Visibility.Visible;
    }

    // The image control can't show text, so reuse the ScottPlot control to display a message.
    private void ShowStaticMessage(string msg)
    {
        QcImage.Visibility = Visibility.Collapsed;
        QcPlot.Visibility = Visibility.Visible;
        var plt = QcPlot.Plot;
        plt.Clear();
        plt.Title(msg);
        QcPlot.Refresh();
    }

    private static void DrawPca(Plot plt, double[,] featuresBySamples, List<string> types, string level, string view, string group)
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
        plt.Title($"{Cap(level)} PCA ({view}, {group})");
        plt.XLabel("PC1");
        plt.YLabel("PC2");
    }

    private static void DrawCv(Plot plt, double[,] featuresBySamples, List<string> types, string level, string view, string group)
    {
        if (group == "all")
        {
            // Overlay one histogram per sample type present (>= 2 samples of that type).
            foreach (var t in types.Distinct().OrderBy(x => x, StringComparer.Ordinal))
            {
                var idx = IndicesOf(types, t);
                if (idx.Count >= 2)
                    AddCvHist(plt, CvMetrics.PerFeatureCvs(featuresBySamples, idx),
                        PlotRenderer.TypeColors.GetValueOrDefault(t, "#1f77b4"), Cap(t));
            }
        }
        else
        {
            AddCvHist(plt, CvMetrics.PerFeatureCvs(featuresBySamples, Enumerable.Range(0, types.Count).ToList()),
                PlotRenderer.TypeColors.GetValueOrDefault(types.Count > 0 ? types[0] : "", "#1f77b4"), Cap(group));
        }
        plt.ShowLegend(Alignment.UpperRight); // upper-right so it doesn't overlap the histogram bars
        plt.Title($"{Cap(level)} CV distribution ({view}, {group})");
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

    private static void DrawIntensity(Plot plt, double[,] featuresBySamples, List<string> types, string level, string view, string group)
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
        plt.Title($"{Cap(level)} median log2 intensity ({view}, {group})");
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
