using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
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

    /// <summary>
    /// The inputs to combine in one run - one per batch. Populated with the launching Skyline document
    /// when started from the Tools menu, and freely extended with other open documents, closed .sky files,
    /// or already-exported reports. A non-empty list is all Run needs, so the window is equally usable as a
    /// standalone PRISM GUI with no Skyline running.
    /// </summary>
    private readonly System.Collections.ObjectModel.ObservableCollection<PrismInput> _inputs = new();

    public MainWindow()
    {
        InitializeComponent();
        QcPlot.MouseMove += QcPlot_MouseMove; // show the replicate name when hovering a PCA point
        InputsGrid.ItemsSource = _inputs;
        _inputs.CollectionChanged += (_, _) => UpdateRunEnabled();

        // Run stays disabled until there is an output directory AND at least one input. When connected to a
        // saved document, SetDefaultOutputDirAsync pre-fills "<document folder>/PRISM-Output"; otherwise the
        // box stays empty so the user must choose (no defaulting into OneDrive-synced Documents).
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
                "Standalone mode - not attached to a running Skyline. Add exported reports or .sky files "
                + "on the Inputs tab, or launch this tool from Skyline's Tools menu to use the open document.";
            App.WriteLog("No Skyline connection at startup: " + ex.Message);
        }

        Log("Skyline-PRISM tool ready.");
        Log("Diagnostic log: " + App.LogFilePath);
        Log(_session is not null
            ? "Connected. Set an output directory and click Run PRISM."
            : "Standalone mode. Add inputs on the Inputs tab (exported reports, or .sky documents exported "
              + "via SkylineCmd), set an output directory, then click Run PRISM.");

        // Skyline reinstalls its tools into a NEW versioned folder on every update, so an existing
        // shortcut is re-pointed at this build whenever PRISM starts.
        StandaloneShortcut.Refresh(Log);
        UpdateShortcutButton();

        MetadataReportCombo.Items.Add(DefaultMetadataItem);
        MetadataReportCombo.SelectedIndex = 0;

        TransitionRollupCombo.SelectedIndex = 0;
        PeptideNormCombo.SelectedIndex = 0;
        SharedPeptideCombo.SelectedIndex = 0; // all_groups, matching PrismConfig's default
        ProteinRollupCombo.SelectedIndex = 0;
        ProteinNormCombo.SelectedIndex = 0;
        QcViewCombo.SelectedIndex = 0;
        QcLevelCombo.SelectedIndex = 0;
        QcPlotCombo.SelectedIndex = 0;
        // QcGroupByCombo / QcGroupCombo are populated from the Replicates report after a run.

        if (_session is not null)
        {
            _ = AddLaunchingDocumentAsync();
            _ = SetDefaultOutputDirAsync();
            _ = LoadReportsAsync();
            _ = LoadLibrariesAsync();
        }
        else
        {
            AddOpenDocButton.IsEnabled = false; // re-enabled if a running instance turns up when clicked
            MainTabs.SelectedItem = InputsTab;  // standalone: the first thing to do is add an input
        }
    }

    /// <summary>Seed the Inputs list with the document of the Skyline that launched the tool.</summary>
    private async Task AddLaunchingDocumentAsync()
    {
        if (_session is null)
            return;
        var session = _session;
        var docPath = await Task.Run(() =>
        {
            try { return session.Execute(c => c.GetDocumentPath()); }
            catch { return null; }
        });
        AddInput(PrismInput.FromRunningSkyline(session, docPath));
        Log($"Input added: {(string.IsNullOrWhiteSpace(docPath) ? "the open (unsaved) document" : docPath)}");
    }

    /// <summary>Add an input, keeping batch labels unique, and refresh the grid + Run state.</summary>
    private void AddInput(PrismInput input)
    {
        _inputs.Add(input);
        PrismInput.EnsureUniqueLabels(_inputs);
        InputsGrid.Items.Refresh();
        UpdateRunEnabled();
    }

    private void UpdateRunEnabled()
    {
        if (RunButton is not null)
            RunButton.IsEnabled = !_isRunning && _inputs.Count > 0
                && !string.IsNullOrWhiteSpace(OutputDirBox?.Text);
    }

    // Offer the documents of every running Skyline instance we can reach. Discovery relies on Skyline having
    // registered itself in ~/.skyline-mcp/, so an instance may not be listed; a closed-document or
    // exported-report input always works as a fallback.
    private async void OnAddOpenDocument(object sender, RoutedEventArgs e)
    {
        AddOpenDocButton.IsEnabled = false;
        try
        {
            var launching = _session;
            var instances = await Task.Run(
                () => SkylineSession.DiscoverRunning(launching, log: Log));
            var already = _inputs
                .Where(i => i.Kind == PrismInputKind.RunningSkyline && !string.IsNullOrWhiteSpace(i.Path))
                .Select(i => i.Path)
                .ToHashSet(StringComparer.OrdinalIgnoreCase);
            var candidates = instances
                .Where(i => string.IsNullOrWhiteSpace(i.DocumentPath) || !already.Contains(i.DocumentPath!))
                .ToList();

            if (candidates.Count == 0)
            {
                MessageBox.Show(
                    instances.Count == 0
                        ? "No running Skyline instance answered.\n\n"
                          + "Skyline is only discoverable while its JSON-RPC server is running. Use "
                          + "\"Add Skyline document (.sky)...\" to export a document with SkylineCmd instead."
                        : "Every running document is already in the list.",
                    "Add open document", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var picked = PickFromList(
                "Add open document",
                "Documents open in running Skyline instances:",
                candidates.Select(c => c.DisplayName).ToList());
            if (picked < 0)
                return;
            var chosen = candidates[picked];
            AddInput(PrismInput.FromRunningSkyline(chosen.Session, chosen.DocumentPath));
            Log($"Input added (open in Skyline): {chosen.DocumentPath ?? chosen.DisplayName}");
        }
        catch (Exception ex)
        {
            Log("Could not list running Skyline instances: " + ex.Message);
            MessageBox.Show("Could not list running Skyline instances: " + ex.Message,
                "Add open document", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
        finally
        {
            AddOpenDocButton.IsEnabled = true;
        }
    }

    private void OnAddClosedDocument(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "Select Skyline document(s)",
            Filter = "Skyline documents (*.sky)|*.sky|All files (*.*)|*.*",
            Multiselect = true,
        };
        var start = DialogStartDir();
        if (start is not null)
            dlg.InitialDirectory = start;
        if (dlg.ShowDialog() != true)
            return;

        foreach (var path in dlg.FileNames)
        {
            var input = PrismInput.FromClosedDocument(path);
            // Read the header now so the user sees replicate counts (and any problem) before running.
            var info = SkyDocumentInfo.TryRead(path, Log);
            input.Status = info is null
                ? "could not read the document header"
                : $"{info.Replicates.Count} replicate(s)"
                  + (info.ReplicateAnnotationNames.Count > 0
                      ? "; annotations: " + string.Join(", ", info.ReplicateAnnotationNames)
                      : "");
            AddInput(input);
            Log($"Input added (closed document): {path} - {input.Status}");
        }
    }

    private void OnAddReportFile(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "Select exported PRISM report(s)",
            Filter = "PRISM reports (*.parquet;*.csv;*.tsv)|*.parquet;*.csv;*.tsv"
                     + "|Parquet (*.parquet)|*.parquet|CSV (*.csv)|*.csv|All files (*.*)|*.*",
            Multiselect = true,
        };
        var start = DialogStartDir();
        if (start is not null)
            dlg.InitialDirectory = start;
        if (dlg.ShowDialog() != true)
            return;

        foreach (var path in dlg.FileNames)
        {
            // Pick up a metadata CSV sitting next to the report under either naming convention.
            var dir = Path.GetDirectoryName(path);
            var stem = Path.GetFileNameWithoutExtension(path);
            string? metadata = null;
            if (dir is not null)
            {
                metadata = new[] { stem + ".metadata.csv", "Metadata.csv" }
                    .Select(n => Path.Combine(dir, n))
                    .FirstOrDefault(File.Exists);
            }
            var input = PrismInput.FromReportFile(path, metadata);
            input.Status = metadata is null ? "no metadata file found" : "metadata: " + Path.GetFileName(metadata);
            AddInput(input);
            Log($"Input added (report file): {path}"
                + (metadata is null ? " (no metadata CSV alongside)" : $" + {metadata}"));
        }
    }

    private void OnRemoveInput(object sender, RoutedEventArgs e)
    {
        foreach (var item in InputsGrid.SelectedItems.Cast<object>().OfType<PrismInput>().ToList())
            _inputs.Remove(item);
        InputsGrid.Items.Refresh();
        UpdateRunEnabled();
    }

    /// <summary>Minimal single-choice picker (WPF has no built-in one), returning the index or -1.</summary>
    private int PickFromList(string title, string prompt, IReadOnlyList<string> items)
    {
        var list = new System.Windows.Controls.ListBox { Margin = new Thickness(0, 6, 0, 8) };
        foreach (var i in items)
            list.Items.Add(i);
        list.SelectedIndex = 0;

        var ok = new System.Windows.Controls.Button
        { Content = "Add", Padding = new Thickness(16, 4, 16, 4), Margin = new Thickness(0, 0, 8, 0), IsDefault = true };
        var cancel = new System.Windows.Controls.Button
        { Content = "Cancel", Padding = new Thickness(16, 4, 16, 4), IsCancel = true };

        var buttons = new System.Windows.Controls.StackPanel
        {
            Orientation = System.Windows.Controls.Orientation.Horizontal,
            HorizontalAlignment = System.Windows.HorizontalAlignment.Right,
        };
        buttons.Children.Add(ok);
        buttons.Children.Add(cancel);

        var panel = new System.Windows.Controls.DockPanel { Margin = new Thickness(12) };
        var label = new System.Windows.Controls.TextBlock { Text = prompt, TextWrapping = TextWrapping.Wrap };
        System.Windows.Controls.DockPanel.SetDock(label, System.Windows.Controls.Dock.Top);
        System.Windows.Controls.DockPanel.SetDock(buttons, System.Windows.Controls.Dock.Bottom);
        panel.Children.Add(label);
        panel.Children.Add(buttons);
        panel.Children.Add(list);

        var win = new Window
        {
            Title = title,
            Width = 560,
            Height = 320,
            Content = panel,
            Owner = this,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
        };
        var result = -1;
        ok.Click += (_, _) => { result = list.SelectedIndex; win.Close(); };
        win.ShowDialog();
        return result;
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

    private void OnParsimonyChanged(object sender, RoutedEventArgs e) => UpdateSharedPeptideRow();

    /// <summary>
    /// Shared-peptide handling only means something once proteins are grouped: with parsimony off every
    /// accession is its own group, so there is nothing for a peptide to be shared between.
    /// </summary>
    private void UpdateSharedPeptideRow()
    {
        if (SharedPeptideRow is not null)
            SharedPeptideRow.IsEnabled = ParsimonyCheck?.IsChecked == true;
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
            Filter = "Spectral library (*.blib;*.tsv)|*.blib;*.tsv|BiblioSpec (*.blib)|*.blib"
                + "|Carafe/DIA-NN (*.tsv)|*.tsv|All files (*.*)|*.*",
        };
        var start = DialogStartDir();
        if (start is not null)
            dlg.InitialDirectory = start;
        if (dlg.ShowDialog() == true)
        {
            if (!LibraryCombo.Items.Contains(dlg.FileName))
                LibraryCombo.Items.Add(dlg.FileName);
            LibraryCombo.Text = dlg.FileName;
        }
    }

    /// <summary>
    /// Create the Start Menu shortcut that opens PRISM standalone. A Skyline tool zip has no install
    /// script, so this is the only point at which a shortcut can be made.
    /// </summary>
    private void OnCreateShortcut(object sender, RoutedEventArgs e)
    {
        try
        {
            var path = StandaloneShortcut.Create();
            Log("Start Menu shortcut created: " + path);
            MessageBox.Show(
                "Created a Start Menu shortcut:" + Environment.NewLine + path + Environment.NewLine
                + Environment.NewLine
                + "It opens Skyline-PRISM on its own, without Skyline running." + Environment.NewLine
                + Environment.NewLine
                + "Note: Skyline installs its tools inside its own versioned program folder, so a Skyline "
                + "update moves this program. PRISM re-points the shortcut automatically the next time you "
                + "start it from Skyline's Tools menu.",
                "Skyline-PRISM shortcut", MessageBoxButton.OK, MessageBoxImage.Information);
            UpdateShortcutButton();
        }
        catch (Exception ex)
        {
            Log("Could not create the shortcut: " + ex.Message);
            MessageBox.Show("Could not create the shortcut: " + ex.Message,
                "Skyline-PRISM shortcut", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void UpdateShortcutButton()
    {
        if (ShortcutButton is not null)
            ShortcutButton.Content = StandaloneShortcut.Exists
                ? "Start Menu shortcut installed"
                : "Add Start Menu shortcut";
    }

    private void OnOpenProvenance(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title = "Open provenance (parameters.json)",
            Filter = "PRISM provenance|parameters.json;metadata.json|JSON (*.json)|*.json|All files (*.*)|*.*",
        };
        var start = DialogStartDir();
        if (start is not null)
            dlg.InitialDirectory = start;
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
        ReferenceAnchoredCheck.IsChecked = c.BatchCorrection.ReferenceAnchored;
        ParsimonyCheck.IsChecked = c.Parsimony.Enabled;
        SelectCombo(SharedPeptideCombo, c.Parsimony.SharedPeptideHandling);
        UpdateSharedPeptideRow();
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
        // reference_type stays at its default "reference", which is what Skyline's "Standard" sample
        // type maps to (ReplicateMetadata.MapSampleType) - so the checkbox's wording is literal.
        c.BatchCorrection.ReferenceAnchored = ReferenceAnchoredCheck.IsChecked == true;

        c.Parsimony.Enabled = ParsimonyCheck.IsChecked == true;
        c.Parsimony.SharedPeptideHandling = ComboText(SharedPeptideCombo, "all_groups");

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

    // Default: read Skyline's built-in "Replicates" document grid directly, dynamically capturing all of
    // its column headings (curated built-ins + every user annotation) into a custom metadata report.
    private const string DefaultMetadataItem = "(default) Replicates";

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

    // Nearest existing directory at or above the "Output directory" box, so Browse dialogs open there.
    // Walks up parents so a not-yet-created output dir still opens in its closest existing ancestor.
    private string? DialogStartDir()
    {
        var dir = OutputDirBox?.Text?.Trim();
        while (!string.IsNullOrEmpty(dir))
        {
            if (Directory.Exists(dir))
                return dir;
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }

    private void OnBrowse(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog { Title = "Select output directory" };
        var start = DialogStartDir();
        if (start is not null)
            dlg.InitialDirectory = start;
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
        var reportsDir = Path.Combine(outputDir, "skyline-reports");
        var configPath = Path.Combine(outputDir, "prism-config.yaml");
        var yaml = SerializeConfig(config);

        // One -i / -m pair per input, in the same order, named after each input's batch label - which is how
        // the CLI attributes replicates to their source document (see ReplicateMetadata document scoping).
        var labels = _inputs.Count > 0
            ? _inputs.Select(i => PrismInput.SanitizeLabel(i.BatchLabel)).ToList()
            : new List<string> { "PRISM" };
        var reportArgs = string.Join(" ", labels.Select(l =>
        {
            // A pre-exported report is read in place; exported ones land in skyline-reports as parquet (live
            // Skyline) or CSV (headless SkylineCmd).
            var input = _inputs.FirstOrDefault(i => PrismInput.SanitizeLabel(i.BatchLabel) == l);
            if (input?.Kind == PrismInputKind.ReportFile)
                return $"\"{input.Path}\"";
            var ext = input?.Kind == PrismInputKind.ClosedDocument ? ".csv" : ".parquet";
            return $"\"{Path.Combine(reportsDir, l + ext)}\"";
        }));
        var metadataArgs = string.Join(" ", labels.Select(l =>
        {
            var input = _inputs.FirstOrDefault(i => PrismInput.SanitizeLabel(i.BatchLabel) == l);
            if (input?.Kind == PrismInputKind.ReportFile)
                return input.MetadataPath is null ? null : $"\"{input.MetadataPath}\"";
            return $"\"{Path.Combine(reportsDir, l + ".metadata.csv")}\"";
        }).Where(s => s is not null));

        var command = $"prism run -i {reportArgs} -o \"{outputDir}\" -c \"{configPath}\""
                      + (metadataArgs.Length > 0 ? $" -m {metadataArgs}" : "");
        var body =
            "Command-line equivalent of the current GUI settings.\r\n\r\n" +
            "The tool exports each input into <output>\\skyline-reports\\ (named after its batch label) and\r\n" +
            "runs the pipeline once over all of them. To reproduce it with the prism CLI: save the config\r\n" +
            "below as prism-config.yaml, then run:\r\n\r\n" +
            command + "\r\n\r\n" +
            "The -i and -m lists are positional: metadata file N describes input N. That pairing is what\r\n" +
            "keeps identically named replicates in different documents from overwriting each other.\r\n\r\n" +
            "# ---------- prism-config.yaml ----------\r\n" + yaml;

        ShowCopyableTextWindow("PRISM command line", body, command, yaml);
    }

    // ConfigWriter emits only the keys that apply to the selected methods (a plain object dump
    // spells out all ~60 properties, including the ones the chosen methods ignore). Both the text
    // shown above and the "Copy config (YAML)" button use this same string.
    private static string SerializeConfig(PrismConfig config) => ConfigWriter.ToYaml(config);

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
        if (_inputs.Count == 0)
        {
            Log("No inputs. Add a document or an exported report on the Inputs tab.");
            MainTabs.SelectedItem = InputsTab;
            return;
        }

        // Labels double as exported file stems and as batch labels, so they must be unique before the run.
        PrismInput.EnsureUniqueLabels(_inputs);
        InputsGrid.Items.Refresh();

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
        var inputs = _inputs.ToList(); // snapshot: the grid stays editable while the run proceeds

        try
        {
            _runCancellation = new CancellationTokenSource();
            StopButton.IsEnabled = true;
            var token = _runCancellation.Token;
            await Task.Run(() => RunPipeline(inputs, outputDir, metadataReport, config, token), token);
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
            InvalidateDensity();      // new merged_data.parquet: reload the Spectrum density tab when shown
            InvalidateDynamicRange(); // and new corrected matrices for the Dynamic Range tab
            RenderQc(); // draws on the UI thread (cheap; the ScottPlot control requires it)
            Log("Done.");
            MainTabs.SelectedItem = QcTab; // land on the plots when the run finishes
        }
        catch (Exception ex) when (IsCancellation(ex))
        {
            // Cancelling is a normal outcome, not a failure: no dialog, no stack trace.
            Log("STOPPED: the run was cancelled. Partial files in the output directory are "
                + "intermediates of an incomplete run - re-run to rebuild them.");
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
            _runCancellation?.Dispose();
            _runCancellation = null;
            StopButton.IsEnabled = false;
            UpdateRunEnabled();
        }
    }

    /// <summary>Cancellation arrives in several shapes depending on which stage was interrupted.</summary>
    private static bool IsCancellation(Exception ex) => ex switch
    {
        OperationCanceledException => true,
        AggregateException agg => agg.InnerExceptions.Count > 0
                                  && agg.InnerExceptions.All(IsCancellation),
        _ => false,
    };

    /// <summary>
    /// Cancels the running pipeline. What that can and cannot reach is worth being precise about:
    /// the PRISM pipeline and any headless Skyline we launched stop within seconds, but a report
    /// export already handed to a RUNNING Skyline is Skyline's work, in Skyline's process, and there
    /// is no RPC to recall it - it will finish writing its file. Saying so here beats leaving the
    /// user watching a file grow after pressing Stop.
    /// </summary>
    private void OnStop(object sender, RoutedEventArgs e)
    {
        if (_runCancellation is null || _runCancellation.IsCancellationRequested)
            return;

        StopButton.IsEnabled = false;
        Log("Stopping...");
        if (_inputs.Any(i => i.Kind == PrismInputKind.RunningSkyline))
            Log("  NOTE: a report export already running inside Skyline cannot be recalled - Skyline "
                + "will finish writing that file. PRISM will not use it.");
        _runCancellation.Cancel();
    }

    /// <summary>
    /// Closing the window must not leave the run going. It previously did: the pipeline ran on a
    /// background thread with nothing watching the window, so closing PRISM left the export running
    /// and the file still growing.
    /// </summary>
    protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
    {
        if (_isRunning)
        {
            var answer = MessageBox.Show(
                "A PRISM run is in progress. Close anyway and stop it?"
                + Environment.NewLine + Environment.NewLine
                + "A report export already running inside Skyline cannot be recalled and will finish "
                + "writing its file.",
                "PRISM is running", MessageBoxButton.YesNo, MessageBoxImage.Warning);
            if (answer != MessageBoxResult.Yes)
            {
                e.Cancel = true;
                return;
            }
            _runCancellation?.Cancel();
        }
        base.OnClosing(e);
    }

    /// <summary>Non-null only while a run is in progress; the Stop button's handle on it.</summary>
    private CancellationTokenSource? _runCancellation;

    // Enable Run only once an output directory is set and an input exists (and no run is in progress).
    private void OnOutputDirChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
    {
        UpdateRunEnabled();
        // Both plot tabs read their inputs from this directory.
        InvalidateDensity();
        InvalidateDynamicRange();
    }

    /// <summary>
    /// Export every input, then run the pipeline once over all of them. Each input is exported under its own
    /// batch label, and the labels become the merge's Source Document labels - so several Skyline documents
    /// (one per batch/plate) are processed as a single cohort with ComBat correcting between them.
    /// Runs on a background thread; <see cref="Log"/> marshals to the UI.
    /// </summary>
    private void RunPipeline(
        IReadOnlyList<PrismInput> inputs, string outputDir, string? metadataReport, PrismConfig config,
        CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(outputDir);
        var reportsDir = Path.Combine(outputDir, "skyline-reports");

        // Derive the digestion enzyme from the source document so FASTA-based parsimony matches the search
        // that produced the data instead of the config default. Only relevant when a FASTA map is used;
        // harmless otherwise. The first input that can answer wins.
        foreach (var input in inputs)
        {
            var enzyme = input.TryGetDigestionEnzyme(Log);
            if (enzyme is null)
                continue;
            config.Parsimony.Enzyme = enzyme;
            break;
        }

        // batchAnnotation ensures the user's batch column reaches the generated Replicates report. For a
        // live document the dynamic grid read already picks up every annotation; for a closed document it is
        // how the column gets into the .skyr at all.
        var batchAnnotation = metadataReport is null ? config.Metadata.BatchColumn : null;

        // Results are indexed BY INPUT POSITION, never appended: the pipeline pairs metadata to inputs
        // positionally, so completion order must not leak into the lists (see the assembly below).
        var exports = new ExportedReports?[inputs.Count];

        var degree = ExportParallelism(inputs);
        Log($"Preparing {inputs.Count} input(s) into {reportsDir}"
            + (degree > 1 ? $" ({degree} at a time)..." : "..."));

        var done = 0;
        try
        {
            Parallel.For(0, inputs.Count, new ParallelOptions { MaxDegreeOfParallelism = degree }, i =>
            {
                var input = inputs[i];
                Log($"[{Interlocked.Increment(ref done)}/{inputs.Count}] {input.KindLabel}: "
                    + $"{input.DisplayName} (batch '{input.BatchLabel}')");
                SetInputStatus(input, "exporting...");
                try
                {
                    var exported = input.Prepare(
                        reportsDir, metadataReport, batchAnnotation, SkylineCmdPathOverride, Log,
                        cancellationToken);
                    exports[i] = exported;
                    var size = File.Exists(exported.InputPath) ? new FileInfo(exported.InputPath).Length : 0;
                    SetInputStatus(input,
                        $"{(exported.InputIsParquet ? "parquet" : "CSV")}, {size / (1024.0 * 1024.0):N1} MB"
                        + (exported.ReplicatesCsv is null ? " (no metadata)" : ""));
                }
                catch (Exception ex)
                {
                    SetInputStatus(input, "FAILED: " + ex.Message);
                    throw new InvalidOperationException(
                        $"Input '{input.DisplayName}' (batch '{input.BatchLabel}') failed: {ex.Message}", ex);
                }
            });
        }
        catch (AggregateException ex)
        {
            // Parallel.For wraps worker failures; surface the first one so the user sees which INPUT
            // failed and why, not "One or more errors occurred."
            throw ex.Flatten().InnerExceptions[0];
        }

        var reportPaths = new List<string>(inputs.Count);
        var metadataPaths = new List<string>(inputs.Count);
        var missingMetadata = 0;
        foreach (var exported in exports)
        {
            if (exported is null)
                continue; // unreachable: Parallel.For propagates any failure as an AggregateException
            reportPaths.Add(exported.InputPath);
            if (exported.ReplicatesCsv is not null)
                metadataPaths.Add(exported.ReplicatesCsv);
            else
                missingMetadata++;
        }

        // Metadata is matched to inputs positionally, so a partial set would mis-assign sample types and
        // batches. Use it only when every input produced one; otherwise fall back to name-based typing.
        List<string>? metadata = metadataPaths;
        if (missingMetadata > 0)
        {
            Log($"WARNING: {missingMetadata} of {inputs.Count} input(s) produced no replicate metadata. "
                + "Sample types will be inferred from replicate names for ALL inputs, because per-document "
                + "metadata must line up 1:1 with the inputs to be attributed correctly.");
            metadata = null;
        }

        // Capture what each input knows about DIA isolation windows and save it beside the outputs, so the
        // Spectrum density tab can bin on the real windows later - including when this output directory is
        // reopened with no Skyline running.
        var isolationCatalog = new IsolationSchemeCatalog();
        foreach (var input in inputs)
            input.CollectIsolationSchemes(isolationCatalog, Log, SkylineCmdPathOverride, cancellationToken);
        if (!isolationCatalog.IsEmpty)
            isolationCatalog.Save(Path.Combine(outputDir, IsolationSchemeCatalog.FileName));

        Log($"Running the PRISM pipeline on {reportPaths.Count} report(s): "
            + string.Join(", ", reportPaths.Select(Path.GetFileName)));
        var result = PrismPipeline.Run(
            reportPaths, outputDir, config, metadata, Log, cancellationToken: cancellationToken);
        Log($"Pipeline complete: {result.NPeptides} peptides, {result.NProteins} proteins, "
            + $"{result.NSamples} samples, {result.Batches.Count} batch(es).");
    }

    /// <summary>Optional explicit SkylineCmd.exe path; null means auto-discover (see SkylineCmdLocator).</summary>
    private string? SkylineCmdPathOverride { get; set; }

    /// <summary>
    /// Peak RAM to assume per concurrent headless export. Measured: a 5 MB .sky with a 116 MB .skyd and 87
    /// replicates peaked at ~1.26 GB resident, and memory scales linearly with concurrency (3 at once
    /// peaked at 3.7 GB). 2.5 GB leaves headroom for documents larger than that.
    /// </summary>
    private const double GbPerConcurrentExport = 2.5;

    /// <summary>Hard cap on concurrent exports: past this the work is disk-bound, not CPU-bound.</summary>
    private const int MaxConcurrentExports = 4;

    /// <summary>
    /// How many inputs to export at once. Exporting several documents in parallel is a large win -
    /// measured 5.2 s versus ~23 s sequential for three documents - because each export is dominated by
    /// Skyline's startup and its single-threaded document load, so the machine is mostly idle otherwise.
    ///
    /// <para>The binding constraint is MEMORY, not CPU: every headless export is a whole Skyline process
    /// with the document loaded, so RAM scales linearly with concurrency. The degree is therefore budgeted
    /// against installed RAM rather than core count, and capped.</para>
    ///
    /// <para>Inputs that need no export (an already-exported report file) are excluded from the budget -
    /// they cost nothing and would otherwise inflate the estimate.</para>
    /// </summary>
    private int ExportParallelism(IReadOnlyList<PrismInput> inputs)
    {
        var exporting = inputs.Count(i => i.Kind != PrismInputKind.ReportFile);
        if (exporting <= 1)
            return 1; // nothing to overlap

        double totalGb;
        try
        {
            totalGb = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (1024.0 * 1024 * 1024);
        }
        catch (Exception)
        {
            return 1; // can't size the budget - stay sequential
        }

        // Spend at most 60% of RAM on exports; the merge that follows sizes itself against RAM too, but it
        // runs after every export has exited, so the two never overlap.
        var byMemory = (int)Math.Floor(totalGb * 0.6 / GbPerConcurrentExport);
        var degree = Math.Max(1, Math.Min(Math.Min(byMemory, MaxConcurrentExports), exporting));
        if (degree > 1)
            Log($"Exporting up to {degree} documents at a time "
                + $"({totalGb:N0} GB RAM, budgeting {GbPerConcurrentExport:N1} GB per export).");
        else
            Log($"Exporting one document at a time ({totalGb:N0} GB RAM is not enough to overlap safely).");
        return degree;
    }

    private void SetInputStatus(PrismInput input, string status)
    {
        if (Dispatcher.CheckAccess())
            input.Status = status;
        else
            Dispatcher.BeginInvoke(new Action(() => input.Status = status));
    }

    // Cached QC matrices ([features, samples], LOG2) keyed by "view|level" (view in raw/corrected).
    private readonly Dictionary<string, (double[,] FeaturesBySamples, List<string> Samples, double[]? MeanRt)> _qcData = new();
    private Dictionary<string, string> _qcTypes = new();

    // PCA hover: the current PCA points (data coordinate + replicate name) and the highlight overlay
    // (a ring marker + a text label). Recreated on every render because Plot.Clear() drops all plottables;
    // null when the active QC plot is not PCA, which makes the hover handler a no-op.
    private List<(ScottPlot.Coordinates Loc, string Name)>? _pcaHoverPoints;
    private ScottPlot.Plottables.Marker? _pcaHoverMarker;
    private ScottPlot.Plottables.Text? _pcaHoverText;
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
            LoadReplicatesReports(Path.Combine(outputDir, "skyline-reports"));
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

    /// <summary>
    /// Load every per-document Replicates report in the export directory ("&lt;label&gt;.metadata.csv", plus
    /// the legacy single-document "Metadata.csv"). Each file's rows are stored under BOTH the
    /// document-qualified sample ID ("&lt;replicate&gt;__@__&lt;label&gt;") and the bare replicate name, so a QC
    /// injection named the same in several documents keeps its own document's annotations in the plots.
    /// </summary>
    private void LoadReplicatesReports(string reportsDir)
    {
        _replicateAnn.Clear();
        _groupColumns = new List<string>();
        if (!Directory.Exists(reportsDir))
            return;

        var files = Directory.EnumerateFiles(reportsDir, "*.metadata.csv")
            .Concat(Directory.EnumerateFiles(reportsDir, "Metadata.csv"))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();

        foreach (var file in files)
        {
            var name = Path.GetFileName(file);
            // "<label>.metadata.csv" -> "<label>"; the legacy "Metadata.csv" has no document label.
            var label = name.EndsWith(".metadata.csv", StringComparison.OrdinalIgnoreCase)
                ? name[..^".metadata.csv".Length]
                : null;
            LoadReplicatesReport(file, label);
        }
    }

    // Parse one exported Replicates report (dynamic annotation columns) into replicate -> column -> value.
    private void LoadReplicatesReport(string path, string? documentLabel)
    {
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
        // Union across documents: a column present in any Replicates report can be grouped by.
        foreach (var name in cols.Select(c => c.Name))
            if (!_groupColumns.Contains(name, StringComparer.Ordinal))
                _groupColumns.Add(name);

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
            // Document-qualified key first (this is the merged Sample ID), then the bare replicate name as
            // a fallback for single-document runs and legacy Metadata.csv exports.
            if (!string.IsNullOrEmpty(documentLabel))
                _replicateAnn[rep + "__@__" + documentLabel] = map;
            if (!_replicateAnn.ContainsKey(rep) || string.IsNullOrEmpty(documentLabel))
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

    // Sample IDs carry a "<replicate>__@__<batch>" suffix, added during merge so identical replicate names
    // from different batches / source documents stay distinct. When every sample in the dataset shares the
    // SAME suffix (a single batch/source), it is redundant noise, so strip it for display. When suffixes
    // differ (a real multi-batch merge), keep them so the replicates remain distinguishable.
    private static List<string> StripSharedBatchSuffix(IReadOnlyList<string> display, IReadOnlyList<string> allSamples)
    {
        const string sep = "__@__";
        string? suffix = null;
        foreach (var n in allSamples)
        {
            var i = n.IndexOf(sep, StringComparison.Ordinal);
            if (i < 0)
                return display.ToList(); // a sample has no suffix -> leave everything as-is
            var s = n[i..];
            if (suffix is null)
                suffix = s;
            else if (!string.Equals(suffix, s, StringComparison.Ordinal))
                return display.ToList(); // suffixes differ -> keep them
        }
        return display.Select(n =>
        {
            var i = n.IndexOf(sep, StringComparison.Ordinal);
            return i >= 0 ? n[..i] : n;
        }).ToList();
    }

    private string SampleAnnotation(string sampleId, string column)
    {
        // The full sample ID is the document-qualified key; fall back to the bare replicate name.
        if (_replicateAnn.TryGetValue(sampleId, out var qualified) && qualified.TryGetValue(column, out var qv))
            return qv;
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

    /// <summary>
    /// One tickable value in the Group-by value dropdown. Several can be on at once so related groups can
    /// be compared without the rest - the case that motivated it is the control-correlation heatmap, where
    /// you want Quality Control AND Standard together but not the unknowns.
    /// </summary>
    private sealed class QcGroupValue : System.ComponentModel.INotifyPropertyChanged
    {
        private bool _isSelected;

        public required string Name { get; init; }

        /// <summary>Raised on tick/untick so the window can re-render and refresh the summary text.</summary>
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
    }

    private List<QcGroupValue> _qcGroupValues = new();

    /// <summary>
    /// The ticked values, or an empty set meaning "no filter - every sample". Callers compare with
    /// <see cref="StringComparer.OrdinalIgnoreCase"/> because annotation spellings vary by source.
    /// </summary>
    private HashSet<string> SelectedGroupValues()
    {
        var set = new HashSet<string>(QcGroupFilter.Comparer);
        foreach (var v in _qcGroupValues.Where(v => v.IsSelected))
            set.Add(v.Name);
        return set;
    }

    private void PopulateValueCombo()
    {
        var column = ComboText(QcGroupByCombo, "");
        var values = new SortedSet<string>(StringComparer.Ordinal);
        if (!string.IsNullOrEmpty(column) && _qcData.Count > 0)
        {
            foreach (var s in _qcData.Values.First().Samples)
            {
                var v = SampleAnnotation(s, column);
                if (!string.IsNullOrEmpty(v))
                    values.Add(v);
            }
        }

        _qcGroupValues = values
            .Select(v => new QcGroupValue { Name = v, Changed = OnGroupValueToggled })
            .ToList();
        QcGroupCombo.ItemsSource = _qcGroupValues;
        UpdateGroupSummary();
    }

    private void OnGroupValueToggled()
    {
        UpdateGroupSummary();
        if (!_suppressQcRender)
            RenderQc();
    }

    /// <summary>The closed-state text: the dropdown itself shows tick boxes, not a selected item.</summary>
    private void UpdateGroupSummary()
    {
        if (QcGroupCombo is null)
            return;
        var selected = _qcGroupValues.Where(v => v.IsSelected).Select(v => v.Name).ToList();
        QcGroupCombo.Text = QcGroupFilter.Summarize(selected, _qcGroupValues.Count);
    }

    /// <summary>
    /// Tick exactly <paramref name="wanted"/> (those that exist), leaving the rest clear. Used to default
    /// the control-correlation heatmap to the control types.
    /// </summary>
    private bool SelectGroupValues(IEnumerable<string> wanted)
    {
        var want = new HashSet<string>(wanted, StringComparer.OrdinalIgnoreCase);
        var matched = _qcGroupValues.Where(v => want.Contains(v.Name)).ToList();
        if (matched.Count == 0)
            return false;
        foreach (var v in _qcGroupValues)
            v.IsSelected = want.Contains(v.Name);
        UpdateGroupSummary();
        return true;
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
    private static readonly HashSet<string> InteractivePlots = new() { "PCA", "CV distribution", "Intensity distribution" };

    private void RenderQc()
    {
        if (QcViewCombo is null || _qcData.Count == 0)
            return;
        var kind = ComboText(QcPlotCombo, "PCA");
        UpdateQcControls(kind); // may force Level=Peptide and grey out Level/View for RT plots
        var view = ComboText(QcViewCombo, "Corrected").ToLowerInvariant();
        var level = ComboText(QcLevelCombo, "Peptide").ToLowerInvariant();
        if (InteractivePlots.Contains(kind))
            RenderInteractiveQc(kind, view, level);
        else
            RenderStaticQc(kind, view, level);
    }

    // RT plots are peptide-only (proteins have no RT); RT-binned CV is inherently a before-vs-after plot.
    // Force Level to Peptide and grey out the selectors that don't apply so they can't be mis-set.
    private void UpdateQcControls(string kind)
    {
        var isRt = kind is "RT-lowess" or "RT-binned CV" or "RT-bin boxplot";
        var beforeAfter = kind == "RT-binned CV";
        if (isRt && QcLevelCombo.SelectedIndex != 0)
        {
            _suppressQcRender = true;
            QcLevelCombo.SelectedIndex = 0; // Peptide
            _suppressQcRender = false;
        }
        QcLevelCombo.IsEnabled = !isRt;       // RT plots: peptide only
        QcViewCombo.IsEnabled = !beforeAfter; // RT-binned CV shows before AND after

        // "Control correlation" is about the CONTROLS: the useful reading is that QC samples correlate
        // with each other and standards with each other, but the two do not cross. Showing every sample
        // buries that in the unknowns, and disagrees with the HTML report, which computes this heatmap
        // over reference+QC only. So default to the control types - but only when nothing is ticked, so
        // an explicit choice is never overridden.
        if (kind == "Control correlation" && SelectedGroupValues().Count == 0)
        {
            // Only when this column actually has control values - on a Condition annotation there are
            // none, and ticking nothing would leave an empty plot.
            var controls = QcGroupFilter.ControlsAmong(_qcGroupValues.Select(v => v.Name));
            if (controls.Count > 0)
            {
                _suppressQcRender = true;
                SelectGroupValues(controls);
                _suppressQcRender = false;
            }
        }
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
        // Optional group filter driven by a Replicates-report column; several values may be ticked.
        var column = ComboText(QcGroupByCombo, "Sample Type");
        var selected = SelectedGroupValues();
        var cols = Enumerable.Range(0, d.Samples.Count).ToList();
        var groupLabel = "all";
        if (selected.Count > 0)
        {
            cols = cols.Where(i => selected.Contains(SampleAnnotation(d.Samples[i], column))).ToList();
            if (cols.Count < 2)
            {
                plt.Title($"Fewer than 2 samples with {column} = {GroupLabel()}.");
                QcPlot.Refresh();
                return;
            }
            groupLabel = GroupLabel();
        }
        var matrix = cols.Count == d.Samples.Count ? d.FeaturesBySamples : SelectColumns(d.FeaturesBySamples, cols);
        var types = cols.Select(i => _qcTypes.GetValueOrDefault(d.Samples[i], "unknown")).ToList();
        // PCA colors each point by its Group-by column value, so the groups show as distinct colors.
        var colorLabels = cols.Select(i =>
        {
            var v = SampleAnnotation(d.Samples[i], column);
            return string.IsNullOrEmpty(v) ? "(none)" : v;
        }).ToList();

        _pcaHoverPoints = null; // only PCA populates it; disables the hover handler for the other plots
        _pcaHoverMarker = null;
        _pcaHoverText = null;
        // Replicate name per plotted column, with the shared __@__<batch> suffix stripped when redundant.
        var sampleNames = StripSharedBatchSuffix(cols.Select(i => d.Samples[i]).ToList(), d.Samples);
        try
        {
            switch (kind)
            {
                case "CV distribution": DrawCv(plt, matrix, colorLabels, level, view, groupLabel); break;
                case "Intensity distribution": DrawIntensity(plt, matrix, colorLabels, level, view, groupLabel); break;
                default: _pcaHoverPoints = DrawPca(plt, matrix, colorLabels, sampleNames, level, view, groupLabel); break;
            }
        }
        catch (Exception ex)
        {
            _pcaHoverPoints = null;
            plt.Title("render failed: " + ex.Message);
        }
        // The Plot is reused across view/level/type switches, so fit the axes to the current data.
        PlotRenderer.StyleQcPlot(plt); // big fonts, thick left+bottom axes (matches the static plots)
        plt.Axes.AutoScale();
        if (kind == "CV distribution") // bar plot: sit the bars on the x-axis (y starts at 0)
            plt.Axes.SetLimitsY(0, plt.Axes.GetLimits().Top);
        // Added AFTER autoscaling so the initially-hidden overlay doesn't skew the axis limits.
        if (_pcaHoverPoints is { Count: > 0 })
            AddPcaHoverOverlay(plt, _pcaHoverPoints[0].Loc);
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
    // Column indices (into the given sample list) selected by the Group-by column + value; "All" = every sample.
    private List<int> GroupColumns(IReadOnlyList<string> samples)
    {
        var column = ComboText(QcGroupByCombo, "Sample Type");
        return QcGroupFilter.Matching(
            samples.Count, i => SampleAnnotation(samples[i], column), SelectedGroupValues());
    }

    /// <summary>Human-readable description of the current filter, for plot titles and messages.</summary>
    private string GroupLabel() => QcGroupFilter.Describe(SelectedGroupValues());

    // Control-correlation heatmap + RT plots, computed on the fly for the selected Group. delta-LOWESS
    // made recompute fast enough to do this per selection, which fixed report PNGs could not respect.
    private void RenderStaticQc(string kind, string view, string level)
    {
        try
        {
            if (!_qcData.TryGetValue($"{view}|{level}", out var d))
            {
                ShowStaticMessage($"No {view} {level} data yet - run PRISM first.");
                return;
            }
            var groupCols = GroupColumns(d.Samples);
            var column = ComboText(QcGroupByCombo, "Sample Type");
            var groupLabel = GroupLabel();
            var types = d.Samples.Select(s => _qcTypes.GetValueOrDefault(s, "unknown")).ToList();
            var cap = view == "raw" ? "Raw" : "Corrected";
            double[,] Subset(double[,] m, List<int> c) => c.Count == d.Samples.Count ? m : SelectColumns(m, c);

            byte[]? png = null;
            switch (kind)
            {
                case "Control correlation":
                    if (groupCols.Count < 2) { ShowStaticMessage($"Correlation needs >= 2 samples in '{groupLabel}'."); return; }
                    var corrTypes = groupCols.Select(i => _qcTypes.GetValueOrDefault(d.Samples[i], "unknown")).ToList();
                    png = PlotRenderer.CorrelationHeatmap(d.FeaturesBySamples, groupCols, "", corrTypes);
                    break;
                case "RT-lowess":
                    if (d.MeanRt is null) { ShowStaticMessage("RT plots are peptide-level only."); return; }
                    if (groupCols.Count < 1) { ShowStaticMessage($"No samples in '{groupLabel}'."); return; }
                    png = PlotRenderer.RtLowessCurves(Subset(d.FeaturesBySamples, groupCols), d.MeanRt,
                        groupCols.Select(i => SampleAnnotation(d.Samples[i], column)).ToList(), "");
                    break;
                case "RT-bin boxplot":
                    if (d.MeanRt is null) { ShowStaticMessage("RT plots are peptide-level only."); return; }
                    if (groupCols.Count < 1) { ShowStaticMessage($"No samples in '{groupLabel}'."); return; }
                    png = PlotRenderer.RtBinBoxplot(Subset(d.FeaturesBySamples, groupCols), d.MeanRt, "", "#1f77b4");
                    break;
                case "RT-binned CV":
                    // before vs after (View ignored), peptide only, over the selected Group's samples.
                    if (!_qcData.TryGetValue("raw|peptide", out var rawD) ||
                        !_qcData.TryGetValue("corrected|peptide", out var corrD) || rawD.MeanRt is null)
                    {
                        ShowStaticMessage("RT-binned CV needs both raw and corrected peptide data.");
                        return;
                    }
                    var cvCols = GroupColumns(rawD.Samples);
                    if (cvCols.Count < 2) { ShowStaticMessage($"RT-binned CV needs >= 2 samples in '{groupLabel}'."); return; }
                    png = PlotRenderer.RtBinCv(rawD.FeaturesBySamples, corrD.FeaturesBySamples, rawD.MeanRt, cvCols, "", "#1f77b4");
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

    // Create the (initially hidden) PCA hover overlay - a ring marker + a text label seeded at a point.
    private void AddPcaHoverOverlay(Plot plt, Coordinates seed)
    {
        var marker = plt.Add.Marker(seed.X, seed.Y, MarkerShape.OpenCircle, 20, Colors.Black);
        marker.IsVisible = false;
        _pcaHoverMarker = marker;

        var text = plt.Add.Text(" ", seed.X, seed.Y);
        PlotRenderer.StyleTextLabel(text, 14, bold: true);
        text.LabelFontColor = Colors.Black;
        text.LabelBackgroundColor = Colors.White.WithAlpha(0.85);
        text.LabelAlignment = Alignment.LowerLeft;
        text.IsVisible = false;
        _pcaHoverText = text;
    }

    // Show the replicate name when the cursor is within ~18 px of a PCA point. Active only while the PCA
    // plot is shown (_pcaHoverPoints non-null); a no-op for the CV / intensity plots.
    private void QcPlot_MouseMove(object sender, System.Windows.Input.MouseEventArgs e)
    {
        var points = _pcaHoverPoints;
        if (points is null || points.Count == 0 || _pcaHoverMarker is null || _pcaHoverText is null)
            return;

        var plt = QcPlot.Plot;
        var pos = e.GetPosition(QcPlot);
        var scale = QcPlot.DisplayScale;
        double mx = pos.X * scale, my = pos.Y * scale;

        var best = double.MaxValue;
        var bestIdx = -1;
        for (var i = 0; i < points.Count; i++)
        {
            var px = plt.GetPixel(points[i].Loc);
            double dx = px.X - mx, dy = px.Y - my;
            var d2 = dx * dx + dy * dy;
            if (d2 < best) { best = d2; bestIdx = i; }
        }

        const double thresholdPx = 18;
        if (bestIdx >= 0 && best <= thresholdPx * thresholdPx)
        {
            var p = points[bestIdx];
            _pcaHoverMarker.Location = p.Loc;
            _pcaHoverMarker.IsVisible = true;
            _pcaHoverText.Location = p.Loc;
            _pcaHoverText.LabelText = p.Name;
            _pcaHoverText.IsVisible = true;
            QcPlot.Refresh();
        }
        else if (_pcaHoverMarker.IsVisible)
        {
            _pcaHoverMarker.IsVisible = false;
            _pcaHoverText.IsVisible = false;
            QcPlot.Refresh();
        }
    }

    private static List<(Coordinates Loc, string Name)> DrawPca(
        Plot plt, double[,] featuresBySamples, List<string> types, List<string> names,
        string level, string view, string group)
    {
        var nF = featuresBySamples.GetLength(0);
        var nS = featuresBySamples.GetLength(1);
        var samplesByFeatures = new double[nS, nF];
        for (var f = 0; f < nF; f++)
            for (var s = 0; s < nS; s++)
                samplesByFeatures[s, f] = featuresBySamples[f, s];
        var scores = Pca.Fit2D(samplesByFeatures);
        var groups = new Dictionary<string, (List<double> X, List<double> Y)>();
        var points = new List<(Coordinates Loc, string Name)>(nS);
        for (var i = 0; i < nS; i++)
        {
            var t = types[i];
            if (!groups.TryGetValue(t, out var g))
                groups[t] = g = (new List<double>(), new List<double>());
            g.X.Add(scores[i, 0]);
            g.Y.Add(scores[i, 1]);
            points.Add((new Coordinates(scores[i, 0], scores[i, 1]), names[i]));
        }
        var colorIndex = 0;
        foreach (var (label, g) in groups.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            var markers = plt.Add.Markers(g.X.ToArray(), g.Y.ToArray());
            // Standardized colors (same across all plots): known sample types get their fixed color,
            // any other Group-by value (e.g. a Condition annotation) gets a distinct cycled color.
            markers.Color = PlotRenderer.GroupColor(label, colorIndex++);
            markers.MarkerSize = 15; // sized to match the figure-scale axis text
            markers.LegendText = label;
        }
        plt.ShowLegend();
        // No title in the tool - the View/Level/Group/Plot selectors above already describe the plot.
        plt.XLabel("PC1");
        plt.YLabel("PC2");
        return points;
    }

    private static void DrawCv(Plot plt, double[,] featuresBySamples, List<string> labels, string level, string view, string group)
    {
        // Groups = distinct Group-by column values with >= 2 samples (e.g. Standard / Quality Control).
        var byLabel = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        for (var i = 0; i < labels.Count; i++)
        {
            if (!byLabel.TryGetValue(labels[i], out var l))
                byLabel[labels[i]] = l = new List<int>();
            l.Add(i);
        }
        var present = byLabel.Where(kv => kv.Value.Count >= 2)
            .OrderBy(kv => kv.Key, StringComparer.Ordinal)
            .Select(kv => (Label: kv.Key, Cols: kv.Value))
            .ToList();
        var isAll = string.Equals(group, "all", StringComparison.OrdinalIgnoreCase);

        if (isAll && present.Count >= 2 && present.Count <= 4)
        {
            AddGroupedCvHists(plt, featuresBySamples, present); // bars side by side
        }
        else
        {
            // One group chosen, a single group present, or > 4 groups: a single aggregate histogram.
            var allIdx = Enumerable.Range(0, labels.Count).ToList();
            var lbl = !isAll && present.Count >= 1 ? present[0].Label
                : present.Count == 1 ? present[0].Label
                : "all samples";
            AddCvHist(plt, CvMetrics.PerFeatureCvs(featuresBySamples, allIdx), PlotRenderer.GroupColor(lbl), lbl);
        }
        plt.ShowLegend(Alignment.UpperRight); // upper-right so it doesn't overlap the histogram bars
        // No title in the tool - the selectors above already describe the plot.
        plt.XLabel("CV (%)");
        plt.YLabel("Count");
    }

    // Overlaid single-group CV histogram + median line.
    private static void AddCvHist(Plot plt, double[] cvs, Color color, string label)
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
        bars.Color = color.WithAlpha((byte)140);
        var med = Stats.NanMedian(cvs);
        var line = plt.Add.VerticalLine(med);
        line.Color = color;
        line.LineWidth = 4;
        line.LinePattern = LinePattern.Dashed;
        line.LegendText = $"{Cap(label)} median {med:0.0}%";
    }

    // 2-4 groups: histograms drawn as bars side by side per CV bin, with a median line each.
    private static void AddGroupedCvHists(Plot plt, double[,] matrix, List<(string Label, List<int> Cols)> groups)
    {
        const int bins = 30;
        var cvsPerGroup = groups
            .Select(g => CvMetrics.PerFeatureCvs(matrix, g.Cols).Where(c => !double.IsNaN(c)).ToArray())
            .ToList();
        var maxCv = 1.0;
        foreach (var cvs in cvsPerGroup)
            foreach (var c in cvs)
                if (c < 100.0 && c > maxCv) maxCv = c;
        maxCv = Math.Min(maxCv, 100.0);
        var binW = maxCv / bins;
        var nG = groups.Count;
        var slot = binW / (nG + 0.6); // width per group within a bin, leaving a small inter-bin gap

        for (var gi = 0; gi < nG; gi++)
        {
            var counts = new double[bins];
            foreach (var c in cvsPerGroup[gi])
            {
                var b = (int)Math.Min(c / binW, bins - 1);
                if (b >= 0) counts[b]++;
            }
            var color = PlotRenderer.GroupColor(groups[gi].Label, gi);
            var barList = new List<Bar>();
            for (var b = 0; b < bins; b++)
            {
                if (counts[b] <= 0) continue;
                var x = (b + 0.5) * binW + (gi - (nG - 1) / 2.0) * slot;
                barList.Add(new Bar { Position = x, Value = counts[b], Size = slot * 0.9, FillColor = color });
            }
            if (barList.Count > 0)
            {
                var bars = plt.Add.Bars(barList);
                bars.LegendText = Cap(groups[gi].Label);
            }
            var med = Stats.NanMedian(cvsPerGroup[gi]);
            var line = plt.Add.VerticalLine(med);
            line.Color = color;
            line.LineWidth = 3;
            line.LinePattern = LinePattern.Dashed;
        }
    }

    private static void DrawIntensity(Plot plt, double[,] featuresBySamples, List<string> labels, string level, string view, string group)
    {
        // Per-sample KDE density curves, colored by Group-by value (N groups -> N colors). All samples shown.
        PlotRenderer.DrawIntensityDensity(plt, featuresBySamples, labels);
        // No title in the tool - the selectors above already describe the plot.
        plt.XLabel("Log2 Abundance");
        plt.YLabel("Density");
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
