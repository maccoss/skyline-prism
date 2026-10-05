using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using SkylinePrism.Core.DifferentialAnalysis;

namespace SkylinePrism.App;

/// <summary>
/// The "Quant report" button: run the Differential pane's current contrast across every view and write a
/// self-contained quant_report.html plus the result tables (CSV) to a quant/ folder, the way the pipeline
/// writes its own outputs. The pane only gathers its controls into a <see cref="QuantRequest"/>; the
/// analysis is <see cref="QuantAnalysis.Run"/>, the same code <c>prism differential --report</c> runs.
/// </summary>
public partial class MainWindow
{
    private async void OnGenerateQuantReport(object sender, RoutedEventArgs e)
    {
        try
        {
            await GenerateQuantReportAsync();
        }
        catch (Exception ex)
        {
            ReportHandlerFailure(nameof(OnGenerateQuantReport), ex);
        }
    }

    private async Task GenerateQuantReportAsync()
    {
        if (_diffDataset is null)
        {
            DiffStatusText.Text = "Load a run and set up a contrast before writing a quant report.";
            return;
        }

        var dir = OutputDirBox.Text?.Trim();
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
        {
            DiffStatusText.Text = "Set a PRISM output directory above.";
            return;
        }

        if (_isRunning)
        {
            DiffStatusText.Text = "A PRISM run is in progress - wait for it to finish.";
            return;
        }

        // Everything that reads a WPF control is gathered here, on the UI thread; the analysis below runs
        // on a worker.
        var ds = _diffDataset;
        var options = DiffOptions(WithoutTestedTerm(SelectedCovariatesFor(ds.SampleIds)));
        string? groupBy = null;
        List<int> groupA = new(), groupB = new();
        List<string> aLevels = new(), bLevels = new();
        if (DiffIsTrend())
        {
            if (DiffTrendColumn() is null)
            {
                DiffStatusText.Text = "Pick a numeric trend column first.";
                return;
            }
        }
        else if (!TryGetGroups(out var col, out groupA, out groupB, out _, out _))
        {
            DiffStatusText.Text = "Pick a group-by column and two values before writing a report.";
            return;
        }
        else
        {
            groupBy = col;
            (aLevels, bLevels) = SelectedArmLevels();
        }

        var request = new QuantRequest
        {
            OutputDir = dir,
            Dataset = ds,
            Options = options,
            Rule = DiffRule(),
            SubjectColumn = DiffPairByCombo.SelectedItem as string,
            GroupBy = groupBy,
            GroupA = groupA,
            GroupB = groupB,
            ALevels = aLevels,
            BLevels = bLevels,
            // The pane's restriction, or the report would fit the trend over every sample while the
            // pane shows a subset - and its recorded command would leave out --restrict-to.
            Restrictions = DiffIsTrend() ? DiffRestrictions() : Array.Empty<QuantRestriction>(),
            MarkerPanels = _markersPanelItems.Where(i => i.IsSelected).Select(i => i.List).ToList(),
            MarkerGroupBy = MarkersGroupByCombo.SelectedItem as string,
            // Enrichment always runs from the button; with no network it becomes a note, not a failure.
            EnrichmentPoster = DiffPoster,
            CachedDetection = _detectionData is not null && _detectionDir == dir ? _detectionData : null,
        };

        DiffStatusText.Text = "Generating quant report...";
        QuantReportButton.IsEnabled = false;
        try
        {
            var result = await Task.Run(() => QuantAnalysis.Run(request));

            // Cache the detection matrix so a later Detection-pane run on the same folder reuses it.
            if (result.DetectionMatrix is not null)
            {
                _detectionData = result.DetectionMatrix;
                _detectionDir = dir;
            }

            DiffStatusText.Text = $"Quant report written to {result.HtmlPath}."
                + string.Concat(result.Notes.Select(n => " " + n));
            try
            {
                Process.Start(new ProcessStartInfo(result.HtmlPath) { UseShellExecute = true });
            }
            catch
            {
                // Opening the browser is best-effort; the file is written regardless.
            }
        }
        finally
        {
            QuantReportButton.IsEnabled = true;
        }
    }
}
