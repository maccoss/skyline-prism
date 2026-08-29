using System;
using System.IO;
using System.Text.RegularExpressions;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The Settings tab's FASTA picker.
///
/// <para>Before it existed, nothing in the tool could set <c>parsimony.fasta_path</c> or
/// <c>protein_rollup.ibaq.fasta_path</c>, so every GUI run recorded both as null. Two consequences, both
/// silent: protein groups always came from the Skyline accession column rather than enzyme-aware FASTA
/// parsimony, and the Dynamic Range tab's iBAQ view could never be a real iBAQ - it read those keys from
/// parameters.json, found nothing, and fell back to the observed peptide count.</para>
///
/// <para>Source checks, in the same style and for the same reason as <see cref="UiThreadSafetyTests"/>:
/// the code lives in <c>MainWindow</c> and needs a real window and dispatcher to exercise, but the
/// defect is visible in the source.</para>
/// </summary>
public class FastaPickerTests
{
    private static string AppDir =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));

    private static string CodeBehind => File.ReadAllText(Path.Combine(AppDir, "MainWindow.xaml.cs"));

    private static string Xaml => File.ReadAllText(Path.Combine(AppDir, "MainWindow.xaml"));

    /// <summary>
    /// The path has to reach the config, or the picker is decoration. BuildConfigFromUi is the single
    /// source for both Run and Show Command Line.
    /// </summary>
    [Fact]
    public void TheChosenFastaReachesTheConfig()
    {
        var build = Between(CodeBehind, "private PrismConfig BuildConfigFromUi()", "\n    private ");

        Assert.Contains("FastaBox.Text", build);
        Assert.Contains("c.Parsimony.FastaPath", build);
    }

    /// <summary>
    /// Blank means "not set", not an empty path. An empty string would be a path that fails to open,
    /// and PRISM distinguishes null (use the accession column) from a path it cannot read.
    /// </summary>
    [Fact]
    public void AnEmptyBoxIsWrittenAsNull()
    {
        var build = Between(CodeBehind, "private PrismConfig BuildConfigFromUi()", "\n    private ");

        Assert.Matches(new Regex(@"IsNullOrWhiteSpace\(fasta\)\s*\?\s*null"), build);
    }

    /// <summary>
    /// iBAQ is offered now that a FASTA can be supplied. It was withheld for exactly that reason, so if
    /// the picker is ever removed this should go with it.
    /// </summary>
    [Fact]
    public void TheRollupPickerOffersTheMethodsThatNeededAFasta()
    {
        var combo = Between(Xaml, "x:Name=\"ProteinRollupCombo\"", "</ComboBox>");

        foreach (var method in new[] { "median_polish", "sum", "topn", "maxlfq", "ibaq" })
            Assert.Contains(method, combo);
    }

    /// <summary>
    /// The hint is set when the window opens, not only when a provenance file is opened - the mistake
    /// the marker-normalization picker shipped with. See <see cref="MarkerListPickerTests"/>.
    /// </summary>
    [Fact]
    public void TheHintIsSetAtStartup()
    {
        var ctor = Between(CodeBehind, "public MainWindow()", "\n    private ");

        Assert.Contains("UpdateFastaHint()", ctor);
    }

    /// <summary>
    /// The one case worth warning about in the window rather than the log: iBAQ chosen with no database.
    /// The run succeeds, dividing by the observed peptide count, which is close to a per-peptide mean
    /// and not the absolute-abundance estimate iBAQ is picked for.
    /// </summary>
    [Fact]
    public void ChoosingIbaqWithoutAFastaIsCalledOut()
    {
        var hint = Between(CodeBehind, "private void UpdateFastaHint()", "\n    private ");

        Assert.Contains("ibaq", hint, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("OBSERVED", hint);
        // ...and a path that does not exist is its own case, not silently treated as unset.
        Assert.Contains("File.Exists", hint);
    }

    private static string Between(string source, string start, string end)
    {
        var from = source.IndexOf(start, StringComparison.Ordinal);
        Assert.True(from >= 0, $"'{start}' not found - the test needs updating with the code.");
        var to = source.IndexOf(end, from, StringComparison.Ordinal);
        return to < 0 ? source[from..] : source[from..to];
    }
}
