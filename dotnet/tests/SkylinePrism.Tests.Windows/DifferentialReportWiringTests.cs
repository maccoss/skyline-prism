using System;
using System.IO;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// What the Differential pane hands to its Quant report and to the covariate list.
///
/// <para>Two defects, both silent. The Quant report button built its request without the pane's
/// Restrict-to choice, so a report fitted the trend over every sample while the pane showed a subset -
/// and its recorded command left out <c>--restrict-to</c>. And the tested term was dropped from the
/// covariates by the axis LABEL ("Longitudinal Draw Description (Week)"), which names no column, so
/// the column's own categories stayed in the design, collinear with the timepoints tested.</para>
///
/// <para>Source checks, in the same style and for the same reason as <see cref="FastaPickerTests"/>:
/// the code lives in <c>MainWindow</c> and needs a real window, a loaded run and a dispatcher to
/// exercise, but the defect is visible in the source. The restriction itself is tested where it is
/// computed (<c>TrendRestrictionTests</c>), which is the one place both the pane and the report call.</para>
/// </summary>
public class DifferentialReportWiringTests
{
    private static string AppDir =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));

    // Line endings normalized: a checkout may hold either, and the markers below are written with LF.
    private static string Source(string file) => File.ReadAllText(Path.Combine(AppDir, file)).Replace("\r\n", "\n");

    [Fact]
    public void TheQuantReport_CarriesThePanesRestriction()
    {
        var request = Between(Source("MainWindow.QuantReport.cs"), "var request = new QuantRequest", "};");
        Assert.Contains("Restrictions = DiffIsTrend() ? DiffRestrictions()", request, StringComparison.Ordinal);
    }

    [Fact]
    public void ThePanesTrend_AndItsReport_RestrictThroughTheSameResolver()
    {
        // The pane's own trend uses the restrictions the report is given, through the resolver the
        // report runs - so the samples the pane shows and the samples the report fits cannot differ.
        var columns = Between(Source("MainWindow.Differential.cs"), "private int[] DiffTrendColumns()", ";\n");
        Assert.Contains("QuantAnalysis.KeptColumns(DiffRestrictions()", columns, StringComparison.Ordinal);
    }

    [Fact]
    public void TheTrendPicker_IsRebuiltWhenAnAxisReadsDifferently_NotOnlyWhenItsLabelChanges()
    {
        // Another output directory can offer the same label at a different position; compared on
        // labels alone, the old option kept reading every sample as NaN.
        var populate = Between(Source("MainWindow.Differential.cs"),
            "private void PopulateTrendColumns(", "\n    }\n");
        Assert.Contains("a.Position", populate, StringComparison.Ordinal);
        Assert.Contains("TrendAxis.DescribePreview(a)", populate, StringComparison.Ordinal);
    }

    [Fact]
    public void TheTestedTerm_IsDroppedByItsColumn_NotItsLabel()
    {
        var drop = Between(Source("MainWindow.Differential.cs"),
            "private IReadOnlyList<Covariate>? WithoutTestedTerm(", "\n    }\n");
        Assert.Contains("axis.Column", drop, StringComparison.Ordinal);
        Assert.DoesNotContain("DiffTrendColumn()", drop, StringComparison.Ordinal);
    }

    private static string Between(string source, string start, string end)
    {
        var from = source.IndexOf(start, StringComparison.Ordinal);
        Assert.True(from >= 0, $"'{start}' not found - the test needs updating with the code.");
        var to = source.IndexOf(end, from, StringComparison.Ordinal);
        Assert.True(to >= 0, $"'{end}' not found after '{start}' - the test needs updating with the code.");
        return source[from..to];
    }
}
