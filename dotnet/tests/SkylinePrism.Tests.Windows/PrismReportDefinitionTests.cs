using System;
using System.IO;
using System.Linq;
using System.Xml.Linq;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The two bundled PRISM transition report definitions and the relation between them: PRISM-Ions is
/// PRISM plus exactly one column, Skyline's per-transition LC Peak ion count. The files are copied
/// beside the test assembly the same way they are copied beside the tool, so this reads what ships.
///
/// <para>The relation matters because a document exported either way must merge identically - the
/// merge auto-detects columns by name - and because the ion-count variant exists only to add the one
/// column the accounting reads. The Apex and precursor-level ion metrics are what made a five-column
/// trial 27x slower per byte on a real document; they must not creep back in under this name.</para>
/// </summary>
public class PrismReportDefinitionTests
{
    private static string ReportsDir => Path.Combine(AppContext.BaseDirectory, "Reports");

    private static XElement View(string fileName) =>
        XDocument.Load(Path.Combine(ReportsDir, fileName)).Root!.Element("view")!;

    private static string[] Columns(string fileName) =>
        View(fileName).Elements("column").Select(c => (string)c.Attribute("name")!).ToArray();

    [Fact]
    public void BothDefinitionsAreBundledBesideTheExecutable()
    {
        Assert.True(File.Exists(Path.Combine(ReportsDir, PrismReport.FileName)), PrismReport.FileName);
        Assert.True(File.Exists(Path.Combine(ReportsDir, PrismReport.IonsFileName)), PrismReport.IonsFileName);
    }

    [Fact]
    public void TheViewNamesAreTheOnesTheExportersAskFor()
    {
        Assert.Equal(PrismReport.Name, (string)View(PrismReport.FileName).Attribute("name")!);
        Assert.Equal(PrismReport.IonsName, (string)View(PrismReport.IonsFileName).Attribute("name")!);

        // Same row source and results sublist, or the two would not be the same report at all.
        foreach (var attr in new[] { "rowsource", "sublist", "uimode" })
        {
            Assert.Equal(
                (string?)View(PrismReport.FileName).Attribute(attr),
                (string?)View(PrismReport.IonsFileName).Attribute(attr));
        }
    }

    /// <summary>PRISM-Ions = PRISM + the ion count, same columns in the same order otherwise.</summary>
    [Fact]
    public void PrismIonsIsPrismPlusTheTransitionIonCountAndNothingElse()
    {
        var prism = Columns(PrismReport.FileName);
        var ions = Columns(PrismReport.IonsFileName);

        Assert.Equal(1, ions.Count(c => c == PrismReport.TransitionIonCountColumn));
        Assert.Equal(prism, ions.Where(c => c != PrismReport.TransitionIonCountColumn).ToArray());
        Assert.Equal(prism.Length + 1, ions.Length);
    }

    /// <summary>
    /// The standard report stays cheap: no ion metrics of any kind. They cost 27x per byte on the FLARE
    /// document, and every run would pay it for a section that is off by default.
    /// </summary>
    [Fact]
    public void TheStandardReportCarriesNoIonMetrics()
    {
        // "FragmentIon" is a column, so match the ion-METRIC families, not the substring "Ion".
        Assert.DoesNotContain(Columns(PrismReport.FileName), c =>
            c.Contains("IonMetrics", StringComparison.Ordinal)
            || c.Contains("IonCount", StringComparison.Ordinal)
            || c.Contains("IonCurrent", StringComparison.Ordinal));
    }

    /// <summary>
    /// Only the LC Peak total per transition. An Apex value is one spectrum, not the peak, and the
    /// precursor-level metrics are not what the accounting reads.
    /// </summary>
    [Fact]
    public void PrismIonsCarriesNoApexOrPrecursorLevelIonMetrics()
    {
        var ions = Columns(PrismReport.IonsFileName);
        Assert.DoesNotContain(ions, c => c.Contains("Apex", StringComparison.Ordinal));
        Assert.DoesNotContain(ions, c => c.Contains("LcPeakIonMetrics", StringComparison.Ordinal));
    }

    [Fact]
    public void TheVariantIsChosenByOneFlag()
    {
        Assert.Equal(PrismReport.Name, PrismReport.NameFor(false));
        Assert.Equal(PrismReport.IonsName, PrismReport.NameFor(true));
        Assert.Equal(PrismReport.FileName, PrismReport.FileFor(false));
        Assert.Equal(PrismReport.IonsFileName, PrismReport.FileFor(true));
    }
}
