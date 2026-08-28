using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Parsing Skyline's isolation-scheme XML (the acquisition's real DIA window layout) and persisting it
/// beside a run's outputs. Both XML spellings below are copied from a live Skyline 26.1.
/// </summary>
public class IsolationSchemeTests
{
    // GetSettingsListItem("IsolationSchemeList", "SWATH (25 m/z)") - PascalCase root.
    private const string SettingsListXml = """
        <IsolationScheme name="SWATH (25 m/z)">
          <isolation_window start="400" end="424" margin="0.5" ce_range="5" />
          <isolation_window start="424" end="448" margin="0.5" ce_range="5" />
          <isolation_window start="448" end="472" margin="0.5" ce_range="5" />
        </IsolationScheme>
        """;

    // The same thing inside a saved .sky - snake_case root, extra attributes.
    private const string DocumentXml = """
        <isolation_scheme name="SWATH (25 m/z)" precursor_filter="24">
          <isolation_window start="400" end="424" margin="0.5" />
          <isolation_window start="424" end="448" margin="0.5" />
        </isolation_scheme>
        """;

    [Fact]
    public void Parse_ReadsTheSettingsListSpelling()
    {
        var scheme = IsolationScheme.Parse(SettingsListXml);
        Assert.NotNull(scheme);
        Assert.Equal("SWATH (25 m/z)", scheme!.Name);
        Assert.Equal(3, scheme.Windows.Count);
        Assert.Equal(400, scheme.MzLow);
        Assert.Equal(472, scheme.MzHigh);
        Assert.Equal(24, scheme.Windows[0].Width);
        Assert.Equal(0.5, scheme.Windows[0].Margin);
    }

    [Fact]
    public void Parse_ReadsTheDocumentSpelling()
    {
        var scheme = IsolationScheme.Parse(DocumentXml);
        Assert.NotNull(scheme);
        Assert.Equal("SWATH (25 m/z)", scheme!.Name);
        Assert.Equal(2, scheme.Windows.Count);
    }

    [Fact]
    public void Parse_ResultsOnlyIsNamedButHasNoWindows()
    {
        // The normal setting for a DIA analysis document: Skyline takes the windows from the data files
        // and stores none here. It must parse (so the UI can say which scheme the document names) but
        // report no windows (so it is never used as a grid).
        var scheme = IsolationScheme.Parse("""<isolation_scheme name="Results only" />""");
        Assert.NotNull(scheme);
        Assert.Equal(IsolationScheme.ResultsOnlyName, scheme!.Name);
        Assert.False(scheme.HasWindows);
    }

    [Fact]
    public void Parse_RejectsUnrelatedOrBrokenXml()
    {
        Assert.Null(IsolationScheme.Parse(null));
        Assert.Null(IsolationScheme.Parse(""));
        Assert.Null(IsolationScheme.Parse("<enzyme name=\"Trypsin\" cut=\"KR\" />"));
        Assert.Null(IsolationScheme.Parse("<isolation_scheme name=\"broken\""));
    }

    [Fact]
    public void Parse_IsInvariantAndSkipsUnusableWindows()
    {
        // Skyline writes invariant decimals regardless of the machine locale, and a window whose end is
        // not past its start is not a window.
        var scheme = IsolationScheme.Parse("""
            <IsolationScheme name="mixed">
              <isolation_window start="500.25" end="510.75" />
              <isolation_window start="600" end="600" />
              <isolation_window start="700" />
            </IsolationScheme>
            """);
        Assert.NotNull(scheme);
        Assert.Single(scheme!.Windows);
        Assert.Equal(10.5, scheme.Windows[0].Width, 6);
    }

    [Fact]
    public void Windows_AreHalfOpenSoBoundariesLandInOneWindow()
    {
        var w = new IsolationWindow(400, 425);
        Assert.True(w.Contains(400));
        Assert.True(w.Contains(424.999));
        Assert.False(w.Contains(425)); // belongs to the next window
        Assert.False(w.Contains(399.999));
    }

    [Fact]
    public void Describe_TreatsInstrumentRoundingAsUniform()
    {
        // Real windows imported from a .raw: nominally 3.0014 Th, but the instrument's own rounding makes
        // consecutive widths differ in the 4th decimal. That is one width, not a variable-width scheme.
        var imported = IsolationScheme.Parse("""
            <isolation_scheme name="from-raw">
              <isolation_window start="400.431890003052" end="403.433289996948" />
              <isolation_window start="403.43330995174404" end="406.434610048256" />
              <isolation_window start="406.43460000305197" end="409.435999996948" />
            </isolation_scheme>
            """);
        Assert.NotNull(imported);
        Assert.DoesNotContain("variable", imported!.Describe());
        Assert.Contains("3.001 Th", imported.Describe());
    }

    [Fact]
    public void Describe_SummarizesUniformAndVariableSchemes()
    {
        Assert.Contains("3 windows", IsolationScheme.Parse(SettingsListXml)!.Describe());
        Assert.Contains("24 Th", IsolationScheme.Parse(SettingsListXml)!.Describe());
        var vw = new IsolationScheme("vw", new[]
        {
            new IsolationWindow(400, 410), new IsolationWindow(410, 500),
        });
        Assert.Contains("variable width", vw.Describe());
    }

    [Fact]
    public void Catalog_RoundTripsThroughDisk()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_iso_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var catalog = new IsolationSchemeCatalog();
            // Plate1's document names a window-less scheme; Plate2 and Plate3 declare real windows -
            // the same scheme, in the two different XML spellings Skyline produces for it, which is what
            // the deduplication below has to survive.
            catalog.AddDocumentScheme("Plate1", IsolationScheme.Parse("""<isolation_scheme name="Results only" />""")!);
            catalog.AddDocumentScheme("Plate2", IsolationScheme.Parse(DocumentXml)!);
            catalog.AddDocumentScheme("Plate3", IsolationScheme.Parse(SettingsListXml)!);

            var path = Path.Combine(dir, IsolationSchemeCatalog.FileName);
            catalog.Save(path);
            var loaded = IsolationSchemeCatalog.Load(path);

            Assert.NotNull(loaded);
            // Plate1 has no usable scheme - the tool must ask - but its name survives for the UI to explain.
            Assert.Null(loaded!.DocumentSchemeFor("Plate1"));
            Assert.Equal(IsolationScheme.ResultsOnlyName, loaded.DocumentSchemeNameFor("Plate1"));
            // Plate2's document scheme is authoritative and comes back with its windows attached.
            var plate2 = loaded.DocumentSchemeFor("Plate2");
            Assert.NotNull(plate2);
            Assert.Equal(2, plate2!.Windows.Count);
            Assert.Equal(424, plate2.Windows[0].End);
            // Both named schemes are offered, deduplicated (Plate2's document scheme shares its name).
            Assert.Single(loaded.UsableSchemes);
            Assert.Equal("SWATH (25 m/z)", loaded.UsableSchemes[0].Name);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Catalog_RoundTripsTheAcquisitionMethod()
    {
        // The map means different things for DIA and for a targeted method, so the acquisition has to
        // survive to the output directory alongside the windows.
        var dir = Path.Combine(Path.GetTempPath(), "prism_iso_acq_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var catalog = new IsolationSchemeCatalog();
            catalog.SetAcquisition("PlateDia", "DIA");
            catalog.SetAcquisition("PlatePrm", "PRM");
            catalog.AddDocumentScheme("PlateDia", IsolationScheme.Parse(DocumentXml)!);
            // A batch with an acquisition but no scheme must still be written out.
            Assert.False(catalog.IsEmpty);

            var path = Path.Combine(dir, IsolationSchemeCatalog.FileName);
            catalog.Save(path);
            var loaded = IsolationSchemeCatalog.Load(path)!;

            Assert.Equal("DIA", loaded.AcquisitionFor("PlateDia"));
            Assert.Equal("PRM", loaded.AcquisitionFor("PlatePrm"));
            Assert.False(loaded.IsNonDia("PlateDia"));
            Assert.True(loaded.IsNonDia("PlatePrm"));
            // Unknown acquisition must not be reported as non-DIA - that would warn on every old output.
            Assert.False(loaded.IsNonDia("NeverSeen"));
            Assert.Null(loaded.AcquisitionFor("NeverSeen"));
            // The scheme still round-trips alongside it.
            Assert.NotNull(loaded.DocumentSchemeFor("PlateDia"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Catalog_LoadReturnsNullWhenAbsent()
    {
        Assert.Null(IsolationSchemeCatalog.Load(
            Path.Combine(Path.GetTempPath(), "prism_no_such_" + Guid.NewGuid().ToString("N") + ".xml")));
    }
}
