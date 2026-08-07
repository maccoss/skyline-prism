using System;
using System.IO;
using System.Linq;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Parsing the header of a CLOSED Skyline document - the replicate-targeted annotation names that shape the
/// generated PRISM-Replicates report, the digestion enzyme, and the replicate list. The fixtures mirror the
/// real .sky layout (settings_summary wrapping peptide_settings / data_settings / measured_results, with the
/// protein list after it).
/// </summary>
public class SkyDocumentInfoTests
{
    private static string WriteSky(string body)
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_sky_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, "doc.sky");
        File.WriteAllText(path, body);
        return path;
    }

    private const string FullDocument = """
<?xml version="1.0" encoding="utf-8"?>
<srm_settings format_version="25.1" software_version="Skyline (64-bit) 25.1.0.237">
  <settings_summary name="Default">
    <peptide_settings>
      <enzyme name="Trypsin" cut="KR" no_cut="P" sense="C" />
      <digest_settings max_missed_cleavages="0" />
    </peptide_settings>
    <data_settings document_guid="3f1a0893-9070-4bdc-9910-d8005ed2e76f">
      <annotation name="Plate" targets="replicate" type="text" />
      <annotation name="Condition" targets="protein, peptide, replicate" type="value_list">
        <value>A</value>
        <value>B</value>
      </annotation>
      <annotation name="PeptideOnly" targets="peptide" type="text" />
      <annotation name="ResultOnly" targets="precursor_result, transition_result" type="text" />
    </data_settings>
    <measured_results time_normal_area="true">
      <replicate name="Ref_01" sample_type="standard">
        <sample_file id="f0" file_path="C:\data\Ref_01.raw" acquired_time="2025-08-22T06:17:54" />
        <annotation name="Plate">P1</annotation>
        <annotation name="Condition">A</annotation>
      </replicate>
      <replicate name="Study_07" sample_type="unknown">
        <sample_file id="f1" file_path="C:\data\Study_07.raw" acquired_time="2025-08-22T07:01:12" />
        <annotation name="Plate">P1</annotation>
      </replicate>
      <replicate name="Blank_01" sample_type="blank" />
    </measured_results>
  </settings_summary>
  <protein name="sp|P12345|TEST_HUMAN" description="ignored">
    <peptide sequence="PEPTIDER" />
  </protein>
</srm_settings>
""";

    [Fact]
    public void Read_CollectsOnlyReplicateTargetedAnnotations()
    {
        var info = SkyDocumentInfo.Read(WriteSky(FullDocument));

        // "replicate" must match as its own entry in the comma list - "precursor_result"/"transition_result"
        // contain the substring but are not replicate annotations.
        Assert.Equal(new[] { "Plate", "Condition" }, info.ReplicateAnnotationNames);
        Assert.DoesNotContain("PeptideOnly", info.ReplicateAnnotationNames);
        Assert.DoesNotContain("ResultOnly", info.ReplicateAnnotationNames);
    }

    [Fact]
    public void Read_ParsesReplicatesWithSampleTypesAndAnnotationValues()
    {
        var info = SkyDocumentInfo.Read(WriteSky(FullDocument));

        Assert.Equal(3, info.Replicates.Count);
        var first = info.Replicates[0];
        Assert.Equal("Ref_01", first.Name);
        Assert.Equal("standard", first.SampleType);
        Assert.Equal("P1", first.Annotations["Plate"]);
        Assert.Equal("A", first.Annotations["Condition"]);

        // An annotation as the LAST child must not swallow </replicate> and merge the next replicate in.
        Assert.Equal("Study_07", info.Replicates[1].Name);
        Assert.Equal("P1", info.Replicates[1].Annotations["Plate"]);
        Assert.False(info.Replicates[1].Annotations.ContainsKey("Condition"));

        // Self-closing <replicate /> is valid and carries no annotations.
        Assert.Equal("Blank_01", info.Replicates[2].Name);
        Assert.Empty(info.Replicates[2].Annotations);
    }

    [Fact]
    public void Read_CollectsSampleFilePathsInDocumentOrder()
    {
        // The raw files are the only place a "Results only" document's DIA isolation windows exist, so
        // PRISM needs their paths to have Skyline read the windows back out of one.
        var info = SkyDocumentInfo.Read(WriteSky(FullDocument));

        Assert.Equal(new[] { @"C:\data\Ref_01.raw", @"C:\data\Study_07.raw" }, info.SampleFilePaths);
    }

    [Fact]
    public void Read_SampleFilesDoNotDisturbAnnotationParsing()
    {
        // Regression: <sample_file> is handled inside the replicate loop, and must not consume the
        // annotations that follow it or the </replicate> that ends the block.
        var info = SkyDocumentInfo.Read(WriteSky("""
<?xml version="1.0" encoding="utf-8"?>
<srm_settings format_version="25.1">
  <settings_summary name="Default">
    <data_settings document_guid="g">
      <annotation name="Plate" targets="replicate" type="text" />
    </data_settings>
    <measured_results>
      <replicate name="R1" sample_type="standard">
        <sample_file id="f0" file_path="C:\d\R1.raw" acquired_time="2025-08-22T06:17:54">
          <instrument_info_list>
            <instrument_info><model>Orbitrap</model></instrument_info>
          </instrument_info_list>
        </sample_file>
        <annotation name="Plate">P9</annotation>
      </replicate>
      <replicate name="R2" sample_type="unknown">
        <sample_file id="f1" file_path="C:\d\R2.raw" />
      </replicate>
    </measured_results>
  </settings_summary>
</srm_settings>
"""));

        Assert.Equal(2, info.Replicates.Count);
        Assert.Equal("R1", info.Replicates[0].Name);
        Assert.Equal("P9", info.Replicates[0].Annotations["Plate"]);  // survived a nested <sample_file>
        Assert.Equal("R2", info.Replicates[1].Name);
        Assert.Equal(new[] { @"C:\d\R1.raw", @"C:\d\R2.raw" }, info.SampleFilePaths);
    }

    [Theory]
    [InlineData("DIA", true)]
    [InlineData("dia", true)]     // Skyline's casing is not guaranteed
    [InlineData("PRM", false)]
    [InlineData("DDA", false)]
    [InlineData("SureQuant", false)]
    [InlineData("None", false)]
    public void Read_ReportsTheAcquisitionMethod(string method, bool expectedDia)
    {
        // Only DIA has the repeating isolation cycle Skyline can import from a data file; on anything else
        // it fails with "No repeating isolation scheme found", so PRISM checks this before launching it.
        var info = SkyDocumentInfo.Read(WriteSky($"""
<?xml version="1.0" encoding="utf-8"?>
<srm_settings format_version="25.1">
  <settings_summary name="Default">
    <transition_settings>
      <transition_full_scan acquisition_method="{method}" product_mass_analyzer="orbitrap">
        <isolation_scheme name="Results only" />
      </transition_full_scan>
    </transition_settings>
  </settings_summary>
</srm_settings>
"""));

        Assert.Equal(method, info.AcquisitionMethod);
        Assert.Equal(expectedDia, info.IsDia);
    }

    [Fact]
    public void Read_AcquisitionMethodIsNullWithoutFullScanSettings()
    {
        // An SRM document has no full-scan section at all - unknown, which must not be mistaken for
        // "not DIA" (the caller only skips the import when it positively knows it is another method).
        var info = SkyDocumentInfo.Read(WriteSky(FullDocument));
        Assert.Null(info.AcquisitionMethod);
        Assert.False(info.IsDia);
    }

    [Fact]
    public void ReadIsolationSchemeXml_ReturnsTheSchemeWithItsWindows()
    {
        var path = WriteSky("""
<?xml version="1.0" encoding="utf-8"?>
<srm_settings format_version="25.1">
  <settings_summary name="Default">
    <transition_settings>
      <transition_full_scan acquisition_method="DIA">
        <isolation_scheme name="SWATH (25 m/z)" precursor_filter="24">
          <isolation_window start="400" end="424" margin="0.5" />
          <isolation_window start="424" end="448" margin="0.5" />
        </isolation_scheme>
      </transition_full_scan>
    </transition_settings>
  </settings_summary>
  <protein name="ignored" />
</srm_settings>
""");

        var xml = SkyDocumentInfo.ReadIsolationSchemeXml(path);
        Assert.NotNull(xml);
        Assert.Contains("isolation_window", xml);
        // And the lazy property on a parsed document agrees with the standalone reader.
        Assert.Equal(xml, SkyDocumentInfo.Read(path).IsolationSchemeXml);

        var scheme = SkylinePrism.Core.Qc.IsolationScheme.Parse(xml);
        Assert.NotNull(scheme);
        Assert.Equal("SWATH (25 m/z)", scheme!.Name);
        Assert.Equal(2, scheme.Windows.Count);
    }

    [Fact]
    public void ReadIsolationSchemeXml_ReturnsResultsOnlyAsAWindowlessScheme()
    {
        // The normal DIA analysis setting. It must come back (so the tool can say what the document
        // declares) but parse to a scheme with no windows, which is what triggers reading them from the
        // raw data file instead.
        var path = WriteSky("""
<?xml version="1.0" encoding="utf-8"?>
<srm_settings format_version="25.1">
  <settings_summary name="Default">
    <transition_settings>
      <transition_full_scan acquisition_method="DIA">
        <isolation_scheme name="Results only" />
      </transition_full_scan>
    </transition_settings>
  </settings_summary>
</srm_settings>
""");

        var scheme = SkylinePrism.Core.Qc.IsolationScheme.Parse(SkyDocumentInfo.ReadIsolationSchemeXml(path));
        Assert.NotNull(scheme);
        Assert.Equal("Results only", scheme!.Name);
        Assert.False(scheme.HasWindows);
    }

    [Fact]
    public void ReadIsolationSchemeXml_IsNullWhenAbsentAndStopsAtTheSettingsHeader()
    {
        // No full-scan settings at all (an SRM document).
        Assert.Null(SkyDocumentInfo.ReadIsolationSchemeXml(WriteSky(FullDocument)));

        // The reader must stop at </settings_summary> rather than scanning a multi-gigabyte target list,
        // so an element after that point is deliberately NOT found.
        var trailing = WriteSky("""
<?xml version="1.0" encoding="utf-8"?>
<srm_settings format_version="25.1">
  <settings_summary name="Default">
    <transition_settings />
  </settings_summary>
  <protein name="p">
    <isolation_scheme name="not really here">
      <isolation_window start="1" end="2" />
    </isolation_scheme>
  </protein>
</srm_settings>
""");
        Assert.Null(SkyDocumentInfo.ReadIsolationSchemeXml(trailing));

        // Missing / unreadable files are best-effort, never an exception.
        Assert.Null(SkyDocumentInfo.ReadIsolationSchemeXml(@"Z:\nope\missing.sky"));
        Assert.Null(SkyDocumentInfo.ReadIsolationSchemeXml(""));
    }

    [Fact]
    public void Read_MapsEnzymeToPrismName()
    {
        var info = SkyDocumentInfo.Read(WriteSky(FullDocument));

        Assert.NotNull(info.EnzymeXml);
        // cut="KR" sense="C" with P blocking = plain trypsin (not trypsin/p).
        Assert.Equal("trypsin", info.PrismEnzyme);
        Assert.Equal("25.1", info.FormatVersion);
        Assert.Equal("3f1a0893-9070-4bdc-9910-d8005ed2e76f", info.DocumentGuid);
    }

    [Fact]
    public void Read_HandlesDocumentWithNoAnnotationsOrResults()
    {
        var path = WriteSky("""
<?xml version="1.0" encoding="utf-8"?>
<srm_settings format_version="25.1">
  <settings_summary name="Default">
    <peptide_settings>
      <enzyme name="LysC" cut="K" no_cut="" sense="C" />
    </peptide_settings>
    <data_settings document_guid="g" />
  </settings_summary>
</srm_settings>
""");

        var info = SkyDocumentInfo.Read(path);

        Assert.Empty(info.ReplicateAnnotationNames);
        Assert.Empty(info.Replicates);
        Assert.Equal("lysc", info.PrismEnzyme);
    }

    [Fact]
    public void Read_RejectsANonSkylineFile()
    {
        var path = WriteSky("<?xml version=\"1.0\"?><not_skyline><a /></not_skyline>");

        Assert.Throws<InvalidDataException>(() => SkyDocumentInfo.Read(path));
        Assert.Null(SkyDocumentInfo.TryRead(path)); // the best-effort probe swallows it
    }

    [Fact]
    public void Read_ThrowsForAMissingFile()
        => Assert.Throws<FileNotFoundException>(
            () => SkyDocumentInfo.Read(Path.Combine(Path.GetTempPath(), "does-not-exist-" + Guid.NewGuid() + ".sky")));

    [Fact]
    public void Name_IsTheFileStem_UsedAsTheBatchLabel()
    {
        var info = SkyDocumentInfo.Read(WriteSky(FullDocument));
        Assert.Equal("doc", info.Name);
    }
}
