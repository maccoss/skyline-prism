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
