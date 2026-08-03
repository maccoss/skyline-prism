using System;
using System.IO;
using System.Xml.Linq;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The generated PRISM-Replicates view definition. Both the live-RPC path and the headless SkylineCmd path
/// build it here, so the metadata columns are the same whether or not the document happens to be open.
/// </summary>
public class ReplicatesReportBuilderTests
{
    private static XElement View(string xml) =>
        XDocument.Parse(xml).Root!.Element("view")!;

    private static string[] ColumnNames(string xml) =>
        View(xml).Elements("column").Select(c => (string)c.Attribute("name")!).ToArray();

    [Fact]
    public void BuildXml_EmitsTheStandardReplicateColumns()
    {
        var xml = ReplicatesReportBuilder.BuildXml();

        var view = View(xml);
        Assert.Equal(ReplicatesReportBuilder.ViewName, (string)view.Attribute("name")!);
        Assert.Equal("pwiz.Skyline.Model.Databinding.Entities.Replicate", (string)view.Attribute("rowsource")!);
        // The empty column name is the row label (the replicate name) - that is how Skyline addresses it.
        Assert.Equal(new[] { "", "SampleType", "AnalyteConcentration" }, ColumnNames(xml));
    }

    [Fact]
    public void BuildXml_QuotesAnnotationColumns()
    {
        var xml = ReplicatesReportBuilder.BuildXml(new[] { "Plate", "Condition" });

        // Skyline parses column/@name as a databinding PropertyPath, whose bare-identifier syntax rejects
        // '_' - and the "annotation_" prefix contains one. Unquoted, SkylineCmd aborts the export with
        // "Error parsing annotation_Plate at location 10: Invalid character _" and writes no report.
        Assert.Contains("\"annotation_Plate\"", ColumnNames(xml));
        Assert.Contains("\"annotation_Condition\"", ColumnNames(xml));
        Assert.DoesNotContain("annotation_Plate", ColumnNames(xml)); // never the bare form
    }

    [Fact]
    public void BuildXml_AcceptsNamesThatAlreadyCarryThePrefixOrQuotes()
    {
        var xml = ReplicatesReportBuilder.BuildXml(new[] { "annotation_Plate", "Plate", "\"annotation_Plate\"" });

        // All three spellings mean the same column; it must appear exactly once.
        Assert.Single(ColumnNames(xml), c => c == "\"annotation_Plate\"");
    }

    [Fact]
    public void BuildXml_EscapesXmlSpecialCharactersInAnnotationNames()
    {
        var xml = ReplicatesReportBuilder.BuildXml(new[] { "A&B<>" });

        // Parses (so the escaping is well-formed) and round-trips to the original name.
        Assert.Contains("\"annotation_A&B<>\"", ColumnNames(xml));
    }

    [Fact]
    public void BuildXml_IgnoresBlankNames()
    {
        var xml = ReplicatesReportBuilder.BuildXml(new[] { "", "  ", "Plate" });

        Assert.Equal(4, ColumnNames(xml).Length); // 3 standard + Plate
    }

    [Fact]
    public void WriteSkyr_WritesParseableXmlToDisk()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_skyr_" + Guid.NewGuid().ToString("N"));
        var path = Path.Combine(dir, "nested", "PRISM-Replicates.skyr");

        var written = ReplicatesReportBuilder.WriteSkyr(path, new[] { "Plate" });

        Assert.Equal(path, written);
        Assert.True(File.Exists(path)); // parent directories created
        Assert.Contains("\"annotation_Plate\"", ColumnNames(File.ReadAllText(path)));
    }
}
