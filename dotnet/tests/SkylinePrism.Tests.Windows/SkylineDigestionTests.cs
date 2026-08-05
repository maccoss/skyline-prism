using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>Maps Skyline enzyme XML (cut/no_cut/sense) to PRISM enzyme names (SkylineDigestion).</summary>
public class SkylineDigestionTests
{
    [Theory]
    [InlineData("<enzyme name=\"Trypsin\" cut=\"KR\" no_cut=\"P\" sense=\"C\" />", "trypsin")]
    [InlineData("<enzyme name=\"Trypsin/P\" cut=\"KR\" no_cut=\"\" sense=\"C\" />", "trypsin/p")]
    [InlineData("<enzyme name=\"LysC\" cut=\"K\" no_cut=\"P\" sense=\"C\" />", "lysc")]
    [InlineData("<enzyme name=\"LysN\" cut=\"K\" no_cut=\"\" sense=\"N\" />", "lysn")]
    [InlineData("<enzyme name=\"ArgC\" cut=\"R\" no_cut=\"P\" sense=\"C\" />", "argc")]
    [InlineData("<enzyme name=\"AspN\" cut=\"D\" no_cut=\"\" sense=\"N\" />", "aspn")]
    [InlineData("<enzyme name=\"GluC\" cut=\"E\" no_cut=\"\" sense=\"C\" />", "gluc")]
    [InlineData("<enzyme name=\"Chymotrypsin\" cut=\"FWYL\" no_cut=\"P\" sense=\"C\" />", "chymotrypsin")]
    public void PrismEnzymeFromXml_MapsKnownEnzymes(string xml, string expected)
        => Assert.Equal(expected, SkylineDigestion.PrismEnzymeFromXml(xml));

    [Theory]
    [InlineData("<enzyme name=\"CNBr\" cut=\"M\" no_cut=\"\" sense=\"C\" />")] // no PRISM equivalent
    [InlineData("<enzyme name=\"TrypChymo\" cut_c=\"KRFYWL\" sense_c=\"C\" />")] // dual-terminal, no single cut attr
    [InlineData("not xml at all")]
    [InlineData("")]
    public void PrismEnzymeFromXml_ReturnsNull_WhenUnmappable(string xml)
        => Assert.Null(SkylineDigestion.PrismEnzymeFromXml(xml));

    [Theory]
    // What GetSettingsListItem("Enzymes", ...) actually returns for a live document: PascalCase
    // <Enzyme>, where the same enzyme in a saved .sky file is lowercase <enzyme>. A case-sensitive
    // element check rejected every live-document enzyme and reported "no PRISM equivalent".
    [InlineData("<Enzyme name=\"Trypsin\" cut=\"KR\" no_cut=\"P\" sense=\"C\" />", "trypsin")]
    [InlineData("<Enzyme name=\"Trypsin/P\" cut=\"KR\" no_cut=\"\" sense=\"C\" />", "trypsin/p")]
    [InlineData("<ENZYME name=\"LysC\" cut=\"K\" no_cut=\"P\" sense=\"C\" />", "lysc")]
    [InlineData("<Enzyme name=\"Trypsin\" Cut=\"KR\" No_Cut=\"P\" Sense=\"C\" />", "trypsin")]
    public void PrismEnzymeFromXml_IsCaseInsensitive(string xml, string expected)
        => Assert.Equal(expected, SkylineDigestion.PrismEnzymeFromXml(xml));

    [Fact]
    public void ParseEnzymeXml_ReadsPascalCaseRpcResponse()
    {
        // Verbatim response from Skyline for a Trypsin document (enzyme display name "Trypsin [KR | P]").
        var rule = SkylineDigestion.ParseEnzymeXml(
            "<Enzyme name=\"Trypsin\" cut=\"KR\" no_cut=\"P\" sense=\"C\" />");
        Assert.NotNull(rule);
        Assert.Equal("KR", rule!.Value.Cut);
        Assert.Equal("P", rule.Value.NoCut);
        Assert.Equal("C", rule.Value.Sense);
    }

    [Fact]
    public void ParseEnzymeXml_ReadsCutNoCutSense()
    {
        var rule = SkylineDigestion.ParseEnzymeXml(
            "<enzyme name=\"Trypsin\" cut=\"KR\" no_cut=\"P\" sense=\"C\" />");
        Assert.NotNull(rule);
        Assert.Equal("KR", rule!.Value.Cut);
        Assert.Equal("P", rule.Value.NoCut);
        Assert.Equal("C", rule.Value.Sense);
    }
}
