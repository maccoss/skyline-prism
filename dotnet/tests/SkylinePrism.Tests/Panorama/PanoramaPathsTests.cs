using SkylinePrism.Core.Panorama;
using Xunit;

namespace SkylinePrism.Tests.Panorama;

public class PanoramaPathsTests
{
    [Theory]
    [InlineData("https://panoramaweb.org/MacCoss/maccoss/My-Project/project-begin.view", "/MacCoss/maccoss/My-Project")]
    [InlineData("https://panoramaweb.org/MacCoss/maccoss/X/wiki-page.view?name=Y", "/MacCoss/maccoss/X")]
    [InlineData("https://panoramaweb.org/_webdav/MacCoss/maccoss/X/%40files/Raw%20Files/", "/MacCoss/maccoss/X/@files/Raw Files")]
    [InlineData("/_webdav/MacCoss/maccoss/X/@files/RawFiles/", "/MacCoss/maccoss/X/@files/RawFiles")]
    [InlineData("MacCoss/maccoss/X/", "/MacCoss/maccoss/X")]
    [InlineData("  /MacCoss/maccoss/X  ", "/MacCoss/maccoss/X")]
    public void FromAnything_ReducesWhatAPersonPastesToAFolder(string pasted, string folder) =>
        Assert.Equal(folder, PanoramaPaths.FromAnything(pasted));

    [Fact]
    public void Container_IsEverythingBeforeTheFileRoot()
    {
        Assert.Equal("/MacCoss/maccoss/X", PanoramaPaths.Container("/MacCoss/maccoss/X/@files/RawFiles"));
        Assert.Equal("/MacCoss/maccoss/X", PanoramaPaths.Container("/MacCoss/maccoss/X"));
    }

    [Fact]
    public void FileArea_MeansInsideAnAtRoot()
    {
        Assert.True(PanoramaPaths.IsFileArea("/P/@files/RawFiles"));
        Assert.True(PanoramaPaths.IsFileArea("/P/@files"));
        Assert.False(PanoramaPaths.IsFileArea("/P/Sub"));
    }

    [Fact]
    public void Encode_KeepsTheAtOfAFileRoot_AndEscapesTheRest() =>
        Assert.Equal("/_webdav/P/@files/Raw%20Files/", PanoramaPaths.Encode(PanoramaPaths.ToWebDav("/P/@files/Raw Files")));

    [Fact]
    public void Parent_Combine_Name()
    {
        Assert.Equal("/P/@files", PanoramaPaths.Parent("/P/@files/RawFiles"));
        Assert.Equal("/P/@files/out", PanoramaPaths.Combine("/P/@files", "out"));
        Assert.Equal("RawFiles", PanoramaPaths.Name("/P/@files/RawFiles/"));
    }

    [Fact]
    public void BrowserAddresses()
    {
        var server = new System.Uri("https://panoramaweb.org");
        Assert.Equal("https://panoramaweb.org/_webdav/P/@files/out/", PanoramaPaths.BrowserUrl("/P/@files/out", server));
        Assert.Equal("https://panoramaweb.org/P/Sub/project-begin.view", PanoramaPaths.BrowserUrl("/P/Sub", server));
        Assert.Equal("https://panoramaweb.org/P/wiki-page.view?name=PRISM-QC-a", PanoramaPaths.WikiPageUrl("/P", "PRISM-QC-a", server));
    }
}
