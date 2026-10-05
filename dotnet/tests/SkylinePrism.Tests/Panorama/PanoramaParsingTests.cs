using System;
using System.Linq;
using SkylinePrism.Core.Panorama;
using Xunit;

namespace SkylinePrism.Tests.Panorama;

/// <summary>What PRISM reads back from LabKey, in the shapes panoramaweb.org actually sends.</summary>
public class PanoramaParsingTests
{
    [Fact]
    public void Listing_TellsFoldersFromFiles_AndKeepsSizeAndPermission()
    {
        var entries = PanoramaClient.ParseListing(
            "{\"files\":[{\"text\":\"@files\",\"collection\":true,\"canUpload\":true},"
            + "{\"text\":\"run1.raw\",\"leaf\":true,\"contentlength\":1234,\"lastmodified\":\"2026-10-01T05:07:39-00:00\"}]}",
            "/P");

        Assert.Equal(("@files", "/P/@files", true, true), (entries[0].Name, entries[0].Folder, entries[0].IsFolder, entries[0].CanUpload));
        Assert.Equal(("run1.raw", false, 1234L), (entries[1].Name, entries[1].IsFolder, entries[1].Length));
        Assert.Equal(new DateTimeOffset(2026, 10, 1, 5, 7, 39, TimeSpan.Zero), entries[1].LastModifiedUtc);
    }

    [Fact]
    public void ALoginPage_InsteadOfAListing_IsASignInProblem()
    {
        var ex = Assert.Throws<PanoramaException>(() => PanoramaClient.ParseListing("<html>Sign In</html>", "/P"));
        Assert.True(ex.IsSignInProblem);
    }

    [Fact]
    public void Md5s_TakeTheNameVerbatim_AfterTheAsterisk()
    {
        // Names arrive raw: spaces, @, # and parentheses unescaped (measured by PanoramaBridge).
        var md5s = PanoramaClient.ParseMd5s(
            "558fb082796f8ff111b6e2f2f3c3356c *Panorama (Webinar 2014-08-19).pdf\r\n"
            + "89C1ACD7FCA191AF78E4B1824C3FC3ED *a @#b.csv\n\n");

        Assert.Equal("558fb082796f8ff111b6e2f2f3c3356c", md5s["Panorama (Webinar 2014-08-19).pdf"]);
        Assert.Equal("89c1acd7fca191af78e4b1824c3fc3ed", md5s["a @#b.csv"]); // case folded once, here
    }

    [Fact]
    public void Attachments_AWarning_MeansAFileWasNotAttached()
    {
        var ok = PanoramaClient.ParseAttachments(
            "{\"success\":true,\"attachments\":[{\"name\":\"a.png\",\"downloadUrl\":\"https://x/P/wiki-download.view?name=a.png\"}]}");
        Assert.Equal("a.png", Assert.Single(ok).Name);

        var ex = Assert.Throws<PanoramaException>(() => PanoramaClient.ParseAttachments(
            "{\"success\":true,\"warnings\":{\"files\":\"A file with name a.png already exists.\"},\"attachments\":[]}"));
        Assert.Contains("already exists", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WikiEditor_PropsAreRead_AndTheBodyUnescaped()
    {
        var page = PanoramaClient.ParseWikiEditor(
            "<script>LABKEY._wiki.setProps({\n    entityId: 'e-1',\n    rowId: 7,\n    pageVersionId: 3,\n"
            + "    name: 'PRISM-QC-a',\n    title: 'It\\'s here',\n    body: '\\x3Cp style=\\\"x\\\">caf\\u00e9\\n\\x3C/p>',\n"
            + "    parent: -1,\n    showAttachments: false,\n    shouldIndex: true\n});</script>");

        Assert.Equal(("e-1", 7, 3), (page.EntityId, page.RowId!.Value, page.PageVersionId!.Value));
        Assert.Equal("It's here", page.Title);
        Assert.Equal("<p style=\"x\">café\n</p>", page.Body);
        Assert.False(page.ShowAttachments);
    }

    [Fact]
    public void WikiEditor_ABodyContainingTheBlocksOwnEnding_IsReadWhole()
    {
        // The props end at the first "});" outside a string; one inside the body is text.
        var page = PanoramaClient.ParseWikiEditor(
            "<script>LABKEY._wiki.setProps({\n    entityId: 'e-1',\n    name: 'p',\n    title: 't',\n"
            + "    body: 'f(x = {a});\\x3Cdiv id=\\\"prism-wiki\\\">',\n    showAttachments: false\n});</script>");

        Assert.Equal("f(x = {a});<div id=\"prism-wiki\">", page.Body);
        Assert.False(page.ShowAttachments);
    }

    [Fact]
    public void AnEditorWithoutAPageId_IsRefused_RatherThanSavedOverBlind() =>
        Assert.Throws<PanoramaException>(() => PanoramaClient.ParseWikiEditor("<script>LABKEY._wiki.setProps({\n    name: 'x'\n});</script>"));
}
