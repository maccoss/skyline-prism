using System;
using System.Linq;
using SkylinePrism.Core.Panorama;
using Xunit;

namespace SkylinePrism.Tests.Panorama;

/// <summary>
/// A PRISM report rewritten for Panorama. LabKey refuses a wiki body - and an uploaded <c>.html</c>
/// file - carrying a <c>&lt;style&gt;</c> block from any user who is not a trusted browser
/// developer, so the report's CSS is inlined; and its plots become page attachments.
/// </summary>
public class WikiReportTests
{
    private const string Png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

    private static readonly string Report =
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>PRISM QC Report</title>"
        + "<link rel=\"icon\" href=\"data:image/png;base64," + Png + "\">"
        + "<style>body { font-family: Arial; color: #222; } th, td { text-align: right; padding: 6px; }"
        + " td:first-child, th:first-child { text-align: left; } .box { background: #f6f8fb; }"
        + " .hero { background: url(back.png); color: #fff; } summary { cursor: pointer; color: #1a3c6e; }</style></head>"
        + "<body><div class=\"box\"><table><tr><td>name</td><td>value</td></tr></table></div>"
        + "<div class=\"hero\" style=\"padding: 4px\">headline</div>"
        + "<details><summary>More</summary><p>text</p></details>"
        + "<img alt=\"one\" src=\"data:image/png;base64," + Png + "\"><img alt=\"two\" src=\"data:image/png;base64," + Png + "\"></body></html>";

    [Fact]
    public void Styles_AreInlined_InSpecificityOrder()
    {
        var body = WikiReport.FromReport(Report, "qc").BodyTemplate;

        // td:first-child (one class-level pseudo + one type) outranks "th, td": the first cell is
        // left-aligned, the second keeps the right alignment - the cascade, not document order alone.
        Assert.Contains("<td style=\"text-align: left; padding: 6px\">name</td>", body, StringComparison.Ordinal);
        Assert.Contains("<td style=\"text-align: right; padding: 6px\">value</td>", body, StringComparison.Ordinal);
        Assert.Contains("background: #f6f8fb", body, StringComparison.Ordinal);
    }

    [Fact]
    public void TheElementsOwnStyle_WinsOverTheSheet()
    {
        var body = WikiReport.FromReport(Report, "qc").BodyTemplate;
        Assert.Contains("color: #fff; padding: 4px", body, StringComparison.Ordinal);
    }

    [Fact]
    public void WhatLabKeyRefuses_IsGone()
    {
        var body = WikiReport.FromReport(Report, "qc").BodyTemplate;

        Assert.Empty(WikiReport.Refusals(body));
        Assert.DoesNotContain("<style", body, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("<link", body, StringComparison.OrdinalIgnoreCase);
        // A url() would fail the WHOLE save, so that one declaration is dropped and the rest kept.
        Assert.DoesNotContain("url(", body, StringComparison.Ordinal);
        Assert.Contains("headline", body, StringComparison.Ordinal);
    }

    [Fact]
    public void Classes_AreDropped_SoPanoramasOwnStylesheetCannotRestyleThePage()
    {
        var body = WikiReport.FromReport(Report, "qc").BodyTemplate;
        Assert.DoesNotContain("class=", body, StringComparison.Ordinal);
    }

    [Fact]
    public void BodyStyles_MoveToAWrapper()
    {
        var body = WikiReport.FromReport(Report, "qc").BodyTemplate;
        Assert.StartsWith("<div style=\"font-family: Arial; color: #222\">", body, StringComparison.Ordinal);
    }

    [Fact]
    public void Images_BecomeNumberedAttachments_NamedByContent_AndRenderPointsAtThem()
    {
        var document = WikiReport.FromReport(Report, "prism-qc-run1");

        // Numbered in order, and named by a hash of the bytes, so an unchanged plot keeps its name.
        var hash = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(Convert.FromBase64String(Png)))[..8];
        Assert.Equal(new[] { $"prism-qc-run1-01-{hash}.png", $"prism-qc-run1-02-{hash}.png" }, document.Images.Select(i => i.Name));
        Assert.Equal(Convert.FromBase64String(Png), document.Images[0].Data);
        // The favicon is a <link>, not an image of the report, and is not attached.
        Assert.DoesNotContain("data:image", document.BodyTemplate, StringComparison.Ordinal);

        var rendered = document.Render(name => "/P/wiki-download.view?name=" + name);
        Assert.Contains($"src=\"/P/wiki-download.view?name=prism-qc-run1-01-{hash}.png\"", rendered, StringComparison.Ordinal);
        Assert.DoesNotContain(WikiReport.ImagePlaceholder, rendered, StringComparison.Ordinal);
    }

    [Fact]
    public void AChangedPlot_GetsANewName_AndAnUnchangedOneKeepsItsName()
    {
        var other = Convert.ToBase64String(Convert.FromBase64String(Png).Append((byte)0).ToArray());
        var before = WikiReport.FromReport(Report, "p").Images.Select(i => i.Name).ToList();
        var after = WikiReport.FromReport(ReplaceLastImage(Report, other), "p").Images.Select(i => i.Name).ToList();

        Assert.Equal(before[0], after[0]);
        Assert.NotEqual(before[1], after[1]);
    }

    private static string ReplaceLastImage(string html, string base64)
    {
        var marker = "data:image/png;base64," + Png;
        var last = html.LastIndexOf(marker, StringComparison.Ordinal);
        return html[..last] + "data:image/png;base64," + base64 + html[(last + marker.Length)..];
    }

    [Theory]
    [InlineData("<div onclick=\"x()\">a</div>")]
    [InlineData("<iframe src=\"/x\"></iframe>")]
    [InlineData("<p style=\"background: url(/x.png)\">a</p>")]
    [InlineData("<a href=\"javascript:x()\">a</a>")]
    public void AnHtmlFile_IsRefused_ForAnyRuleOfTheWikis_AndItsPanoramaFormIsNot(string html)
    {
        // Panorama applies all of the wiki's rules to an uploaded .html, not only the style/script ones.
        Assert.True(WikiReport.RefusedAsFile("<html><body>" + html + "</body></html>"));
        Assert.False(WikiReport.RefusedAsFile(WikiReport.ToPanoramaFile("<html><body>" + html + "</body></html>")));
    }

    [Fact]
    public void InteractionOnlyProperties_AreDropped()
    {
        var body = WikiReport.FromReport(Report, "qc").BodyTemplate;
        Assert.DoesNotContain("cursor", body, StringComparison.Ordinal);
        Assert.Contains("<summary style=\"color: #1a3c6e\">", body, StringComparison.Ordinal);
    }

    [Fact]
    public void Refusals_NameEachRuleLabKeyApplies()
    {
        var refusals = WikiReport.Refusals(
            "<style>p{}</style><p onclick=\"x()\">a</p><a href=\"javascript:x()\">b</a><div style=\"background: url(a.png)\">c</div>");

        Assert.Contains(refusals, r => r.StartsWith("Illegal element <style>", StringComparison.Ordinal));
        Assert.Contains(refusals, r => r.StartsWith("Illegal attribute 'onclick'", StringComparison.Ordinal));
        Assert.Contains(refusals, r => r.StartsWith("Script is not allowed in 'href'", StringComparison.Ordinal));
        Assert.Contains(refusals, r => r.StartsWith("Style attribute cannot contain", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("<!doctype html><p>a</p>", true)]   // measured: the doctype alone is refused
    [InlineData("<html><head><title>t</title><meta charset=\"utf-8\"></head><body><p>a</p></body></html>", false)]
    [InlineData("<p>a</p><style>p{}</style>", true)]
    [InlineData("<p>a</p>", false)]
    public void RefusedAsFile_FollowsWhatPanoramaMeasured(string html, bool refused) =>
        Assert.Equal(refused, WikiReport.RefusedAsFile(html));

    [Fact]
    public void TheUploadableFile_IsTheSamePage_InAFormPanoramaAccepts()
    {
        var file = WikiReport.ToPanoramaFile(Report);

        Assert.False(WikiReport.RefusedAsFile(file));
        Assert.StartsWith("<html lang=\"en\"><head><meta charset=\"utf-8\"><title>PRISM QC Report</title>", file, StringComparison.Ordinal);
        // Images stay embedded, so the downloaded file stands on its own.
        Assert.Equal(2, System.Text.RegularExpressions.Regex.Matches(file, "src=\"data:image/png;base64,").Count);
        // Without a doctype a table does not inherit the page font (quirks mode); it is told to.
        Assert.Contains("<table style=\"font: inherit; color: inherit", file, StringComparison.Ordinal);
    }

    [Fact]
    public void Fingerprint_IsStableAndShort()
    {
        Assert.Equal(WikiReport.Fingerprint("abc"), WikiReport.Fingerprint("abc"));
        Assert.NotEqual(WikiReport.Fingerprint("abc"), WikiReport.Fingerprint("abd"));
        Assert.Equal(16, WikiReport.Fingerprint("abc").Length);
    }
}
