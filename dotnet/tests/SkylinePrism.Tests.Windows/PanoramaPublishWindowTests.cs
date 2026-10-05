using System;
using System.IO;
using System.Threading;
using SkylinePrism.App;
using SkylinePrism.Core.Panorama;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The Publish to Panorama window opens on what the output directory has and remembers. Built for
/// real on an STA thread - a window that compiles is not one that constructs, as a handler firing
/// inside InitializeComponent once proved by crashing the tool on startup.
/// </summary>
public sealed class PanoramaPublishWindowTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "prism_pubwin_" + Guid.NewGuid().ToString("N"), "run-a");

    public PanoramaPublishWindowTests()
    {
        Directory.CreateDirectory(_dir);
        File.WriteAllText(Path.Combine(_dir, "qc_report.html"), "<html><body><h1>QC</h1></body></html>");
    }

    public void Dispose()
    {
        try { Directory.Delete(Path.GetDirectoryName(_dir)!, recursive: true); }
        catch (IOException) { /* best-effort */ }
    }

    private static T OnSta<T>(Func<T> body)
    {
        T result = default!;
        Exception? failure = null;
        var thread = new Thread(() =>
        {
            try { result = body(); }
            catch (Exception ex) { failure = ex; }
        });
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        thread.Join();
        if (failure is not null)
            throw new Xunit.Sdk.XunitException("The window threw while being built: " + failure);
        return result;
    }

    [Fact]
    public void WithOnlyAQcReport_TheQuantStepIsOff_AndPublishWaitsForASignIn()
    {
        var state = OnSta(() =>
        {
            var window = new PanoramaPublishWindow(_dir);
            var s = (Qc: window.QcCheck.IsChecked, QuantEnabled: window.QuantCheck.IsEnabled, Quant: window.QuantCheck.IsChecked,
                Page: window.QcPageBox.Text, Publish: window.PublishButton.IsEnabled, Plan: window.LogBox.Text);
            window.Close();
            return s;
        });

        Assert.True(state.Qc);
        Assert.False(state.QuantEnabled);
        Assert.False(state.Quant);
        Assert.Equal("PRISM-QC-run-a", state.Page);
        Assert.False(state.Publish); // not signed in yet - Loaded has not run
        Assert.Contains("Choose a folder for the QC report's wiki page", state.Plan, StringComparison.Ordinal);
    }

    [Fact]
    public void ItOpensOnTheTargetsThisOutputDirectoryRemembers()
    {
        new PanoramaTargets("https://panoramaweb.org", new WikiTarget("/MacCoss/maccoss/X", "My-QC-page"), null,
            "/MacCoss/maccoss/X/@files/RawFiles", null).Save(_dir);

        var (folder, page, raw, destination) = OnSta(() =>
        {
            var window = new PanoramaPublishWindow(_dir);
            var s = (window.QcFolderBox.Text, window.QcPageBox.Text, window.RawFolderBox.Text, window.DestinationText.Text);
            window.Close();
            return s;
        });

        Assert.Equal("/MacCoss/maccoss/X", folder);
        Assert.Equal("My-QC-page", page);
        Assert.Equal("/MacCoss/maccoss/X/@files/RawFiles", raw);
        Assert.Equal("Uploads to /MacCoss/maccoss/X/@files/run-a", destination);
    }

    [Fact]
    public void TheLinksFolder_FollowsTheQcFolder_UntilSomeoneChoosesOne()
    {
        var (on, page, followed, afterChoosing) = OnSta(() =>
        {
            var window = new PanoramaPublishWindow(_dir);
            var o = window.LinksCheck.IsChecked == true;
            var p = window.LinksPageBox.Text;
            window.QcFolderBox.Text = "/MacCoss/maccoss/Y";
            var f = window.LinksFolderBox.Text;
            window.LinksFolderBox.Text = "/MacCoss/maccoss/Z"; // typed by someone
            window.QcFolderBox.Text = "/MacCoss/maccoss/W";
            var a = window.LinksFolderBox.Text;
            window.Close();
            return (o, p, f, a);
        });

        Assert.True(on);
        Assert.Equal("PRISM-run-a", page);
        Assert.Equal("/MacCoss/maccoss/Y", followed);
        Assert.Equal("/MacCoss/maccoss/Z", afterChoosing);
    }

    [Fact]
    public void WithOnlyAQuantReport_TheLinksFolderFollowsTheQuantFolder_AsTheCliDefaults()
    {
        var quantOnly = Path.Combine(Path.GetDirectoryName(_dir)!, "run-q");
        Directory.CreateDirectory(Path.Combine(quantOnly, "quant"));
        File.WriteAllText(Path.Combine(quantOnly, "quant", "quant_report.html"), "<html><body><h1>Quant</h1></body></html>");

        var links = OnSta(() =>
        {
            var window = new PanoramaPublishWindow(quantOnly);
            window.QuantFolderBox.Text = "/MacCoss/maccoss/Q";
            var s = window.LinksFolderBox.Text;
            window.Close();
            return s;
        });

        Assert.Equal("/MacCoss/maccoss/Q", links);
    }

    [Fact]
    public void ADirectoryWithNoReport_OpensWithLinksOff_SoAnUploadIsNotHeldUp()
    {
        var bare = Path.Combine(Path.GetDirectoryName(_dir)!, "run-bare");
        Directory.CreateDirectory(bare);

        var (links, plan) = OnSta(() =>
        {
            var window = new PanoramaPublishWindow(bare);
            var s = (window.LinksCheck.IsChecked == true, window.LogBox.Text);
            window.Close();
            return s;
        });

        Assert.False(links);
        Assert.DoesNotContain("Choose a folder for the links page", plan, StringComparison.Ordinal);
    }

    [Fact]
    public void ARememberedLinksFolder_IsKept_AndDoesNotFollowTheQcFolder()
    {
        new PanoramaTargets("https://panoramaweb.org", new WikiTarget("/MacCoss/maccoss/X", "My-QC-page"), null, null, null,
            LinksWiki: new WikiTarget("/MacCoss/maccoss/Lab", "PRISM-run-a")).Save(_dir);

        var links = OnSta(() =>
        {
            var window = new PanoramaPublishWindow(_dir);
            window.QcFolderBox.Text = "/MacCoss/maccoss/Elsewhere";
            var s = window.LinksFolderBox.Text;
            window.Close();
            return s;
        });

        Assert.Equal("/MacCoss/maccoss/Lab", links);
    }

    [Theory]
    [InlineData("https://panoramaweb.org", "https://PanoramaWeb.org/", true)]
    [InlineData("https://panoramaweb.org", "http://panoramaweb.org", false)]
    [InlineData("https://panoramaweb.org", "https://panoramaweb.org:8443", false)]
    [InlineData("https://panoramaweb.org", "https://panoramaweb.org:443", true)]
    public void AServerIsTheSameOnlyIfSchemeHostAndPortAre(string a, string b, bool same) =>
        // The host alone took http for https, or another port, for the server already signed in to.
        Assert.Equal(same, PanoramaPublishWindow.SameServer(new Uri(a), new Uri(b)));

    [Fact]
    public void TheSignInWindow_ConstructsWithItsPreselectedChoice()
    {
        // ApiKeyChoice is IsChecked in the XAML, so its handler runs inside InitializeComponent.
        OnSta(() =>
        {
            var type = typeof(PanoramaSignInWindow);
            var ctor = type.GetConstructor(System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance,
                new[] { typeof(PanoramaSignIn), typeof(string) })!;
            var window = (System.Windows.Window)ctor.Invoke(new object[]
            {
                new PanoramaSignIn(null, new Uri("https://panoramaweb.org"), _ => null), "why",
            });
            window.Close();
            return 0;
        });
    }
}
