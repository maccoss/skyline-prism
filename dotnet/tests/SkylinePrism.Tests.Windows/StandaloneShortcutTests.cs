using System;
using System.IO;
using SkylinePrism.App;
using Xunit;
using static SkylinePrism.App.StandaloneShortcut;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Only an installed Skyline tool may re-point the Start Menu shortcut, and never to an older
/// version. Any build used to, so starting a development build once took the shortcut over: every
/// later Start Menu launch ran a build without the instrument-file reader, and ion accounting was
/// skipped on a machine whose installed tool had one.
/// </summary>
public class StandaloneShortcutTests
{
    private const string Daily = @"C:\Users\u\AppData\Local\Apps\2.0\X\Y\skyl..tion_new";
    private const string Release = @"C:\Users\u\AppData\Local\Apps\2.0\X\Y\skyl..tion_release";

    /// <summary>The Skyline installations these tests pretend exist.</summary>
    private static bool IsSkyline(string folder) =>
        folder.Equals(Daily, StringComparison.OrdinalIgnoreCase)
        || folder.Equals(Release, StringComparison.OrdinalIgnoreCase)
        || folder.Equals(@"C:\Program Files\Skyline", StringComparison.OrdinalIgnoreCase);

    [Theory]
    // A Skyline installation's own Tools folder, ClickOnce or not.
    [InlineData(Daily + @"\Tools\~SK6C53\", true)]
    [InlineData(Daily + @"\Tools\SkylinePrism", true)]
    [InlineData(@"C:\Program Files\Skyline\tools\SkylinePrism\", true)]
    // A folder merely NAMED Tools - where portable copies are often kept - is not one.
    [InlineData(@"C:\Tools\prism\", false)]
    [InlineData(@"D:\Users\u\Tools\SkylinePrism\", false)]
    // Build outputs and loose copies.
    [InlineData(@"C:\src\skyline-prism\dotnet\src\SkylinePrism.App\bin\Debug\net10.0-windows10.0.19041.0\", false)]
    [InlineData(@"C:\src\skyline-prism\dotnet\src\SkylinePrism.App\bin\Release\net10.0-windows10.0.19041.0", false)]
    [InlineData(@"C:\Users\u\Downloads\SkylinePrism\", false)]
    [InlineData(@"C:\Tools\", false)]
    [InlineData(@"C:\", false)]
    public void IsInstalledTool_OnlyInsideASkylineInstallationsToolsFolder(string exeDirectory, bool expected)
        => Assert.Equal(expected, IsInstalledTool(exeDirectory, IsSkyline));

    private static RefreshDecision DecideFor(string? target, string exeDirectory, Version? theirs = null, Version? ours = null) =>
        Decide(target, exeDirectory, IsSkyline,
            fileExists: path => path == target,
            versionOf: dir => target is not null && dir == Path.GetDirectoryName(target) ? theirs
                : dir.TrimEnd('\\') == exeDirectory.TrimEnd('\\') ? ours : null);

    /// <summary>The fix itself: a development build leaves the shortcut where it is.</summary>
    [Fact]
    public void ADevelopmentBuild_DoesNotTakeTheShortcutOver()
    {
        var installed = Daily + @"\Tools\~SK6C53\SkylinePrism.exe";
        var dev = @"C:\src\skyline-prism\dotnet\src\SkylinePrism.App\bin\Debug\net10.0-windows10.0.19041.0\";

        Assert.Equal(RefreshDecision.NotInstalledTool, DecideFor(installed, dev));
    }

    /// <summary>What the refresh is for: Skyline updated and reinstalled the tool in a new folder.</summary>
    [Fact]
    public void AnInstalledTool_FollowsSkylinesReinstall()
    {
        var old = Release + @"\Tools\~SK1A3A\SkylinePrism.exe";

        Assert.Equal(RefreshDecision.Repoint,
            DecideFor(old, Daily + @"\Tools\~SK6C53\", theirs: new Version(26, 32, 0), ours: new Version(26, 32, 0)));
        // A target that is gone (its old application folder cleaned up) is replaced too.
        Assert.Equal(RefreshDecision.Repoint,
            Decide(old, Daily + @"\Tools\~SK6C53\", IsSkyline, _ => false, _ => new Version(1, 0)));
    }

    /// <summary>
    /// Never backwards: PRISM in a release Skyline does not take the shortcut from Skyline-daily's
    /// newer copy.
    /// </summary>
    [Fact]
    public void AnOlderInstalledTool_LeavesANewerTargetAlone()
    {
        var newer = Daily + @"\Tools\~SK6C53\SkylinePrism.exe";

        Assert.Equal(RefreshDecision.TargetIsNewer,
            DecideFor(newer, Release + @"\Tools\~SK1A3A\", theirs: new Version(26, 32, 0), ours: new Version(26, 24, 3)));
        Assert.Equal(RefreshDecision.Repoint,
            DecideFor(newer, Release + @"\Tools\~SK1A3A\", theirs: new Version(26, 24, 3), ours: new Version(26, 32, 0)));
    }

    [Fact]
    public void TheShortcutAlreadyHere_IsLeftAlone()
    {
        var here = Daily + @"\Tools\~SK6C53";
        Assert.Equal(RefreshDecision.AlreadyCurrent, DecideFor(Path.Combine(here, "SkylinePrism.exe"), here));
    }

    /// <summary>A version that cannot be read does not block the refresh: only a known-newer target does.</summary>
    [Fact]
    public void AnUnreadableVersion_DoesNotBlockTheRefresh()
    {
        var other = Release + @"\Tools\~SK1A3A\SkylinePrism.exe";
        Assert.Equal(RefreshDecision.Repoint, DecideFor(other, Daily + @"\Tools\~SK6C53\", theirs: null, ours: new Version(26, 32, 0)));
        Assert.Equal(RefreshDecision.Repoint, DecideFor(null, Daily + @"\Tools\~SK6C53\"));
    }

    [Fact]
    public void ToolVersion_ReadsTheManifestBesideTheExe()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism-shortcut-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(dir, "tool-inf"));
        try
        {
            File.WriteAllText(Path.Combine(dir, "tool-inf", "info.properties"),
                "Name = Skyline-PRISM\nVersion = 26.32.0\nIdentifier = URN:LSID:maccosslab.org:SkylinePrism\n");
            Assert.Equal(new Version(26, 32, 0), ToolVersion(dir));
            Assert.Null(ToolVersion(Path.Combine(dir, "missing")));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
