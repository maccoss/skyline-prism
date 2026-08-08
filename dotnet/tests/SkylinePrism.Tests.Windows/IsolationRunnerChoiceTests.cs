using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Which Skyline the isolation-window probe drives.
///
/// <para>The app runner is the default everywhere else because it is the only one that can export
/// parquet - but it pays for that by driving a whole UI-less Skyline through a pair of named pipes.
/// Reading isolation windows needs no report at all, and going through the application is what made
/// it hang: on a 4.9 GB Thermo .raw, SkylineCmd returned 167 windows in <b>8.7 s</b> while the app
/// runner produced its first line and then nothing, until it was killed five minutes later.</para>
///
/// <para>Both tests need a real Skyline installed, so they assert only when one is found - on a
/// machine without Skyline there is nothing meaningful to check and nothing to fail.</para>
/// </summary>
public class IsolationRunnerChoiceTests
{
    [Fact]
    public void SettingsProbe_PrefersSkylineCmd()
    {
        if (SkylineCmdLocator.Find() is null)
            return; // no SkylineCmd on this machine; the fallback path is covered below

        var exporter = HeadlessSkylineExporter.Create(preferCmd: true);

        Assert.IsType<SkylineCmdRunner>(exporter.Runner);
        // The trade-off being accepted: no parquet. That is fine here - this probe reads settings.
        Assert.False(exporter.Runner.SupportsParquet);
    }

    /// <summary>
    /// Report export must NOT be moved to SkylineCmd by this change: it needs parquet, and
    /// SkylineCmd cannot write it (its .exe.config lacks the Parquet.Net binding).
    /// </summary>
    [Fact]
    public void ReportExport_StillPrefersTheParquetCapableRunner()
    {
        if (SkylineAppRunner.FindInstallations().Count == 0)
            return; // no installed Skyline to drive headlessly

        var exporter = HeadlessSkylineExporter.Create();

        Assert.IsType<SkylineAppRunner>(exporter.Runner);
        Assert.True(exporter.Runner.SupportsParquet);
    }
}
