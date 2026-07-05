using SkylinePrism.Skyline;
using SkylineTool;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Windows-only smoke tests for the Skyline RPC layer + WPF app assembly. These validate
/// wiring/construction only; a live JSON-RPC round-trip requires a running Skyline instance.
/// </summary>
public class SmokeTests
{
    [Fact]
    public void AppAssembly_IsLoadable()
    {
        var asm = typeof(SkylinePrism.App.App).Assembly;
        Assert.Equal("SkylinePrism", asm.GetName().Name);
    }

    [Fact]
    public void SkylineSession_FromArguments_TransformsLegacyPipeToJson()
    {
        var session = SkylineSession.FromArguments(new[] { "MyLegacyToolServicePipe" });
        Assert.StartsWith(JsonToolConstants.JSON_PIPE_PREFIX, session.PipeName);
    }

    [Fact]
    public void SkylineSession_FromArguments_KeepsExistingJsonPipeName()
    {
        var name = JsonToolConstants.JSON_PIPE_PREFIX + "already-json";
        var session = SkylineSession.FromArguments(new[] { name });
        Assert.Equal(name, session.PipeName);
    }

    [Fact]
    public void SkylineReportDriver_Constructs()
    {
        var session = SkylineSession.FromArguments(new[] { "pipe" });
        var driver = new SkylineReportDriver(session, _ => { });
        Assert.NotNull(driver);
    }
}
