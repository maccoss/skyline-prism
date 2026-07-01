using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Placeholder so the Windows-only test project is exercised. Layer 10/11 add the RPC
/// framing and WPF view-model smoke tests here.
/// </summary>
public class SmokeTests
{
    [Fact]
    public void AppAssembly_IsLoadable()
    {
        var asm = typeof(SkylinePrism.App.App).Assembly;
        Assert.Equal("SkylinePrism", asm.GetName().Name);
    }
}
