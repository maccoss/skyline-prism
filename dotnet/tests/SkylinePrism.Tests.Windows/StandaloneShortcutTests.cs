using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Only an installed Skyline tool may re-point the Start Menu shortcut. Any build used to, so
/// starting a development build once took the shortcut over: every later Start Menu launch ran a
/// build without the instrument-file reader, and ion accounting was skipped on a machine whose
/// installed tool had one.
/// </summary>
public class StandaloneShortcutTests
{
    [Theory]
    // ClickOnce Skyline: each tool in its own "~SK" folder under the version-stamped Tools folder.
    [InlineData(@"C:\Users\u\AppData\Local\Apps\2.0\0V4L7BX4.3AM\L086A0DE.ATG\skyl..tion_9286511f3362df93_001a.0001_6c454ec235f7ebac\Tools\~SK6C53\", true)]
    [InlineData(@"C:\Users\u\AppData\Local\Apps\2.0\X\Y\skyl..tion_abc\Tools\SkylinePrism", true)]
    [InlineData(@"C:\Program Files\Skyline\tools\SkylinePrism\", true)]
    // Build outputs and loose copies: never.
    [InlineData(@"C:\src\skyline-prism\dotnet\src\SkylinePrism.App\bin\Debug\net10.0-windows10.0.19041.0\", false)]
    [InlineData(@"C:\src\skyline-prism\dotnet\src\SkylinePrism.App\bin\Release\net10.0-windows10.0.19041.0", false)]
    [InlineData(@"C:\Users\u\Downloads\SkylinePrism\", false)]
    [InlineData(@"C:\Tools\", false)]
    [InlineData(@"C:\", false)]
    public void IsInstalledTool_OnlyForAFolderDirectlyInsideTools(string exeDirectory, bool expected)
        => Assert.Equal(expected, StandaloneShortcut.IsInstalledTool(exeDirectory));
}
