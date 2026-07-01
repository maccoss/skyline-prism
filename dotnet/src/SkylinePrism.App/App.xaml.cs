using System.Windows;

namespace SkylinePrism.App;

/// <summary>
/// WPF application entry. Skyline passes the JSON-RPC connection string as the first
/// command-line argument ($(SkylineConnection)); Layer 11 captures it here and hands it
/// to SkylineSession.FromArguments.
/// </summary>
public partial class App : Application
{
    /// <summary>Command-line args captured at startup (the Skyline connection string).</summary>
    public static string[] LaunchArgs { get; private set; } = System.Array.Empty<string>();

    protected override void OnStartup(StartupEventArgs e)
    {
        LaunchArgs = e.Args;
        base.OnStartup(e);
    }
}
