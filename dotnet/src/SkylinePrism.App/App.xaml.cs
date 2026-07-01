using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;

namespace SkylinePrism.App;

/// <summary>
/// WPF application entry. Skyline passes the JSON-RPC connection string as the first
/// command-line argument ($(SkylineConnection)); it is captured here and handed to
/// SkylineSession.FromArguments. Installs global exception handlers so a background-thread
/// failure surfaces as a dialog + a persistent log file instead of silently closing the window.
/// </summary>
public partial class App : Application
{
    /// <summary>Command-line args captured at startup (the Skyline connection string).</summary>
    public static string[] LaunchArgs { get; private set; } = Array.Empty<string>();

    /// <summary>Always-available crash/diagnostic log (survives a hard crash of the window).</summary>
    public static string LogFilePath { get; } = InitLogPath();

    private static readonly object LogLock = new();

    private static string InitLogPath()
    {
        try
        {
            var dir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SkylinePrism");
            Directory.CreateDirectory(dir);
            return Path.Combine(dir, "prism-tool.log");
        }
        catch
        {
            return Path.Combine(Path.GetTempPath(), "prism-tool.log");
        }
    }

    public static void WriteLog(string line)
    {
        try
        {
            lock (LogLock)
                File.AppendAllText(LogFilePath,
                    DateTime.Now.ToString("HH:mm:ss.fff") + "  " + line + Environment.NewLine);
        }
        catch
        {
            // Logging must never throw.
        }
    }

    protected override void OnStartup(StartupEventArgs e)
    {
        LaunchArgs = e.Args;

        DispatcherUnhandledException += OnDispatcherUnhandledException;
        AppDomain.CurrentDomain.UnhandledException += OnAppDomainUnhandledException;
        TaskScheduler.UnobservedTaskException += OnUnobservedTaskException;

        WriteLog("==== Skyline-PRISM tool started ====");
        WriteLog("Args: " + string.Join(" ", e.Args));
        base.OnStartup(e);
    }

    private void OnDispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        Report("UI thread", e.Exception);
        e.Handled = true; // keep the window open so the user can read the error
    }

    private void OnAppDomainUnhandledException(object sender, UnhandledExceptionEventArgs e)
        => Report("background thread", e.ExceptionObject as Exception);

    private void OnUnobservedTaskException(object sender, UnobservedTaskExceptionEventArgs e)
    {
        Report("task", e.Exception);
        e.SetObserved();
    }

    private static void Report(string source, Exception? ex)
    {
        WriteLog($"UNHANDLED ({source}): {ex}");
        try
        {
            MessageBox.Show(
                (ex?.ToString() ?? "Unknown error")
                + Environment.NewLine + Environment.NewLine
                + "A copy of this error was written to:" + Environment.NewLine + LogFilePath,
                "Skyline-PRISM error",
                MessageBoxButton.OK, MessageBoxImage.Error);
        }
        catch
        {
            // Never let the error reporter itself take down the process.
        }
    }
}
