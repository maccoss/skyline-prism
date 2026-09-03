using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;
using SkylinePrism.Core.RawData;

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

        // Make the instrument-file reader available, if this build carries one. Done here rather
        // than lazily at the point of use so the log records which build the user is running -
        // "acquired MS2 signal unknown" otherwise looks like a data problem rather than a package
        // built without the reader.
        RegisterRawReader();
        base.OnStartup(e);
    }

    /// <summary>
    /// Registers the pwiz-backed reader when the build has one. Reflection rather than a direct
    /// call: SkylinePrism.Pwiz is referenced only when a pwiz-sharp checkout was present at build
    /// time, so naming its types here would stop a pwiz-less build compiling - and that build is
    /// what a developer without the checkout, and the cross-platform CLI, both use.
    /// </summary>
    private static void RegisterRawReader()
    {
        try
        {
            var type = Type.GetType(
                "SkylinePrism.Pwiz.PwizReaderRegistration, SkylinePrism.Pwiz", throwOnError: false);
            if (type is null)
            {
                WriteLog("Instrument-file reader: not in this build; acquired MS2 signal will read "
                    + "as unknown.");
                return;
            }

            type.GetMethod("Register")?.Invoke(null, null);
            WriteLog("Instrument-file reader: registered ("
                + string.Join(", ", Ms2SignalReaders.All.Select(r => r.Describe())) + ").");
        }
        catch (Exception ex)
        {
            // A reader that cannot load is a missing denominator, not a reason to fail startup.
            WriteLog("Instrument-file reader: could not be registered - " + ex.Message);
        }
    }

    private void OnDispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        Report("UI thread", e.Exception);
        e.Handled = true; // keep the window open so the user can read the error
    }

    private void OnAppDomainUnhandledException(object sender, UnhandledExceptionEventArgs e)
        => Report("background thread", e.ExceptionObject as Exception);

    // EventHandler<T> declares a nullable sender, unlike the two non-generic handlers above.
    private void OnUnobservedTaskException(object? sender, UnobservedTaskExceptionEventArgs e)
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
