using System;
using System.Collections.Generic;
using System.IO;
using System.Windows;

namespace SkylinePrism.App;

/// <summary>
/// Detects a PRISM process whose .NET runtime, or whose own tool folder, has been removed from disk
/// since it started, and words the message asking the user to restart.
/// </summary>
/// <remarks>
/// <para>The case this exists for: a PRISM window left open for five days across a .NET update. The
/// update (10.0.14 to 10.0.15) removed <c>shared\Microsoft.WindowsDesktop.App\10.0.14\</c>, the folder
/// the process had loaded WPF from. Everything already loaded kept working; the next thing WPF loaded
/// on demand did not. That was <c>PenImc_cor3.dll</c>, WPF's pen and touch support, which loads when
/// Windows first reports a tablet - on a touch-screen laptop, that can be days after start. It failed
/// with <c>DllNotFoundException</c>, then failed again on every tablet message after it, one dialog
/// each. No PRISM frame was on the stack, and the only fix was a restart.</para>
///
/// <para>Detected by STATE, not by matching that exception - the same choice as App's shutdown guard.
/// The pen DLL was the first casualty, not the only possible one: every assembly and native library the
/// process has not loaded yet is equally gone (a window not yet opened, accessibility support when a
/// screen reader attaches), and each fails with its own exception type and stack. Whether the files are
/// still there is one question with one answer.</para>
///
/// <para>An assembly's <see cref="System.Reflection.Assembly.Location"/> is the path it was loaded from,
/// fixed at load, so it still names the file after the file is gone - which is what makes checking it
/// meaningful. One probe per folder that can disappear under a running process: CoreLib for the .NET
/// runtime, PresentationCore for the WPF runtime (where the pen DLL lives), and PRISM's own assembly
/// for the tool folder, in case reinstalling or updating the tool in Skyline removes the copy a
/// running PRISM started from.</para>
/// </remarks>
internal static class StartupFiles
{
    /// <summary>
    /// The files whose absence means this process can no longer load what it has not loaded yet.
    /// </summary>
    internal static IEnumerable<string> Probes() => new[]
    {
        typeof(object).Assembly.Location,       // Microsoft.NETCore.App\<version>\
        typeof(UIElement).Assembly.Location,    // Microsoft.WindowsDesktop.App\<version>\
        typeof(StartupFiles).Assembly.Location, // the tool folder
    };

    /// <summary>
    /// The first of <paramref name="paths"/> that no longer exists, or null when all are present.
    /// </summary>
    /// <remarks>
    /// An empty path is skipped rather than reported: a single-file publish loads assemblies from the
    /// bundle and reports an empty <c>Location</c>, and that is not a missing file. PRISM does not
    /// publish single-file today, but a restart prompt on every error would be the cost of assuming so.
    /// </remarks>
    internal static string? FirstMissing(IEnumerable<string> paths)
    {
        foreach (var path in paths)
        {
            if (!string.IsNullOrEmpty(path) && !File.Exists(path))
                return path;
        }
        return null;
    }

    /// <summary>What the user is shown instead of the exception once a startup file is missing.</summary>
    internal static string RestartMessage(string missingPath, string logPath) =>
        "Skyline-PRISM needs to be restarted." + Environment.NewLine + Environment.NewLine
        + "A file it was started from is no longer on disk:" + Environment.NewLine
        + missingPath + Environment.NewLine + Environment.NewLine
        + "This happens when a .NET update is installed, or PRISM is reinstalled or updated in Skyline, "
        + "while PRISM is open. Anything PRISM has not loaded yet can no longer load, so more errors "
        + "will follow until it is restarted." + Environment.NewLine + Environment.NewLine
        + "Close PRISM and start it again from Skyline's Tools menu. Results already written to disk "
        + "are not affected." + Environment.NewLine + Environment.NewLine
        + "The error was written to:" + Environment.NewLine + logPath;
}
