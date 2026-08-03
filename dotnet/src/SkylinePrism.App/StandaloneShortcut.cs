using System;
using System.IO;
using System.Linq;

namespace SkylinePrism.App;

/// <summary>
/// Creates and maintains a Start Menu shortcut that opens PRISM as a standalone GUI (no Skyline needed).
///
/// <para><b>Why this is not part of installation.</b> A Skyline tool <c>.zip</c> has no install script -
/// Skyline just extracts it - so nothing can create a shortcut at install time. The tool does it itself,
/// on launch.</para>
///
/// <para><b>Why it has to be refreshed.</b> Skyline installs tools INSIDE its own ClickOnce application
/// folder (<c>%LOCALAPPDATA%\Apps\2.0\…\skyl..tion_&lt;hash&gt;\Tools\SkylinePrism\</c>), and that folder is
/// version-stamped: every Skyline update lands in a NEW hash directory and re-installs the tools there.
/// A shortcut pointing at today's path therefore goes stale on the next Skyline update. So
/// <see cref="Refresh"/> re-points an existing shortcut at the current executable every time PRISM starts
/// from Skyline, which self-heals it. (Old application folders are left behind, so even a stale shortcut
/// keeps opening the previous build rather than breaking outright.)</para>
/// </summary>
public static class StandaloneShortcut
{
    private const string ShortcutName = "Skyline-PRISM.lnk";

    /// <summary>Start Menu ▸ Programs ▸ MacCoss Lab, UW - the folder Skyline itself uses.</summary>
    private static string Folder => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.Programs), "MacCoss Lab, UW");

    public static string ShortcutPath => Path.Combine(Folder, ShortcutName);

    /// <summary>The running SkylinePrism.exe (the shortcut target).</summary>
    public static string CurrentExePath =>
        Path.Combine(AppContext.BaseDirectory, "SkylinePrism.exe");

    public static bool Exists => File.Exists(ShortcutPath);

    /// <summary>
    /// Create (or re-point) the Start Menu shortcut. Returns its path. Throws only for genuinely
    /// unexpected failures; callers surface the message.
    /// </summary>
    public static string Create()
    {
        var exe = CurrentExePath;
        if (!File.Exists(exe))
            throw new FileNotFoundException($"Could not find {exe} to point the shortcut at.", exe);

        Directory.CreateDirectory(Folder);

        // WScript.Shell is the dependency-free way to write a .lnk from .NET; this project is Windows-only
        // (WPF), so the late-bound COM call is safe.
        var shellType = Type.GetTypeFromProgID("WScript.Shell")
            ?? throw new InvalidOperationException("Windows Script Host is unavailable on this machine.");
        dynamic shell = Activator.CreateInstance(shellType)
            ?? throw new InvalidOperationException("Could not create a Windows Script Host shell.");
        try
        {
            dynamic link = shell.CreateShortcut(ShortcutPath);
            link.TargetPath = exe;
            link.WorkingDirectory = AppContext.BaseDirectory;
            link.IconLocation = exe + ",0";
            link.Description = "Skyline-PRISM - normalization and QC for LC-MS proteomics (standalone)";
            link.Save();
        }
        finally
        {
            if (System.Runtime.InteropServices.Marshal.IsComObject(shell))
                System.Runtime.InteropServices.Marshal.FinalReleaseComObject(shell);
        }
        return ShortcutPath;
    }

    /// <summary>
    /// If a shortcut already exists but points somewhere else (a previous Skyline application folder),
    /// re-point it at this build. No-op when absent - creating one is an explicit user choice.
    /// Returns true when it was rewritten.
    /// </summary>
    public static bool Refresh(Action<string>? log = null)
    {
        try
        {
            if (!Exists)
                return false;
            var target = ReadTarget();
            if (string.Equals(target, CurrentExePath, StringComparison.OrdinalIgnoreCase))
                return false;
            Create();
            log?.Invoke($"Updated the Start Menu shortcut to this build ({CurrentExePath}).");
            return true;
        }
        catch (Exception ex)
        {
            log?.Invoke("(could not refresh the Start Menu shortcut: " + ex.Message + ")");
            return false;
        }
    }

    /// <summary>The executable an existing shortcut points at, or null.</summary>
    public static string? ReadTarget()
    {
        if (!Exists)
            return null;
        var shellType = Type.GetTypeFromProgID("WScript.Shell");
        if (shellType is null)
            return null;
        dynamic? shell = Activator.CreateInstance(shellType);
        if (shell is null)
            return null;
        try
        {
            dynamic link = shell.CreateShortcut(ShortcutPath);
            return (string?)link.TargetPath;
        }
        finally
        {
            if (System.Runtime.InteropServices.Marshal.IsComObject(shell))
                System.Runtime.InteropServices.Marshal.FinalReleaseComObject(shell);
        }
    }

    public static void Remove()
    {
        if (File.Exists(ShortcutPath))
            File.Delete(ShortcutPath);
    }
}
