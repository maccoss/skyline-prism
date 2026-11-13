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
/// <see cref="Refresh"/> re-points an existing shortcut at the current executable every time an installed
/// copy of PRISM starts, which self-heals it. (Old application folders are left behind, so even a stale shortcut
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
    /// re-point it at this build - when <see cref="Decide"/> says so: only from an installed Skyline
    /// tool, and never to an older version than the one it points at. No-op when absent - creating
    /// one is an explicit user choice. Returns true when it was rewritten.
    /// </summary>
    /// <remarks>
    /// Any build used to re-point it, so starting a development build once - built without the
    /// instrument-file reader, as an ordinary <c>dotnet build</c> is - took the shortcut over, and
    /// every later Start Menu launch ran that build: ion accounting then reported "this build has no
    /// instrument-file reader" on a machine whose installed tool had one.
    /// <para>A declined refresh goes to <paramref name="diagnostic"/> (the log file), not
    /// <paramref name="log"/>: it is the same on every start of the same copy, and the window's log
    /// is what a user reads.</para>
    /// </remarks>
    public static bool Refresh(Action<string>? log = null, Action<string>? diagnostic = null)
    {
        try
        {
            if (!Exists)
                return false;
            var target = ReadTarget();
            var shown = string.IsNullOrEmpty(target) ? "(a target that could not be read)" : target;
            switch (Decide(target, AppContext.BaseDirectory, IsSkylineFolder, File.Exists, ToolVersion))
            {
                case RefreshDecision.AlreadyCurrent:
                    return false;
                case RefreshDecision.NotInstalledTool:
                    diagnostic?.Invoke($"Start Menu shortcut left pointing at {shown}: this copy is not an "
                                       + "installed Skyline tool, so it does not take the shortcut over.");
                    return false;
                case RefreshDecision.TargetIsNewer:
                    diagnostic?.Invoke($"Start Menu shortcut left pointing at {shown}: that copy is a newer "
                                       + "version than this one.");
                    return false;
            }
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

    internal enum RefreshDecision { Repoint, AlreadyCurrent, NotInstalledTool, TargetIsNewer }

    /// <summary>
    /// Whether the copy running from <paramref name="exeDirectory"/> should re-point a shortcut
    /// whose target is <paramref name="target"/>. File system access is passed in so the rule can be
    /// tested without touching the real Start Menu.
    /// </summary>
    internal static RefreshDecision Decide(
        string? target, string exeDirectory, Func<string, bool> isSkylineFolder,
        Func<string, bool> fileExists, Func<string, Version?> versionOf)
    {
        var current = Path.Combine(exeDirectory, "SkylinePrism.exe");
        if (string.Equals(target, current, StringComparison.OrdinalIgnoreCase))
            return RefreshDecision.AlreadyCurrent;
        if (!IsInstalledTool(exeDirectory, isSkylineFolder))
            return RefreshDecision.NotInstalledTool;
        // Never backwards: PRISM in a release Skyline must not take the shortcut from Skyline-daily's
        // newer copy. A target that is gone, or whose version cannot be read, is replaced.
        if (!string.IsNullOrEmpty(target) && fileExists(target)
            && Path.GetDirectoryName(target) is { } targetDir
            && versionOf(targetDir) is { } theirs && versionOf(exeDirectory) is { } ours
            && theirs > ours)
            return RefreshDecision.TargetIsNewer;
        return RefreshDecision.Repoint;
    }

    /// <summary>
    /// True when <paramref name="exeDirectory"/> is where Skyline installed a tool: its own folder
    /// directly inside the <c>Tools</c> folder of a Skyline installation -
    /// <c>...\skyl..tion_&lt;hash&gt;\Tools\~SK6C53\</c> for a ClickOnce Skyline. The Tools folder's
    /// parent must hold Skyline itself (<paramref name="isSkylineFolder"/>): a folder merely NAMED
    /// Tools, such as <c>C:\Tools\prism\</c> where portable copies are often kept, is not one.
    /// </summary>
    internal static bool IsInstalledTool(string exeDirectory, Func<string, bool> isSkylineFolder)
    {
        var dir = exeDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var tools = Path.GetDirectoryName(dir);
        if (string.IsNullOrEmpty(tools)
            || !string.Equals(Path.GetFileName(tools), "Tools", StringComparison.OrdinalIgnoreCase))
            return false;
        var skyline = Path.GetDirectoryName(tools);
        return !string.IsNullOrEmpty(skyline) && isSkylineFolder(skyline);
    }

    /// <summary>A folder holding Skyline itself - release or daily - as a Skyline tool's grandparent does.</summary>
    private static bool IsSkylineFolder(string folder) =>
        File.Exists(Path.Combine(folder, "Skyline.exe")) || File.Exists(Path.Combine(folder, "Skyline-daily.exe"));

    /// <summary>
    /// The tool version of the copy in <paramref name="exeDirectory"/>, from the <c>Version =</c> line of
    /// its <c>tool-inf\info.properties</c> (which every build copies beside the exe); null if unreadable.
    /// </summary>
    internal static Version? ToolVersion(string exeDirectory)
    {
        try
        {
            var file = Path.Combine(exeDirectory, "tool-inf", "info.properties");
            if (!File.Exists(file))
                return null;
            foreach (var line in File.ReadLines(file))
            {
                var parts = line.Split('=', 2);
                if (parts.Length == 2 && parts[0].Trim().Equals("Version", StringComparison.OrdinalIgnoreCase))
                    return Version.TryParse(parts[1].Trim(), out var v) ? v : null;
            }
            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return null;
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
