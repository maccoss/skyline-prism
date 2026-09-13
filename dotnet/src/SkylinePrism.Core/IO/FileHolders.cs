using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text;

namespace SkylinePrism.Core.IO;

/// <summary>
/// Names the processes holding a file open, on Windows, without needing admin rights.
/// </summary>
/// <remarks>
/// <para><b>Why this exists.</b> "The file is locked by another process" is where a diagnosis
/// stops. Three separate investigations went into one locked cache file here and none of them could
/// say what held it - the user's reasonable answer was "nothing other than PRISM reads or writes
/// it", which is true of every program a person chooses to run and says nothing about the scanners
/// and indexers that open a file the moment it is closed.</para>
///
/// <para>The Restart Manager is the API built for exactly this question. Installers use it to ask
/// "who is using this file so I can ask them to close it"; it needs no elevation, it is present on
/// every supported Windows, and it answers in milliseconds.</para>
///
/// <para><b>It cannot see a holder on another machine.</b> For a file on a share, the handle lives
/// in the server's session table, so a lock taken by a different client - or left behind by one that
/// died - is invisible here. An empty answer therefore means "nobody on THIS machine", which is
/// itself worth knowing: it is the difference between a program to close and a share to investigate.
/// </para>
/// </remarks>
internal static class FileHolders
{
    /// <summary>
    /// The processes on this machine with <paramref name="path"/> open, as a phrase for a log line,
    /// or null when there are none to name.
    /// </summary>
    public static string? Describe(string path)
    {
        if (!OperatingSystem.IsWindows())
            return null;

        try
        {
            var holders = Query(path);
            if (holders.Count == 0)
                return null;
            return string.Join(", ", holders);
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException
                                       or InvalidOperationException or IOException)
        {
            // A diagnostic must never be the thing that fails. Nothing downstream depends on this
            // answer; the caller degrades to the message it had before.
            return null;
        }
    }

    [SupportedOSPlatform("windows")]
    private static List<string> Query(string path)
    {
        var names = new List<string>();

        // The API WRITES a session key of CCH_RM_SESSION_KEY characters into this buffer, so it has
        // to be a writable one of at least that capacity plus the terminator. Handing it a string -
        // which marshals as a read-only pointer to, in the empty case, no characters at all - is a
        // 33-character write into nothing: an 0xC0000005 that takes the whole process with it, not
        // an exception. It did.
        var key = new StringBuilder(CchSessionKey + 1);
        var result = RmStartSession(out var session, 0, key);
        if (result != 0)
            return names;

        try
        {
            // The application and service arrays are genuinely unused, and IntPtr.Zero says so
            // without asking the marshaller to reason about a null typed array.
            result = RmRegisterResources(
                session, 1, new[] { path }, 0, IntPtr.Zero, 0, IntPtr.Zero);
            if (result != 0)
                return names;

            uint needed = 0;
            uint count = 0;
            var reasons = 0u;
            // Two-call idiom: the first asks how many there are, the second fills the buffer. A
            // holder can appear between the two, which is what ERROR_MORE_DATA means here - one
            // retry with the larger count is enough for a question this small.
            result = RmGetList(session, out needed, ref count, null, ref reasons);
            if (result == ErrorMoreData && needed > 0)
            {
                var processes = new RmProcessInfo[needed];
                count = needed;
                result = RmGetList(session, out needed, ref count, processes, ref reasons);
                if (result == 0)
                {
                    for (var i = 0; i < count; i++)
                        names.Add(NameOf(processes[i]));
                }
            }
        }
        finally
        {
            RmEndSession(session);
        }

        return names.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
    }

    [SupportedOSPlatform("windows")]
    private static string NameOf(RmProcessInfo info)
    {
        var name = info.strAppName;
        try
        {
            // The friendly name the Restart Manager reports is often just the window title, and for
            // a service there is none at all - so the executable is what makes it actionable.
            using var process = Process.GetProcessById((int)info.Process.dwProcessId);
            name = process.ProcessName;
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
        {
            // Gone between the list and the lookup, which is the good outcome: it let go.
        }
        return string.IsNullOrWhiteSpace(name)
            ? $"pid {info.Process.dwProcessId}"
            : $"{name} (pid {info.Process.dwProcessId})";
    }

    private const int CchSessionKey = 32;
    private const int ErrorMoreData = 234;

    [StructLayout(LayoutKind.Sequential)]
    private struct RmUniqueProcess
    {
        public uint dwProcessId;
        public System.Runtime.InteropServices.ComTypes.FILETIME ProcessStartTime;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct RmProcessInfo
    {
        public RmUniqueProcess Process;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
        public string strAppName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
        public string strServiceShortName;
        public int ApplicationType;
        public uint AppStatus;
        public uint TSSessionId;
        [MarshalAs(UnmanagedType.Bool)]
        public bool bRestartable;
    }

    [DllImport("rstrtmgr.dll", CharSet = CharSet.Unicode)]
    private static extern int RmStartSession(
        out uint pSessionHandle, int dwSessionFlags, StringBuilder strSessionKey);

    [DllImport("rstrtmgr.dll")]
    private static extern int RmEndSession(uint pSessionHandle);

    [DllImport("rstrtmgr.dll", CharSet = CharSet.Unicode)]
    private static extern int RmRegisterResources(
        uint pSessionHandle, uint nFiles, string[] rgsFilenames,
        uint nApplications, IntPtr rgApplications,
        uint nServices, IntPtr rgsServiceNames);

    [DllImport("rstrtmgr.dll")]
    private static extern int RmGetList(
        uint dwSessionHandle, out uint pnProcInfoNeeded, ref uint pnProcInfo,
        // [In, Out] is load-bearing. RM_PROCESS_INFO carries fixed-length strings, so it is
        // not blittable, and the marshaller defaults a non-blittable array parameter to IN
        // ONLY - it builds a native copy, lets the API fill it, and throws it away. Every
        // entry came back zeroed, which reads as pid 0, the System Idle Process, rather than
        // as a marshalling fault. Verified against a file this process holds: pid 0 without
        // the attribute, its own pid with it.
        [In, Out] RmProcessInfo[]? rgAffectedApps, ref uint lpdwRebootReasons);
}
