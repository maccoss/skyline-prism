#nullable enable

using System;
using System.Runtime.InteropServices;

namespace SkylinePrism.App;

/// <summary>
/// What the machine's physical memory looks like RIGHT NOW.
///
/// <para>Separate from <c>MainWindow</c> deliberately. This is interop - the kind of code most easily got
/// subtly wrong in a struct layout, and most worth a test that checks the numbers against the machine it
/// runs on - and CLAUDE.md records MainWindow as "~1000 lines of code-behind at 0% coverage" where nothing
/// can reach it.</para>
///
/// <para><b>Not <c>GC.GetGCMemoryInfo()</c>.</b> That struct's <c>MemoryLoadBytes</c> is a snapshot taken
/// at the last garbage collection, and the caller runs early in a GUI session where a significant GC may
/// not have happened yet - so it can report a nearly full machine as nearly empty. The number is used to
/// tell a user whether closing Skyline would free enough memory to export, so a stale one is worse than
/// none at all.</para>
/// </summary>
internal static class MachineMemory
{
    /// <summary>Physical RAM free now, in GB, or null when it cannot be read.</summary>
    internal static double? FreePhysicalGb() => Read()?.AvailableGb;

    /// <summary>Total and available physical RAM in GB, or null when the query fails.</summary>
    internal static (double TotalGb, double AvailableGb)? Read()
    {
        try
        {
            var status = new MemoryStatusEx();
            if (!GlobalMemoryStatusEx(status))
                return null;
            const double gb = 1024.0 * 1024 * 1024;
            return (status.ullTotalPhys / gb, status.ullAvailPhys / gb);
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            // Advisory only - every caller has something useful to say without it.
            return null;
        }
    }

    /// <summary>
    /// MEMORYSTATUSEX. A class rather than a struct so the P/Invoke marshals it by reference, which is
    /// what the API expects. No CharSet: there are no character members for one to affect.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    private sealed class MemoryStatusEx
    {
        public uint dwLength = (uint)Marshal.SizeOf(typeof(MemoryStatusEx));
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx([In, Out] MemoryStatusEx lpBuffer);
}
