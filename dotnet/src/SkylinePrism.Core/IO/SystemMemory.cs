using System;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;

namespace SkylinePrism.Core.IO;

/// <summary>
/// How much physical memory the machine has, and how much of it is actually free right now.
/// <para>
/// The distinction matters because DuckDB runs <b>in-process</b> and its buffer pool is native
/// memory the .NET GC knows nothing about: a limit set from TOTAL RAM on a machine that is already
/// half full does not fail, it pages - which shows up as the whole system going to 100% memory and
/// crawling, rather than as a clean spill to the temp directory. Budgeting from what is free keeps
/// the sort inside real RAM and lets DuckDB spill the rest, which is the behaviour we want.
/// </para>
/// </summary>
internal static class SystemMemory
{
    /// <summary>
    /// Total physical memory (or the cgroup/job-object limit, when running under one). The GC
    /// reports the same number it sizes itself against, which is the right one under a container.
    /// </summary>
    public static long TotalPhysicalBytes => GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;

    /// <summary>
    /// Physical memory currently free, or <c>null</c> when the platform gives us no honest answer.
    /// Callers must degrade to a total-based budget rather than assuming a number.
    /// <para>
    /// Deliberately NOT derived from <c>GCMemoryInfo.MemoryLoadBytes</c>: that is a snapshot from the
    /// last garbage collection, which in a process that has just started may never have happened
    /// (reporting 0 in use, i.e. "everything is free" - exactly the wrong answer).
    /// </para>
    /// </summary>
    public static long? AvailablePhysicalBytes()
    {
        try
        {
            if (OperatingSystem.IsWindows())
                return WindowsAvailableBytes();
            if (OperatingSystem.IsLinux())
                return LinuxAvailableBytes();
        }
        catch (Exception)
        {
            // Never let a memory probe break a merge; the caller has a working fallback.
        }
        return null; // macOS and anything else: no cheap reliable probe, so say so.
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatusEx
    {
        public uint dwLength;
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }

    private static long? WindowsAvailableBytes()
    {
        var status = new MemoryStatusEx { dwLength = (uint)Marshal.SizeOf<MemoryStatusEx>() };
        if (!GlobalMemoryStatusEx(ref status))
            return null;
        return (long)status.ullAvailPhys;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatusEx lpBuffer);

    /// <summary>
    /// <c>MemAvailable</c> from <c>/proc/meminfo</c> - the kernel's own estimate of what a new
    /// allocation can have without swapping, which already accounts for reclaimable page cache.
    /// <c>MemFree</c> would badly understate it on any machine that has read files.
    /// </summary>
    private static long? LinuxAvailableBytes()
    {
        const string path = "/proc/meminfo";
        if (!File.Exists(path))
            return null;
        foreach (var line in File.ReadLines(path))
        {
            if (!line.StartsWith("MemAvailable:", StringComparison.Ordinal))
                continue;
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length >= 2
                && long.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var kb))
                return kb * 1024L;
            return null;
        }
        return null;
    }
}
