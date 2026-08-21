using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace Stage2Bench;

/// <summary>
/// Watches what ELSE the machine is doing while a measurement runs, so a contended run is reported as
/// contended instead of as a fact.
/// <para>
/// This exists because it was needed. An identical benchmark configuration measured 2.07 min and
/// 3.70 min hours apart; the difference was two Skyline instances that had started in between, and the
/// first explanation reached for was page-cache warmth. Several hours of analysis were then built on
/// the gap. A benchmark that cannot tell you the machine was busy is a benchmark that will eventually
/// lie to you.
/// </para>
/// </summary>
public sealed class MachineLoad
{
    /// <summary>Foreign CPU above this (as a fraction of one core) marks a sample contended.</summary>
    private const double ForeignCoreThreshold = 0.5;

    /// <summary>Free physical memory below this marks a sample contended.</summary>
    private const double MinFreeGb = 8.0;

    private readonly List<Sample> _samples = new();
    private readonly Process _self = Process.GetCurrentProcess();
    private readonly Dictionary<int, TimeSpan> _lastForeignCpu = new();
    private CancellationTokenSource? _cts;
    private Thread? _thread;
    private DateTime _lastAt;

    private readonly record struct Sample(double ForeignCores, double FreeGb, string TopProcess);

    public void Start()
    {
        _samples.Clear();
        _lastForeignCpu.Clear();
        _lastAt = DateTime.UtcNow;
        SnapshotForeignCpu();
        _cts = new CancellationTokenSource();
        _thread = new Thread(() => Loop(_cts.Token)) { IsBackground = true };
        _thread.Start();
    }

    public Report Stop()
    {
        _cts?.Cancel();
        _thread?.Join(TimeSpan.FromSeconds(2));
        if (_samples.Count == 0)
            return new Report(0, double.NaN, "", false);

        var peakForeign = _samples.Max(s => s.ForeignCores);
        var minFree = _samples.Min(s => s.FreeGb);
        var worst = _samples.OrderByDescending(s => s.ForeignCores).First().TopProcess;
        var contended = peakForeign > ForeignCoreThreshold || minFree < MinFreeGb;
        return new Report(peakForeign, minFree, worst, contended);
    }

    private void Loop(CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            try
            {
                Thread.Sleep(1000);
                if (token.IsCancellationRequested)
                    return;
                _samples.Add(Take());
            }
            catch (Exception)
            {
                // A probe must never break the measurement it is watching.
            }
        }
    }

    private Sample Take()
    {
        var now = DateTime.UtcNow;
        var seconds = Math.Max(0.001, (now - _lastAt).TotalSeconds);
        var (cores, top) = ForeignCpuCores(seconds);
        _lastAt = now;
        return new Sample(cores, FreeGb(), top);
    }

    /// <summary>
    /// CPU consumed by every process other than this one, in cores, since the previous sample - plus
    /// the biggest single contributor, because "something was using 6 cores" is far less actionable
    /// than "Skyline-daily was".
    /// </summary>
    private (double Cores, string Top) ForeignCpuCores(double seconds)
    {
        double total = 0;
        var topName = "";
        double topCores = 0;
        foreach (var p in Process.GetProcesses())
        {
            try
            {
                if (p.Id == _self.Id)
                    continue;
                var cpu = p.TotalProcessorTime;
                if (_lastForeignCpu.TryGetValue(p.Id, out var prev))
                {
                    var cores = (cpu - prev).TotalSeconds / seconds;
                    if (cores > 0.01)
                    {
                        total += cores;
                        if (cores > topCores)
                        {
                            topCores = cores;
                            topName = p.ProcessName;
                        }
                    }
                }
                _lastForeignCpu[p.Id] = cpu;
            }
            catch (Exception)
            {
                // Access denied / exited between enumeration and read; not our concern.
            }
            finally
            {
                p.Dispose();
            }
        }
        return (total, topCores > 0.01 ? $"{topName} ({topCores:n1} cores)" : "");
    }

    private void SnapshotForeignCpu()
    {
        foreach (var p in Process.GetProcesses())
        {
            try
            {
                if (p.Id != _self.Id)
                    _lastForeignCpu[p.Id] = p.TotalProcessorTime;
            }
            catch (Exception) { }
            finally { p.Dispose(); }
        }
    }

    private static double FreeGb()
    {
        if (!OperatingSystem.IsWindows())
            return double.NaN;
        var status = new MemoryStatusEx { dwLength = (uint)Marshal.SizeOf<MemoryStatusEx>() };
        return GlobalMemoryStatusEx(ref status) ? status.ullAvailPhys / 1024.0 / 1024 / 1024 : double.NaN;
    }

    public readonly record struct Report(double PeakForeignCores, double MinFreeGb, string WorstOffender, bool Contended)
    {
        public string Describe() => Contended
            ? $"CONTENDED (peak {PeakForeignCores:n1} foreign cores, min free {MinFreeGb:n1} GB"
              + (WorstOffender.Length > 0 ? $", worst: {WorstOffender})" : ")")
            : $"quiet (peak {PeakForeignCores:n1} foreign cores, min free {MinFreeGb:n1} GB)";
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

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatusEx lpBuffer);
}
