#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SkylinePrism.Skyline;

/// <summary>
/// Drives a running Skyline instance (over JSON-RPC) to export the inputs the PRISM
/// pipeline consumes. Prefers exporting the PRISM transition report as PARQUET (typed,
/// compact, and much faster to read than CSV); falls back to invariant CSV only if a valid
/// parquet is not produced. Also exports the Replicates metadata report when available.
/// Installs the PRISM report definition into the document first if it is not already present.
/// </summary>
public sealed class SkylineReportDriver
{
    private readonly SkylineSession _session;
    private readonly Action<string> _log;

    public SkylineReportDriver(SkylineSession session, Action<string>? log = null)
    {
        _session = session;
        _log = log ?? (_ => { });
    }

    /// <summary>
    /// The report inputs exported into <c>workDir</c>. <see cref="InputPath"/> is the single
    /// PRISM report the pipeline should read (parquet when <see cref="InputIsParquet"/>).
    /// </summary>
    public sealed record ExportedReports(
        string InputPath, bool InputIsParquet, string? ReplicatesCsv, string? DocumentPath);

    public ExportedReports Export(string workDir)
    {
        Directory.CreateDirectory(workDir);

        var docPath = Try(() => _session.Execute(c => c.GetDocumentPath()));
        _log($"Skyline document: {docPath ?? "(unsaved)"}");
        var version = Try(() => _session.Execute(c => c.GetVersion()));
        if (version is not null)
            _log($"Skyline version: {version}");

        EnsurePrismReport();

        // Preferred path: export PRISM directly as parquet. Skyline determines the format
        // from the file extension, the same mechanism that produced the CSV.
        var prismParquet = Path.Combine(workDir, "PRISM.parquet");
        _log("Exporting PRISM report as parquet (this can take a while on large documents)...");
        try
        {
            _session.Execute(c => c.ExportReport("PRISM", prismParquet, "invariant"));
        }
        catch (Exception ex)
        {
            _log($"PRISM parquet export threw: {ex.Message}");
        }

        if (IsValidParquet(prismParquet))
        {
            _log($"Exported PRISM report (parquet): {prismParquet} "
                 + $"({new FileInfo(prismParquet).Length:N0} bytes)");
            return new ExportedReports(prismParquet, true, ExportReplicates(workDir), docPath);
        }

        // Fallback: invariant CSV (older Skyline builds without parquet report export).
        _log("A valid parquet was not produced; falling back to invariant CSV.");
        TryDelete(prismParquet);
        var prismCsv = Path.Combine(workDir, "PRISM.csv");
        _session.Execute(c => c.ExportReport("PRISM", prismCsv, "invariant"));
        _log($"Exported PRISM report (invariant CSV): {prismCsv}");
        return new ExportedReports(prismCsv, false, ExportReplicates(workDir), docPath);
    }

    private string? ExportReplicates(string workDir)
    {
        // Resolve the built-in Replicates report name (its columns are annotation-dependent).
        var reportName = "Replicates";
        try
        {
            var available = GetAvailableReportNames();
            if (available.Count > 0)
            {
                _log("Available Skyline reports: " + string.Join(", ", available));
                var match = available.FirstOrDefault(
                    n => string.Equals(n, "Replicates", StringComparison.OrdinalIgnoreCase));
                if (match is not null)
                    reportName = match;
            }
        }
        catch (Exception ex)
        {
            _log($"(could not enumerate reports: {ex.Message})");
        }

        var replicatesCsv = Path.Combine(workDir, "Replicates.csv");
        try
        {
            _session.Execute(c => c.ExportReport(reportName, replicatesCsv, "invariant"));
            _log($"Exported '{reportName}' report: {replicatesCsv}");
            return replicatesCsv;
        }
        catch (Exception ex)
        {
            _log($"Replicates report export skipped: {ex.Message}");
            return null;
        }
    }

    private List<string> GetAvailableReportNames()
    {
        var names = new List<string>();
        foreach (var group in new string?[] { null, "main", "external_tools" })
        {
            try
            {
                var arr = _session.Execute(c => c.GetSettingsListNames("Reports", group));
                if (arr is not null)
                    names.AddRange(arr);
            }
            catch
            {
                // Group not supported on this Skyline build; ignore.
            }
        }
        return names.Where(n => !string.IsNullOrWhiteSpace(n)).Distinct().ToList();
    }

    private void EnsurePrismReport()
    {
        var skyr = Path.Combine(AppContext.BaseDirectory, "Reports", "Skyline-PRISM.skyr");
        if (!File.Exists(skyr))
        {
            _log($"PRISM report definition not bundled at {skyr}; assuming it is already installed in Skyline.");
            return;
        }
        try
        {
            _session.Execute(c => c.RunCommandSilent(new[]
            {
                $"--report-add={skyr}",
                "--report-conflict-resolution=overwrite",
            }));
            _log("Installed/updated the PRISM report definition in Skyline.");
        }
        catch (Exception ex)
        {
            _log($"Could not add PRISM report (it may already exist): {ex.Message}");
        }
    }

    /// <summary>A parquet file starts and ends with the 4-byte "PAR1" magic marker.</summary>
    private static bool IsValidParquet(string path)
    {
        try
        {
            if (!File.Exists(path))
                return false;
            using var fs = File.OpenRead(path);
            if (fs.Length < 8)
                return false;
            var head = new byte[4];
            fs.ReadExactly(head);
            fs.Seek(-4, SeekOrigin.End);
            var tail = new byte[4];
            fs.ReadExactly(tail);
            ReadOnlySpan<byte> magic = "PAR1"u8;
            return head.AsSpan().SequenceEqual(magic) && tail.AsSpan().SequenceEqual(magic);
        }
        catch
        {
            return false;
        }
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { /* ignore */ }
    }

    private T? Try<T>(Func<T> f) where T : class
    {
        try { return f(); }
        catch (Exception ex) { _log($"(warning) {ex.Message}"); return null; }
    }
}
