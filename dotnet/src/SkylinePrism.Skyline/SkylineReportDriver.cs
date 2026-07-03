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
    private readonly ISkylineExecutor _session;
    private readonly Action<string> _log;

    public SkylineReportDriver(ISkylineExecutor session, Action<string>? log = null)
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

    public ExportedReports Export(string workDir, string? metadataReportName = null, string? batchAnnotation = null)
    {
        Directory.CreateDirectory(workDir);

        var docPath = Try(() => _session.Execute(c => c.GetDocumentPath()));
        _log($"Skyline document: {docPath ?? "(unsaved)"}");
        var version = Try(() => _session.Execute(c => c.GetVersion()));
        if (version is not null)
            _log($"Skyline version: {version}");

        EnsureReportsInstalled(batchAnnotation);

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
            return new ExportedReports(prismParquet, true, ExportMetadataReport(workDir, metadataReportName), docPath);
        }

        // Fallback: invariant CSV (older Skyline builds without parquet report export).
        _log("A valid parquet was not produced; falling back to invariant CSV.");
        TryDelete(prismParquet);
        var prismCsv = Path.Combine(workDir, "PRISM.csv");
        _session.Execute(c => c.ExportReport("PRISM", prismCsv, "invariant"));
        _log($"Exported PRISM report (invariant CSV): {prismCsv}");
        return new ExportedReports(prismCsv, false, ExportMetadataReport(workDir, metadataReportName), docPath);
    }

    /// <summary>
    /// Produce the replicate-metadata CSV. By default we read Skyline's built-in "Replicates" data
    /// grid directly over the RPC (<see cref="TryWriteMetadataFromGrid"/>), which captures every
    /// Document Annotation column dynamically, the way the grid displays them. If the caller names a
    /// specific saved report, or the grid read fails, we fall back to exporting a saved report.
    /// </summary>
    private string? ExportMetadataReport(string workDir, string? requestedName)
    {
        var csv = Path.Combine(workDir, "Metadata.csv");

        // Default: read the Replicates document grid and generate Metadata.csv from it.
        if (string.IsNullOrWhiteSpace(requestedName))
        {
            var fromGrid = TryWriteMetadataFromGrid(csv);
            if (fromGrid is not null)
                return fromGrid;
            // fall through to the saved-report export below
        }

        var available = GetAvailableReportNames();
        if (available.Count > 0)
            _log("Available Skyline reports: " + string.Join(", ", available));

        string? reportName;
        if (!string.IsNullOrWhiteSpace(requestedName))
        {
            reportName = available.FirstOrDefault(
                n => string.Equals(n, requestedName, StringComparison.OrdinalIgnoreCase)) ?? requestedName;
        }
        else
        {
            // Fallback to our installed PRISM-Replicates report if the grid read didn't work.
            reportName = available.FirstOrDefault(
                             n => string.Equals(n, "PRISM-Replicates", StringComparison.OrdinalIgnoreCase))
                         ?? "PRISM-Replicates";
        }

        try
        {
            _session.Execute(c => c.ExportReport(reportName, csv, "invariant"));
            _log($"Exported metadata report '{reportName}': {csv}");
            return csv;
        }
        catch (Exception ex)
        {
            _log($"Metadata report '{reportName}' export skipped: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Read Skyline's built-in "Replicates" document-grid view over the RPC (all Document Annotation
    /// columns) and write it to <paramref name="csvPath"/>. Returns the path on success, or null to
    /// fall back to a saved report.
    /// </summary>
    private string? TryWriteMetadataFromGrid(string csvPath)
    {
        try
        {
            // The Replicates view name can vary by build; discover it, else use the literal name.
            var views = _session.Execute(c => c.ListDocumentGridViews());
            var viewName = views?.FirstOrDefault(v => v.Equals("Replicates", StringComparison.OrdinalIgnoreCase))
                           ?? "Replicates";
            var rows = _session.Execute(c => c.GetReportRows(viewName));
            if (rows is null || rows.Columns.Count == 0)
            {
                _log($"Replicates grid '{viewName}' returned no data; using a saved report instead.");
                return null;
            }
            WriteCsv(csvPath, rows.Columns, rows.Rows);
            _log($"Read the '{viewName}' data grid ({rows.Rows.Count} replicates x {rows.Columns.Count} columns): {csvPath}");
            return csvPath;
        }
        catch (Exception ex)
        {
            _log($"Could not read the Replicates data grid ({ex.Message}); using a saved report instead.");
            return null;
        }
    }

    private static void WriteCsv(string path, IReadOnlyList<string> columns, IReadOnlyList<string[]> rows)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append(string.Join(",", columns.Select(CsvEscape))).Append('\n');
        foreach (var row in rows)
            sb.Append(string.Join(",", row.Select(CsvEscape))).Append('\n');
        File.WriteAllText(path, sb.ToString());
    }

    private static string CsvEscape(string? s)
    {
        s ??= "";
        return s.IndexOfAny(new[] { ',', '"', '\n', '\r' }) >= 0
            ? "\"" + s.Replace("\"", "\"\"") + "\""
            : s;
    }

    /// <summary>The saved report names available in the document (for a metadata-report picker).</summary>
    public IReadOnlyList<string> ListAvailableReports() => GetAvailableReportNames();

    /// <summary>
    /// Spectral libraries (.blib) available to the document: the .blib files that sit next to the
    /// .sky file (Skyline stores the document library and any local BiblioSpec libraries there),
    /// with the document-named library listed first as the likely active one.
    /// </summary>
    public IReadOnlyList<string> ListDocumentLibraries()
    {
        var docPath = Try(() => _session.Execute(c => c.GetDocumentPath()));
        if (string.IsNullOrWhiteSpace(docPath))
            return Array.Empty<string>();
        var dir = Path.GetDirectoryName(docPath);
        if (dir is null || !Directory.Exists(dir))
            return Array.Empty<string>();

        var docBlib = Path.Combine(dir, Path.GetFileNameWithoutExtension(docPath) + ".blib");
        var blibs = Directory.GetFiles(dir, "*.blib", SearchOption.TopDirectoryOnly)
            .OrderByDescending(f => string.Equals(f, docBlib, StringComparison.OrdinalIgnoreCase))
            .ThenBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();
        return blibs;
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

    private void EnsureReportsInstalled(string? batchAnnotation)
    {
        InstallReport("Skyline-PRISM.skyr", "PRISM");
        InstallReplicatesReport(batchAnnotation);
    }

    /// <summary>
    /// Install the PRISM-Replicates report. When a batch annotation name is provided, generate
    /// the view on the fly to include that dynamic annotation column (annotation_&lt;Name&gt;),
    /// since Skyline's built-in Replicates view is not exportable via the RPC. Falls back to the
    /// bundled static report if the dynamic definition is rejected.
    /// </summary>
    private void InstallReplicatesReport(string? batchAnnotation)
    {
        if (string.IsNullOrWhiteSpace(batchAnnotation))
        {
            InstallReport("Skyline-PRISM-Replicates.skyr", "PRISM-Replicates");
            return;
        }

        var ann = batchAnnotation!.Trim();
        var annColumn = ann.StartsWith("annotation_", StringComparison.OrdinalIgnoreCase) ? ann : "annotation_" + ann;
        var xml =
            "<?xml version=\"1.0\"?>\n<views>\n"
            + "  <view name=\"PRISM-Replicates\" rowsource=\"pwiz.Skyline.Model.Databinding.Entities.Replicate\" uimode=\"proteomic\">\n"
            + "    <column name=\"\" />\n"
            + "    <column name=\"SampleType\" />\n"
            + "    <column name=\"BatchName\" />\n"
            + $"    <column name=\"{XmlEscape(annColumn)}\" />\n"
            + "  </view>\n</views>\n";

        var tempSkyr = Path.Combine(Path.GetTempPath(), "PRISM-Replicates.skyr");
        try
        {
            File.WriteAllText(tempSkyr, xml);
            _session.Execute(c => c.RunCommandSilent(new[]
            {
                $"--report-add={tempSkyr}",
                "--report-conflict-resolution=overwrite",
            }));
            _log($"Installed PRISM-Replicates report including the '{ann}' annotation column.");
        }
        catch (Exception ex)
        {
            _log($"Could not add PRISM-Replicates with annotation '{ann}' ({ex.Message}); "
                 + "using the bundled report (SampleType + BatchName only).");
            InstallReport("Skyline-PRISM-Replicates.skyr", "PRISM-Replicates");
        }
    }

    private static string XmlEscape(string s) => s
        .Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");

    private void InstallReport(string fileName, string displayName)
    {
        var skyr = Path.Combine(AppContext.BaseDirectory, "Reports", fileName);
        if (!File.Exists(skyr))
        {
            _log($"{displayName} report definition not bundled at {skyr}; assuming it is already installed in Skyline.");
            return;
        }
        try
        {
            _session.Execute(c => c.RunCommandSilent(new[]
            {
                $"--report-add={skyr}",
                "--report-conflict-resolution=overwrite",
            }));
            _log($"Installed/updated the {displayName} report definition in Skyline.");
        }
        catch (Exception ex)
        {
            _log($"Could not add {displayName} report (it may already exist): {ex.Message}");
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
