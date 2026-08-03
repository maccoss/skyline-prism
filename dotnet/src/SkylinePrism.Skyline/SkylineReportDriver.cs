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
    /// Export this document's PRISM transition report (+ replicate metadata) into <paramref name="workDir"/>.
    /// </summary>
    /// <param name="documentLabel">
    /// Base name for the exported files, and the batch / Source Document label the merge derives from the
    /// file stem. Defaults to "PRISM" (the single-document case); pass the document name when several
    /// documents are being combined, so their replicates stay distinguishable.
    /// </param>
    public ExportedReports Export(
        string workDir, string? metadataReportName = null, string? batchAnnotation = null,
        string? documentLabel = null)
    {
        Directory.CreateDirectory(workDir);

        var docPath = Try(() => _session.Execute(c => c.GetDocumentPath()));
        _log($"Skyline document: {docPath ?? "(unsaved)"}");
        var version = Try(() => _session.Execute(c => c.GetVersion()));
        if (version is not null)
            _log($"Skyline version: {version}");

        var label = string.IsNullOrWhiteSpace(documentLabel) ? "PRISM" : documentLabel!;

        EnsureReportsInstalled();

        // Preferred path: export PRISM directly as parquet. Skyline determines the format
        // from the file extension, the same mechanism that produced the CSV.
        var prismParquet = Path.Combine(workDir, label + ".parquet");
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
            return new ExportedReports(
                prismParquet, true,
                ExportMetadataReport(workDir, metadataReportName, batchAnnotation, label), docPath, label);
        }

        // Fallback: invariant CSV (older Skyline builds without parquet report export).
        _log("A valid parquet was not produced; falling back to invariant CSV.");
        TryDelete(prismParquet);
        var prismCsv = Path.Combine(workDir, label + ".csv");
        _session.Execute(c => c.ExportReport("PRISM", prismCsv, "invariant"));
        _log($"Exported PRISM report (invariant CSV): {prismCsv}");
        return new ExportedReports(
            prismCsv, false,
            ExportMetadataReport(workDir, metadataReportName, batchAnnotation, label), docPath, label);
    }

    /// <summary>The metadata CSV file name paired with a given transition-report label.</summary>
    public static string MetadataFileName(string documentLabel) => documentLabel + ".metadata.csv";

    /// <summary>
    /// Produce the replicate-metadata CSV. By default we read Skyline's built-in "Replicates" data
    /// grid directly over the RPC (<see cref="TryWriteMetadataFromGrid"/>), which captures every
    /// Document Annotation column dynamically, the way the grid displays them. If the caller names a
    /// specific saved report, or the grid read fails, we fall back to exporting a saved report.
    /// </summary>
    private string? ExportMetadataReport(
        string workDir, string? requestedName, string? batchAnnotation, string documentLabel)
    {
        var csv = Path.Combine(workDir, MetadataFileName(documentLabel));

        // Default: read the Replicates document grid and generate Metadata.csv from it.
        if (string.IsNullOrWhiteSpace(requestedName))
        {
            var fromGrid = TryWriteMetadataFromGrid(csv);
            if (fromGrid is not null)
                return fromGrid;
            // Grid read failed: install our PRISM-Replicates saved report on demand (with the batch
            // annotation column, if any) and export it as the fallback below.
            InstallReplicatesReport(batchAnnotation);
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

    // The built-in Replicate-entity columns. Any Replicate column NOT in this set is a user-defined
    // document annotation (e.g. Condition, Subject) and is always carried into the metadata.
    private static readonly HashSet<string> ReplicateBuiltinColumns = new(StringComparer.OrdinalIgnoreCase)
    {
        "Replicate", "ReplicateName", "FileName", "FilePath", "SampleName", "ModifiedTime", "AcquiredTime",
        "ExplicitGlobalStandardArea", "TicArea", "IonMobilityUnits", "ResultFileLocator", "SampleId",
        "InstrumentSerialNumber", "MedianPeakArea", "NormalizationDivisor", "SampleType",
        "AnalyteConcentration", "SampleDilutionFactor", "BatchName", "ReplicateLocator",
    };

    // The standard columns Skyline's built-in "Replicates" document-grid view shows: the replicate name
    // plus Sample Type and Analyte Concentration. We deliberately do NOT carry the other Replicate-entity
    // properties (BatchName, SampleDilutionFactor, SampleId, file/instrument/computed columns): the
    // built-in Replicates view does not display them, so including them put phantom fields into the
    // metadata / QC "Group by" list that are not in the user's actual Replicates grid. Every replicate
    // annotation is still carried dynamically (see TryWriteMetadataFromGrid), so the reconstructed report
    // matches the Replicates view: standard columns + annotations.
    private static readonly string[] ReplicateCuratedColumns =
        { "ReplicateName", "SampleType", "AnalyteConcentration" };

    /// <summary>
    /// Reconstruct Skyline's built-in "Replicates" document-grid view (which is not itself a named,
    /// RPC-exportable report) and write it to <paramref name="csvPath"/>: enumerate the Replicate entity's
    /// columns, then run a report definition selecting the view's standard columns (Sample Type, Analyte
    /// Concentration) plus every replicate annotation - the columns not in <see cref="ReplicateBuiltinColumns"/>.
    /// The result matches the columns the user sees in the Replicates grid. Returns the path on success, or
    /// null to fall back to a saved report.
    /// </summary>
    private string? TryWriteMetadataFromGrid(string csvPath)
    {
        try
        {
            var columns = _session.Execute(c => c.GetReplicateColumns());
            if (columns is null || columns.Length == 0)
            {
                _log("Replicate columns unavailable; using a saved report for metadata instead.");
                return null;
            }

            var present = new HashSet<string>(columns, StringComparer.OrdinalIgnoreCase);
            var annotations = columns.Where(c => !ReplicateBuiltinColumns.Contains(c)).ToList();
            var select = ReplicateCuratedColumns.Where(present.Contains)
                .Concat(annotations)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            var rows = _session.Execute(c => c.GetReplicateReport(select));
            if (rows is null || rows.Columns.Count == 0)
            {
                _log("Replicate report returned no data; using a saved report for metadata instead.");
                return null;
            }
            WriteCsv(csvPath, rows.Columns, rows.Rows);
            _log($"Read the Replicates grid ({rows.Rows.Count} replicates, {rows.Columns.Count} columns"
                 + (annotations.Count > 0 ? "; annotations: " + string.Join(", ", annotations) : "; no annotations")
                 + $"): {csvPath}");
            return csvPath;
        }
        catch (Exception ex)
        {
            _log($"Could not read the Replicates grid ({ex.Message}); using a saved report for metadata instead.");
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

    /// <summary>
    /// The document's active digestion enzyme, mapped to a PRISM enzyme name (parsimony.enzyme). Reads
    /// the selected "Enzymes" settings item over JSON-RPC and parses its cut/no_cut/sense XML. Returns
    /// null when the document has no enzyme, the RPC is unavailable, or the enzyme has no PRISM
    /// equivalent - the caller then keeps the config default. Used so the Skyline external tool's
    /// FASTA-based parsimony matches the document's digestion instead of a hardcoded default.
    /// </summary>
    public string? GetDigestionEnzyme()
    {
        var selected = Try(() => _session.Execute(c => c.GetSettingsListSelectedItems("Enzymes")));
        var name = selected?.FirstOrDefault(n => !string.IsNullOrWhiteSpace(n));
        if (string.IsNullOrWhiteSpace(name))
            return null;

        var xml = Try(() => _session.Execute(c => c.GetSettingsListItem("Enzymes", name!)));
        if (string.IsNullOrWhiteSpace(xml))
            return null;

        var prism = SkylineDigestion.PrismEnzymeFromXml(xml);
        if (prism is null)
            _log($"Document enzyme '{name}' has no PRISM equivalent; using the configured default enzyme.");
        else
            _log($"Document digestion enzyme: {name} -> PRISM enzyme '{prism}'.");
        return prism;
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

    private void EnsureReportsInstalled()
    {
        // Only the PRISM transition report is always needed. The replicate metadata comes from the
        // Replicates document grid (read dynamically), so the PRISM-Replicates saved report is installed
        // on demand only if that grid read fails - keeping it out of the document's saved-report list.
        InstallReport("Skyline-PRISM.skyr", "PRISM");
    }

    /// <summary>
    /// Install the PRISM-Replicates report. When a batch annotation name is provided, generate
    /// the view on the fly to include that dynamic annotation column (annotation_&lt;Name&gt;),
    /// since Skyline's built-in Replicates view is not exportable via the RPC. Falls back to the
    /// bundled static report if the dynamic definition is rejected. The view XML comes from
    /// <see cref="ReplicatesReportBuilder"/>, shared with the headless (SkylineCmd) path.
    /// </summary>
    private void InstallReplicatesReport(string? batchAnnotation)
    {
        if (string.IsNullOrWhiteSpace(batchAnnotation))
        {
            InstallReport("Skyline-PRISM-Replicates.skyr", "PRISM-Replicates");
            return;
        }

        var ann = batchAnnotation!.Trim();
        var tempSkyr = Path.Combine(Path.GetTempPath(), "PRISM-Replicates.skyr");
        try
        {
            ReplicatesReportBuilder.WriteSkyr(tempSkyr, new[] { ann });
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

    private static bool IsValidParquet(string path) => ParquetMagic.IsValid(path);

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
