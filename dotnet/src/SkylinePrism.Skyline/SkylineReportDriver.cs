#nullable enable

using System;
using System.IO;

namespace SkylinePrism.Skyline;

/// <summary>
/// Drives a running Skyline instance (over JSON-RPC) to export the inputs the PRISM
/// pipeline consumes: the PRISM transition report (invariant CSV + parquet) and the
/// default Replicates metadata report. Installs the PRISM report definition into the
/// document first if it is not already present.
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

    /// <summary>Paths of the reports exported into <c>workDir</c> (null when an export was skipped).</summary>
    public sealed record ExportedReports(
        string PrismCsv, string? PrismParquet, string? ReplicatesCsv, string? DocumentPath);

    public ExportedReports Export(string workDir)
    {
        Directory.CreateDirectory(workDir);

        var docPath = Try(() => _session.Execute(c => c.GetDocumentPath()));
        _log($"Skyline document: {docPath ?? "(unsaved)"}");
        var version = Try(() => _session.Execute(c => c.GetVersion()));
        if (version is not null)
            _log($"Skyline version: {version}");

        EnsurePrismReport();

        // PRISM report - invariant CSV (the pipeline's primary input).
        var prismCsv = Path.Combine(workDir, "PRISM.csv");
        _session.Execute(c => c.ExportReport("PRISM", prismCsv, "invariant"));
        _log($"Exported PRISM report (invariant CSV): {prismCsv}");

        // PRISM report - parquet (best-effort; also a user deliverable).
        string? prismParquet = Path.Combine(workDir, "PRISM.parquet");
        try
        {
            _session.Execute(c => c.RunCommandSilent(new[]
            {
                "--report-name=PRISM",
                $"--report-file={prismParquet}",
                "--report-format=parquet",
                "--report-invariant",
            }));
            _log($"Exported PRISM report (parquet): {prismParquet}");
        }
        catch (Exception ex)
        {
            _log($"PRISM parquet export skipped: {ex.Message}");
            prismParquet = null;
        }

        // Replicates report - default Skyline report with sample annotations (standards, QC,
        // Batch, etc.). Used as the pipeline's metadata input.
        string? replicatesCsv = Path.Combine(workDir, "Replicates.csv");
        try
        {
            _session.Execute(c => c.ExportReport("Replicates", replicatesCsv, "invariant"));
            _log($"Exported Replicates report: {replicatesCsv}");
        }
        catch (Exception ex)
        {
            _log($"Replicates report export skipped: {ex.Message}");
            replicatesCsv = null;
        }

        return new ExportedReports(prismCsv, prismParquet, replicatesCsv, docPath);
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

    private T? Try<T>(Func<T> f) where T : class
    {
        try { return f(); }
        catch (Exception ex) { _log($"(warning) {ex.Message}"); return null; }
    }
}
