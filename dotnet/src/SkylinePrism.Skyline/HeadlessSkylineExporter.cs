#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;

namespace SkylinePrism.Skyline;

/// <summary>
/// Exports the PRISM reports from a Skyline document that is NOT open in a running Skyline. This is what
/// lets a user hand PRISM several documents - one per batch/plate - without opening and exporting each
/// one by hand.
///
/// <para><b>Two ways to reach Skyline</b>, behind <see cref="ISkylineCommandRunner"/>:
/// <see cref="SkylineAppRunner"/> drives the installed <c>Skyline.exe</c> headlessly (the SkylineRunner
/// mechanism) and is PREFERRED because it can export parquet; <see cref="SkylineCmdRunner"/> is the
/// faster-starting fallback that cannot. See <see cref="ExportTransitionReport"/>.</para>
///
/// <para><b>Format follows the file extension.</b> There is no <c>--report-format=parquet</c> (that flag
/// takes only csv|tsv) - Skyline picks parquet from a <c>.parquet</c> output name. The transition report
/// is requested as parquet when the runner supports it and falls back to invariant CSV otherwise.
/// <c>DuckDbMerge</c> reads CSV and parquet inputs in the same merge, so a run may freely mix open and
/// closed documents whichever way each landed.</para>
///
/// <para><b>Two invocations per document.</b> <c>--report-name</c> takes a single report, so the
/// transition report and the replicate metadata report are exported separately - which means the
/// document is loaded twice. That is the cost of not having it open.</para>
///
/// <para><b>Side effect.</b> <c>--report-add</c> installs the PRISM report definitions into Skyline's
/// saved-report list, which is a per-user program setting shared with the GUI. The live RPC path already
/// does exactly this, so behavior is consistent between the two. Nothing here writes to the .sky - the
/// document is opened with <c>--in</c> and never saved.</para>
/// </summary>
public sealed class HeadlessSkylineExporter
{
    private readonly ISkylineCommandRunner _runner;
    private readonly string _reportsDir;
    private readonly Action<string> _log;

    /// <param name="reportsDir">
    /// Where the bundled .skyr report definitions live; defaults to <c>Reports/</c> next to the executable.
    /// </param>
    public HeadlessSkylineExporter(
        ISkylineCommandRunner runner, Action<string>? log = null, string? reportsDir = null)
    {
        _runner = runner ?? throw new ArgumentNullException(nameof(runner));
        _reportsDir = reportsDir ?? Path.Combine(AppContext.BaseDirectory, "Reports");
        _log = log ?? (_ => { });
    }

    /// <summary>How this exporter is reaching Skyline (for the UI/log).</summary>
    public string RunnerDescription => _runner.Description;

    /// <summary>
    /// The resolved way to drive Skyline. Exposed so other headless jobs can reuse the same discovery
    /// (Skyline app first, SkylineCmd fallback) instead of repeating it - e.g.
    /// <see cref="SkylineIsolationImporter"/>, which reads DIA isolation windows out of a raw data file.
    /// </summary>
    public ISkylineCommandRunner Runner => _runner;

    /// <summary>
    /// Pick the best available way to drive Skyline, or throw a message the UI can show verbatim.
    /// Prefers the installed Skyline application (parquet-capable) over SkylineCmd.
    /// </summary>
    /// <param name="preferCmd">
    /// Prefer <see cref="SkylineCmdRunner"/> over the full application.
    /// <para>
    /// <b>Pass true for anything that uses <c>--new</c>.</b> A scratch-document command HANGS through
    /// the app runner: <c>SkylineDailyRunner --new=x.sky --overwrite --save</c> prints
    /// <c>File x.sky opened.</c> and then nothing, forever, never writing the file - while SkylineCmd
    /// runs the identical arguments in 0.9 s. Reproduced with the official runner (see the field
    /// guide's <c>repro/</c>), so it is not something in our protocol implementation; root cause is
    /// with the Skyline developers.
    /// </para>
    /// <para>
    /// The default is the app runner because it is the only one that can write parquet, and that is
    /// what report export needs. Report export opens an EXISTING document (<c>--in</c>), which the
    /// same runner handles in ~1.6 s - so the two are not in conflict, and the rule is about
    /// <c>--new</c> versus <c>--in</c>, NOT about "settings versus reports".
    /// </para>
    /// <para>
    /// Note the trap this leaves: a future <c>--new</c> command that also wants parquet has no good
    /// option, because SkylineCmd cannot write parquet at all. Reach for the live RPC session on an
    /// open document, or wait for the runner fix.
    /// </para>
    /// </param>
    public static HeadlessSkylineExporter Create(
        string? explicitCmdPath = null, Action<string>? log = null, string? reportsDir = null,
        bool preferCmd = false)
    {
        // An explicit SkylineCmd path is a deliberate override, so honour it ahead of discovery.
        ISkylineCommandRunner? runner = string.IsNullOrWhiteSpace(explicitCmdPath) && !preferCmd
            ? SkylineAppRunner.Find(log)
            : SkylineCmdRunner.Find(explicitCmdPath, log);
        runner ??= preferCmd
            ? (ISkylineCommandRunner?)SkylineAppRunner.Find(log)
            : SkylineCmdRunner.Find(explicitCmdPath, log);

        if (runner is null)
        {
            throw new InvalidOperationException(
                "Could not find an installed Skyline to export with. Install Skyline (or Skyline-daily), "
                + $"or set the {SkylineCmdLocator.OverrideEnvVar} environment variable to the full path of "
                + "SkylineCmd.exe. Documents that are already open in Skyline do not need this.");
        }
        log?.Invoke($"Headless export will use: {runner.Description}"
                    + (runner.SupportsParquet ? " - parquet capable." : " - CSV only."));
        return new HeadlessSkylineExporter(runner, log, reportsDir);
    }

    /// <summary>Sidecar recording what a previous export of this document produced, and from what.</summary>
    /// <param name="Tool">
    /// The PRISM version that wrote it. Without this, a release that changes the bundled
    /// Skyline-PRISM.skyr or the generated Replicates view would reuse an export made with the OLD
    /// report definition - the merge would then read a report missing a column PRISM now expects, while
    /// the log said the document and report were unchanged. StageCache folds the version in for exactly
    /// this reason; the export sidecar has to as well.
    /// </param>
    private sealed record ExportStamp(
        string Document, long Length, long LastWriteUtcTicks, string BatchAnnotation,
        string ReportPath, string? MetadataPath, string Tool);

    private static string ToolVersion =>
        typeof(HeadlessSkylineExporter).Assembly.GetName().Version?.ToString() ?? "0";

    private static string StampPath(string workDir, string label) =>
        Path.Combine(workDir, label + ".export.json");

    /// <summary>
    /// The previous export of this document, when the document and the report definition are unchanged
    /// and the files are still there; null otherwise. Never throws - a bad sidecar means "export again".
    /// </summary>
    private ExportedReports? TryReuseExport(
        string skyPath, string workDir, string label, string? batchAnnotation)
    {
        try
        {
            var stampPath = StampPath(workDir, label);
            if (!File.Exists(stampPath))
                return null;
            var stamp = System.Text.Json.JsonSerializer.Deserialize<ExportStamp>(File.ReadAllText(stampPath));
            if (stamp is null)
                return null;

            var info = new FileInfo(skyPath);
            if (!info.Exists
                || !string.Equals(stamp.Document, info.FullName, StringComparison.OrdinalIgnoreCase)
                || stamp.Length != info.Length
                || stamp.LastWriteUtcTicks != info.LastWriteTimeUtc.Ticks
                || !string.Equals(stamp.BatchAnnotation, batchAnnotation ?? "", StringComparison.Ordinal))
                return null;

            if (!string.Equals(stamp.Tool, ToolVersion, StringComparison.Ordinal))
                return null;

            if (!File.Exists(stamp.ReportPath) || new FileInfo(stamp.ReportPath).Length == 0)
                return null;

            // The metadata report is not optional on reuse. Silently returning null for it would drop
            // the run back to inferring sample types from replicate names and batches from the source
            // document - different reference/QC assignment, different ComBat, different numbers, and no
            // message. If it is gone or empty, export again.
            // A stamp is only ever written WITH a metadata path (see Export), so one lacking it is from
            // an older tool or was hand-edited. Re-export rather than reason about it: that costs one
            // export, and the alternative is inheriting the very gap this guard exists to prevent.
            if (stamp.MetadataPath is null
                || !File.Exists(stamp.MetadataPath)
                || new FileInfo(stamp.MetadataPath).Length == 0)
                return null;
            var metadata = stamp.MetadataPath;

            _log($"Reusing the previous export of {Path.GetFileName(skyPath)} "
                 + "(document and report unchanged since it was written).");
            var isParquet = Path.GetExtension(stamp.ReportPath)
                .Equals(".parquet", StringComparison.OrdinalIgnoreCase);
            // info.FullName, matching what the fresh path returns: a caller that deduplicates inputs by
            // DocumentPath must not see one document as two because one run took the cache.
            return new ExportedReports(stamp.ReportPath, isParquet, metadata, info.FullName, label);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Record what an export produced, so an unchanged document can skip it next time.</summary>
    private static void WriteExportStamp(
        string skyPath, string workDir, string label, string? batchAnnotation, ExportedReports reports)
    {
        try
        {
            var info = new FileInfo(skyPath);
            var stamp = new ExportStamp(
                info.FullName, info.Length, info.LastWriteTimeUtc.Ticks, batchAnnotation ?? "",
                reports.InputPath, reports.ReplicatesCsv, ToolVersion);
            File.WriteAllText(
                StampPath(workDir, label),
                System.Text.Json.JsonSerializer.Serialize(stamp));
        }
        catch
        {
            // Losing the stamp costs an export next time; failing here costs the run.
        }
    }

    /// <summary>
    /// Export the PRISM transition report and the replicate metadata report for <paramref name="skyPath"/>
    /// into <paramref name="workDir"/>, named after <paramref name="documentLabel"/> so the merge derives
    /// the right Source Document / batch label from each file stem.
    /// </summary>
    /// <param name="batchAnnotation">
    /// Replicate annotation carrying the batch/plate, forced into the metadata report even if it is not one
    /// of the document's declared annotations.
    /// </param>
    public ExportedReports Export(
        string skyPath, string workDir, string? documentLabel = null,
        string? batchAnnotation = null, CancellationToken cancellationToken = default)
    {
        if (!File.Exists(skyPath))
            throw new FileNotFoundException($"Skyline document not found: {skyPath}", skyPath);
        Directory.CreateDirectory(workDir);

        var label = string.IsNullOrWhiteSpace(documentLabel)
            ? Path.GetFileNameWithoutExtension(skyPath)
            : documentLabel!;

        // Reuse the previous export when nothing about it could differ. This is the single most
        // expensive step of a re-run - a large cohort's transition report is tens of GB - and re-doing
        // it to change a downstream setting is pure waste.
        //
        // Safe here because the document is CLOSED: a file cannot change without its size or
        // last-write-time changing. The running-Skyline path deliberately has no equivalent, because a
        // live document can hold unsaved edits that the .sky on disk knows nothing about.
        if (TryReuseExport(skyPath, workDir, label, batchAnnotation) is { } reused)
            return reused;

        // Read the header for the replicate annotation names, so the generated metadata report carries the
        // same columns the Replicates grid would show for an open document.
        var info = SkyDocumentInfo.TryRead(skyPath, _log);
        var annotations = new List<string>();
        if (info is not null)
        {
            annotations.AddRange(info.ReplicateAnnotationNames);
            _log($"{info.Replicates.Count} replicate(s); annotations: "
                 + (info.ReplicateAnnotationNames.Count > 0
                     ? string.Join(", ", info.ReplicateAnnotationNames)
                     : "(none)"));
        }
        if (!string.IsNullOrWhiteSpace(batchAnnotation))
            annotations.Add(batchAnnotation!);

        var metadataCsv = Path.Combine(workDir, SkylineReportDriver.MetadataFileName(label));

        // 1. Transition report (the bundled PRISM.skyr is fixed, so install and export in one load).
        var prismSkyr = Path.Combine(_reportsDir, "Skyline-PRISM.skyr");
        if (!File.Exists(prismSkyr))
        {
            // Not bundled next to the executable: fall back to whatever "PRISM" report is already in the
            // user's Skyline settings (usually installed by an earlier run against a live document).
            _log($"PRISM report definition not found at {prismSkyr}; "
                 + "assuming a 'PRISM' report is already installed in Skyline.");
            prismSkyr = null;
        }

        var (prismPath, isParquet) = ExportTransitionReport(skyPath, workDir, label, prismSkyr, cancellationToken);

        // 2. Replicate metadata: generate a .skyr carrying this document's annotation columns, then export.
        string? metadataResult = null;
        try
        {
            var replicatesSkyr = Path.Combine(workDir, label + ".PRISM-Replicates.skyr");
            ReplicatesReportBuilder.WriteSkyr(replicatesSkyr, annotations);
            var metadataSidecar = SidecarFor(metadataCsv);
            try
            {
                _runner.Run(
                    BuildArgs(skyPath, replicatesSkyr, ReplicatesReportBuilder.ViewName, metadataSidecar),
                    _log, cancellationToken);
            }
            catch
            {
                TryDelete(metadataSidecar);
                throw;
            }
            if (HasContent(metadataSidecar) && TryPromote(metadataSidecar, metadataCsv))
            {
                metadataResult = metadataCsv;
                _log($"Exported replicate metadata to {metadataCsv}.");
            }
            else
            {
                TryDelete(metadataSidecar);
                _log($"The replicate metadata report produced no file; "
                     + "sample types will be inferred from replicate names.");
            }
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            // Metadata is optional - the pipeline can still run and infer sample types from names.
            _log($"Replicate metadata export failed ({ex.Message}); "
                 + "sample types will be inferred from replicate names.");
        }

        var exported = new ExportedReports(
            prismPath, isParquet, metadataResult, Path.GetFullPath(skyPath), label);
        // Stamped only now, with the files written: a stamp recorded earlier would vouch for an export
        // that a cancellation or a SkylineCmd failure left half-written.
        //
        // And not stamped AT ALL when the metadata is missing, or the next run of an unchanged document
        // reuses this one and inherits the gap permanently: TryReuseExport only re-exports when a stamped
        // metadata file has gone missing, so a stamp saying there was never any metadata reads as
        // "correctly has none" and is honoured forever. Sample types would then come from replicate names
        // on every future run - a different reference/QC split, different ComBat, and nothing in the log
        // but "Reusing the previous export". Losing the transition report's cache costs one re-export;
        // this costs the numbers.
        if (metadataResult is null)
        {
            _log("Not recording this export for reuse: the replicate metadata is missing, and a cached "
                 + "export without it would keep inferring sample types from replicate names.");
            return exported;
        }
        WriteExportStamp(skyPath, workDir, label, batchAnnotation, exported);
        return exported;
    }

    /// <summary>
    /// Export the transition report, preferring parquet (typed, and ~15x smaller than the CSV, so the
    /// merge is much faster) and falling back to invariant CSV.
    ///
    /// <para>Skyline picks the report format from the output file's EXTENSION - there is no
    /// <c>--report-format=parquet</c>, that flag only accepts csv|tsv. Whether the extension actually
    /// yields parquet depends on WHICH host runs the command: the full Skyline application can
    /// (<see cref="SkylineAppRunner"/>), while <c>SkylineCmd.exe</c> cannot, because its config file omits
    /// the Parquet.Net assembly bindings that <c>Skyline.exe.config</c> carries (the managed assembly ships
    /// as <c>ParquetNet.dll</c> and needs a <c>codeBase</c>, since a NATIVE <c>parquet.dll</c> occupies the
    /// default probe path) and it dies with "Could not load file or assembly 'Parquet'".</para>
    ///
    /// <para>So parquet is attempted whenever the runner claims support, and the result is still verified
    /// rather than trusted - it must be parquet by its PAR1 magic AND be a file this run actually wrote
    /// (<see cref="ClearPreviousExport"/>). If anything goes wrong we silently produce CSV, which the
    /// pipeline handles identically; if THAT fails too, the export fails loudly rather than quietly
    /// reporting whatever was left at the path.</para>
    /// </summary>
    private (string Path, bool IsParquet) ExportTransitionReport(
        string skyPath, string workDir, string label, string? prismSkyr, CancellationToken cancellationToken)
    {
        _log($"Exporting the PRISM transition report via {_runner.Description} "
             + "(this can take a while on a large document)...");

        var parquet = Path.Combine(workDir, label + ".parquet");
        if (_runner.SupportsParquet)
        {
            // Skyline writes to a sidecar this run owns, never to the destination. See SidecarFor.
            var sidecar = SidecarFor(parquet);
            try
            {
                // No --report-format: the extension selects parquet.
                _runner.Run(BuildArgs(skyPath, prismSkyr, "PRISM", sidecar, format: null), _log, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                TryDelete(sidecar);
                throw;
            }
            catch (Exception ex)
            {
                _log($"Parquet export failed ({ex.Message}); falling back to CSV.");
            }

            if (HasContent(sidecar) && ParquetMagic.IsValid(sidecar) && TryPromote(sidecar, parquet))
            {
                _log($"Exported {parquet} ({new FileInfo(parquet).Length:N0} bytes, parquet).");
                // A CSV from a previous run that fell back is superseded by this parquet; the two are
                // alternatives for the same input, and a large cohort's CSV is ~20x the parquet.
                var superseded = Path.Combine(workDir, label + ".csv");
                if (File.Exists(superseded))
                {
                    var bytes = new FileInfo(superseded).Length;
                    TryDelete(superseded);
                    if (!File.Exists(superseded))
                        _log($"Removed the superseded CSV export {superseded} ({bytes:N0} bytes).");
                }
                return (parquet, true);
            }
            TryDelete(sidecar); // a failed run can leave a stub behind; the destination is untouched
        }

        var csv = Path.Combine(workDir, label + ".csv");
        var csvSidecar = SidecarFor(csv);
        try
        {
            _runner.Run(BuildArgs(skyPath, prismSkyr, "PRISM", csvSidecar, format: "csv"), _log, cancellationToken);
        }
        catch
        {
            TryDelete(csvSidecar);
            throw;
        }
        if (!HasContent(csvSidecar) || !TryPromote(csvSidecar, csv))
        {
            TryDelete(csvSidecar);
            throw new InvalidOperationException(
                $"Skyline did not produce a PRISM report for {Path.GetFileName(skyPath)}. "
                + "See the log above for its output.");
        }
        _log($"Exported {csv} ({new FileInfo(csv).Length:N0} bytes, invariant CSV).");
        return (csv, false);
    }

    /// <summary>
    /// Where Skyline is told to write, so its output is distinguishable from whatever is already at
    /// the destination. The extension is preserved, because it is what selects the export format.
    ///
    /// <para><b>Why not write straight to the destination.</b> The question that has to be answered
    /// after an export is "did THIS run write this file", and answering it at the destination needs a
    /// proxy. Accepting <see cref="ParquetMagic.IsValid"/> alone asked only whether the bytes there
    /// are parquet: when an export failed before Skyline received its arguments, the file a PREVIOUS
    /// export left was still there, still valid, and was logged as "Exported ... bytes, parquet" and
    /// handed to the merge - a 2-plate cohort silently analyzed a report exported 25 days earlier,
    /// from a version of the .sky that had since been re-integrated.</para>
    ///
    /// <para>Deleting the destination first and comparing size and mtime afterwards fixed that, but
    /// cost two things. The delete destroyed a usable cached export whenever the re-export then failed
    /// - the most expensive step of a re-run, thrown away on a transient error. And size+mtime is not
    /// an identity: the delete is best-effort, so on a share that refuses it, a re-export of an
    /// UNCHANGED document is byte-identical, and on a filesystem with second-granular timestamps it
    /// reads as "not replaced" - failing an export that in fact just succeeded.</para>
    ///
    /// <para>A sidecar answers the question directly instead. This run named the path, so anything
    /// there is this run's output; and the destination is not touched until there is something good to
    /// put in it, so a failed export leaves the previous one usable.</para>
    /// </summary>
    /// <remarks>
    /// The whole file name is kept and the marker goes in FRONT of it. Appending before the extension
    /// would have turned "PlateA.metadata.csv" into "PlateA.metadata.prism-partial-1a2b.csv", which no
    /// longer ends in ".metadata.csv" - and the report kind is told apart by exactly that suffix. Skyline
    /// also picks the export format from the extension, which this keeps intact either way.
    /// </remarks>
    private static string SidecarFor(string finalPath) =>
        Path.Combine(
            Path.GetDirectoryName(finalPath) ?? ".",
            ".prism-partial-" + Guid.NewGuid().ToString("N")[..8] + "." + Path.GetFileName(finalPath));

    private static bool HasContent(string path)
    {
        try
        {
            var info = new FileInfo(path);
            return info.Exists && info.Length > 0;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// Move a finished export into place, replacing any previous one. Returns false - leaving the
    /// previous export where it is - rather than throwing, so a destination that cannot be written
    /// falls through to the caller's own failure path with the cache still intact.
    /// </summary>
    private bool TryPromote(string sidecar, string finalPath)
    {
        var previous = HasContent(finalPath) ? new FileInfo(finalPath).Length : 0L;
        try
        {
            File.Move(sidecar, finalPath, overwrite: true);
            if (previous > 0)
                _log($"Replaced the previous export at {finalPath} ({previous:N0} bytes).");
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            _log($"Could not put the new export in place at {finalPath} ({ex.Message}); "
                 + "the previous export there is unchanged.");
            return false;
        }
    }

    private void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            _log($"(could not remove {path}: {ex.Message})");
        }
    }

    /// <summary>
    /// One SkylineCmd invocation: open the document, install the report definition (when we have one), and
    /// export it. No <c>--save</c>, so the document itself is never modified.
    /// </summary>
    /// <param name="format">
    /// <c>csv</c> / <c>tsv</c>, or null to let the output file's EXTENSION choose - the only way to ask for
    /// parquet, since <c>--report-format</c> accepts nothing else.
    /// </param>
    public static string[] BuildArgs(
        string skyPath, string? skyrPath, string reportName, string outPath, string? format = "csv")
    {
        var args = new List<string> { $"--in={skyPath}" };
        if (!string.IsNullOrEmpty(skyrPath))
        {
            args.Add($"--report-add={skyrPath}");
            args.Add("--report-conflict-resolution=overwrite");
        }
        args.Add($"--report-name={reportName}");
        args.Add($"--report-file={outPath}");
        if (!string.IsNullOrEmpty(format))
            args.Add($"--report-format={format}");
        args.Add("--report-invariant"); // invariant numbers, per the culture rule
        return args.ToArray();
    }

}
