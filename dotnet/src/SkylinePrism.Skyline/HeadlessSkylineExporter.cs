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
            _runner.Run(
                BuildArgs(skyPath, replicatesSkyr, ReplicatesReportBuilder.ViewName, metadataCsv),
                _log, cancellationToken);
            if (File.Exists(metadataCsv) && new FileInfo(metadataCsv).Length > 0)
            {
                metadataResult = metadataCsv;
                _log($"Exported replicate metadata to {metadataCsv}.");
            }
            else
            {
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

        return new ExportedReports(prismPath, isParquet, metadataResult, Path.GetFullPath(skyPath), label);
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
    /// by its PAR1 magic rather than trusted - if anything goes wrong we silently produce CSV, which the
    /// pipeline handles identically.</para>
    /// </summary>
    private (string Path, bool IsParquet) ExportTransitionReport(
        string skyPath, string workDir, string label, string? prismSkyr, CancellationToken cancellationToken)
    {
        _log($"Exporting the PRISM transition report via {_runner.Description} "
             + "(this can take a while on a large document)...");

        var parquet = Path.Combine(workDir, label + ".parquet");
        if (_runner.SupportsParquet)
        {
            try
            {
                // No --report-format: the extension selects parquet.
                _runner.Run(BuildArgs(skyPath, prismSkyr, "PRISM", parquet, format: null), _log, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                _log($"Parquet export failed ({ex.Message}); falling back to CSV.");
            }

            if (ParquetMagic.IsValid(parquet))
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
            TryDelete(parquet); // a failed run can leave a stub behind
        }

        var csv = Path.Combine(workDir, label + ".csv");
        _runner.Run(BuildArgs(skyPath, prismSkyr, "PRISM", csv, format: "csv"), _log, cancellationToken);
        if (!File.Exists(csv) || new FileInfo(csv).Length == 0)
            throw new InvalidOperationException(
                $"Skyline did not produce a PRISM report for {Path.GetFileName(skyPath)}. "
                + "See the log above for its output.");
        _log($"Exported {csv} ({new FileInfo(csv).Length:N0} bytes, invariant CSV).");
        return (csv, false);
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
