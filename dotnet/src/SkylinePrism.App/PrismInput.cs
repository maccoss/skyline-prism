using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using SkylinePrism.Skyline;

namespace SkylinePrism.App;

/// <summary>Where one PRISM input comes from.</summary>
public enum PrismInputKind
{
    /// <summary>The Skyline instance that launched this tool (or another running instance), over JSON-RPC.</summary>
    RunningSkyline,

    /// <summary>A .sky file that is not open anywhere; exported headlessly with SkylineCmd.</summary>
    ClosedDocument,

    /// <summary>An already-exported PRISM report (parquet/CSV/TSV) - no Skyline involved at all.</summary>
    ReportFile,
}

/// <summary>
/// One input row in the tool's Inputs list: a Skyline document (open or closed) or a pre-exported report,
/// which <see cref="Prepare"/> resolves to the report + metadata files the pipeline consumes.
///
/// <para>Several inputs may be combined in a single run - that is how multiple batches held in multiple
/// Skyline documents are processed together without exporting each by hand. Each input contributes its
/// <see cref="BatchLabel"/> as the Source Document / Batch label; the merge stamps it into every sample ID
/// ("&lt;replicate&gt;__@__&lt;batch&gt;"), which is what keeps identically-named reference/QC injections from
/// different documents distinct.</para>
/// </summary>
public sealed class PrismInput : INotifyPropertyChanged
{
    private string _batchLabel = "";
    private string _status = "";

    private PrismInput(PrismInputKind kind, string displayName, string batchLabel)
    {
        Kind = kind;
        DisplayName = displayName;
        _batchLabel = batchLabel;
    }

    public PrismInputKind Kind { get; }

    /// <summary>What the user sees in the Source column (file name, or the open document's name).</summary>
    public string DisplayName { get; }

    /// <summary>Full path to the .sky / report file, when there is one ("" for an unsaved open document).</summary>
    public string Path { get; private init; } = "";

    /// <summary>Set for <see cref="PrismInputKind.RunningSkyline"/>: the live RPC session to export from.</summary>
    public SkylineSession? Session { get; private init; }

    /// <summary>
    /// Batch / Source Document label. Editable in the grid, and used as the exported file stem so
    /// <c>DuckDbMerge</c> derives the same label. Must be unique and file-name safe within a run.
    /// </summary>
    public string BatchLabel
    {
        get => _batchLabel;
        set => Set(ref _batchLabel, value ?? "");
    }

    /// <summary>Free-text progress/result shown in the grid ("exported 1.2 GB", "queued", an error).</summary>
    public string Status
    {
        get => _status;
        set => Set(ref _status, value ?? "");
    }

    /// <summary>Human-readable source kind for the grid.</summary>
    /// <summary>
    /// Whether this input is a <c>.sky.zip</c>, which has to be extracted before Skyline's command
    /// line can open it - see <see cref="SharedDocumentArchive"/>.
    /// </summary>
    public bool IsSharedArchive =>
        Kind == PrismInputKind.ClosedDocument && SharedDocumentArchive.IsArchive(Path);

    public string KindLabel => Kind switch
    {
        PrismInputKind.RunningSkyline => "Open in Skyline",
        PrismInputKind.ClosedDocument => "Skyline document",
        _ => "Report file",
    };

    public static PrismInput FromRunningSkyline(SkylineSession session, string? documentPath, string? displayName = null)
    {
        var name = displayName
            ?? (string.IsNullOrWhiteSpace(documentPath)
                ? "(unsaved document)"
                : System.IO.Path.GetFileNameWithoutExtension(documentPath));
        return new PrismInput(PrismInputKind.RunningSkyline, name, SanitizeLabel(name))
        {
            Path = documentPath ?? "",
            Session = session,
        };
    }

    /// <param name="skyPath">
    /// A <c>.sky</c>, or a <c>.sky.zip</c> shared document archive as downloaded from PanoramaWeb.
    /// The label drops <b>both</b> extensions, so an archive and the document inside it produce the
    /// same batch label - <c>Path.GetFileNameWithoutExtension</c> alone would leave a trailing
    /// <c>.sky</c> in every sample ID.
    /// </param>
    public static PrismInput FromClosedDocument(string skyPath)
    {
        var name = SharedDocumentArchive.StemOf(skyPath);
        return new PrismInput(PrismInputKind.ClosedDocument, System.IO.Path.GetFileName(skyPath), SanitizeLabel(name))
        {
            Path = System.IO.Path.GetFullPath(skyPath),
        };
    }

    /// <param name="metadataPath">Optional replicate metadata CSV exported alongside the report.</param>
    public static PrismInput FromReportFile(string reportPath, string? metadataPath = null)
    {
        var name = System.IO.Path.GetFileNameWithoutExtension(reportPath);
        return new PrismInput(PrismInputKind.ReportFile, System.IO.Path.GetFileName(reportPath), SanitizeLabel(name))
        {
            Path = System.IO.Path.GetFullPath(reportPath),
            MetadataPath = metadataPath is null ? null : System.IO.Path.GetFullPath(metadataPath),
        };
    }

    /// <summary>For <see cref="PrismInputKind.ReportFile"/>: a metadata CSV the user picked explicitly.</summary>
    public string? MetadataPath { get; set; }

    /// <summary>
    /// Produce this input's report + metadata files under <paramref name="reportsDir"/>. Exports from
    /// Skyline when needed; a pre-exported report is used in place (no copy).
    /// </summary>
    /// <param name="skylineCmdPath">Optional explicit SkylineCmd.exe for the closed-document path.</param>
    public ExportedReports Prepare(
        string reportsDir, string? metadataReportName, string? batchAnnotation,
        string? skylineCmdPath, Action<string> log, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        // StemOf, not the display name as it stands: DisplayName is the file NAME (so the Inputs grid
        // can show "Plate1.sky.zip" and not leave the user guessing which kind it is), and a batch
        // label ending in ".sky.zip" would carry that into every merged sample ID.
        var label = string.IsNullOrWhiteSpace(BatchLabel)
            ? SanitizeLabel(SharedDocumentArchive.StemOf(DisplayName))
            : SanitizeLabel(BatchLabel);

        // The two report variants get their own directories, because the exported file is named after
        // the batch label (the merge derives Batch from the file stem) and so cannot carry the variant
        // in its name. Sharing one path meant switching the measure back OVERWROTE the ion-count
        // export - four hours of Skyline, unrecoverable without paying again - and the export cache
        // could only ever hold one of the two. The stem, and therefore every batch label, is unchanged.
        // Every line produced while preparing THIS input is tagged with its document. Inputs are exported
        // concurrently, and the deepest lines come from Skyline's own console ("Opening file...", "2%"),
        // which say nothing about which document they belong to - two documents at once produce a stream
        // of identical-looking pairs. Tagging here rather than inside each exporter covers all three input
        // kinds, plus the runner output and the Skyline-selection messages, from one place.
        var scoped = Scoped(log, label);

        // Captured before the ion-count rewrite below moves reportsDir into a subdirectory: an
        // extraction that cannot go beside its archive belongs under the OUTPUT directory, not nested
        // inside the reports folder - and nesting it made toggling the measure extract 17 GB twice.
        var outputDir = System.IO.Path.GetDirectoryName(reportsDir);

        switch (Kind)
        {
            case PrismInputKind.RunningSkyline:
            {
                var session = Session
                    ?? throw new InvalidOperationException($"{DisplayName}: no Skyline connection for this input.");
                var driver = new SkylineReportDriver(session, scoped);
                // Deliberately never reused. An open document can carry edits that have not been
                // saved, so the file on disk does not describe what Skyline would export - and there
                // is no way to ask: neither the tool service nor the JSON-RPC surface reports a
                // document hash, a revision or a modified flag. Exporting is the only honest answer.
                return driver.Export(reportsDir, metadataReportName, batchAnnotation, label);
            }

            case PrismInputKind.ClosedDocument:
            {
                // A closed document cannot have changed without its file changing, so an export whose
                // document and settings are unchanged is still current - and exporting is minutes of
                // Skyline per document, plus an extraction for a .sky.zip. Recorded in the run's own
                // stage_cache.json, as one more stage beside the merge and the rollups.
                var stage = ExportStageId(label);
                var fingerprint = ExportFingerprint(Path, label, metadataReportName, batchAnnotation);
                var cacheable = outputDir is not null && fingerprint is not null;
                if (cacheable
                    && TryReuseExport(outputDir!, stage, fingerprint!, Path!, label) is { } reused)
                {
                    scoped($"Reusing the report already exported from {DisplayName} - the document "
                        + "has not changed since it was written.");
                    return reused;
                }

                var document = ResolveDocumentForExport(outputDir, scoped, cancellationToken);
                var exporter = HeadlessSkylineExporter.Create(skylineCmdPath, scoped);
                var exported = exporter.Export(
                    document, reportsDir, label, batchAnnotation, cancellationToken);
                // After the export, never before: an entry written first would survive a failure and
                // vouch for a report that was never finished. StageCache.Record claims only what exists.
                if (cacheable)
                    RecordExport(outputDir!, stage, fingerprint!, exported);
                return exported;
            }

            default:
            {
                if (!File.Exists(Path))
                    throw new FileNotFoundException($"Report file not found: {Path}", Path);
                var isParquet = System.IO.Path.GetExtension(Path)
                    .Equals(".parquet", StringComparison.OrdinalIgnoreCase);
                scoped($"Using the existing report {Path}"
                    + (MetadataPath is not null ? $" with metadata {MetadataPath}" : " (no metadata file)"));
                return new ExportedReports(Path, isParquet, MetadataPath, null, label);
            }
        }
    }

    /// <summary>
    /// The document Skyline is actually asked to open: this input's path, or - for a
    /// <c>.sky.zip</c> - the document inside its extraction, made on the first run and reused after.
    ///
    /// <para>A shared archive cannot be handed to Skyline as it stands: its command line XML-parses
    /// <c>--in</c> directly, so a <c>.sky.zip</c> fails with a generic "does not appear to be a
    /// Skyline document". Only the GUI extracts, which is why a document already OPEN in Skyline
    /// needs none of this - it was extracted on the way in.</para>
    /// </summary>
    /// <param name="outputDir">
    /// The run's output directory - where an extraction goes when the archive's own folder cannot be
    /// written to. Null is allowed; the extraction then falls back to the temp directory.
    /// </param>
    internal string ResolveDocumentForExport(
        string? outputDir, Action<string> log, CancellationToken cancellationToken) =>
        IsSharedArchive
            ? SharedDocumentArchive.Extract(Path, outputDir, log, cancellationToken)
            : Path;

    /// <summary>
    /// The stage id an input's export is recorded under. One per batch label, because that is what
    /// names the exported file and therefore what a second document would collide with.
    /// </summary>
    internal static string ExportStageId(string label) => "export." + label;

    /// <summary>
    /// Serializes this pane's reads and writes of <c>stage_cache.json</c>.
    /// </summary>
    /// <remarks>
    /// Inputs are exported CONCURRENTLY (<c>Parallel.For</c> over the input list), and
    /// <see cref="StageCache"/> is a read-whole-file, write-whole-file sidecar: two workers that each
    /// loaded it before either recorded would each write back a snapshot taken before the other's
    /// entry existed, so the second write erased the first - and the next run re-exported whichever
    /// document lost, silently. Loading INSIDE this lock rather than once per worker is the point: a
    /// shared instance would still be a stale snapshot by the time the second worker wrote it.
    /// </remarks>
    private static readonly object ExportCacheLock = new();

    /// <summary>
    /// The reports an already-recorded export claimed, or null when there is nothing to stand on.
    /// </summary>
    internal static ExportedReports? TryReuseExport(
        string outputDir, string stage, string fingerprint, string documentPath, string label)
    {
        lock (ExportCacheLock)
        {
            var cache = StageCache.Load(outputDir);
            // CanReuse has checked the entry's fingerprint and that its files exist and are not empty.
            if (!cache.CanReuse(stage, fingerprint))
                return null;

            var recorded = cache.OutputsOf(stage);
            if (recorded.Count == 0)
                return null;

            // Recorded relative to the output directory (StageCache.Relative), which is what lets one
            // machine read what another wrote; Path.Combine returns an absolute entry unchanged.
            var report = System.IO.Path.Combine(outputDir, recorded[0]);
            var metadata = recorded.Count > 1 ? System.IO.Path.Combine(outputDir, recorded[1]) : null;
            return new ExportedReports(
                report,
                System.IO.Path.GetExtension(report).Equals(".parquet", StringComparison.OrdinalIgnoreCase),
                metadata,
                documentPath,
                label);
        }
    }

    /// <summary>Record a finished export, against the cache as it stands at this moment.</summary>
    internal static void RecordExport(
        string outputDir, string stage, string fingerprint, ExportedReports exported)
    {
        lock (ExportCacheLock)
        {
            StageCache.Load(outputDir)
                .Record(stage, fingerprint, exported.InputPath, exported.ReplicatesCsv);
        }
    }

    /// <summary>
    /// What makes an already-exported report still current: the document, plus everything about the
    /// export that decides the file's content. Null when the document cannot be stamped, which means
    /// "export it" - the safe direction.
    /// </summary>
    /// <remarks>
    /// The document's NAME, size and last-write time, not its full path. The same document on a share
    /// is <c>Z:\...</c> from one machine and <c>Y:\...</c> from another, and a full path would re-export
    /// a report that is already correct - which is the whole case this exists for. Size and write time
    /// are properties of the file itself and read the same from either machine. The PRISM version is in
    /// it because a change to what PRISM asks Skyline for changes the report without touching anything
    /// here.
    /// </remarks>
    internal static string? ExportFingerprint(
        string? documentPath, string label, string? metadataReportName, string? batchAnnotation)
    {
        if (string.IsNullOrWhiteSpace(documentPath))
            return null;
        try
        {
            var info = new FileInfo(documentPath);
            if (!info.Exists)
                return null;
            return string.Join('|',
                System.IO.Path.GetFileName(documentPath), info.Length, info.LastWriteTimeUtc.Ticks,
                label, metadataReportName ?? "", batchAnnotation ?? "", PrismVersion.Current);
        }
        catch (Exception ex) when (ex is IOException or ArgumentException or NotSupportedException
                                       or UnauthorizedAccessException or System.Security.SecurityException)
        {
            return null;
        }
    }

    /// <summary>
    /// Prefix every message with <paramref name="label"/> so interleaved output from concurrent exports
    /// stays attributable. Blank lines are passed through untouched so they still separate sections.
    /// </summary>
    public static Action<string> Scoped(Action<string> log, string label) =>
        message => log(string.IsNullOrWhiteSpace(message) ? message : $"[{label}] {message}");

    /// <summary>
    /// The digestion enzyme from this input's document, mapped to a PRISM enzyme name, or null when it is
    /// unavailable or has no PRISM equivalent (the caller then keeps the configured default). Read over the
    /// RPC for an open document and straight from the .sky header for a closed one; a bare report file has
    /// no document to ask.
    /// </summary>
    public string? TryGetDigestionEnzyme(Action<string> log)
    {
        try
        {
            return Kind switch
            {
                PrismInputKind.RunningSkyline when Session is not null =>
                    new SkylineReportDriver(Session, log).GetDigestionEnzyme(),
                PrismInputKind.ClosedDocument => SkyDocumentInfo.TryRead(Path, log)?.PrismEnzyme,
                _ => null,
            };
        }
        catch (Exception ex)
        {
            log($"({DisplayName}: could not read the digestion enzyme: {ex.Message})");
            return null;
        }
    }

    private readonly object _documentBytesLock = new();
    private long _documentBytes;
    private (long Length, long Ticks) _documentBytesStamp;

    /// <summary>
    /// How big the DOCUMENT is, for the export memory budget: the <c>.sky</c>'s length, or - for a
    /// <c>.sky.zip</c> - the uncompressed length of the document inside it, since that is what a
    /// headless Skyline loads. 0 for an input that is not exported, or one that cannot be sized.
    ///
    /// <para>Remembered per (length, last-write-time) of the input file, and a failure is not
    /// remembered at all, and for a sharper reason than caching usually has. A
    /// zero here does not read as "unknown", it reads as "small": the budget falls back to its floor
    /// and may start four concurrent exports of a document that needs ~9 GB each, which is exactly
    /// the memory exhaustion the budget exists to prevent, and a starved Skyline does not recover.
    /// An archive added while it is still downloading answers 0 once; it must not answer 0 forever.</para>
    /// </summary>
    public long DocumentBytes()
    {
        if (Kind == PrismInputKind.ReportFile || string.IsNullOrEmpty(Path))
            return 0;
        lock (_documentBytesLock)
        {
            var stamp = FileStamp();
            if (_documentBytes > 0 && stamp == _documentBytesStamp)
                return _documentBytes;
            try
            {
                _documentBytes = IsSharedArchive
                    ? SharedDocumentArchive.DocumentBytes(Path)
                    : new FileInfo(Path) is { Exists: true } info ? info.Length : 0;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                _documentBytes = 0;   // unsizable: the budget's floor still applies
            }
            // Remember WHICH file was measured only when the measurement answered.
            _documentBytesStamp = _documentBytes > 0 ? stamp : default;
            return _documentBytes;
        }
    }

    /// <summary>
    /// The m/z range Skyline extracted each product ion over, from the document's
    /// <c>&lt;transition_full_scan&gt;</c> settings - what the MS2 signal accounting needs to decide when
    /// two transitions read the same detector counts. Null when this input cannot say: a pre-exported
    /// report carries no settings, an unsaved live document has no file to read, and a document with no
    /// full-scan settings has no tolerance. The caller then keeps the configured value.
    ///
    /// <para>For a live Skyline the SAVED document is read, the way the closed-document path reads it:
    /// there is no RPC for the transition settings, and the extraction tolerance is not something that
    /// is edited between saves the way peak boundaries are.</para>
    /// </summary>
    public ProductMassTolerance? TryGetExtractionTolerance(Action<string> log) =>
        TryGetExtractionTolerances(log).Product;

    /// <summary>
    /// Where this input's instrument files are, from the document's own record of where they were
    /// imported from.
    /// </summary>
    /// <remarks>
    /// <para>A Skyline document stores the <c>file_path</c> of every <c>&lt;sample_file&gt;</c> it
    /// imported, so the usual case needs no guessing at all - the answer is written down. Asking the
    /// user to find a directory the document already names is work they should not have to do.</para>
    ///
    /// <para>Falls back through <see cref="SkylineIsolationImporter.ResolveDataFile"/>, which tries
    /// the recorded path, then beside the document, then one directory above it - the layouts that
    /// cover a moved cohort, a <c>.sky.zip</c> extracted into a subfolder, and a document kept below
    /// its acquisition directory. Returns null when nothing resolves, and the caller then asks.</para>
    ///
    /// <para>The directory of the FIRST file that resolves, not a directory per replicate: ion
    /// accounting takes one raw directory for the cohort, and a document whose replicates live in
    /// several places cannot be expressed to it anyway.</para>
    /// </remarks>
    public string? GuessRawDirectory(Action<string> log)
    {
        try
        {
            var documentPath = Kind switch
            {
                PrismInputKind.RunningSkyline when Session is not null =>
                    Session.Execute(c => c.GetDocumentPath()),
                PrismInputKind.ClosedDocument => Path,
                _ => null,
            };
            if (string.IsNullOrWhiteSpace(documentPath) || !File.Exists(documentPath))
                return null;

            var info = SkyDocumentInfo.TryRead(documentPath, log);
            if (info is null || info.SampleFilePaths.Count == 0)
                return null;

            var resolved = SkylineIsolationImporter.ResolveDataFile(
                info.SampleFilePaths, documentPath);
            if (resolved is null)
                return null;

            // A Bruker/Agilent acquisition IS a directory, so the containing directory is what is
            // wanted either way - GetDirectoryName of a directory path gives its parent.
            return System.IO.Path.GetDirectoryName(
                resolved.TrimEnd(
                    System.IO.Path.DirectorySeparatorChar,
                    System.IO.Path.AltDirectorySeparatorChar));
        }
        catch (Exception ex)
        {
            log($"({DisplayName}: could not work out where the data files are: {ex.Message})");
            return null;
        }
    }

    /// <summary>
    /// Both extraction windows the document states, product and precursor.
    /// </summary>
    /// <remarks>
    /// Ion accounting wants both: the product window decides when two fragments read the same
    /// detector counts, and the precursor window does the same for the MS1 half. Reading them
    /// together costs one parse of the document header rather than two, and keeps them from
    /// disagreeing about which document they came from.
    ///
    /// <para>Either may be null on its own - a document can state a product analyzer and no
    /// precursor one - and the caller must treat a null precursor as "compute the MS2 half only"
    /// rather than substituting the product value, because a guessed window changes how much sharing
    /// is found with nothing on the plot to say the number moved.</para>
    /// </remarks>
    public (ProductMassTolerance? Product, ProductMassTolerance? Precursor)
        TryGetExtractionTolerances(Action<string> log)
    {
        try
        {
            var documentPath = Kind switch
            {
                PrismInputKind.RunningSkyline when Session is not null =>
                    Session.Execute(c => c.GetDocumentPath()),
                PrismInputKind.ClosedDocument => Path,
                _ => null,
            };
            if (string.IsNullOrWhiteSpace(documentPath) || !File.Exists(documentPath))
                return (null, null);

            var info = SkyDocumentInfo.TryRead(documentPath, log);
            return (info?.ProductTolerance, info?.PrecursorTolerance);
        }
        catch (Exception ex)
        {
            log($"({DisplayName}: could not read the extraction settings: {ex.Message})");
            return (null, null);
        }
    }

    private readonly object _ionCountLock = new();

    private (long Length, long Ticks) FileStamp()
    {
        try
        {
            var info = new FileInfo(Path);
            return info.Exists ? (info.Length, info.LastWriteTimeUtc.Ticks) : default;
        }
        catch
        {
            return default;
        }
    }

    /// <summary>
    /// The column names of an exported report: the parquet schema, or the first line of a CSV/TSV.
    ///
    /// <para><b>The delimiter is chosen by EXTENSION</b>, deliberately the same rule as
    /// <c>DuckDbMerge.ReadHeader</c>, which is what will actually read this file. Sniffing the header
    /// for a tab instead - the obvious thing - lets the two disagree about the same file: a
    /// comma-separated report saved as <c>.txt</c> would have its columns read here and not by the
    /// merge, so the Settings tab would offer a measure the run then cannot deliver. The parquet half
    /// reads the footer through Parquet.Net rather than DuckDB, because this is asked interactively.</para>
    /// </summary>
    internal static IReadOnlyList<string> ReadReportColumnNames(string path)
    {
        var suffix = System.IO.Path.GetExtension(path).ToLowerInvariant();
        if (suffix == ".parquet")
            return ParquetTable.ReadColumnNames(path);

        using var reader = new StreamReader(path);
        var header = reader.ReadLine() ?? "";
        // CsvLine.Split honours quoting, so a quoted column name keeps its commas and loses its
        // quotes; a tab-delimited header has neither problem.
        return suffix is ".tsv" or ".txt"
            ? header.Split('\t').Select(h => h.Trim()).ToArray()
            : CsvLine.Split(header).Select(h => h.Trim()).ToArray();
    }

    /// <summary>
    /// What this input can tell us about DIA isolation windows, for the Spectrum density map: the
    /// document's own isolation scheme (usually "Results only", i.e. named but window-less) plus, when a
    /// live Skyline is attached, every isolation scheme saved in it - the layouts the acquisition could
    /// have used. Best-effort: any failure just contributes nothing.
    /// </summary>
    public void CollectIsolationSchemes(
        IsolationSchemeCatalog catalog, Action<string> log, string? skylineCmdPath = null,
        CancellationToken cancellationToken = default)
    {
        var label = SanitizeLabel(BatchLabel);
        try
        {
            // The document's declared scheme. An OPEN document is read from its saved .sky (the scheme
            // lives in Full-Scan settings, which the RPC exposes no selected-item accessor for); an
            // unsaved document simply has no path to read.
            var documentPath = Kind == PrismInputKind.ReportFile ? null : Path;
            var documentXml = string.IsNullOrWhiteSpace(documentPath)
                ? null
                : SkyDocumentInfo.ReadIsolationSchemeXml(documentPath!);
            var documentScheme = IsolationScheme.Parse(documentXml);
            if (documentScheme is { HasWindows: true })
            {
                catalog.AddDocumentScheme(label, documentScheme);
                // Record the acquisition method here too - the explicit-scheme branch skips the data-file
                // import, which is where it would otherwise be read.
                if (!string.IsNullOrWhiteSpace(documentPath) && File.Exists(documentPath))
                    catalog.SetAcquisition(label, SkyDocumentInfo.TryRead(documentPath!, _ => { })?.AcquisitionMethod);
                log($"Isolation scheme from the document: {documentScheme.Name} ({documentScheme.Describe()}).");
            }
            else
            {
                // "Results only" (the normal DIA analysis setting): the document names a scheme but stores
                // no windows, because Skyline reads them from the data files at import. So have Skyline
                // read them back out of one of those files - the same thing Transition Settings >
                // Isolation scheme > Add > Import from a data file does, run against a throwaway document
                // so the user's own is never modified.
                if (documentScheme is not null)
                    log($"Document isolation scheme is '{documentScheme.Name}' - it stores no windows.");
                var imported = ImportIsolationSchemeFromData(
                    documentPath, skylineCmdPath, log, cancellationToken, catalog, label);
                if (imported is not null)
                    catalog.AddDocumentScheme(label, imported);
                else if (documentScheme is not null)
                    catalog.AddDocumentScheme(label, documentScheme); // record the name for the UI to explain
            }

            // The saved isolation schemes in Skyline's settings list are deliberately NOT collected.
            // They are generic templates - SWATH (25 m/z), SWATH (VW 64) and the like - that have
            // nothing to do with how this data was acquired, and offering them invites picking one:
            // binning a 3.0014 Th forbidden-zone acquisition on a 25 Th SWATH grid produces a map that
            // looks plausible and is wrong. The acquisition's own windows come from the data file
            // (above), and where those cannot be read the tab says so and uses labelled uniform bins,
            // which at least does not misrepresent itself.
        }
        catch (Exception ex)
        {
            log($"({DisplayName}: could not read isolation schemes: {ex.Message})");
        }
    }

    /// <summary>
    /// Have Skyline read the acquisition's real isolation windows out of one of the document's raw data
    /// files. Null when there is no reachable data file or no installed Skyline to read it with - the
    /// tool then falls back to asking the user which saved scheme to use.
    /// </summary>
    private static IsolationScheme? ImportIsolationSchemeFromData(
        string? documentPath, string? skylineCmdPath, Action<string> log, CancellationToken cancellationToken,
        IsolationSchemeCatalog? catalog = null, string? batchLabel = null)
    {
        if (string.IsNullOrWhiteSpace(documentPath) || !File.Exists(documentPath))
            return null;

        var info = SkyDocumentInfo.TryRead(documentPath!, log);
        if (info is null)
            return null;
        if (catalog is not null && batchLabel is not null)
            catalog.SetAcquisition(batchLabel, info.AcquisitionMethod);

        // Only DIA has the REPEATING isolation cycle Skyline's importer looks for; on anything else it
        // fails with "No repeating isolation scheme found in <file>". Scheduled methods (PRM, and the
        // multiplexed targeted variants) acquire different windows at different retention times, so there
        // is no cycle to find. Skip the ~10 s Skyline launch instead of provoking that error.
        if (info.AcquisitionMethod is not null && !info.IsDia)
        {
            log($"Acquisition method is {info.AcquisitionMethod}, not DIA - isolation windows cannot "
                + "be read from the data files, because only DIA has a repeating isolation cycle to "
                + "read. Scheduled methods acquire different windows at different retention times.");
            return null;
        }
        if (info.SampleFilePaths.Count == 0)
            return null;

        var dataFile = SkylineIsolationImporter.ResolveDataFile(info.SampleFilePaths, documentPath);
        if (dataFile is null)
        {
            log("None of this document's raw data files could be found, so its DIA isolation windows "
                + "cannot be read. The Spectrum density tab will ask which saved scheme to use.");
            return null;
        }

        // OUR OWN READER FIRST. The windows are scan headers in the first couple of acquisition
        // cycles, so pwiz answers for the cost of opening the file - measured at 4.2 s on a 3.3 GB
        // Thermo .raw over SMB, against ~10 s to launch Skyline and drive it through a scratch
        // document, and verified to give the same 167 windows edge for edge on the acquisition this
        // was built against.
        //
        // Reading them ourselves also removes three things that were never worth paying for: a
        // Skyline installation as a requirement for a step that only looks at a data file (the
        // standalone GUI and the CLI have no Skyline at all), a temporary document written to the
        // temp folder, and that document's audit trail - which reported Skyline's defaults for a
        // brand new document, "Product mass analyzer changed from None to QIT", in the middle of a
        // run on Astral data. Filtering that message was treating the symptom; not creating the
        // document is the cure.
        if (IsolationWindowProbe.Available)
        {
            log($"Reading isolation windows from {System.IO.Path.GetFileName(dataFile)}...");
            var windows = IsolationWindowProbe.Read(dataFile, null, cancellationToken);
            if (windows.Count > 0)
            {
                var read = new IsolationScheme(
                    System.IO.Path.GetFileNameWithoutExtension(dataFile), windows);
                log($"Isolation windows read from the data file: {read.Describe()}.");
                return read;
            }

            // No repeating cycle. Skyline would not find one either - it looks for the same thing -
            // so there is nothing to gain by launching it.
            log($"{System.IO.Path.GetFileName(dataFile)} reports no repeating isolation windows, so "
                + "it has no scheme to read. The Spectrum density tab will ask which saved "
                + "scheme to use.");
            return null;
        }

        // No reader in this build. Every PUBLISHED build carries one - the release workflow and CI
        // both assert it - so this is the developer build that packaged without pwiz.
        //
        // preferCmd because this probe uses --new (a throwaway document, so the user's is never
        // touched), and --new hangs through the app runner - it prints the "opened" line and then
        // nothing. SkylineCmd reads the same 4.9 GB Thermo .raw in 8.7 s. NOT because it is a
        // "settings" command: report export goes through the app runner and is fine, because it
        // opens an existing document with --in.
        var exporter = HeadlessSkylineExporter.Create(skylineCmdPath, _ => { }, preferCmd: true);
        return SkylineIsolationImporter.ImportFromDataFile(
            dataFile, exporter.Runner, log, cancellationToken);
    }

    /// <summary>
    /// Make <paramref name="label"/> safe to use as a file stem and as a batch label. Characters that are
    /// illegal in a file name become '_', because the label IS the exported report's file name.
    /// </summary>
    public static string SanitizeLabel(string? label)
    {
        var trimmed = (label ?? "").Trim();
        if (trimmed.Length == 0)
            return "batch";
        var invalid = System.IO.Path.GetInvalidFileNameChars();
        var chars = trimmed.Select(ch => invalid.Contains(ch) ? '_' : ch).ToArray();
        return new string(chars);
    }

    /// <summary>
    /// Give every input a unique, file-safe batch label, appending _2, _3, ... to duplicates. Two documents
    /// sharing a label would merge into one batch and silently defeat batch correction.
    /// </summary>
    public static void EnsureUniqueLabels(IEnumerable<PrismInput> inputs)
    {
        var used = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var input in inputs)
        {
            var baseLabel = SanitizeLabel(input.BatchLabel);
            var label = baseLabel;
            var n = 2;
            while (!used.Add(label))
                label = $"{baseLabel}_{n++}";
            input.BatchLabel = label;
        }
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private void Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
            return;
        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
