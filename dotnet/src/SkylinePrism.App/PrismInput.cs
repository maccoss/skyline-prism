using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using SkylinePrism.Core.Qc;
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

    public static PrismInput FromClosedDocument(string skyPath)
    {
        var name = System.IO.Path.GetFileNameWithoutExtension(skyPath);
        return new PrismInput(PrismInputKind.ClosedDocument, name, SanitizeLabel(name))
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
        var label = string.IsNullOrWhiteSpace(BatchLabel) ? SanitizeLabel(DisplayName) : SanitizeLabel(BatchLabel);

        // Every line produced while preparing THIS input is tagged with its document. Inputs are exported
        // concurrently, and the deepest lines come from Skyline's own console ("Opening file...", "2%"),
        // which say nothing about which document they belong to - two documents at once produce a stream
        // of identical-looking pairs. Tagging here rather than inside each exporter covers all three input
        // kinds, plus the runner output and the Skyline-selection messages, from one place.
        var scoped = Scoped(log, label);

        switch (Kind)
        {
            case PrismInputKind.RunningSkyline:
            {
                var session = Session
                    ?? throw new InvalidOperationException($"{DisplayName}: no Skyline connection for this input.");
                var driver = new SkylineReportDriver(session, scoped);
                return driver.Export(reportsDir, metadataReportName, batchAnnotation, label);
            }

            case PrismInputKind.ClosedDocument:
            {
                var exporter = HeadlessSkylineExporter.Create(skylineCmdPath, scoped);
                return exporter.Export(Path, reportsDir, label, batchAnnotation, cancellationToken);
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
            log($"Acquisition method is {info.AcquisitionMethod}, not DIA - isolation windows cannot be "
                + "read from the data files (Skyline can only import a repeating DIA cycle).");
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
