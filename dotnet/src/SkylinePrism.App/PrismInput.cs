using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
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

        switch (Kind)
        {
            case PrismInputKind.RunningSkyline:
            {
                var session = Session
                    ?? throw new InvalidOperationException($"{DisplayName}: no Skyline connection for this input.");
                var driver = new SkylineReportDriver(session, log);
                return driver.Export(reportsDir, metadataReportName, batchAnnotation, label);
            }

            case PrismInputKind.ClosedDocument:
            {
                var exporter = HeadlessSkylineExporter.Create(skylineCmdPath, log);
                return exporter.Export(Path, reportsDir, label, batchAnnotation, cancellationToken);
            }

            default:
            {
                if (!File.Exists(Path))
                    throw new FileNotFoundException($"Report file not found: {Path}", Path);
                var isParquet = System.IO.Path.GetExtension(Path)
                    .Equals(".parquet", StringComparison.OrdinalIgnoreCase);
                log($"{label}: using the existing report {Path}"
                    + (MetadataPath is not null ? $" with metadata {MetadataPath}" : " (no metadata file)"));
                return new ExportedReports(Path, isParquet, MetadataPath, null, label);
            }
        }
    }

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
