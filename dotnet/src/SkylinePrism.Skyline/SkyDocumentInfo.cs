#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Xml;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.Skyline;

/// <summary>
/// The header facts PRISM needs from a Skyline document that is NOT open in a running Skyline:
/// the replicate-targeted annotation names (so the PRISM-Replicates report can be generated with the
/// right columns before <see cref="HeadlessSkylineExporter"/> exports it), the digestion enzyme, and
/// the replicate list.
///
/// Everything here lives inside &lt;settings_summary&gt;, which precedes the (huge) &lt;protein&gt; list,
/// so parsing stops at &lt;/settings_summary&gt; and never reads the bulk of the file:
/// <code>
/// &lt;srm_settings format_version="25.1" software_version="Skyline (64-bit) 25.1.0.237"&gt;
///   &lt;settings_summary name="Default"&gt;
///     &lt;peptide_settings&gt;&lt;enzyme name="Trypsin" cut="KR" no_cut="P" sense="C" /&gt; ... &lt;/peptide_settings&gt;
///     &lt;data_settings document_guid="..."&gt;
///       &lt;annotation name="Plate" targets="replicate" type="text" /&gt;
///     &lt;/data_settings&gt;
///     &lt;measured_results&gt;
///       &lt;replicate name="Ref_01" sample_type="standard"&gt;
///         &lt;sample_file ... acquired_time="2025-08-22T06:17:54" /&gt;
///         &lt;annotation name="Plate"&gt;P1&lt;/annotation&gt;
///       &lt;/replicate&gt;
///     &lt;/measured_results&gt;
///   &lt;/settings_summary&gt;
///   &lt;protein ...&gt;   &lt;!-- parsing stops before here --&gt;
/// </code>
/// This is a READ-ONLY view of the user's document; nothing here writes to the .sky.
/// </summary>
public sealed class SkyDocumentInfo
{
    /// <summary>One replicate as recorded in &lt;measured_results&gt;.</summary>
    /// <param name="SampleFiles">The <c>file_path</c> of every <c>&lt;sample_file&gt;</c> under this
    /// replicate - normally one, several for a multi-injection replicate. Kept per replicate as well as
    /// in the document-wide <see cref="SkyDocumentInfo.SampleFilePaths"/>, because measuring acquired
    /// MS2 signal means opening the raw file belonging to a NAMED replicate, which a flat list cannot
    /// say. Still in Skyline's own spelling (<c>file.wiff|sample|1</c>, and paths that have since
    /// moved), so pass them through
    /// <see cref="SkylineIsolationImporter.ResolveDataFile"/> before opening one.</param>
    public sealed record SkyReplicate(
        string Name, string SampleType, IReadOnlyDictionary<string, string> Annotations,
        IReadOnlyList<string> SampleFiles);

    public string DocumentPath { get; private init; } = "";
    public string? FormatVersion { get; private init; }
    public string? SoftwareVersion { get; private init; }
    public string? DocumentGuid { get; private init; }

    /// <summary>Raw <c>&lt;enzyme .../&gt;</c> element, in the same shape the settings-list RPC returns.</summary>
    public string? EnzymeXml { get; private init; }

    /// <summary>
    /// Full-Scan acquisition method - <c>DIA</c>, <c>PRM</c>, <c>DDA</c>, <c>SureQuant</c>, <c>None</c> -
    /// or null when the document has no full-scan settings (e.g. SRM). Only DIA has the repeating
    /// isolation cycle that <c>SkylineIsolationImporter</c> can read out of a data file.
    /// </summary>
    public string? AcquisitionMethod { get; private init; }

    /// <summary>True when the document was acquired by DIA (case-insensitive).</summary>
    public bool IsDia =>
        string.Equals(AcquisitionMethod, "DIA", StringComparison.OrdinalIgnoreCase);

    /// <summary>The document enzyme mapped to a PRISM <c>parsimony.enzyme</c> name, or null if unmappable.</summary>
    public string? PrismEnzyme => SkylineDigestion.PrismEnzymeFromXml(EnzymeXml);

    /// <summary>
    /// Product-ion extraction settings, verbatim from <c>&lt;transition_full_scan&gt;</c>:
    /// <c>product_mass_analyzer</c> (centroided, qit, tof, orbitrap, ft_icr), <c>product_res</c>,
    /// <c>product_res_mz</c> and <c>selective_extraction</c>. Kept raw as well as parsed, so a document
    /// that yields no <see cref="ProductTolerance"/> can still say why.
    /// </summary>
    public string? ProductMassAnalyzer { get; private init; }

    /// <inheritdoc cref="ProductMassAnalyzer"/>
    public string? ProductResolution { get; private init; }

    /// <inheritdoc cref="ProductMassAnalyzer"/>
    public string? ProductResolutionMz { get; private init; }

    /// <inheritdoc cref="ProductMassAnalyzer"/>
    public string? SelectiveExtraction { get; private init; }

    /// <summary>
    /// The m/z range Skyline extracted each product ion over, or null when the document does not say
    /// enough to know (no full-scan settings, or a resolving-power analyzer with no calibration m/z).
    ///
    /// <para>MS2 signal accounting needs it to decide when two transitions read the same detector
    /// counts. Null is not fatal there: it means the accounting cannot be computed for this document,
    /// which is a thing to report rather than to guess a default for - a guessed tolerance would change
    /// how much fragment sharing is found with nothing on the plot to show it.</para>
    /// </summary>
    public ProductMassTolerance? ProductTolerance => ProductMassTolerance.Parse(
        ProductMassAnalyzer, ProductResolution, ProductResolutionMz, SelectiveExtraction);

    private string? _isolationSchemeXml;
    private bool _isolationSchemeRead;

    /// <summary>
    /// Raw <c>&lt;isolation_scheme&gt;</c> element (with its windows) from Transition Settings &gt;
    /// Full-Scan, or null when the document has none. Note that a DIA analysis document normally carries
    /// only <c>&lt;isolation_scheme name="Results only" /&gt;</c> - named, but with no windows, because
    /// Skyline reads them from the data files at import and never writes them here.
    /// <para>Read lazily with its own pass (see <see cref="ReadIsolationSchemeXml"/>) rather than during
    /// <see cref="Read"/>: capturing an element WITH ITS CHILDREN consumes the reader past that element,
    /// which the main loop's advance-on-every-iteration structure cannot absorb safely.</para>
    /// </summary>
    public string? IsolationSchemeXml
    {
        get
        {
            if (!_isolationSchemeRead)
            {
                _isolationSchemeRead = true;
                _isolationSchemeXml = ReadIsolationSchemeXml(DocumentPath);
            }
            return _isolationSchemeXml;
        }
    }

    /// <summary>
    /// Read just the <c>&lt;isolation_scheme&gt;</c> element from a .sky (or the document inside a
    /// <c>.sky.zip</c>), or null if it has none. Stops at
    /// <c>&lt;/transition_settings&gt;</c>, so it reads only the head of the file. Never throws: an
    /// unreadable or non-Skyline file simply yields null.
    /// </summary>
    public static string? ReadIsolationSchemeXml(string skyPath)
    {
        if (string.IsNullOrWhiteSpace(skyPath) || !File.Exists(skyPath))
            return null;
        try
        {
            var settings = new XmlReaderSettings
            {
                IgnoreComments = true,
                IgnoreWhitespace = true,
                DtdProcessing = DtdProcessing.Prohibit,
                XmlResolver = null,
            };
            using var stream = OpenDocument(skyPath);
            using var reader = XmlReader.Create(stream, settings);
            while (reader.Read())
            {
                if (reader.NodeType == XmlNodeType.EndElement
                    && (reader.LocalName == "transition_settings" || reader.LocalName == "settings_summary"))
                {
                    return null; // passed the full-scan settings without finding one
                }
                if (reader.NodeType == XmlNodeType.Element && reader.LocalName == "isolation_scheme")
                    return reader.ReadOuterXml();
            }
        }
        catch (Exception ex) when (ex is IOException or XmlException or UnauthorizedAccessException)
        {
            // Best-effort: the isolation scheme is a nicety, never a reason to fail an input.
        }
        return null;
    }

    /// <summary>Names of annotations whose <c>targets</c> include <c>replicate</c> (Document Annotations).</summary>
    public IReadOnlyList<string> ReplicateAnnotationNames { get; private init; } = Array.Empty<string>();

    public IReadOnlyList<SkyReplicate> Replicates { get; private init; } = Array.Empty<SkyReplicate>();

    /// <summary>
    /// Raw data files this document imported, in document order, as recorded at import time (they may
    /// since have moved - see <c>SkylineIsolationImporter.ResolveDataFile</c>). Needed because the DIA
    /// isolation windows of a "Results only" document exist only inside these files.
    /// </summary>
    public IReadOnlyList<string> SampleFilePaths { get; private init; } = Array.Empty<string>();

    /// <summary>Display name used as the batch label for this document (the .sky file stem).</summary>
    public string Name => Path.GetFileNameWithoutExtension(DocumentPath);

    /// <summary>
    /// Read the document header. Throws <see cref="FileNotFoundException"/> when the path does not exist and
    /// <see cref="InvalidDataException"/> when the file is not a Skyline document.
    /// </summary>
    public static SkyDocumentInfo Read(string skyPath)
    {
        if (!File.Exists(skyPath))
            throw new FileNotFoundException($"Skyline document not found: {skyPath}", skyPath);

        using var documentStream = OpenDocument(skyPath);
        return Read(skyPath, documentStream);
    }

    /// <summary>
    /// The document's XML, whether the path names a <c>.sky</c> or a <c>.sky.zip</c> - in which case
    /// the archive's document entry is read in place, with nothing extracted. Every parser here stops
    /// at the end of the settings, so an archive answers as cheaply as a loose document does even
    /// when the entry inside it is gigabytes.
    /// </summary>
    private static Stream OpenDocument(string skyPath) =>
        SharedDocumentArchive.IsArchive(skyPath)
            ? SharedDocumentArchive.OpenDocumentEntry(skyPath)
            : new FileStream(
                skyPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);

    private static SkyDocumentInfo Read(string skyPath, Stream documentStream)
    {
        string? formatVersion = null, softwareVersion = null, documentGuid = null, enzymeXml = null;
        string? acquisitionMethod = null;
        string? productMassAnalyzer = null, productRes = null, productResMz = null;
        string? selectiveExtraction = null;
        var annotationNames = new List<string>();
        var replicates = new List<SkyReplicate>();
        var sampleFilePaths = new List<string>();
        var sawRoot = false;

        var settings = new XmlReaderSettings
        {
            IgnoreComments = true,
            IgnoreWhitespace = true,
            DtdProcessing = DtdProcessing.Prohibit, // never fetch/expand external entities from a user file
            XmlResolver = null,
        };
        using var reader = XmlReader.Create(documentStream, settings);

        while (reader.Read())
        {
            if (reader.NodeType == XmlNodeType.EndElement && reader.LocalName == "settings_summary")
                break; // everything we need precedes the protein list
            if (reader.NodeType != XmlNodeType.Element)
                continue;

            switch (reader.LocalName)
            {
                case "srm_settings":
                    sawRoot = true;
                    formatVersion = reader.GetAttribute("format_version");
                    softwareVersion = reader.GetAttribute("software_version");
                    break;

                case "data_settings":
                    documentGuid = reader.GetAttribute("document_guid");
                    break;

                case "transition_full_scan":
                    // Attribute-only read, so it is safe in this advance-every-iteration loop (unlike the
                    // <isolation_scheme> child element - see ReadIsolationSchemeXml).
                    acquisitionMethod ??= reader.GetAttribute("acquisition_method");
                    productMassAnalyzer ??= reader.GetAttribute("product_mass_analyzer");
                    productRes ??= reader.GetAttribute("product_res");
                    productResMz ??= reader.GetAttribute("product_res_mz");
                    selectiveExtraction ??= reader.GetAttribute("selective_extraction");
                    break;

                case "enzyme":
                    // Re-emit as a standalone element so SkylineDigestion parses it exactly as it parses
                    // the XML returned by GetSettingsListItem("Enzymes", name).
                    enzymeXml ??= EnzymeElementXml(reader);
                    break;

                case "annotation":
                    // In <data_settings> these are DEFINITIONS (name + targets); inside a <replicate> they
                    // are VALUES and are handled by ReadReplicate. Only definitions carry "targets".
                    var targets = reader.GetAttribute("targets");
                    var annName = reader.GetAttribute("name");
                    if (targets is not null && !string.IsNullOrWhiteSpace(annName)
                        && TargetsReplicate(targets)
                        && !annotationNames.Contains(annName!, StringComparer.Ordinal))
                    {
                        annotationNames.Add(annName!);
                    }
                    break;

                case "replicate":
                    replicates.Add(ReadReplicate(reader, sampleFilePaths));
                    break;

                case "protein":
                case "peptide_list":
                    // Defensive: a document without <settings_summary> would otherwise scan the whole file.
                    goto done;
            }
        }

    done:
        if (!sawRoot)
            throw new InvalidDataException(
                $"{Path.GetFileName(skyPath)} is not a Skyline document (no <srm_settings> root element).");

        return new SkyDocumentInfo
        {
            DocumentPath = Path.GetFullPath(skyPath),
            FormatVersion = formatVersion,
            SoftwareVersion = softwareVersion,
            DocumentGuid = documentGuid,
            EnzymeXml = enzymeXml,
            AcquisitionMethod = acquisitionMethod,
            ProductMassAnalyzer = productMassAnalyzer,
            ProductResolution = productRes,
            ProductResolutionMz = productResMz,
            SelectiveExtraction = selectiveExtraction,
            ReplicateAnnotationNames = annotationNames,
            Replicates = replicates,
            SampleFilePaths = sampleFilePaths,
        };
    }

    /// <summary>Like <see cref="Read"/> but returns null instead of throwing (for best-effort UI probes).</summary>
    public static SkyDocumentInfo? TryRead(string skyPath, Action<string>? log = null)
    {
        try
        {
            return Read(skyPath);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or XmlException or UnauthorizedAccessException)
        {
            // InvalidDataException covers both a file that is not a Skyline document and an archive
            // that is not a shared document (no .sky inside, or several) - its message says which.
            log?.Invoke($"Could not read {Path.GetFileName(skyPath)}: {ex.Message}");
            return null;
        }
    }

    // targets is a comma-separated list, e.g. "protein, peptide, replicate, precursor_result". Match the
    // "replicate" entry exactly so "precursor_result"/"transition_result" don't produce false positives.
    private static bool TargetsReplicate(string targets) => targets
        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
        .Any(t => t.Equals("replicate", StringComparison.OrdinalIgnoreCase));

    private static string EnzymeElementXml(XmlReader reader)
    {
        var attrs = new List<string>();
        if (reader.HasAttributes)
        {
            for (var i = 0; i < reader.AttributeCount; i++)
            {
                reader.MoveToAttribute(i);
                attrs.Add($"{reader.Name}=\"{XmlEscapeAttr(reader.Value)}\"");
            }
            reader.MoveToElement();
        }
        return attrs.Count > 0 ? $"<enzyme {string.Join(" ", attrs)} />" : "<enzyme />";
    }

    private static string XmlEscapeAttr(string s) => s
        .Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");

    // Reads one <replicate> element, collecting its annotation VALUES and appending any <sample_file>
    // paths to sampleFilePaths. Positioned on <replicate>; leaves the reader on </replicate> (or on the
    // element itself when it is empty).
    private static SkyReplicate ReadReplicate(XmlReader reader, List<string> sampleFilePaths)
    {
        var name = reader.GetAttribute("name") ?? "";
        var sampleType = reader.GetAttribute("sample_type") ?? "";
        var annotations = new Dictionary<string, string>(StringComparer.Ordinal);
        var mine = new List<string>();

        if (reader.IsEmptyElement)
            return new SkyReplicate(name, sampleType, annotations, mine);

        var depth = reader.Depth;
        reader.Read(); // step into the element
        while (!reader.EOF)
        {
            if (reader.NodeType == XmlNodeType.EndElement && reader.Depth == depth
                && reader.LocalName == "replicate")
            {
                break; // leave the reader ON </replicate> so the caller's Read() lands on the next node
            }

            if (reader.NodeType == XmlNodeType.Element && reader.LocalName == "sample_file")
            {
                var filePath = reader.GetAttribute("file_path");
                if (!string.IsNullOrWhiteSpace(filePath))
                {
                    mine.Add(filePath!);
                    sampleFilePaths.Add(filePath!);
                }
                // Fall through to the plain Read() below: <sample_file> children (instrument info) are
                // not needed, and letting the loop walk them keeps the reader's position predictable.
            }

            if (reader.NodeType == XmlNodeType.Element && reader.LocalName == "annotation")
            {
                var annName = reader.GetAttribute("name");
                if (reader.IsEmptyElement)
                {
                    if (!string.IsNullOrWhiteSpace(annName))
                        annotations[annName!] = "";
                    reader.Read();
                }
                else
                {
                    // ReadElementContentAsString consumes through </annotation> and positions the reader
                    // on the FOLLOWING node - so do not Read() again, or the next node gets skipped
                    // (which would swallow </replicate> when an annotation is the last child).
                    var value = reader.ReadElementContentAsString();
                    if (!string.IsNullOrWhiteSpace(annName))
                        annotations[annName!] = value;
                }
                continue;
            }

            reader.Read();
        }
        return new SkyReplicate(name, sampleType, annotations, mine);
    }
}
