using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>Feature level a differential dataset is loaded at.</summary>
public enum FeatureLevel
{
    Protein,
    Peptide,
}

/// <summary>
/// Everything one feature is known by: what to show a reader, and what to look it up by in a Skyline
/// document.
/// </summary>
/// <remarks>
/// The plural lists are not a convenience - a peptide shared between protein groups genuinely belongs
/// to all of them, and which one quantifies it is a decision about the rollup rather than a fact about
/// the peptide. PRISM also runs its OWN parsimony, so its groups need not line up with the document's;
/// offering every accession and name is what lets a group whose leading protein is absent from the
/// document still be found by one of its other members.
/// </remarks>
/// <param name="FeatureId">The matrix row's id: a protein-group id, or a modified peptide sequence.</param>
/// <param name="Label">What to show for it - a gene name for a protein, the sequence for a peptide.</param>
/// <param name="ProteinGroups">PRISM's protein-group ids, empty if the matrix carried none.</param>
/// <param name="Accessions">Leading accession per group, index-aligned to <paramref name="ProteinGroups"/>.</param>
/// <param name="ProteinNames">Leading protein name per group, index-aligned.</param>
/// <param name="Genes">Leading gene name per group, index-aligned.</param>
public sealed record FeatureIdentity(
    string FeatureId,
    string Label,
    IReadOnlyList<string> ProteinGroups,
    IReadOnlyList<string> Accessions,
    IReadOnlyList<string> ProteinNames,
    IReadOnlyList<string> Genes)
{
    /// <summary>True when this feature maps to more than one protein group.</summary>
    public bool IsShared => ProteinGroups.Count > 1;

    /// <summary>
    /// A one-line description for a hover readout or a status line: the label, then the gene and
    /// protein it belongs to, then every further group it is shared with. Nothing is invented - a
    /// part is left out entirely when the matrix did not carry it.
    /// </summary>
    public string Describe()
    {
        var parts = new List<string>();
        var gene = Genes.FirstOrDefault(g => !string.IsNullOrEmpty(g));
        if (gene is not null && gene != Label)
            parts.Add(gene);
        var protein = ProteinNames.FirstOrDefault(n => !string.IsNullOrEmpty(n))
                      ?? Accessions.FirstOrDefault(a => !string.IsNullOrEmpty(a));
        if (protein is not null && protein != Label)
            parts.Add(protein);
        if (IsShared)
            parts.Add($"shared across {ProteinGroups.Count} protein groups");

        return parts.Count == 0 ? Label : $"{Label}  ({string.Join(" - ", parts)})";
    }
}

/// <summary>Outcome of joining a clinical metadata CSV to a dataset.</summary>
public sealed record ClinicalAttachResult(
    string? KeyColumn, double MatchRate, int ClinicalColumnCount, IReadOnlyList<string> AddedColumns);

/// <summary>
/// A loaded PRISM output ready for differential analysis, ported from the explorer's
/// <c>load_prism</c>. Reads the corrected (LINEAR-scale) protein or peptide matrix and
/// <c>sample_metadata.csv</c> from a run's output directory, identifies the sample columns by matching
/// the metadata's sample ids, and log2-transforms the abundances (non-positive values become NaN). The
/// sample metadata is exposed so a contrast can be built from any annotation column.
/// </summary>
public sealed class DifferentialDataset
{
    private readonly Dictionary<string, string?[]> _metaByColumn;
    private readonly List<string> _metadataColumns;
    private readonly string[] _featureGroups;
    private readonly string[] _featureAccessions;
    private readonly string[] _featureProteinNames;

    private DifferentialDataset(FeatureLevel level, double[,] exprLog2, string[] featureIds,
        string[] featureLabels, string[] featureGenes, string[] featureGroups,
        string[] featureAccessions, string[] featureProteinNames, double[]? peptideCounts,
        string[] sampleIds, string idColumn, string labelColumn,
        IReadOnlyList<string> metadataColumns, Dictionary<string, string?[]> metaByColumn)
    {
        PeptideCounts = peptideCounts;
        _featureGroups = featureGroups;
        _featureAccessions = featureAccessions;
        _featureProteinNames = featureProteinNames;
        Level = level;
        ExprLog2 = exprLog2;
        FeatureIds = featureIds;
        FeatureLabels = featureLabels;
        FeatureGenes = featureGenes;
        SampleIds = sampleIds;
        IdColumn = idColumn;
        LabelColumn = labelColumn;
        _metadataColumns = metadataColumns.ToList();
        _metaByColumn = metaByColumn;
    }

    /// <summary>The level this was loaded at.</summary>
    public FeatureLevel Level { get; }

    /// <summary>LOG2 abundance matrix, <c>[feature, sample]</c> (NaN where the linear value was &lt;= 0).</summary>
    public double[,] ExprLog2 { get; }

    /// <summary>Feature identifiers (matrix rows).</summary>
    public string[] FeatureIds { get; }

    /// <summary>
    /// Each feature's leading UniProt entry name(s) (<c>ICAM1_MOUSE</c>), from <c>leading_name</c>;
    /// <c>;</c>-separated for a shared peptide, empty where the matrix has none. Parallel to
    /// <see cref="FeatureIds"/>. The species code in them is what enrichment infers its organism from.
    /// </summary>
    public IReadOnlyList<string> FeatureProteinNames => _featureProteinNames;

    /// <summary>
    /// Human-readable feature labels, parallel to <see cref="FeatureIds"/>: for a protein its gene,
    /// else its UniProt accession (<c>leading_uniprot_id</c>, then <c>leading_protein</c>), else empty
    /// (callers then show the feature id); for a peptide its modified sequence. Never a placeholder
    /// such as "NA".
    /// </summary>
    public string[] FeatureLabels { get; }

    /// <summary>
    /// Gene symbol per feature, or "" where none is known; parallel to <see cref="FeatureIds"/>.
    /// A shared peptide names every group it belongs to, <c>;</c>-separated, which is what
    /// <c>Enrichment.CleanSymbols</c> splits on.
    /// </summary>
    /// <remarks>
    /// Separate from <see cref="FeatureLabels"/> on purpose. At peptide level the label is the
    /// modified sequence - the right thing to show on a Volcano point or a boxplot title - but a
    /// sequence is not a gene symbol, and enrichment submitted it as one: it passes CleanSymbols
    /// untouched (that only drops empty/"nan" and splits delimiters), so g:Profiler was asked about
    /// a list of peptide sequences and the empty result was captioned as a gene enrichment.
    /// <c>corrected_peptides.parquet</c> carries <c>leading_gene_name</c> for exactly this
    /// (PrismPipeline.PeptideGroupColumns stamps it on), so the gene is available at both levels -
    /// it just has to be kept apart from the display label. Empty for a peptide file written before
    /// those columns existed, which callers must treat as "no genes", never as a symbol.
    /// </remarks>
    public string[] FeatureGenes { get; }

    /// <summary>
    /// Peptides behind each feature (<c>n_peptides</c>), parallel to <see cref="FeatureIds"/>, or
    /// null where the matrix does not carry it.
    /// </summary>
    /// <remarks>
    /// Null is the ordinary PEPTIDE-level case, not an error: a peptide has no peptide count. The
    /// count-based variance priors are unavailable there and the caller must say so rather than
    /// substitute a count of one, which would make the trend a flat line and look like it worked.
    /// </remarks>
    public double[]? PeptideCounts { get; }

    /// <summary>
    /// Everything feature <paramref name="index"/> is known by - what to show a reader, and what to
    /// look it up by in a Skyline document. Every list is empty where the matrix carried no such
    /// column, never a placeholder: a caller finding an element by name must not be handed a
    /// protein-group id or a peptide sequence to search for.
    /// </summary>
    public FeatureIdentity IdentityOf(int index) => new(
        FeatureIds[index],
        FeatureLabels[index],
        SplitGroups(_featureGroups[index]),
        SplitGroups(_featureAccessions[index]),
        SplitGroups(_featureProteinNames[index]),
        SplitGroups(FeatureGenes[index]));

    /// <summary>
    /// <see cref="IdentityOf(int)"/> for a feature id, or null when the id is not in this matrix.
    /// </summary>
    public FeatureIdentity? IdentityOf(string featureId)
    {
        // Indexed, not scanned. A linear Array.IndexOf here is invisible for one hover and
        // quadratic for a caller that walks a whole result: the CLI's CSV writer calls this once
        // per row, so a 75,000-peptide cohort was 75,000 scans of a 75,000-element array - billions
        // of string comparisons, and a command that looks hung after printing its hit count.
        _indexById ??= BuildIndex(FeatureIds);
        return _indexById.TryGetValue(featureId, out var i) ? IdentityOf(i) : null;
    }

    /// <summary>Built on first use: most callers never look a feature up by id at all.</summary>
    private Dictionary<string, int>? _indexById;

    private static Dictionary<string, int> BuildIndex(string[] ids)
    {
        var map = new Dictionary<string, int>(ids.Length, StringComparer.Ordinal);
        for (var i = 0; i < ids.Length; i++)
            // First wins, matching Array.IndexOf, in the pathological case of a duplicated id.
            map.TryAdd(ids[i], i);
        return map;
    }


    /// <summary>Sample identifiers (matrix columns), matching the metadata's sample ids.</summary>
    public string[] SampleIds { get; }

    /// <summary>The parquet column used as the feature id.</summary>
    public string IdColumn { get; }

    /// <summary>
    /// The parquet column the feature label is primarily taken from: <c>leading_gene_name</c> for a
    /// protein matrix that has it - where a protein has no gene, its label falls back to its accession
    /// (see <see cref="FeatureLabels"/>) - otherwise the id column.
    /// </summary>
    public string LabelColumn { get; }

    /// <summary>Metadata columns available for building a contrast (e.g. sample_type, batch, ...).</summary>
    public IReadOnlyList<string> MetadataColumns => _metadataColumns;

    /// <summary>Metadata values for <paramref name="column"/>, aligned to <see cref="SampleIds"/>.</summary>
    public string?[] MetadataValues(string column) =>
        _metaByColumn.TryGetValue(column, out var v)
            ? v
            : throw new ArgumentException($"Unknown metadata column '{column}'.", nameof(column));

    /// <summary>
    /// <paramref name="column"/> read as numbers, aligned to <see cref="SampleIds"/>: invariant
    /// culture, NaN where a sample has no value or one that does not parse. What a trend is fitted
    /// against - one parse, so the pane, the CLI and the report cannot disagree on a sample's x.
    /// </summary>
    public double[] NumericValues(string column)
    {
        var raw = MetadataValues(column);
        var values = new double[raw.Length];
        for (var i = 0; i < raw.Length; i++)
            values[i] = raw[i] is { } v
                && double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var d)
                ? d
                : double.NaN;
        return values;
    }

    private readonly List<string> _attachedClinical = new();

    /// <summary>
    /// The clinical CSVs that were actually joined - those that added at least one column - in the
    /// order attached. A report records these, because a contrast grouped or adjusted by a clinical
    /// column cannot be reproduced from the output directory alone.
    /// </summary>
    public IReadOnlyList<string> AttachedClinicalCsvs => _attachedClinical;

    /// <summary>
    /// Join a clinical metadata CSV to the samples, ported from the explorer's <c>attach_clinical</c>.
    /// The identifier column is auto-detected by value (<c>infer_clinical_key</c>): the column whose
    /// values best match the sample names, preferring near one-to-one columns so a low-cardinality
    /// column cannot win by coincidence. Every other clinical column is added to
    /// <see cref="MetadataColumns"/> (suffixed <c>_clin</c> on a name clash), aligned to
    /// <see cref="SampleIds"/>, so it becomes available for grouping and covariates. Returns null key
    /// column (and adds nothing) when nothing matches at least half the samples.
    /// </summary>
    public ClinicalAttachResult AttachClinical(string clinicalCsvPath)
    {
        if (!File.Exists(clinicalCsvPath))
            throw new FileNotFoundException($"Clinical CSV not found: {clinicalCsvPath}", clinicalCsvPath);

        var lines = SharedFile.ReadAllLines(clinicalCsvPath);
        if (lines.Length < 2)
            return new ClinicalAttachResult(null, 0.0, 0, Array.Empty<string>());

        var header = CsvLine.Split(lines[0]);
        var rows = new List<string[]>(lines.Length - 1);
        for (var r = 1; r < lines.Length; r++)
            if (!string.IsNullOrEmpty(lines[r]))
                rows.Add(CsvLine.Split(lines[r]));

        var n = SampleIds.Length;
        var names = _metaByColumn.TryGetValue("sample", out var s) ? s : null;
        var sampleTok = new List<string>[n];
        var nameUp = new string[n];
        for (var i = 0; i < n; i++)
        {
            var nm = names?[i] ?? SampleIds[i];
            sampleTok[i] = SampleKeys(nm);
            nameUp[i] = nm.Trim().ToUpperInvariant();
        }

        string? bestCol = null;
        double bestRate = 0;
        Dictionary<int, int>? bestMap = null;
        var bestScore = -1.0;
        for (var c = 0; c < header.Length; c++)
        {
            var mapping = MatchClinicalColumn(rows, c, sampleTok, nameUp, n);
            if (mapping.Count == 0)
                continue;
            var rate = mapping.Count / (double)n;
            var oneToOne = mapping.Values.Distinct().Count() / (double)mapping.Count;
            var score = rate * oneToOne;
            if (score > bestScore)
            {
                bestScore = score;
                bestCol = header[c];
                bestRate = rate;
                bestMap = mapping;
            }
        }

        if (bestCol is null || bestMap is null || bestRate < 0.5)
            return new ClinicalAttachResult(null, bestRate, 0, Array.Empty<string>());

        var keyIdx = Array.IndexOf(header, bestCol);
        var added = new List<string>();
        for (var c = 0; c < header.Length; c++)
        {
            if (c == keyIdx)
                continue;
            var name = _metaByColumn.ContainsKey(header[c]) ? header[c] + "_clin" : header[c];
            var values = new string?[n];
            for (var i = 0; i < n; i++)
            {
                if (bestMap.TryGetValue(i, out var r) && c < rows[r].Length && !string.IsNullOrEmpty(rows[r][c]))
                    values[i] = rows[r][c];
                else
                    values[i] = null;
            }

            _metaByColumn[name] = values;
            _metadataColumns.Add(name);
            added.Add(name);
        }

        if (added.Count > 0)
            _attachedClinical.Add(Path.GetFullPath(clinicalCsvPath));
        return new ClinicalAttachResult(bestCol, bestRate, header.Length - 1, added);
    }

    /// <summary>Candidate identifier tokens for a sample name (full name, trailing token, then every
    /// token), upper-cased and de-duplicated - the explorer's <c>_sample_keys</c>.</summary>
    private static List<string> SampleKeys(string name)
    {
        var nm = (name ?? string.Empty).Trim();
        var parts = Regex.Split(nm, @"[-_\s/]+").Where(p => p.Length > 0).ToList();
        var raw = new List<string> { nm };
        if (parts.Count > 0)
            raw.Add(parts[^1]);
        raw.AddRange(parts);

        var outp = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var k in raw)
        {
            var ku = k.ToUpperInvariant();
            if (ku.Length > 0 && seen.Add(ku))
                outp.Add(ku);
        }

        return outp;
    }

    /// <summary>Map sample index to clinical row index for one column by exact token then substring
    /// match - the explorer's <c>_match_clinical_column</c>.</summary>
    private static Dictionary<int, int> MatchClinicalColumn(List<string[]> rows, int colIdx,
        List<string>[] sampleTok, string[] nameUp, int n)
    {
        var valToRow = new Dictionary<string, int>(StringComparer.Ordinal);
        var longVals = new List<(string Val, int Row)>();
        for (var r = 0; r < rows.Count; r++)
        {
            if (colIdx >= rows[r].Length)
                continue;
            var raw = rows[r][colIdx];
            if (string.IsNullOrEmpty(raw))
                continue;
            var key = raw.Trim().ToUpperInvariant();
            if (key.Length == 0 || valToRow.ContainsKey(key))
                continue;
            valToRow[key] = r;
            if (key.Length >= 3)
                longVals.Add((key, r));
        }

        var mapping = new Dictionary<int, int>();
        for (var i = 0; i < n; i++)
        {
            var hit = -1;
            foreach (var k in sampleTok[i])
                if (valToRow.TryGetValue(k, out var r))
                {
                    hit = r;
                    break;
                }

            if (hit < 0)
            {
                var up = nameUp[i];
                foreach (var k in sampleTok[i])
                {
                    if (k.Length < 4)
                        continue;
                    foreach (var (val, r) in longVals)
                        if (val.Contains(k, StringComparison.Ordinal) || up.Contains(val, StringComparison.Ordinal))
                        {
                            hit = r;
                            break;
                        }

                    if (hit >= 0)
                        break;
                }
            }

            if (hit >= 0)
                mapping[i] = hit;
        }

        return mapping;
    }

    /// <summary>
    /// Load the corrected matrix and sample metadata from a run's <paramref name="outputDir"/>.
    /// </summary>
    public static DifferentialDataset Load(string outputDir, FeatureLevel level)
    {
        var matrixName = level == FeatureLevel.Protein ? "corrected_proteins.parquet" : "corrected_peptides.parquet";
        var matrixPath = Path.Combine(outputDir, matrixName);
        var metaPath = Path.Combine(outputDir, "sample_metadata.csv");
        if (!File.Exists(matrixPath))
            throw new FileNotFoundException($"Corrected matrix not found: {matrixPath}", matrixPath);
        if (!File.Exists(metaPath))
            throw new FileNotFoundException($"sample_metadata.csv not found: {metaPath}", metaPath);

        var (sampleIdToRow, metaColumns) = ReadSampleMetadata(metaPath);

        var table = ParquetTable.Load(matrixPath);
        var alignment = AlignSampleColumns(table.ColumnNames, sampleIdToRow, metaColumns);
        if (alignment.Count == 0)
            throw new InvalidOperationException(
                "No sample columns in the matrix matched sample_metadata.csv sample ids " +
                "(neither the full sample_id nor the bare replicate name before '__@__').");

        // A metadata row with no column means the two files describe different cohorts. Refuse rather
        // than analyze whichever samples happened to line up: the result would be a contrast over a
        // silently chosen subset, and nothing in the output would say which samples were in it.
        var unmatched = UnmatchedMetadata(sampleIdToRow, alignment);
        if (unmatched.Count > 0)
            throw new InvalidOperationException(
                $"{unmatched.Count} of {sampleIdToRow.Count} replicates in sample_metadata.csv have no "
                + $"column in {matrixName} ({NamePreview.Of(unmatched, show: 3)}). The two files "
                + "describe different sample sets, so a contrast would silently run on only the "
                + $"{alignment.Count} that matched. Re-run the pipeline so the matrix and the metadata "
                + "come from the same run.");
        var sampleCols = alignment.Select(a => a.Col).ToArray();
        var alignedMeta = alignment.Select(a => a.Meta).ToArray();
        var matched = new HashSet<string>(sampleCols, StringComparer.Ordinal);
        var annotCols = table.ColumnNames.Where(c => !matched.Contains(c)).ToList();

        // The other direction: a SAMPLE column of the matrix with no metadata row. Sample columns are
        // the only ones carrying the document separator, so they cannot be mistaken for annotation
        // columns. Unchecked, they fell through to annotCols and the contrast ran on the rest - which
        // is exactly what reading sample_metadata.csv while a run is rewriting it produces: a header
        // and a prefix of the rows, every sample past the cut silently gone.
        var unmatchedSamples = annotCols.Where(c => c.Contains(ReplicateMetadata.SampleIdSeparator, StringComparison.Ordinal)).ToList();
        if (unmatchedSamples.Count > 0)
            throw new InvalidOperationException(
                $"{unmatchedSamples.Count} of the {sampleCols.Length + unmatchedSamples.Count} samples in "
                + $"{matrixName} have no row in sample_metadata.csv ({NamePreview.Of(unmatchedSamples, show: 3)}). "
                + "If a PRISM run is writing this directory, wait for it to finish and load again; "
                + "otherwise re-run the pipeline so the matrix and the metadata come from the same run.");

        var idColumn = ChooseColumn(table, level == FeatureLevel.Protein
            ? new[] { "protein_group" }
            : new[] { "PeptideModifiedSequenceUnimodIds" }, annotCols);
        var labelColumn = level == FeatureLevel.Protein && table.HasColumn("leading_gene_name")
            ? "leading_gene_name"
            : idColumn;

        var nFeatures = table.RowCount;
        var featureIds = table.GetString(idColumn).Select(s => s ?? string.Empty).ToArray();
        // Genes come from leading_gene_name at BOTH levels - the C# engine stamps it onto
        // corrected_peptides as well (PrismPipeline.PeptideGroupColumns) - and are kept apart from
        // the display label, which at peptide level is the modified sequence. Empty when the column
        // is absent, which is a peptide file written before it existed; enrichment must then report
        // that it has no genes rather than fall back to the label and submit sequences as symbols.
        var featureGenes = IdentityColumn(table, "leading_gene_name", nFeatures);
        // The rest of the identity: what the feature is called in a Skyline document. Both corrected
        // matrices carry these - PrismPipeline stamps the protein-group columns onto the peptide
        // output too - and a shared peptide names every group it belongs to, ";"-separated and
        // index-aligned across the four.
        // n_peptides, for the DEqMS-style priors. Null at peptide level, where a feature IS a
        // peptide and the count means nothing - the caller must then not offer those priors rather
        // than invent a count of one.
        var peptideCounts = table.HasColumn("n_peptides")
            ? table.GetDouble("n_peptides").Select(v => v ?? double.NaN).ToArray()
            : null;
        var featureGroups = IdentityColumn(table, "protein_group", nFeatures);
        var featureAccessions = IdentityColumn(table, "leading_protein", nFeatures);
        var featureProteinNames = IdentityColumn(table, "leading_name", nFeatures);
        // A protein is labeled by its gene, then - where none is known, as for most proteins of a
        // non-model organism - by its UniProt accession, which still names it uniquely and can be
        // looked up. leading_uniprot_id before leading_protein, as the Dynamic Range labels do:
        // leading_protein is whatever Skyline called the protein and can be a whole FASTA id
        // ("sp|P02768|ALBU_HUMAN"), which does not read on a plot. An empty label is left for the
        // caller to replace with the protein-group id. Before this, the label was leading_gene_name
        // verbatim, so every unannotated protein carried parsimony's "NA" placeholder and a volcano
        // showed a column of identical "NA" labels.
        var featureUniProtIds = IdentityColumn(table, "leading_uniprot_id", nFeatures);
        var featureLabels = labelColumn == "leading_gene_name"
            ? featureGenes.Select((g, i) => FirstNamed(g, featureUniProtIds[i], featureAccessions[i])).ToArray()
            : table.GetString(labelColumn).Select(s => s ?? string.Empty).ToArray();

        var exprLog2 = new double[nFeatures, sampleCols.Length];
        for (var j = 0; j < sampleCols.Length; j++)
        {
            var col = table.GetDouble(sampleCols[j]);
            for (var i = 0; i < nFeatures; i++)
            {
                var v = col[i];
                exprLog2[i, j] = v.HasValue && v.Value > 0.0 ? Math.Log2(v.Value) : double.NaN;
            }
        }

        // Metadata aligned to the sample-column order.
        var metaByColumn = new Dictionary<string, string?[]>(StringComparer.Ordinal);
        for (var m = 0; m < metaColumns.Count; m++)
        {
            var values = new string?[sampleCols.Length];
            for (var j = 0; j < sampleCols.Length; j++)
                values[j] = alignedMeta[j][m];
            metaByColumn[metaColumns[m]] = values;
        }

        return new DifferentialDataset(level, exprLog2, featureIds, featureLabels, featureGenes,
            featureGroups, featureAccessions, featureProteinNames, peptideCounts,
            sampleCols, idColumn, labelColumn, metaColumns, metaByColumn);
    }

    /// <summary>
    /// Align the matrix's sample columns to the metadata rows. Each column is resolved on its own:
    /// an exact <c>sample_id</c> match first, then the bare replicate name (the part before the
    /// <c>__@__</c> document separator), which equals the metadata <c>sample</c> value. The bare-name
    /// fallback exists because some PRISM runs wrote the corrected matrix with one document/batch stem
    /// (e.g. <c>...__@__PRISM</c>) but <c>sample_metadata.csv</c> with another (e.g.
    /// <c>...__@__merged_data</c>); the bare replicate name is identical in both. A bare name matched
    /// to more than one metadata row is ambiguous and skipped, and a row already taken by an exact
    /// match is never handed to a second column. Returned columns keep the matrix's column names
    /// (they index the matrix), each paired with its resolved metadata row.
    /// </summary>
    /// <remarks>
    /// <para><b>Per column, not per file.</b> This used to choose between the two strategies globally -
    /// return the exact matches if there were ANY, otherwise try bare names - which meant a PARTIAL
    /// exact match silently discarded the rest. The leftovers did not merely drop out of the analysis:
    /// they fell through to <c>annotCols</c> and were treated as feature-annotation columns, so a
    /// 96-sample contrast could quietly run on 90. Nothing reported it, because the only guard was
    /// <c>alignment.Count == 0</c>. The stage cache makes the mismatch reachable - it reuses a stage
    /// whose declared keys are unchanged, so a matrix can carry an older run's sample ids beside a
    /// freshly written <c>sample_metadata.csv</c>.</para>
    /// <para>Resolving each column independently removes the failure rather than reporting it: the
    /// columns that match exactly do, and the stragglers get the fallback that was put there for them.
    /// <see cref="UnmatchedMetadata"/> then states what is left over, and the caller refuses.</para>
    /// </remarks>
    internal static List<(string Col, string?[] Meta)> AlignSampleColumns(
        IReadOnlyList<string> parquetCols,
        IReadOnlyDictionary<string, string?[]> sampleIdToRow,
        IReadOnlyList<string> metaColumns)
    {
        var result = new List<(string, string?[])>();
        // Reference equality: two metadata rows may be identical field for field (a duplicated
        // replicate in the CSV) and are still two rows, so they must be tracked as two objects.
        var claimed = new HashSet<string?[]>(ReferenceEqualityComparer.Instance);
        var pending = new List<string>();

        foreach (var c in parquetCols)
        {
            if (sampleIdToRow.TryGetValue(c, out var row))
            {
                result.Add((c, row));
                claimed.Add(row);
            }
            else
            {
                pending.Add(c);
            }
        }

        if (pending.Count == 0)
            return result;

        // Fallback: match on the bare replicate name (before "__@__"), requiring it to be unique.
        var sampleIdx = -1;
        for (var i = 0; i < metaColumns.Count; i++)
            if (string.Equals(metaColumns[i], "sample", StringComparison.Ordinal))
            {
                sampleIdx = i;
                break;
            }

        var bareToRows = new Dictionary<string, List<string?[]>>(StringComparer.Ordinal);
        foreach (var kv in sampleIdToRow)
        {
            var bare = sampleIdx >= 0 && !string.IsNullOrEmpty(kv.Value[sampleIdx])
                ? StripDocSuffix(kv.Value[sampleIdx]!)
                : StripDocSuffix(kv.Key);
            if (!bareToRows.TryGetValue(bare, out var lst))
            {
                lst = new List<string?[]>();
                bareToRows[bare] = lst;
            }

            lst.Add(kv.Value);
        }

        foreach (var c in pending)
        {
            if (!bareToRows.TryGetValue(StripDocSuffix(c), out var lst) || lst.Count != 1)
                continue;
            // A row an exact match already took is not available: handing it to a second column would
            // give two different samples one sample's type, batch and subject, which is worse than
            // leaving the column out - that at least gets reported.
            if (!claimed.Add(lst[0]))
                continue;
            result.Add((c, lst[0]));
        }

        return result;
    }

    /// <summary>
    /// The <c>sample_id</c>s in <paramref name="sampleIdToRow"/> that no column of
    /// <paramref name="alignment"/> resolved to, in metadata order.
    /// </summary>
    /// <remarks>
    /// Every row of <c>sample_metadata.csv</c> is a replicate that must have a column: the pipeline
    /// writes that file from the same sample list it writes the matrix from, AFTER outlier exclusion
    /// has already shortened it (<c>PrismPipeline</c> Stage 2a reassigns <c>samples</c>, and Stage 5
    /// writes both from it). So a leftover row is never a legitimate "this one was excluded" - it means
    /// the two files describe different cohorts, and any contrast run across them would be over a
    /// silently chosen subset. Leftover COLUMNS are not an error by contrast: the matrix carries
    /// feature-annotation columns (protein_group, leading_gene_name, n_peptides) that match no sample
    /// by construction.
    /// </remarks>
    internal static List<string> UnmatchedMetadata(
        IReadOnlyDictionary<string, string?[]> sampleIdToRow,
        IReadOnlyList<(string Col, string?[] Meta)> alignment)
    {
        var claimed = new HashSet<string?[]>(ReferenceEqualityComparer.Instance);
        foreach (var (_, meta) in alignment)
            claimed.Add(meta);

        var missing = new List<string>();
        foreach (var kv in sampleIdToRow)
            if (!claimed.Contains(kv.Value))
                missing.Add(kv.Key);
        return missing;
    }

    /// <summary>The replicate name before the <c>__@__</c> document separator (or the whole string).</summary>
    private static string StripDocSuffix(string s)
    {
        var idx = s.IndexOf("__@__", StringComparison.Ordinal);
        return idx >= 0 ? s.Substring(0, idx) : s;
    }

    private static string ChooseColumn(ParquetTable table, string[] preferred, List<string> fallback)
    {
        foreach (var name in preferred)
            if (table.HasColumn(name))
                return name;
        if (fallback.Count == 0)
            throw new InvalidOperationException("No annotation column available for the feature id.");
        return fallback[0];
    }

    private static (Dictionary<string, string?[]> ByRow, IReadOnlyList<string> Columns) ReadSampleMetadata(
        string path)
    {
        var lines = SharedFile.ReadAllLines(path);
        if (lines.Length == 0)
            throw new InvalidOperationException("sample_metadata.csv is empty.");

        var header = CsvLine.Split(lines[0]);
        var idIdx = CsvLine.IndexOf(header, "sample_id");
        if (idIdx < 0)
            throw new InvalidOperationException("sample_metadata.csv has no 'sample_id' column.");

        // Every column except sample_id, preserving file order.
        var otherIdx = Enumerable.Range(0, header.Length).Where(i => i != idIdx).ToArray();
        var columns = otherIdx.Select(i => header[i]).ToList();

        var byRow = new Dictionary<string, string?[]>(StringComparer.Ordinal);
        for (var r = 1; r < lines.Length; r++)
        {
            if (string.IsNullOrEmpty(lines[r]))
                continue;
            var fields = CsvLine.Split(lines[r]);
            if (idIdx >= fields.Length)
                continue;
            var sampleId = fields[idIdx];
            var values = new string?[otherIdx.Length];
            for (var k = 0; k < otherIdx.Length; k++)
            {
                // PRISM writes "" for a sample with no value; pandas reads that as NaN, and the
                // covariate/finder contracts treat missing as null - so an empty cell is null, not "".
                var raw = otherIdx[k] < fields.Length ? fields[otherIdx[k]] : null;
                values[k] = string.IsNullOrEmpty(raw) ? null : raw;
            }

            byRow[sampleId] = values;
        }

        return (byRow, columns);
    }

    /// <summary>
    /// An identity column, or all-empty when the matrix does not carry it. A placeholder for an
    /// unknown name becomes empty, member by member of a ";"-separated list, so the members stay
    /// aligned with the other identity columns.
    /// </summary>
    private static string[] IdentityColumn(ParquetTable table, string name, int nFeatures) =>
        table.HasColumn(name)
            ? table.GetString(name).Select(NamePlaceholder.Clean).ToArray()
            : Enumerable.Repeat(string.Empty, nFeatures).ToArray();

    /// <summary>
    /// The first of <paramref name="candidates"/> that names something, with the empty members of a
    /// ";"-separated list left out; empty when none does.
    /// </summary>
    private static string FirstNamed(params string[] candidates)
    {
        foreach (var c in candidates)
        {
            var named = c.Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            if (named.Length > 0)
                return string.Join(";", named);
        }
        return string.Empty;
    }

    /// <summary>
    /// The members of a ";"-separated group column. A shared peptide lists every protein group it
    /// belongs to; which one quantifies it is a decision about the rollup, not a fact about the
    /// peptide, so all of them are kept and the caller decides.
    /// </summary>
    private static IReadOnlyList<string> SplitGroups(string? value) =>
        string.IsNullOrEmpty(value)
            ? Array.Empty<string>()
            : value.Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
}
