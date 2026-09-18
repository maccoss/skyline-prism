using System;
using System.Collections.Generic;
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

    private DifferentialDataset(FeatureLevel level, double[,] exprLog2, string[] featureIds,
        string[] featureLabels, string[] sampleIds, string idColumn, string labelColumn,
        IReadOnlyList<string> metadataColumns, Dictionary<string, string?[]> metaByColumn)
    {
        Level = level;
        ExprLog2 = exprLog2;
        FeatureIds = featureIds;
        FeatureLabels = featureLabels;
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

    /// <summary>Human-readable feature labels (gene names for proteins); parallel to <see cref="FeatureIds"/>.</summary>
    public string[] FeatureLabels { get; }

    /// <summary>Sample identifiers (matrix columns), matching the metadata's sample ids.</summary>
    public string[] SampleIds { get; }

    /// <summary>The parquet column used as the feature id.</summary>
    public string IdColumn { get; }

    /// <summary>The parquet column used as the feature label.</summary>
    public string LabelColumn { get; }

    /// <summary>Metadata columns available for building a contrast (e.g. sample_type, batch, ...).</summary>
    public IReadOnlyList<string> MetadataColumns => _metadataColumns;

    /// <summary>Metadata values for <paramref name="column"/>, aligned to <see cref="SampleIds"/>.</summary>
    public string?[] MetadataValues(string column) =>
        _metaByColumn.TryGetValue(column, out var v)
            ? v
            : throw new ArgumentException($"Unknown metadata column '{column}'.", nameof(column));

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

        var lines = File.ReadAllLines(clinicalCsvPath);
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
        var sampleCols = alignment.Select(a => a.Col).ToArray();
        var alignedMeta = alignment.Select(a => a.Meta).ToArray();
        var matched = new HashSet<string>(sampleCols, StringComparer.Ordinal);
        var annotCols = table.ColumnNames.Where(c => !matched.Contains(c)).ToList();

        var idColumn = ChooseColumn(table, level == FeatureLevel.Protein
            ? new[] { "protein_group" }
            : new[] { "PeptideModifiedSequenceUnimodIds" }, annotCols);
        var labelColumn = level == FeatureLevel.Protein && table.HasColumn("leading_gene_name")
            ? "leading_gene_name"
            : idColumn;

        var nFeatures = table.RowCount;
        var featureIds = table.GetString(idColumn).Select(s => s ?? string.Empty).ToArray();
        var featureLabels = table.GetString(labelColumn).Select(s => s ?? string.Empty).ToArray();

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

        return new DifferentialDataset(level, exprLog2, featureIds, featureLabels, sampleCols,
            idColumn, labelColumn, metaColumns, metaByColumn);
    }

    /// <summary>
    /// Align the matrix's sample columns to the metadata rows. Exact <c>sample_id</c> match is
    /// preferred; if it matches nothing, fall back to the bare replicate name (the part before the
    /// <c>__@__</c> document separator), which equals the metadata <c>sample</c> value. The fallback
    /// exists because some PRISM runs wrote the corrected matrix with one document/batch stem
    /// (e.g. <c>...__@__PRISM</c>) but <c>sample_metadata.csv</c> with another (e.g.
    /// <c>...__@__merged_data</c>); the bare replicate name is identical in both. A bare name matched
    /// to more than one metadata row is ambiguous and skipped. Returned columns keep the matrix's
    /// column names (they index the matrix), each paired with its resolved metadata row.
    /// </summary>
    internal static List<(string Col, string?[] Meta)> AlignSampleColumns(
        IReadOnlyList<string> parquetCols,
        IReadOnlyDictionary<string, string?[]> sampleIdToRow,
        IReadOnlyList<string> metaColumns)
    {
        var exact = new List<(string, string?[])>();
        foreach (var c in parquetCols)
            if (sampleIdToRow.TryGetValue(c, out var row))
                exact.Add((c, row));
        if (exact.Count > 0)
            return exact;

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

        var result = new List<(string, string?[])>();
        foreach (var c in parquetCols)
            if (bareToRows.TryGetValue(StripDocSuffix(c), out var lst) && lst.Count == 1)
                result.Add((c, lst[0]));

        return result;
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
        var lines = File.ReadAllLines(path);
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
}
