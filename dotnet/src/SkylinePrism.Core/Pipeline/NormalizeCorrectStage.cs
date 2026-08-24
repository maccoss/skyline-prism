using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using SkylinePrism.Core.BatchCorrection;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Normalization;
using SkylinePrism.Core.Qc;
using System.Threading;

namespace SkylinePrism.Core.Pipeline;

/// <summary>Element type of a metadata (non-sample) column carried through the stage.</summary>
internal enum MetaType { Str, Long, Double, Bool }

/// <summary>
/// Everything Stage 2b/2c (peptide normalization + ComBat) or Stage 4b/4c (protein) needs. Bundled
/// into one object rather than a 19-argument call so the in-memory and streaming implementations
/// take the same input and a test can run both over one request.
/// </summary>
internal sealed record NormalizeCorrectRequest
{
    /// <summary>Wide LOG2 input: meta columns + one float64 column per sample.</summary>
    public required string WideParquet { get; init; }

    /// <summary>Meta columns to carry from the input to both outputs, in output order.</summary>
    public required IReadOnlyList<(string Name, MetaType Type)> MetaSpec { get; init; }

    public required IReadOnlyList<string> Samples { get; init; }

    /// <summary>Batch label per sample, aligned to <see cref="Samples"/>.</summary>
    public required IReadOnlyList<string> BatchLabels { get; init; }

    public required bool CombatEnabled { get; init; }

    /// <summary>median | quantile | vsn | rt_lowess | none.</summary>
    public required string NormMethod { get; init; }

    /// <summary>LOG2 intermediate to write (peptide stage), or null to write only the corrected output.</summary>
    public string? InternalLog2Path { get; init; }

    /// <summary>Published output, always LINEAR. .parquet, else delimited by extension.</summary>
    public required string CorrectedLinearPath { get; init; }

    public required Action<string> Report { get; init; }

    /// <summary>Reference / QC sample column indices for the before-vs-after control CV report.</summary>
    public required IReadOnlyList<int> RefIdx { get; init; }

    public required IReadOnlyList<int> QcIdx { get; init; }

    public bool ReferenceAnchored { get; init; }

    /// <summary>Per-sample reference flag for reference-anchored ComBat.</summary>
    public IReadOnlyList<bool>? ReferenceMask { get; init; }

    /// <summary>
    /// What reference-anchored ComBat does with a batch that has no reference samples: "fallback"
    /// (correct it on its own center, location only), "skip" (leave it untouched), or "error".
    /// Held here rather than defaulted at each call site so the in-memory and streaming paths cannot
    /// end up using different policies.
    /// </summary>
    public string NoReferenceBatch { get; init; } = "fallback";

    /// <summary>Meta column holding retention time, used only by rt_lowess.</summary>
    public string? RtColumn { get; init; }

    public double RtLowessFrac { get; init; } = 0.3;

    public int RtLowessGridPoints { get; init; } = 100;

    public bool AutoRevert { get; init; }

    /// <summary>
    /// Columns computed elsewhere and stamped onto the CORRECTED output only, keyed by
    /// <see cref="DerivedKeyColumn"/> (the peptide stage's protein-group columns).
    /// </summary>
    public IReadOnlyList<(string Name, Func<string, string> Value)>? DerivedMeta { get; init; }

    public string? DerivedKeyColumn { get; init; }

    /// <summary>
    /// Optional sink for which implementation ran. Deliberately separate from <see cref="Report"/>:
    /// the parity harness compares the report line for line, and the two implementations have to
    /// produce the same one.
    /// </summary>
    public Action<string>? PathReport { get; init; }

    /// <summary>Checked once per row group, so a Stop lands within a row group rather than a stage.</summary>
    public CancellationToken CancellationToken { get; init; }
}

/// <summary>
/// Stage 2b/2c and 4b/4c: load a wide LOG2 parquet, drop all-NaN feature rows, globally normalize,
/// optionally ComBat-correct, then write the LOG2 "internal" parquet (if a path is given) and the
/// LINEAR corrected output.
/// <para>
/// <see cref="Run"/> is the entry point and picks an implementation; <see cref="RunInMemory"/> is the
/// original one, which materializes the whole feature x sample matrix. It stays directly reachable so
/// a streaming implementation can be diffed against it in one test process - the cross-language
/// goldens are far too loose (3e-2 relative on the corrected outputs) to protect that refactor.
/// </para>
/// </summary>
internal static class NormalizeCorrectStage
{
    /// <summary>
    /// Run the stage, streaming when the request allows it. Returns the number of features written.
    /// Both implementations produce the same numbers; the choice is only about peak memory, so a
    /// request the streaming path does not cover falls back rather than failing.
    /// </summary>
    public static int Run(NormalizeCorrectRequest r)
    {
        var (canStream, reason) = StreamingNormalizeCorrect.Eligibility(r);
        r.PathReport?.Invoke(reason);
        return canStream ? StreamingNormalizeCorrect.Run(r) : RunInMemory(r);
    }

    /// <summary>
    /// Report what ComBat could not estimate from the data, rather than letting it pass silently.
    /// Shared by both implementations so they emit identical lines - the parity harness compares the
    /// report line for line, and a difference here would mean the two disagree about the data.
    /// </summary>
    internal static void ReportComBatDiagnostics(
        Action<string> report, int heldOutFeatures, long unestimableScales)
    {
        if (heldOutFeatures > 0)
            report($"  ComBat: {heldOutFeatures:N0} feature(s) held out (no variance, or absent from a "
                + "batch); passed through uncorrected.");
        if (unestimableScales > 0)
            report($"  ComBat: {unestimableScales:N0} (batch, feature) scale(s) not estimable "
                + "(< 2 observations, or no spread); location corrected only.");
    }

    /// <summary>
    /// The whole-matrix implementation: peak memory is O(features x samples), several times over.
    /// </summary>
    internal static int RunInMemory(NormalizeCorrectRequest r)
    {
        var samples = r.Samples;
        var report = r.Report;

        var table = ParquetTable.Load(r.WideParquet);
        var nAll = table.RowCount;

        // Read matrix + meta.
        var matrixAll = new double[nAll, samples.Count];
        for (var j = 0; j < samples.Count; j++)
        {
            var col = table.GetDouble(samples[j]);
            for (var i = 0; i < nAll; i++)
                matrixAll[i, j] = col[i] ?? double.NaN;
        }

        // The sample columns have been copied into matrixAll and are never read again - only meta columns
        // and the RT column are taken from the table below. Release them now: on a large cohort they are
        // the single biggest live allocation here (double?[] at 16 bytes/cell, twice the matrix), and
        // holding them through normalization + ComBat roughly doubles this stage's peak for nothing.
        table.ReleaseColumns(samples.Where(s => !r.MetaSpec.Any(
            m => string.Equals(m.Name, s, StringComparison.Ordinal))
            && !string.Equals(s, r.RtColumn, StringComparison.Ordinal)));

        // Drop all-NaN rows.
        var keep = new List<int>(nAll);
        for (var i = 0; i < nAll; i++)
        {
            var any = false;
            for (var j = 0; j < samples.Count && !any; j++)
                any = !double.IsNaN(matrixAll[i, j]);
            if (any)
                keep.Add(i);
        }

        // Reuse matrixAll when nothing was dropped (the dense case) instead of copying it.
        var n = keep.Count;
        double[,] matrix;
        if (n == nAll)
        {
            matrix = matrixAll;
        }
        else
        {
            matrix = new double[n, samples.Count];
            for (var row = 0; row < n; row++)
                for (var j = 0; j < samples.Count; j++)
                    matrix[row, j] = matrixAll[keep[row], j];
        }
        matrixAll = null!; // free the [nAll] copy (or clear the alias) - dead from here

        // Control-sample median CV (linear scale) BEFORE normalization/correction (matrix is freed below).
        var beforeRefCv = r.RefIdx.Count >= 2 ? CvMetrics.MedianCv(matrix, r.RefIdx) : double.NaN;
        var beforeQcCv = r.QcIdx.Count >= 2 ? CvMetrics.MedianCv(matrix, r.QcIdx) : double.NaN;

        double[]? rtKept = null;
        if (r.NormMethod is "rt_lowess" && r.RtColumn is not null && table.HasColumn(r.RtColumn))
        {
            var rtAll = table.GetDouble(r.RtColumn);
            rtKept = new double[n];
            for (var row = 0; row < n; row++)
                rtKept[row] = rtAll[keep[row]] ?? double.NaN;
        }

        var normalized = rtKept is not null
            ? Normalizer.RtLowessNormalize(matrix, rtKept, r.RtLowessFrac, r.RtLowessGridPoints)
            : r.NormMethod switch
            {
                "quantile" => Normalizer.QuantileNormalize(matrix),
                "vsn" => Normalizer.VsnNormalize(matrix),
                "none" => matrix,
                _ => Normalizer.MedianNormalize(matrix),
            };
        if (!ReferenceEquals(normalized, matrix))
            matrix = null!; // dead once a distinct normalized matrix exists

        // The internal LOG2 file is written HERE - after normalization, BEFORE ComBat - because it
        // is what the protein arm consumes, and the protein arm must not inherit the peptide arm's
        // batch correction. Correcting at both levels would batch-correct the protein output twice:
        // once through its already-corrected peptide inputs and again at Stage 4c. PRISM's design is
        // one correction per reporting level (CLAUDE.md, "Batch correction at reporting level"), and
        // this is the line that makes that true. It also matches what docs/output_files.md has always
        // said this file is: "peptide quantities after normalization".
        //
        // Writing it here rather than holding the normalized matrix until after ComBat also keeps
        // peak memory unchanged - `normalized` is still freed as soon as correction produces a
        // distinct matrix.
        if (r.InternalLog2Path is not null)
        {
            var normCols = new double[samples.Count][];
            for (var j = 0; j < samples.Count; j++)
            {
                normCols[j] = new double[n];
                for (var row = 0; row < n; row++)
                    normCols[j][row] = normalized[row, j];
            }
            ParquetWideWriter.Write(r.InternalLog2Path, BuildMetaCols(), samples, normCols, n);
        }

        double[,] corrected;
        if (!r.CombatEnabled)
        {
            corrected = normalized;
        }
        else
        {
            var diagnostics = new ComBatDiagnostics();
            var combatOut = r.ReferenceAnchored && r.ReferenceMask is not null && r.ReferenceMask.Any(m => m)
                ? ReferenceAnchoredComBat.Run(
                    normalized, r.BatchLabels, r.ReferenceMask,
                    noReferenceBatch: r.NoReferenceBatch, diagnostics: diagnostics)
                : ComBat.Run(normalized, r.BatchLabels, diagnostics: diagnostics);
            ReportComBatDiagnostics(report, diagnostics.HeldOutFeatures, diagnostics.UnestimableScales);

            // Safety net (opt-in): if ComBat worsened the control CV by >10%, revert to the uncorrected
            // (post-normalization) data; separately warn on reference/QC overfitting.
            if (r.AutoRevert)
            {
                var eval = BatchCorrectionEvaluator.Evaluate(normalized, combatOut, r.QcIdx, r.RefIdx);
                if (eval.OverfittingWarning is not null)
                    report($"  WARNING: ComBat {eval.OverfittingWarning}");
                if (eval.Revert)
                {
                    report($"  ComBat REVERTED: {eval.ControlName} CV worsened "
                        + $"{eval.ControlCvBefore:F1}% -> {eval.ControlCvAfter:F1}% (>10%); keeping uncorrected data.");
                    corrected = normalized;
                }
                else
                {
                    corrected = combatOut;
                }
            }
            else
            {
                corrected = combatOut;
            }
        }
        if (!ReferenceEquals(corrected, normalized))
            normalized = null!; // dead after correction

        // Median control-sample CV before vs after normalization + batch correction (linear scale).
        // Only a type with >= 2 samples is meaningful; skip the other (or both if no controls).
        if (r.RefIdx.Count >= 2)
            report($"  Reference CV (median): {beforeRefCv:F1}% -> {CvMetrics.MedianCv(corrected, r.RefIdx):F1}% (before -> after)");
        if (r.QcIdx.Count >= 2)
            report($"  QC CV (median): {beforeQcCv:F1}% -> {CvMetrics.MedianCv(corrected, r.QcIdx):F1}% (before -> after)");

        // Meta columns (filtered to kept rows). A local function because the internal LOG2 file is
        // written BEFORE correction (see below) and the corrected file after, so both need them.
        var metaCols = BuildMetaCols();

        List<ParquetWideWriter.MetaColumn> BuildMetaCols()
        {
        var metaCols = new List<ParquetWideWriter.MetaColumn>();
        foreach (var (name, type) in r.MetaSpec)
        {
            switch (type)
            {
                case MetaType.Str:
                    var sv = table.GetString(name);
                    metaCols.Add(ParquetWideWriter.Strings(name, keep.Select(i => sv[i] ?? "").ToArray()));
                    break;
                case MetaType.Long:
                    var lv = table.GetLong(name);
                    metaCols.Add(ParquetWideWriter.Longs(name, keep.Select(i => lv[i]).ToArray()));
                    break;
                case MetaType.Double:
                    var dv = table.GetDouble(name);
                    metaCols.Add(ParquetWideWriter.Doubles(name, keep.Select(i => dv[i] ?? double.NaN).ToArray()));
                    break;
                case MetaType.Bool:
                    var bv = table.GetBool(name);
                    metaCols.Add(ParquetWideWriter.Bools(name, keep.Select(i => bv[i]).ToArray()));
                    break;
            }
        }
        return metaCols;
        }

        // Columns computed from another stage rather than read from this file - the peptide output's
        // protein groups, which parsimony knows and the peptide rollup does not. These go on the
        // CORRECTED (published) output ONLY: the internal log2 file is a pipeline intermediate whose
        // readers treat every non-declared column as a sample, and a string column there is read as an
        // abundance and throws.
        var correctedMetaCols = metaCols;
        if (r.DerivedMeta is { Count: > 0 } && r.DerivedKeyColumn is not null && table.HasColumn(r.DerivedKeyColumn))
        {
            var keys = table.GetString(r.DerivedKeyColumn);
            correctedMetaCols = new List<ParquetWideWriter.MetaColumn>(metaCols);
            foreach (var (name, value) in r.DerivedMeta)
                correctedMetaCols.Add(ParquetWideWriter.Strings(
                    name, keep.Select(i => value(keys[i] ?? "")).ToArray()));
        }

        // Corrected output is LINEAR (2^log2).
        var linearCols = new double[samples.Count][];
        for (var j = 0; j < samples.Count; j++)
        {
            linearCols[j] = new double[n];
            for (var row = 0; row < n; row++)
                linearCols[j][row] = Math.Pow(2.0, corrected[row, j]);
        }

        if (r.CorrectedLinearPath.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase))
            ParquetWideWriter.Write(r.CorrectedLinearPath, correctedMetaCols, samples, linearCols, n);
        else
            WriteDelimited(r.CorrectedLinearPath, correctedMetaCols, samples, linearCols, n);

        return n;
    }

    private static void WriteDelimited(
        string path, IReadOnlyList<ParquetWideWriter.MetaColumn> meta,
        IReadOnlyList<string> samples, IReadOnlyList<double[]> sampleCols, int n)
    {
        var delim = path.EndsWith(".tsv", StringComparison.OrdinalIgnoreCase) ? '\t' : ',';
        var sb = new StringBuilder();
        var headers = meta.Select(m => m.Name).Concat(samples);
        sb.Append(string.Join(delim, headers)).Append('\n');
        for (var row = 0; row < n; row++)
        {
            var fields = new List<string>();
            foreach (var m in meta)
                fields.Add(Convert.ToString(m.Values.GetValue(row), CultureInfo.InvariantCulture) ?? "");
            for (var j = 0; j < samples.Count; j++)
                fields.Add(sampleCols[j][row].ToString("R", CultureInfo.InvariantCulture));
            sb.Append(string.Join(delim, fields)).Append('\n');
        }
        File.WriteAllText(path, sb.ToString());
    }
}
