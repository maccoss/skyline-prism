using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.BatchCorrection;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// Derives the batch-corrected residual file from the raw one, by applying the same per-batch SCALE
/// that ComBat applied to the values the residuals belong to.
/// <para>
/// A residual is a deviation from a fitted profile, so ComBat's location terms cancel out of it: with
/// <c>y* = y / sqrt(delta) + c(batch, feature)</c>, the constant disappears from any difference and
/// only the scale survives. So the corrected residual is
/// <c>e* = e / sqrt(deltaStar[batch, feature])</c> - no gamma, no centering.
/// </para>
/// <para>
/// Three cases deliberately scale by 1.0 rather than guessing, each because ComBat itself did nothing
/// there: a feature ComBat held out (no variance, or missing from a batch entirely), a sample in no
/// corrected batch, and a (batch, feature) whose scale was not estimable - fewer than two observations
/// or no spread among them, which is common when a batch carries a single reference injection.
/// </para>
/// </summary>
internal static class ResidualScaler
{
    /// <summary>
    /// Read <paramref name="rawPath"/> and write <paramref name="outPath"/> with every value divided by
    /// the square root of its (batch, feature) scale.
    /// </summary>
    /// <param name="featureKeys">Feature key per matrix row, in matrix row order.</param>
    /// <param name="scaling">
    /// ComBat's scale and index maps, or <c>null</c> when no correction was applied - disabled, or
    /// reverted by <c>auto_revert</c>. Null means every scale is 1.0, so the corrected file is written
    /// as a faithful copy of the raw one rather than being skipped: the file is always present, which
    /// is what makes it safe for a script to read unconditionally.
    /// </param>
    public static void Write(
        string rawPath,
        string outPath,
        IReadOnlyList<string> samples,
        IReadOnlyList<string> featureKeys,
        ComBatScaling? scaling)
    {
        if (!File.Exists(rawPath))
            return;

        using var reader = ParquetColumnReader.Open(rawPath);
        var sampleSet = new HashSet<string>(samples, StringComparer.Ordinal);
        var metaNames = reader.ColumnNames.Where(c => !sampleSet.Contains(c)).ToList();
        if (metaNames.Count == 0)
            throw new InvalidOperationException(
                $"{Path.GetFileName(rawPath)} has no key columns; cannot map rows to ComBat features.");

        // The FIRST meta column is the ComBat feature: peptide for the transition-stage residuals,
        // protein_group for the protein-stage ones. The remaining meta columns pass through untouched.
        var keyCol = metaNames[0];

        // Everything below is indexed by PRESENT sample, not by the caller's sample list - a residual
        // file need not carry every sample, and mixing the two orders would apply one sample's batch
        // scale to another's column.
        var srcIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var j = 0; j < samples.Count; j++)
            srcIndex[samples[j]] = j;

        var presentSamples = samples.Where(reader.HasColumn).ToList();
        var batchOfPresent = new int[presentSamples.Count];
        for (var j = 0; j < presentSamples.Count; j++)
            batchOfPresent[j] = scaling is not null
                                && srcIndex.TryGetValue(presentSamples[j], out var src)
                                && src < scaling.BatchOfSample.Length
                ? scaling.BatchOfSample[src]
                : -1;

        // Feature key -> scale per present sample, precomputed once: the residual file has one row per
        // (feature x transition) or (group x peptide), so a key recurs for every transition and
        // recomputing per row would repeat the same lookups thousands of times.
        var scaleByKey = new Dictionary<string, double[]>(StringComparer.Ordinal);
        for (var row = 0; row < featureKeys.Count; row++)
        {
            var key = featureKeys[row];
            if (key is null || scaleByKey.ContainsKey(key))
                continue;
            var active = scaling is not null && row < scaling.ActiveOfRow.Length
                ? scaling.ActiveOfRow[row]
                : -1;
            var scale = new double[presentSamples.Count];
            for (var j = 0; j < presentSamples.Count; j++)
            {
                var batch = batchOfPresent[j];
                if (scaling is null || active < 0 || batch < 0)
                {
                    scale[j] = 1.0;
                    continue;
                }
                var delta = scaling.DeltaStar[batch, active];
                scale[j] = delta > 0 && !double.IsNaN(delta) ? 1.0 / Math.Sqrt(delta) : 1.0;
            }
            scaleByKey[key] = scale;
        }

        var metaTypes = new List<(string, Type)>();
        using (var probe = reader.OpenRowGroup(0))
        {
            foreach (var m in metaNames)
                metaTypes.Add((m, probe.ReadRaw(m).GetType().GetElementType()!));
        }

        using var writer = StreamingWideWriter.Create(outPath, metaTypes.ToArray(), presentSamples);
        var one = new double[presentSamples.Count];
        Array.Fill(one, 1.0);

        for (var g = 0; g < reader.RowGroupCount; g++)
        {
            using var rg = reader.OpenRowGroup(g);
            var meta = metaNames.Select(rg.ReadRaw).ToList();
            var keys = (string?[])rg.ReadRaw(keyCol);
            var n = rg.RowCount;

            var cols = new double[presentSamples.Count][];
            for (var j = 0; j < presentSamples.Count; j++)
                cols[j] = rg.ReadDoubles(presentSamples[j]);

            for (var i = 0; i < n; i++)
            {
                var scale = keys[i] is { } k && scaleByKey.TryGetValue(k, out var s) ? s : one;
                for (var j = 0; j < presentSamples.Count; j++)
                    cols[j][i] *= scale[j];
            }

            writer.WriteRowGroup(meta, cols);
        }
    }
}
