using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// What a finished run recorded about its marker normalization: the per-sample score and the per-marker
/// PC1 loadings, read back from <c>marker_normalization.csv</c>.
///
/// <para>This exists so the score can be LOOKED AT rather than trusted. The panel has to be
/// proportional to how much of the marked material a sample contributed and <b>not</b> to the
/// phenotype - a marker that tracks the biology gets the finding regressed out along with the capture,
/// and nothing in the pipeline can detect that for you. The two things that give it away are whether
/// the score separates the study's own groups, and whether one marker is carrying the whole axis; both
/// are visible from what this type returns.</para>
///
/// <para>The file is written by Stage 5a. Reading it is deliberately forgiving: a run that predates the
/// stage, or one that did not enable it, simply has no file, and that is not an error.</para>
/// </summary>
public sealed class MarkerNormalizationReport
{
    /// <summary>
    /// The flag column Stage 5a stamps on both corrected outputs, marking the features that DEFINED the
    /// score. Their residual is near zero by construction, so they must be excluded from results - and
    /// they are also exactly the rows a diagnostic plot of the panel should select.
    /// </summary>
    public const string MarkerColumn = "normalization_marker";

    /// <summary>The file Stage 5a writes into the output directory.</summary>
    public const string FileName = "marker_normalization.csv";

    private MarkerNormalizationReport(
        IReadOnlyList<string> samples, IReadOnlyList<double> scores,
        IReadOnlyList<string> markerNames, IReadOnlyList<double> loadings)
    {
        Samples = samples;
        Scores = scores;
        MarkerNames = markerNames;
        Loadings = loadings;
    }

    /// <summary>Replicate ids, in the order the run scored them.</summary>
    public IReadOnlyList<string> Samples { get; }

    /// <summary>Per-sample marker score; higher means more of the marked material.</summary>
    public IReadOnlyList<double> Scores { get; }

    /// <summary>Marker names, aligned with <see cref="Loadings"/>.</summary>
    public IReadOnlyList<string> MarkerNames { get; }

    /// <summary>PC1 loading per marker. Signs are meaningful: markers can and do oppose each other.</summary>
    public IReadOnlyList<double> Loadings { get; }

    /// <summary>
    /// Read the report from an output directory, or null when the run did not use marker normalization
    /// (no file), or the file is unreadable. Never throws - a diagnostic plot must not be able to take
    /// the window down.
    /// </summary>
    public static MarkerNormalizationReport? Read(string outputDir)
    {
        try
        {
            var path = Path.Combine(outputDir, FileName);
            if (!File.Exists(path))
                return null;
            return Parse(File.ReadAllLines(path));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FormatException)
        {
            return null;
        }
    }

    /// <summary>
    /// Parse the file's two blocks: <c>sample_id,marker_score</c> rows, then the loadings, which are
    /// written as <c>#</c> comment lines so the file still reads as a plain two-column CSV in a
    /// spreadsheet. Returns null when neither block yields anything usable.
    /// </summary>
    public static MarkerNormalizationReport? Parse(IReadOnlyList<string> lines)
    {
        var samples = new List<string>();
        var scores = new List<double>();
        var markers = new List<string>();
        var loadings = new List<double>();

        foreach (var raw in lines)
        {
            var line = raw.Trim();
            if (line.Length == 0)
                continue;

            if (line.StartsWith('#'))
            {
                var comment = line.TrimStart('#').Trim();
                var parts = CsvLine.Split(comment);
                // The block's own header ("marker,loading") is a comment too; skip anything that is
                // not a name followed by a number.
                if (parts.Length >= 2 && TryNumber(parts[1], out var loading))
                {
                    markers.Add(parts[0]);
                    loadings.Add(loading);
                }
                continue;
            }

            var row = CsvLine.Split(line);
            if (row.Length >= 2 && TryNumber(row[1], out var score))
            {
                samples.Add(row[0]);
                scores.Add(score);
            }
        }

        return samples.Count == 0 && markers.Count == 0
            ? null
            : new MarkerNormalizationReport(samples, scores, markers, loadings);
    }

    /// <summary>
    /// Loadings ordered by contribution, largest magnitude first - which is the order that answers "is
    /// one marker carrying this axis on its own".
    /// </summary>
    public IReadOnlyList<(string Marker, double Loading)> LoadingsByMagnitude() =>
        MarkerNames
            .Select((m, i) => (Marker: m, Loading: i < Loadings.Count ? Loadings[i] : 0.0))
            .OrderByDescending(t => Math.Abs(t.Loading))
            .ToList();

    /// <summary>
    /// How concentrated the axis is: the largest marker's share of the total absolute loading. A panel
    /// where one marker dominates is a single-protein normalization wearing a panel's clothes; NaN when
    /// there are no loadings to judge.
    /// </summary>
    public double LargestLoadingShare()
    {
        var total = Loadings.Sum(Math.Abs);
        return total > 0 ? Loadings.Max(Math.Abs) / total : double.NaN;
    }

    private static bool TryNumber(string s, out double value) =>
        double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
}
