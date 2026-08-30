using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Normalization;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// Stage 5a: normalization against a set of marker proteins, applied to both corrected outputs.
///
/// <para>Runs LAST, after both arms have been normalized and batch-corrected, for two reasons. The
/// score has to be estimated from data whose per-sample loading is already removed - on raw
/// abundances PC1 loads on injection volume, and residualizing then re-does the loading step with
/// eighteen proteins' worth of noise. And it is estimated at the PROTEIN level and applied to both
/// matrices, because how much marked material a sample contributed is a property of the sample, not
/// of the table being analyzed; re-estimating it from peptides would mostly re-measure the same
/// quantity with more noise.</para>
///
/// <para>The two matrices are read, adjusted and rewritten in place. That is a deliberate extra pass
/// rather than folding the adjustment into the writers: the score does not exist until the protein
/// arm is finished, and the peptide output was written long before that.</para>
/// </summary>
internal static class MarkerNormalizeStage
{
    /// <summary>
    /// The flag column added to both outputs, marking the features that defined the score. Defined on
    /// the public <see cref="MarkerNormalizationReport"/> because the GUI selects these rows to plot the
    /// panel, and a second copy of the name here would be a second thing to keep in step.
    /// </summary>
    public const string MarkerColumn = MarkerNormalizationReport.MarkerColumn;

    public sealed record Result(
        MarkerNormalization.MarkerScore Score,
        IReadOnlyList<string> Missing,
        int ProteinsAdjusted,
        int PeptidesAdjusted);

    /// <summary>
    /// Residualize <paramref name="correctedProteins"/> and <paramref name="correctedPeptides"/> on a
    /// score built from the markers in <paramref name="list"/>. Both files are LINEAR on disk; the fit
    /// is done in log2 and the values written back linear, the scale the rest of PRISM contracts for.
    /// </summary>
    public static Result Run(
        string correctedProteins, string? correctedPeptides, ProteinList list,
        MarkerScoreMethod method, IReadOnlyList<string> samples, Action<string> report)
    {
        // MatcherFor, not BuildMatcher: the latter keeps only VISIBLE lists, which is a plot setting.
        // Every shipped panel ships unticked, so routing normalization through it found zero markers.
        var matcher = ProteinListSet.MatcherFor(list);

        // ---- the marker block, from the protein matrix -------------------------------------------
        var protein = WideMatrix.Read(correctedProteins, ProteinMetaNames);
        var gene = protein.MetaStrings("leading_gene_name");
        var accession = protein.MetaStrings("leading_protein");
        var name = protein.MetaStrings("leading_name");

        var markerRows = new List<int>();
        for (var i = 0; i < protein.RowCount; i++)
            if (matcher.Match(accession?[i], gene?[i], name?[i]) is not null)
                markerRows.Add(i);

        // Which MEMBERS were found, asked of the matcher rather than by comparing strings. Both cheaper
        // ways of answering this are wrong, and both were shipped: comparing a member against the raw
        // identifier columns misses every member the matcher reached through tokenization (a member
        // "H2AC18" against the group "H2AC11 / H2AC18 / H2AJ", "ALBU" against "ALBU_HUMAN", "P02768"
        // against "P02768-2"), and comparing it against the gene symbols found misses every member
        // written as an accession. Asking a one-member matcher uses the identical rules the match itself
        // used, so the answer cannot drift from it.
        var unmatched = list.Members
            .Where(m =>
            {
                var probe = ProteinListSet.MatcherFor(
                    new ProteinList { Name = list.Name, Visible = true, Members = { m } });
                return !markerRows.Any(i => probe.Match(accession?[i], gene?[i], name?[i]) is not null);
            })
            .ToList();

        // Counted in MEMBERS, so the two halves of the sentence share a basis. Reporting the distinct
        // gene symbols found against the member count let the line contradict itself: a protein with an
        // empty gene column counted toward neither, so a panel that matched every member could report
        // "0 of 14 quantified" and list nothing as missing.
        //
        // Counted from the MEMBERS that did not match, not by looking their labels back up in the
        // deduplicated list below: two members may deliberately share a label, and a found member sharing
        // one with a missing member would then be counted missing too.
        var quantified = list.Members.Count - unmatched.Count;

        var missing = unmatched
            .Select(ProteinList.DisplayName)
            // Two members may share a label - the panel deliberately lists some proteins twice - and
            // naming the same protein twice in one line reads as a bug in the report.
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(m => m, StringComparer.OrdinalIgnoreCase)
            .ToList();

        report($"  Markers: {quantified} of {list.Members.Count} quantified from '{list.Name}'"
            + $" ({markerRows.Count} protein rows)"
            + (missing.Count > 0 ? $" (not quantified: {string.Join(", ", missing)})" : "") + ".");

        if (markerRows.Count < MarkerNormalization.MinMarkers)
            throw new InvalidOperationException(
                $"marker_normalization needs at least {MarkerNormalization.MinMarkers} quantified "
                + $"markers from '{list.Name}'; only {markerRows.Count} were found. Check that the list "
                + "uses identifiers this document reports (gene symbol, accession or protein name).");

        var block = new double[markerRows.Count, protein.SampleCount];
        for (var i = 0; i < markerRows.Count; i++)
            for (var j = 0; j < protein.SampleCount; j++)
                block[i, j] = Log2(protein.Values[markerRows[i], j]);

        var score = MarkerNormalization.ComputeScore(
            block, markerRows.Select(i => gene?[i] ?? accession?[i] ?? $"row{i}").ToList(), method);

        var opposing = score.MarkerNames.Where((_, k) => score.Loadings[k] < 0).ToList();
        report($"  Score: {method.ToString().ToLowerInvariant()} of the marker block"
            + (double.IsNaN(score.VarianceExplained)
                ? ""
                : $", {score.VarianceExplained:P1} of marker variance")
            + $"; r={score.CorrelationWithMean:0.00} with the plain marker mean"
            + (opposing.Count > 0
                ? $"; {opposing.Count} marker(s) load opposite: {string.Join(", ", opposing)}"
                : "") + ".");
        if (!double.IsNaN(score.VarianceExplained) && score.VarianceExplained < 0.4)
            report("  WARNING: the markers do not move together (PC1 explains under 40% of their "
                + "variance), so this score is a weak summary of them - treat the result with care.");

        // ---- apply to both matrices ---------------------------------------------------------------
        var proteinsAdjusted = Adjust(protein, score.Score, markerRows, correctedProteins, report);

        var peptidesAdjusted = 0;
        if (correctedPeptides is not null && File.Exists(correctedPeptides))
        {
            var peptide = WideMatrix.Read(correctedPeptides, PeptideMetaNames);
            // The same score, aligned to this file's own column order rather than assumed equal.
            var aligned = AlignScore(score.Score, protein.Samples, peptide.Samples);
            var peptideMarkers = MarkerPeptideRows(peptide, matcher);
            peptidesAdjusted = Adjust(peptide, aligned, peptideMarkers, correctedPeptides, report);
        }

        return new Result(score, missing, proteinsAdjusted, peptidesAdjusted);
    }

    /// <summary>
    /// Residualize, flag the markers, and rewrite the file. Returns the number of features adjusted.
    /// </summary>
    private static int Adjust(
        WideMatrix matrix, double[] score, IReadOnlyList<int> markerRows, string path,
        Action<string> report)
    {
        var log2 = new double[matrix.RowCount, matrix.SampleCount];
        for (var i = 0; i < matrix.RowCount; i++)
            for (var j = 0; j < matrix.SampleCount; j++)
                log2[i, j] = Log2(matrix.Values[i, j]);

        MarkerNormalization.Residualize(log2, score);

        for (var i = 0; i < matrix.RowCount; i++)
            for (var j = 0; j < matrix.SampleCount; j++)
                matrix.Values[i, j] = double.IsNaN(log2[i, j]) ? double.NaN : Math.Pow(2, log2[i, j]);

        // The markers are kept, not dropped - they are real measurements - but their residual is near
        // zero by construction, so any test among them is circular. The flag is how a reader knows.
        var isMarker = new bool[matrix.RowCount];
        foreach (var i in markerRows)
            isMarker[i] = true;
        matrix.SetFlag(MarkerColumn, isMarker);

        matrix.Write(path);
        report($"  Rewrote {Path.GetFileName(path)}: {matrix.RowCount:N0} features residualized, "
            + $"{markerRows.Count:N0} flagged as markers.");
        return matrix.RowCount;
    }

    /// <summary>Peptides belonging to a marker protein, via the group columns the peptide output carries.</summary>
    private static List<int> MarkerPeptideRows(WideMatrix peptide, ProteinListMatcher matcher)
    {
        var rows = new List<int>();
        var gene = peptide.MetaStrings("leading_gene_name");
        var accession = peptide.MetaStrings("leading_protein");
        var name = peptide.MetaStrings("leading_name");
        if (gene is null && accession is null && name is null)
            return rows; // an older peptide file with no group columns: nothing to flag

        for (var i = 0; i < peptide.RowCount; i++)
        {
            // A shared peptide lists every group it belongs to; being in one marker group is enough.
            var hit = Split(accession, i).Any(a => matcher.Match(a, null, null) is not null)
                || Split(gene, i).Any(g => matcher.Match(null, g, null) is not null)
                || Split(name, i).Any(n => matcher.Match(null, null, n) is not null);
            if (hit)
                rows.Add(i);
        }
        return rows;
    }

    private static IEnumerable<string> Split(string[]? column, int row) =>
        column is null || row >= column.Length || string.IsNullOrEmpty(column[row])
            ? Array.Empty<string>()
            : column[row].Split(';', StringSplitOptions.RemoveEmptyEntries).Select(s => s.Trim());

    /// <summary>
    /// The protein-level score, put in another file's sample order. Both matrices come from the same
    /// run and normally agree, but matching by name rather than trusting position is the difference
    /// between a correct adjustment and a silently scrambled one.
    /// </summary>
    private static double[] AlignScore(
        double[] score, IReadOnlyList<string> from, IReadOnlyList<string> to)
    {
        var byName = new Dictionary<string, double>(StringComparer.Ordinal);
        for (var i = 0; i < from.Count && i < score.Length; i++)
            byName[from[i]] = score[i];

        var aligned = new double[to.Count];
        for (var j = 0; j < to.Count; j++)
            aligned[j] = byName.TryGetValue(to[j], out var v) ? v : double.NaN;

        var missing = aligned.Count(double.IsNaN);
        if (missing > 0)
            throw new InvalidOperationException(
                $"{missing} sample(s) in the peptide output have no protein-level marker score. The two "
                + "outputs are from the same run and should carry the same samples.");
        return aligned;
    }

    /// <summary>Linear to log2, with the same floor the analysis this came from used.</summary>
    private static double Log2(double linear) =>
        double.IsNaN(linear) ? double.NaN : Math.Log2(Math.Max(linear, 1e-6));

    private static readonly string[] ProteinMetaNames = Rollup.ProteinRollup.MetadataColumns;

    private static readonly string[] PeptideMetaNames =
    {
        "n_transitions", "mean_rt", "protein_group", "leading_protein", "leading_name",
        "leading_gene_name",
    };

    /// <summary>Write the per-sample score beside the outputs, so the adjustment can be inspected.</summary>
    public static void WriteScoreCsv(
        string path, IReadOnlyList<string> samples, MarkerNormalization.MarkerScore score)
    {
        var sb = new StringBuilder("sample_id,marker_score\n");
        for (var i = 0; i < samples.Count && i < score.Score.Length; i++)
            sb.Append(CsvLine.Quote(samples[i])).Append(',')
              .Append(score.Score[i].ToString("R", CultureInfo.InvariantCulture)).Append('\n');
        sb.Append("\n# marker,loading\n");
        for (var i = 0; i < score.MarkerNames.Count; i++)
            sb.Append("# ").Append(CsvLine.Quote(score.MarkerNames[i])).Append(',')
              .Append(score.Loadings[i].ToString("R", CultureInfo.InvariantCulture)).Append('\n');
        File.WriteAllText(path, sb.ToString());
    }
}
