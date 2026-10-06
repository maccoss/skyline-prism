using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.DifferentialAnalysis.Detection;

/// <summary>Which detection-rate test a <see cref="DetectionAnalysis"/> ran.</summary>
public enum DetectionMethod
{
    /// <summary>Two-sided Fisher exact, unpaired, Benjamini-Hochberg across peptides.</summary>
    FisherExact,

    /// <summary>McNemar's exact test over subjects matched across the arms.</summary>
    McNemarPaired,

    /// <summary>Firth-penalized logistic regression adjusted for covariates (unpaired).</summary>
    FirthGlm,
}

/// <summary>Why a paired design's detection ran unpaired, when it did.</summary>
public enum UnpairedReason
{
    /// <summary>It did not: the design is unpaired, or McNemar ran.</summary>
    None,

    /// <summary>Covariates were set, and a paired adjusted test (conditional logistic) is not implemented.</summary>
    Covariates,

    /// <summary>No subject could be matched across the arms, or no subject column was given.</summary>
    NoMatchedSubjects,

    /// <summary>Subjects were matched, but no matched pair has both its samples in merged_data.</summary>
    NoPairInMergedData,

    /// <summary>
    /// The design is blocked by subject and some subject has several samples in the arms, which the
    /// detection tests - all of which count samples as independent - cannot take into account.
    /// </summary>
    RepeatedSubjects,
}

/// <summary>
/// No sample of an arm is in the detection matrix, so there is nothing to test. Its own type so a caller
/// can tell "your arms are not in merged_data" from a failure, without reading the message.
/// </summary>
public sealed class DetectionSamplesNotFoundException : ArgumentException
{
    /// <summary>Create the exception with its standard message.</summary>
    public DetectionSamplesNotFoundException()
        : base("The selected samples were not found in the detection matrix.")
    {
    }
}

/// <summary>The outcome of <see cref="DetectionAnalysis.Run"/>: which test ran, and on what.</summary>
public sealed class DetectionAnalysisResult
{
    internal DetectionAnalysisResult(DetectionMethod method, IReadOnlyList<DetectionRow> rows,
        int nA, int nB, int droppedSamples, UnpairedReason unpairedReason,
        IReadOnlyList<DetectionPairedRow>? pairedRows = null, int pairsNotInMerged = 0,
        IReadOnlyList<string>? pairMessages = null, DetectionGlmResult? glm = null,
        int samplesWithoutSubject = 0)
    {
        SamplesWithoutSubject = samplesWithoutSubject;
        Method = method;
        Rows = rows;
        NA = nA;
        NB = nB;
        DroppedSamples = droppedSamples;
        UnpairedReason = unpairedReason;
        PairedRows = pairedRows;
        PairsNotInMerged = pairsNotInMerged;
        PairMessages = pairMessages ?? Array.Empty<string>();
        Glm = glm;
    }

    /// <summary>The test that ran.</summary>
    public DetectionMethod Method { get; }

    /// <summary>
    /// One row per peptide in the unpaired shape, whichever test ran, for the plot, the table and the
    /// CSV. Under McNemar each arm's n is the pair count, which is what a paired rate is out of; empty
    /// when the adjusted model was not identifiable.
    /// </summary>
    public IReadOnlyList<DetectionRow> Rows { get; }

    /// <summary>Samples per arm in the detection matrix, or matched pairs under McNemar.</summary>
    public int NA { get; }

    /// <inheritdoc cref="NA"/>
    public int NB { get; }

    /// <summary>Arm samples that are absent from merged_data and so took no part.</summary>
    public int DroppedSamples { get; }

    /// <summary>
    /// Picked samples a blocked design left out for having no subject - the same ones the
    /// differential leaves out. Counted apart from <see cref="DroppedSamples"/>, which are samples
    /// missing from merged_data.
    /// </summary>
    public int SamplesWithoutSubject { get; }

    /// <summary>Why a paired design ran an unpaired test, or <see cref="Detection.UnpairedReason.None"/>.</summary>
    public UnpairedReason UnpairedReason { get; }

    /// <summary>McNemar's rows, carrying the discordant counts only the paired test has.</summary>
    public IReadOnlyList<DetectionPairedRow>? PairedRows { get; }

    /// <summary>Matched pairs dropped because either half is absent from merged_data.</summary>
    public int PairsNotInMerged { get; }

    /// <summary>What <see cref="PairedSamples.Resolve"/> could not match, and why.</summary>
    public IReadOnlyList<string> PairMessages { get; }

    /// <summary>The Firth GLM's full result - log odds ratios, covariates used, identifiability.</summary>
    public DetectionGlmResult? Glm { get; }

    /// <summary>False only when the adjusted model is confounded (group collinear with covariates).</summary>
    public bool Identifiable => Glm?.Identifiable ?? true;
}

/// <summary>
/// Choosing and running the detection-rate test a contrast's design calls for, over the binary
/// detection matrix from merged_data.
/// </summary>
/// <remarks>
/// <para>One implementation for the Detection pane, the quant report and <c>prism differential
/// --report</c>. It used to live in the pane alone, so the report ran Fisher whatever the design
/// and the CLI had no detection at all.</para>
/// <para>The choice, in the pane's long-standing order:</para>
/// <list type="bullet">
/// <item><b>Covariates set</b> - the Firth-penalized GLM, unpaired whatever the design. Adjusting a
/// paired binary outcome needs conditional logistic regression, which is not implemented.</item>
/// <item><b>Paired, no covariates</b> - McNemar's exact test over the matched subjects. Only the
/// discordant pairs carry information, so their counts are kept.</item>
/// <item><b>Otherwise</b> - Fisher exact. Under a paired design that happens only when no subject
/// could be matched, and <see cref="DetectionAnalysisResult.UnpairedReason"/> says so.</item>
/// </list>
/// <para>Arms and subjects are given in the DATASET's column space (as the contrast was resolved);
/// the detection matrix has its own column order, and every sample is re-expressed in it here.</para>
/// </remarks>
public static class DetectionAnalysis
{
    /// <summary>
    /// Run the design's detection test.
    /// </summary>
    /// <param name="detection">The binary peptide x sample matrix (see <see cref="DetectionMatrix.Load"/>).</param>
    /// <param name="dataset">The loaded contrast dataset, whose sample ids and metadata the arms index.</param>
    /// <param name="groupA">Arm A's dataset columns, as picked (not pre-paired).</param>
    /// <param name="groupB">Arm B's dataset columns, as picked.</param>
    /// <param name="design">The contrast's design. A trend has no two groups and is refused.</param>
    /// <param name="subjectLabels">Per dataset sample, the subject key; used only under a paired design.</param>
    /// <param name="covariateColumns">Metadata columns to adjust for; null or empty for none.</param>
    /// <param name="correction">The correction McNemar's p-values get (Fisher and Firth use BH).</param>
    /// <exception cref="DetectionSamplesNotFoundException">No arm sample is present in merged_data.</exception>
    /// <exception cref="ArgumentException">A trend design.</exception>
    public static DetectionAnalysisResult Run(
        DetectionMatrixData detection, DifferentialDataset dataset,
        IReadOnlyList<int> groupA, IReadOnlyList<int> groupB, DifferentialDesign design,
        IReadOnlyList<string?>? subjectLabels, IReadOnlyList<string>? covariateColumns,
        MultipleTesting correction)
    {
        if (design is DifferentialDesign.LinearTrend or DifferentialDesign.LinearTrendWithinSubject)
            throw new ArgumentException("Detection compares two groups, and a trend design has none.");

        // A blocked design tests the samples that have a subject - the same ones the differential
        // fits and the plots draw - so the two sections of one report agree on n.
        var repeatsASubject = false;
        var withoutSubject = 0;
        if (design == DifferentialDesign.BlockedBySubject && subjectLabels is not null)
        {
            var blocked = BlockedSamples.Resolve(subjectLabels, groupA, groupB);
            (groupA, groupB) = (blocked.A, blocked.B);
            repeatsASubject = BlockedSamples.RepeatsASubject(blocked);
            withoutSubject = blocked.Dropped;
        }

        var detIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < detection.SampleIds.Length; i++)
            detIndex[detection.SampleIds[i]] = i;

        List<int> ToDet(IReadOnlyList<int> cols) => cols
            .Select(j => dataset.SampleIds[j]).Where(detIndex.ContainsKey).Select(s => detIndex[s]).ToList();
        var aCols = ToDet(groupA);
        var bCols = ToDet(groupB);
        if (aCols.Count == 0 || bCols.Count == 0)
            throw new DetectionSamplesNotFoundException();

        var dropped = groupA.Count - aCols.Count + (groupB.Count - bCols.Count);
        var covariates = CovariatesFor(dataset, covariateColumns, detection.SampleIds);
        var paired = design == DifferentialDesign.Paired;
        var matchedButMissing = 0;

        if (covariates is null && paired && subjectLabels is not null)
        {
            var (pairs, pairMessages) = PairedSamples.Resolve(subjectLabels, groupA, groupB);

            // A pair with either half missing from merged_data cannot be tested at all.
            var detPairs = pairs
                .Select(pair => (
                    A: detIndex.TryGetValue(dataset.SampleIds[pair.AColumn], out var ia) ? ia : -1,
                    B: detIndex.TryGetValue(dataset.SampleIds[pair.BColumn], out var ib) ? ib : -1,
                    pair.Subject))
                .Where(x => x.A >= 0 && x.B >= 0)
                .Select(x => new SamplePair(x.Subject, x.A, x.B))
                .ToList();

            if (detPairs.Count > 0)
            {
                var pairedRows = DetectionPairedTest.Run(
                    detection.Matrix, detection.PeptideIds, detPairs, correction);
                var rows = pairedRows
                    .Select(r => new DetectionRow(r.PeptideId, r.DetA, r.Pairs, r.DetB, r.Pairs,
                        r.RateA, r.RateB, r.P, r.Q))
                    .ToList();
                return new DetectionAnalysisResult(DetectionMethod.McNemarPaired, rows,
                    detPairs.Count, detPairs.Count, dropped, UnpairedReason.None,
                    pairedRows: pairedRows, pairsNotInMerged: pairs.Count - detPairs.Count,
                    pairMessages: pairMessages);
            }

            // Matched, but every pair lost a half to merged_data: a different reason from "nothing
            // matched", and saying the latter would send someone to fix the wrong column.
            matchedButMissing = pairs.Count;
        }

        var unpaired = repeatsASubject
            ? UnpairedReason.RepeatedSubjects
            : !paired ? UnpairedReason.None
            : covariates is not null ? UnpairedReason.Covariates
            : matchedButMissing > 0 ? UnpairedReason.NoPairInMergedData
            : UnpairedReason.NoMatchedSubjects;

        if (covariates is not null)
        {
            var glm = DetectionGlm.Run(detection.Matrix, detection.PeptideIds, aCols, bCols, covariates);
            var rows = glm.Identifiable
                ? glm.Rows.Select(r => new DetectionRow(r.PeptideId, r.DetA, r.NA, r.DetB, r.NB,
                    r.RateA, r.RateB, r.P, r.Q)).ToList()
                : new List<DetectionRow>();
            return new DetectionAnalysisResult(DetectionMethod.FirthGlm, rows, aCols.Count, bCols.Count,
                dropped, unpaired, glm: glm, samplesWithoutSubject: withoutSubject);
        }

        var fisher = DetectionTest.Run(detection.Matrix, detection.PeptideIds, aCols, bCols);
        return new DetectionAnalysisResult(DetectionMethod.FisherExact, fisher, aCols.Count, bCols.Count,
            dropped, unpaired, pairsNotInMerged: matchedButMissing, samplesWithoutSubject: withoutSubject);
    }

    /// <summary>
    /// The named metadata columns as covariates aligned to <paramref name="targetSampleIds"/> - the
    /// detection matrix's column order, not the dataset's. A sample the dataset does not know gets no
    /// value. Null when no column is named.
    /// </summary>
    public static IReadOnlyList<Covariate>? CovariatesFor(DifferentialDataset dataset,
        IReadOnlyList<string>? columns, IReadOnlyList<string> targetSampleIds)
    {
        if (columns is null || columns.Count == 0)
            return null;

        var indexById = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < dataset.SampleIds.Length; i++)
            indexById[dataset.SampleIds[i]] = i;

        var result = new List<Covariate>(columns.Count);
        foreach (var col in columns)
        {
            var colValues = dataset.MetadataValues(col);
            var aligned = new string?[targetSampleIds.Count];
            for (var k = 0; k < targetSampleIds.Count; k++)
                aligned[k] = indexById.TryGetValue(targetSampleIds[k], out var di) ? colValues[di] : null;
            result.Add(Covariate.FromMetadata(col, aligned));
        }

        return result;
    }

    /// <summary>
    /// Said when a paired design's detection ran unpaired - it must never arrive silently under a
    /// design that says Paired. Null when it did not.
    /// </summary>
    /// <remarks>
    /// The paired form of a detection-rate test is McNemar's, which PRISM does run - but only
    /// unadjusted. Adjusting a paired binary outcome for covariates needs conditional logistic
    /// regression, a different estimator that is not implemented, so a covariate falls back to the
    /// unpaired Firth GLM. That is a reasonable answer to a different question.
    /// </remarks>
    public static string? UnpairedNote(UnpairedReason reason) => reason switch
    {
        UnpairedReason.RepeatedSubjects =>
            "Note: detection does NOT account for repeated samples - the test counts every sample as "
            + "independent, and some subjects contribute several, so its p-values are too small. The "
            + "blocked design's correlation has no counterpart in the detection tests.",
        UnpairedReason.Covariates =>
            "Note: adjusted detection is UNPAIRED - the paired form needs conditional logistic "
            + "regression, which is not implemented - so this uses every sample in the arms, "
            + "including subjects the paired contrast left out. Drop the covariates for "
            + "McNemar's paired test.",
        UnpairedReason.NoMatchedSubjects =>
            "Note: detection is tested UNPAIRED - no subject could be matched across the arms - "
            + "so this uses every sample in them.",
        UnpairedReason.NoPairInMergedData =>
            "Note: detection is tested UNPAIRED - subjects were matched across the arms, but no matched "
            + "pair has both its samples in merged_data - so this uses every sample in the arms that is.",
        _ => null,
    };

    /// <summary>
    /// The test as a reader would name it, for status lines, the report and CSV headers.
    /// </summary>
    public static string Describe(DetectionAnalysisResult result) => result.Method switch
    {
        DetectionMethod.McNemarPaired => $"McNemar's exact test over {result.NA} matched subjects",
        DetectionMethod.FirthGlm =>
            "Firth-penalized GLM adjusted for " + string.Join(", ", result.Glm!.CovariatesUsed),
        _ => "Fisher exact, unpaired, Benjamini-Hochberg",
    };
}
