using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// What a run is about to write over, decided stage by stage.
/// </summary>
/// <remarks>
/// <para>Pointing PRISM at a directory that already holds a cohort's results said nothing at all. It
/// would recompute what the settings changed and replace the rest without a word; the person who did
/// it by pasting the wrong path found out from the file timestamps.</para>
///
/// <para><b>It asks the question the pipeline asks.</b> The first version of this compared the whole
/// recorded config's YAML against the whole current one, which is wrong in both directions: it fired
/// on keys that change nothing any stage reads, and it stayed silent when an input file had been
/// edited in place under settings that had not moved. The comparison is now per stage, over
/// <see cref="StageDependencies"/> - the same declarations <see cref="StageCache"/> keys on - and the
/// files it names are the ones those stages actually recorded, read back from
/// <c>stage_cache.json</c> rather than assumed.</para>
///
/// <para><b>Exact where it can be, a prediction where it cannot, and it says which.</b> Given the
/// input files, the two most expensive stages are answered rather than guessed, each the way the
/// pipeline answers it: the merge against its own sidecar beside <c>merged_data</c> (it is not in
/// the stage cache - its fingerprint appears there only as the rollup's upstream ingredient), and
/// the transition rollup through <see cref="StageCache.CanReuse"/>. That is what catches an input
/// file edited in place under settings that did not move. The three stages below them fold in the
/// RESOLVED sample types and batches, which do not exist until the merge has run, so their
/// fingerprints cannot be computed in advance at all; those are compared on their declared settings
/// alone. The warning can therefore under-report, and says so; a stage whose settings changed is
/// always listed.</para>
/// </remarks>
public sealed record ExistingResults(
    bool Any,
    string? Version,
    string? Date,
    bool SameVersion,
    IReadOnlyList<string> Recomputed,
    IReadOnlyList<string> Files,
    bool Measured)
{
    /// <summary>Nothing was there to begin with.</summary>
    public static readonly ExistingResults None =
        new(false, null, null, true, Array.Empty<string>(), Array.Empty<string>(), false);

    /// <summary>
    /// The stages in the order the pipeline runs them. Each depends on the ones before it, so the
    /// first one to change carries everything after it with it - which is exactly how the
    /// fingerprints chain through <c>upstream</c>. Marker normalization is last because it takes
    /// both the peptide and the protein matrix.
    /// </summary>
    private static readonly string[] Chain =
    {
        StageDependencies.Merge,
        StageDependencies.TransitionRollup,
        StageDependencies.PeptideNormalize,
        StageDependencies.ProteinRollup,
        StageDependencies.ProteinNormalize,
        StageDependencies.MarkerNormalize,
    };

    /// <summary>Whether every stage would reuse what is already here.</summary>
    public bool SameSettings => Recomputed.Count == 0;

    /// <summary>Whether this run would replace results that differ from what it produces.</summary>
    public bool WouldReplace => Any && Files.Count > 0;

    /// <summary>The warning, or null when there is nothing worth saying.</summary>
    public string? Warning()
    {
        if (!WouldReplace)
            return null;

        var what = Version is null
            ? "a previous run"
            : $"a run of PRISM {Version}" + (Date is null ? "" : $" from {Date}");
        var why = SameVersion
            ? $"different settings for {Describe(Recomputed)}"
            : "a different version of PRISM, which recomputes every stage";
        var names = string.Join(", ", Files.Take(4))
            + (Files.Count > 4 ? $" and {Files.Count - 4:N0} more" : "");

        return $"This output directory already holds results from {what}. This run uses {why}, so "
            + $"those results will be replaced: {names}. Stages whose inputs and settings are "
            + "unchanged are reused rather than recomputed"
            + (Measured
                ? "."
                : " - and a stage can also be recomputed for a reason that is only visible once the "
                  + "run has started, such as an input file edited in place, so this list is a lower "
                  + "bound.");
    }

    /// <summary>Stage ids as a reader would name them.</summary>
    private static string Describe(IReadOnlyList<string> stages) =>
        string.Join(", ", stages.Select(s => s switch
        {
            StageDependencies.Merge => "the merge",
            StageDependencies.TransitionRollup => "the transition rollup",
            StageDependencies.PeptideNormalize => "peptide normalization",
            StageDependencies.ProteinRollup => "the protein rollup",
            StageDependencies.ProteinNormalize => "protein normalization",
            StageDependencies.MarkerNormalize => "marker normalization",
            _ => s,
        }));

    /// <summary>
    /// The files a completed run leaves that a reader would recognize as "there are results here".
    /// </summary>
    /// <remarks>
    /// Two jobs. It decides whether the directory holds results at all, and it names them when no
    /// stage cache was recorded. Deliberately the REPORTED outputs rather than every file:
    /// intermediates and caches are working state a re-run is expected to churn, and listing them
    /// would bury the two files someone actually cares about losing.
    /// </remarks>
    private static readonly string[] Reported =
    {
        "corrected_peptides.parquet",
        "corrected_proteins.parquet",
        "corrected_peptides.csv",
        "corrected_proteins.csv",
        "protein_groups.csv",
        "qc_report.html",
    };

    /// <summary>
    /// Look at <paramref name="outputDir"/> and decide what <paramref name="config"/> would replace.
    /// </summary>
    /// <param name="inputs">
    /// The report files this run will merge, when they are known. Given them, the merge and the
    /// transition rollup are answered exactly rather than predicted - including an input edited in
    /// place under settings that did not change. The tool does not have them at the point it asks
    /// (the documents have not been exported yet), which is why they are optional.
    /// </param>
    /// <remarks>
    /// Never throws: an unreadable or half-written <c>parameters.json</c> means the run cannot be
    /// compared, which is reported as "different" rather than as "fine" - the safe direction when the
    /// question is whether someone is about to lose a cohort.
    /// </remarks>
    public static ExistingResults Inspect(
        string outputDir, PrismConfig config, IReadOnlyList<string>? inputs = null)
    {
        if (string.IsNullOrWhiteSpace(outputDir) || !Directory.Exists(outputDir))
            return None;

        var present = Reported
            .Where(name => File.Exists(Path.Combine(outputDir, name)))
            .ToArray();
        if (present.Length == 0)
            return None;

        var provenance = Path.Combine(outputDir, Provenance.FileName);
        if (!File.Exists(provenance))
            return new ExistingResults(true, null, null, false, Chain.ToArray(), present, false);

        // ONE read of the file, on a directory that is routinely a network share.
        string json;
        string? version = null;
        string? date = null;
        try
        {
            json = File.ReadAllText(provenance);
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.TryGetProperty("pipeline_version", out var v))
                version = v.GetString();
            if (doc.RootElement.TryGetProperty("processing_date", out var d))
                date = d.GetString();
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            return new ExistingResults(true, null, null, false, Chain.ToArray(), present, false);
        }

        PrismConfig recorded;
        try
        {
            // ConfigFromJson, not LoadConfig: the latter redirects a FASTA whose original path has
            // gone to the copy the run archived, which is right for re-running and wrong for
            // comparing - it would report "different settings" for the config that produced these
            // very results, and only once the archive had become load-bearing.
            recorded = Provenance.ConfigFromJson(json, provenance);
        }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException
                                       or NotSupportedException or ArgumentException)
        {
            return new ExistingResults(true, version, date, false, Chain.ToArray(), present, false);
        }

        var sameVersion = string.Equals(version, PrismVersion.Current, StringComparison.Ordinal);
        var cache = StageCache.Load(outputDir);

        // Both exact checks need both things: the merge needs the input files, the transition rollup
        // needs a recorded stage cache. With either missing the answer is a prediction and the
        // warning has to say so rather than presenting a lower bound as a complete list.
        var exact = inputs is { Count: > 0 } && !cache.IsEmpty;

        // A version change invalidates every stage by construction: Fingerprint folds PrismVersion in
        // deliberately, because a change to a rollup's arithmetic leaves no trace in the config.
        int from;
        try
        {
            from = sameVersion ? FirstChanged(recorded, config, cache, inputs, outputDir) : 0;
        }
        catch (Exception ex) when (ex is IOException or ArgumentException or NotSupportedException
                                       or UnauthorizedAccessException or System.Security.SecurityException)
        {
            // Stamping the input files touches the filesystem, and a path the caller has not
            // validated yet can be rejected outright. This runs BEFORE the pipeline, whose job it is
            // to report that properly - so the check must not be what fails the run. Treat it as
            // "cannot tell", which reports rather than reassures.
            from = 0;
        }
        if (from < 0)
        {
            return new ExistingResults(
                true, version, date, true, Array.Empty<string>(), Array.Empty<string>(), exact);
        }

        var recomputed = Chain.Skip(from).ToArray();
        var files = recomputed
            .SelectMany(cache.OutputsOf)
            .Select(o => Path.IsPathRooted(o) ? Path.GetFileName(o) : o)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Where(o => File.Exists(Path.Combine(outputDir, o)))
            .OrderBy(o => o, StringComparer.Ordinal)
            .ToArray();

        // A directory written before the stage cache existed, or one whose cache was cleared, has
        // nothing to name - fall back to the outputs a reader would recognize.
        return new ExistingResults(
            true, version, date, sameVersion, recomputed,
            files.Length > 0 ? files : present, exact);
    }

    /// <summary>
    /// The index in <see cref="Chain"/> of the first stage this run would recompute, or -1 when it
    /// would recompute none.
    /// </summary>
    private static int FirstChanged(
        PrismConfig recorded, PrismConfig config, StageCache cache, IReadOnlyList<string>? inputs,
        string outputDir)
    {
        // Exact for as far as the chain can be computed without having run: the merge folds in a
        // stamp of the input files and its own declared keys, and the transition rollup folds in the
        // merge. Everything below them needs the resolved sample types and batches, which only exist
        // once the merge has produced them - so this is where certainty stops, not where care stops.
        if (inputs is { Count: > 0 })
        {
            // The merge is NOT in the stage cache - it keeps its own sidecar beside merged_data, and
            // its fingerprint appears in stage_cache.json only as the rollup's upstream ingredient.
            // Asking CanReuse about it therefore always says no, and a warning that fires on every
            // re-run is the failure this whole check exists to avoid.
            var source = SourceFingerprint.Compute(inputs)
                + "|" + StageDependencies.Values(StageDependencies.Merge, config);
            var mergeFp = StageCache.Fingerprint(
                StageDependencies.Merge, config, extraInputs: new[] { source });

            var mergedPath = Path.Combine(outputDir, "merged_data");
            var merged = SourceFingerprint.TryRead(mergedPath + ".cache.json");
            if (merged is null
                || !string.Equals(merged.Fingerprint, source, StringComparison.Ordinal)
                || !MergedDataset.Exists(mergedPath))
            {
                return 0;
            }

            var rollupFp = StageCache.Fingerprint(
                StageDependencies.TransitionRollup, config, upstream: new[] { mergeFp });
            if (!cache.IsEmpty && !cache.CanReuse(StageDependencies.TransitionRollup, rollupFp))
                return 1;
        }

        for (var i = 0; i < Chain.Length; i++)
        {
            if (!string.Equals(
                    StageDependencies.Values(Chain[i], recorded),
                    StageDependencies.Values(Chain[i], config),
                    StringComparison.Ordinal))
            {
                return i;
            }
        }
        return -1;
    }
}
