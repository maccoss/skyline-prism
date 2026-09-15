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
/// <param name="Host">The machine that produced the results, as that run recorded it.</param>
/// <param name="Present">
/// The result files that are in the directory NOW - what a run here destroys. Independent of every
/// judgement about what would be recomputed: see <see cref="OverwritePrompt"/>.
/// </param>
public sealed record ExistingResults(
    bool Any,
    string? Version,
    string? Date,
    bool SameVersion,
    IReadOnlyList<string> Recomputed,
    IReadOnlyList<string> Files,
    bool Measured,
    bool InputsChanged = false,
    string? Host = null,
    IReadOnlyList<string>? Present = null)
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

    /// <summary>
    /// What to ask before running here, or null when the directory holds no results to lose.
    /// </summary>
    /// <remarks>
    /// <para><b>Presence, not difference.</b> <see cref="Warning"/> answers "would this run produce
    /// something different from what is here", which is the right question for a log and the wrong one
    /// for a person: a re-run that recomputes the identical numbers still deletes and rewrites every
    /// file in the folder, and if those files are somebody's finished analysis they are just as gone.
    /// So this fires whenever a completed analysis is there, whatever the settings say.</para>
    ///
    /// <para>The default output directory is <c>&lt;document folder&gt;/PRISM-Output</c>, so landing on
    /// a previous analysis takes no mistake at all - it is what happens unless the path is changed.
    /// That is the case this exists for, and it is why the question is asked every time rather than
    /// only when something looks unusual.</para>
    /// </remarks>
    public string? OverwritePrompt()
    {
        var files = Present ?? Files;
        if (!Any || files.Count == 0)
            return null;

        var names = string.Join(", ", files.Take(4))
            + (files.Count > 4 ? $" and {files.Count - 4:N0} more" : "");
        // The files are named as what IS THERE, not as what this run will rewrite. A stage whose
        // inputs and settings have not moved is reused rather than recomputed, so a particular file
        // may survive untouched - and naming it as overwritten would be a claim this cannot make.
        // What is certainly replaced is the analysis: its provenance, its report, and every output
        // any stage does recompute.
        return $"This output directory already holds a finished analysis from {Describe()}, "
            + $"including {names}. Running here overwrites it.";
    }

    /// <summary>The run that produced what is here, as a reader would recognize it.</summary>
    private string Describe()
    {
        var what = Version is null
            ? "a previous run"
            : $"a run of PRISM {Version}" + (Date is null ? "" : $" from {Date}");
        // Naming your own machine is noise; naming someone else's is the point - it is the difference
        // between overwriting your own re-run and overwriting a colleague's cohort.
        return Host is { Length: > 0 }
               && !string.Equals(Host, Environment.MachineName, StringComparison.OrdinalIgnoreCase)
            ? what + $" on {Host}"
            : what;
    }

    /// <summary>The warning, or null when there is nothing worth saying.</summary>
    public string? Warning()
    {
        if (!WouldReplace)
            return null;

        var what = Describe();
        // What actually changed, not what usually changes. An input file rewritten under settings
        // that did not move invalidates the merge and everything below it, and reporting that as
        // "different settings" sends the reader to a config diff that shows nothing.
        var why = !SameVersion
            ? "a different version of PRISM, which recomputes every stage"
            : InputsChanged
                ? "input files that have changed since that run, so every stage below the merge"
                : $"different settings for {Describe(Recomputed)}";
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
        stages.Count >= Chain.Length
            // Which is what the three cannot-compare paths report, and naming all six there buries
            // the file list that follows behind a sentence that only means "all of them".
            ? "every stage"
            : string.Join(", ", stages.Select(s => s switch
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
    /// <remarks>
    /// Every extension <c>output.format</c> can produce, not just the default: a cohort written as
    /// tsv leaves <c>corrected_peptides.tsv</c>, and listing only the parquet and csv spellings would
    /// name <c>protein_groups.csv</c> and the report while omitting the two files someone actually
    /// minds losing. (The directory is still recognized as holding results either way -
    /// <c>protein_groups.csv</c> is rewritten on every run whatever the format - so this is about
    /// naming them, not about noticing them.)
    /// </remarks>
    private static readonly string[] Reported =
    {
        "corrected_peptides.parquet",
        "corrected_proteins.parquet",
        "corrected_peptides.csv",
        "corrected_proteins.csv",
        "corrected_peptides.tsv",
        "corrected_proteins.tsv",
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
            return new ExistingResults(true, null, null, false, Chain.ToArray(), present, false, Present: present);

        // ONE read of the file, on a directory that is routinely a network share.
        string json;
        string? version = null;
        string? date = null;
        string? host = null;
        try
        {
            json = File.ReadAllText(provenance);
            using var doc = JsonDocument.Parse(json);
            version = Text(doc.RootElement, "pipeline_version");
            date = Text(doc.RootElement, "processing_date");
            host = Text(doc.RootElement, "host");
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            return new ExistingResults(true, null, null, false, Chain.ToArray(), present, false, Present: present);
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
            return new ExistingResults(
                true, version, date, false, Chain.ToArray(), present, false, Host: host, Present: present);
        }

        var sameVersion = string.Equals(version, PrismVersion.Current, StringComparison.Ordinal);
        var cache = StageCache.Load(outputDir);

        // Both exact checks need both things: the merge needs the input files, the transition rollup
        // needs a recorded stage cache. With either missing the answer is a prediction and the
        // warning has to say so rather than presenting a lower bound as a complete list.
        var exact = inputs is { Count: > 0 } && !cache.IsEmpty;

        // A version change invalidates every stage by construction: Fingerprint folds PrismVersion in
        // deliberately, because a change to a rollup's arithmetic leaves no trace in the config.
        var (from, inputsChanged) = sameVersion
            ? FirstChanged(recorded, config, cache, inputs, outputDir)
            : (0, false);
        if (from < 0)
        {
            // Nothing DIFFERENT would be written, so there is nothing to warn a log about - but the
            // files are still there and a run still rewrites them, which is what Present carries.
            return new ExistingResults(
                true, version, date, true, Array.Empty<string>(), Array.Empty<string>(), exact,
                Host: host, Present: present);
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
            files.Length > 0 ? files : present, exact, inputsChanged, host, present);
    }

    /// <summary>
    /// One string property of the provenance, or null when it is absent or is not a string.
    /// </summary>
    /// <remarks>
    /// <see cref="JsonElement.GetString"/> THROWS on a value of another kind, and
    /// <see cref="InvalidOperationException"/> is not among the exceptions the caller catches - so a
    /// provenance file carrying, say, a numeric host would have aborted the whole pre-run check and
    /// with it the run, over a field used for nothing but a sentence. These three are display
    /// metadata: absent, blank and malformed all mean the same thing here.
    /// </remarks>
    private static string? Text(JsonElement root, string name) =>
        root.TryGetProperty(name, out var e) && e.ValueKind == JsonValueKind.String
            ? e.GetString()
            : null;

    /// <summary>
    /// The index in <see cref="Chain"/> of the first stage this run would recompute, or -1 when it
    /// would recompute none - and whether it was the INPUT FILES rather than the settings that moved.
    /// </summary>
    private static (int From, bool InputsChanged) FirstChanged(
        PrismConfig recorded, PrismConfig config, StageCache cache, IReadOnlyList<string>? inputs,
        string outputDir)
    {
        // Exact for as far as the chain can be computed without having run: the merge folds in a
        // stamp of the input files and its own declared keys, and the transition rollup folds in the
        // merge. Everything below them needs the resolved sample types and batches, which only exist
        // once the merge has produced them - so this is where certainty stops, not where care stops.
        if (inputs is { Count: > 0 })
        {
            // Only the filesystem work is guarded, and deliberately not the comparison below it.
            // Stamping the inputs touches paths the caller has not validated yet, and this runs
            // BEFORE the pipeline whose job it is to report a bad input properly - so it must not be
            // what fails the run. Wrapping the whole method instead would turn a defect in
            // StageDependencies into a silent "warn about everything", which is a bug that hides
            // itself.
            try
            {
                // The merge is NOT in the stage cache - it keeps its own sidecar beside merged_data,
                // and its fingerprint appears in stage_cache.json only as the rollup's upstream
                // ingredient. Asking CanReuse about it therefore always says no, and a warning that
                // fires on every re-run is the failure this whole check exists to avoid.
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
                    // The merge stamp covers the input files AND the merge's own settings. When
                    // those settings are unchanged, the files are what moved - which is the half a
                    // config comparison can never see, and worth naming as itself.
                    var sameMergeSettings = string.Equals(
                        StageDependencies.Values(StageDependencies.Merge, recorded),
                        StageDependencies.Values(StageDependencies.Merge, config),
                        StringComparison.Ordinal);
                    return (0, sameMergeSettings);
                }

                var rollupFp = StageCache.Fingerprint(
                    StageDependencies.TransitionRollup, config, upstream: new[] { mergeFp });
                if (!cache.IsEmpty && !cache.CanReuse(StageDependencies.TransitionRollup, rollupFp))
                    return (1, false);
            }
            catch (Exception ex) when (ex is IOException or ArgumentException or NotSupportedException
                                           or UnauthorizedAccessException
                                           or System.Security.SecurityException)
            {
                // Cannot stamp the inputs, so cannot tell - which reports rather than reassures.
                return (0, false);
            }
        }

        for (var i = 0; i < Chain.Length; i++)
        {
            if (!string.Equals(
                    StageDependencies.Values(Chain[i], recorded),
                    StageDependencies.Values(Chain[i], config),
                    StringComparison.Ordinal))
            {
                return (i, false);
            }
        }
        return (-1, false);
    }
}
