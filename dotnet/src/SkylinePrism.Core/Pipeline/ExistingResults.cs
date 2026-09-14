using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using SkylinePrism.Core.Config;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// What a run is about to write over, and whether it would be writing anything different.
/// </summary>
/// <remarks>
/// <para>Pointing PRISM at a directory that already holds a cohort's results said nothing at all. It
/// would recompute what the settings changed - <see cref="StageCache"/> gets that right - and
/// replace the rest without a word. The person who did it by pasting the wrong path found out from
/// the file timestamps.</para>
///
/// <para><b>Silent when nothing changes.</b> Re-running the same version with the same settings is
/// how a QC report gets regenerated and how a partial ion accounting gets topped up; warning there
/// would be noise, and noise is how a warning stops being read. So the previous run's version and
/// settings are compared with this one's, and only a genuine difference is reported.</para>
///
/// <para>This is a PREDICTION, from <c>parameters.json</c>, not a promise: the stage cache decides
/// per stage at run time and also weighs the input files, which can change underneath identical
/// settings. It is deliberately the conservative direction - it can warn about a run that turns out
/// to reuse everything, and does not stay silent about one that replaces things.</para>
/// </remarks>
public sealed record ExistingResults(
    bool Any,
    string? Version,
    string? Date,
    bool SameVersion,
    bool SameSettings,
    IReadOnlyList<string> Files)
{
    /// <summary>Nothing was there to begin with.</summary>
    public static readonly ExistingResults None =
        new(false, null, null, true, true, Array.Empty<string>());

    /// <summary>
    /// Whether this run would replace results that are meaningfully different from what it produces.
    /// </summary>
    public bool WouldReplace => Any && !(SameVersion && SameSettings);

    /// <summary>The warning, or null when there is nothing worth saying.</summary>
    public string? Warning()
    {
        if (!WouldReplace)
            return null;

        var what = Version is null
            ? "a previous run"
            : $"a run of PRISM {Version}" + (Date is null ? "" : $" from {Date}");
        var why = (SameVersion, SameSettings) switch
        {
            (false, false) => "a different version and different settings",
            (false, true) => "a different version",
            _ => "different settings",
        };
        var names = string.Join(", ", Files.Take(4))
            + (Files.Count > 4 ? $" and {Files.Count - 4:N0} more" : "");

        return $"This output directory already holds results from {what}. This run uses {why}, so "
            + $"those results will be replaced: {names}. Stages whose inputs and settings are "
            + "unchanged are reused rather than recomputed; everything else is overwritten.";
    }

    /// <summary>
    /// The files a completed run leaves that a reader would recognize as "there are results here".
    /// </summary>
    /// <remarks>
    /// Deliberately the REPORTED outputs rather than every file: intermediates and caches are
    /// working state that a re-run is expected to churn, and listing them would bury the two files
    /// someone actually cares about losing.
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
    /// <remarks>
    /// Never throws: an unreadable or half-written <c>parameters.json</c> means the version and
    /// settings cannot be compared, which is reported as "different" rather than as "fine" - the
    /// safe direction when the question is whether someone is about to lose a cohort.
    /// </remarks>
    public static ExistingResults Inspect(string outputDir, PrismConfig config)
    {
        if (string.IsNullOrWhiteSpace(outputDir) || !Directory.Exists(outputDir))
            return None;

        var files = Reported
            .Where(name => File.Exists(Path.Combine(outputDir, name)))
            .ToArray();
        if (files.Length == 0)
            return None;

        var provenance = Path.Combine(outputDir, Provenance.FileName);
        if (!File.Exists(provenance))
            return new ExistingResults(true, null, null, false, false, files);

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
            return new ExistingResults(true, null, null, false, false, files);
        }

        var sameVersion = string.Equals(version, PrismVersion.Current, StringComparison.Ordinal);

        bool sameSettings;
        try
        {
            // ConfigFromJson, not LoadConfig: the latter redirects a FASTA whose original path has
            // gone to the copy the run archived, which is right for re-running and wrong for
            // comparing - it would report "different settings" for the config that produced these
            // very results, and only once the archive had become load-bearing.
            //
            // Through ConfigWriter so the comparison is over the settings that are round-tripped and
            // recorded, not over object identity - two configs that write the same YAML produce the
            // same outputs, which is the question being asked.
            sameSettings = string.Equals(
                ConfigWriter.ToYaml(Provenance.ConfigFromJson(json, provenance)),
                ConfigWriter.ToYaml(config),
                StringComparison.Ordinal);
        }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException
                                       or NotSupportedException or ArgumentException)
        {
            sameSettings = false;
        }

        return new ExistingResults(true, version, date, sameVersion, sameSettings, files);
    }
}
