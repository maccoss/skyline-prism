using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Text;
using SkylinePrism.Core.Config;

namespace SkylinePrism.Core.Pipeline;

/// <summary>
/// Which configuration keys each pipeline stage actually reads - the map that lets a re-run reuse a
/// stage whose inputs and settings have not changed.
///
/// <para><b>This table is the safety property, not the caching.</b> Claim too much for a stage and
/// nothing is ever reused (changing the protein rollup would re-run the peptide arm, which is the
/// problem this exists to solve). Claim too little and a changed parameter silently reuses a stale
/// output - numbers that look fine and are wrong, the worst failure this codebase has. So every leaf
/// key of <see cref="PrismConfig"/> must appear either under a stage here or in
/// <see cref="OutputIrrelevant"/>, and <c>StageDependencyCoverageTests</c> fails the build when a new
/// key is added without being classified.</para>
///
/// <para>Keys are the snake_case paths of the YAML, so this table reads against `prism
/// config-template` output and against docs/parameters.md. Treat it as part of the configuration
/// contract in CLAUDE.md: adding a key means classifying it here in the same change.</para>
/// </summary>
public static class StageDependencies
{
    // Stage identifiers, also the keys of the on-disk cache. Changing one invalidates that stage for
    // every existing output directory, which is a safe (if wasteful) thing to do.
    public const string Merge = "merge";
    public const string TransitionRollup = "rollup.transition";
    public const string PeptideNormalize = "normalize.peptide";
    public const string ProteinRollup = "rollup.protein";
    public const string ProteinNormalize = "normalize.protein";

    // Deliberately NOT cached:
    //
    // Parsimony. protein_groups.csv stores only the COUNT of each group's all-mapped peptides, not the
    // list (see ProteinGroupsCsv), so reading it back gives a group whose AllMappedPeptides is the
    // parsimony-ASSIGNED set. That is the very list the default shared_peptide_handling ("all_groups")
    // quantifies from, so reusing a group read from the CSV would silently drop every shared peptide
    // from every protein - a different answer, with nothing to see. Recomputing it is cheap next to the
    // rollups. Making it cacheable means persisting the all-mapped set losslessly first.
    //
    // Outlier detection. It has no output file of its own; what it produces is a smaller sample list,
    // and that reaches the stages below through SampleContextKey. Its keys are therefore declared on
    // the stages whose matrices those samples are columns of.

    /// <summary>
    /// Stage -> the config keys whose values change that stage's OUTPUT. Upstream stages are chained
    /// separately (a stage's fingerprint includes its inputs'), so this lists only what the stage
    /// itself reads.
    /// </summary>
    public static readonly IReadOnlyDictionary<string, string[]> ByStage =
        new Dictionary<string, string[]>(StringComparer.Ordinal)
        {
            // The merged table's schema and contents. Column overrides decide which input columns are
            // read; the input FILES are fingerprinted separately (SourceFingerprint).
            [Merge] = new[] { "data" },

            // Transition -> peptide. Every transition_rollup key, plus whether residuals are written.
            [TransitionRollup] = new[]
            {
                "transition_rollup.method", "transition_rollup.min_transitions",
                "transition_rollup.topn_count", "transition_rollup.topn_selection",
                "transition_rollup.topn_weighting", "transition_rollup.consensus_regularization",
                "transition_rollup.use_ms1", "transition_rollup.library_path",
                "transition_rollup.library_min_fragments", "transition_rollup.library_mz_tolerance",
                "transition_rollup.library_outlier_threshold", "transition_rollup.library_remove_outliers",
                "transition_rollup.library_fitting_method",
                "output.include_residuals",
                "data",
            },

            // Peptide normalization + peptide ComBat, written in one pass (peptides_log2_internal and
            // corrected_peptides). Sample types and batches decide what ComBat corrects and what the
            // control CVs are computed over, so every source of them counts. parsimony is here too:
            // corrected_peptides carries the protein-group columns.
            [PeptideNormalize] = new[]
            {
                "global_normalization",
                "batch_correction",
                "sample_annotations", "metadata", "batch_estimation",
                "sample_outlier_detection",
                "parsimony",
                "output.format", "output.include_residuals",
            },

            // Peptide -> protein. shared_peptide_handling is read here as well as by parsimony, and
            // parsimony.fasta_path is iBAQ's fallback when ibaq.fasta_path is unset - a dependency the
            // coverage test caught before this table did.
            [ProteinRollup] = new[]
            {
                "protein_rollup",
                "parsimony.shared_peptide_handling",
                "parsimony.fasta_path",
                "sample_outlier_detection",
                "output.include_residuals",
            },

            // Protein normalization + protein ComBat.
            [ProteinNormalize] = new[]
            {
                "protein_normalization",
                "batch_correction",
                "sample_annotations", "metadata", "batch_estimation",
                "sample_outlier_detection",
                "output.format", "output.include_residuals",
            },
        };

    /// <summary>
    /// Keys that cannot change any cached output, with the reason. Listed explicitly rather than
    /// defaulted, so a new key is never treated as irrelevant by omission.
    /// </summary>
    public static readonly IReadOnlyDictionary<string, string> OutputIrrelevant =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            // Performance only. The rollup is asserted reproducible run to run (dotnet-v26.14.1), so
            // thread count and buffer sizes cannot move a number; they only decide how fast and in how
            // much memory it is produced.
            ["processing.n_workers"] = "performance only - the rollup is reproducible run to run",
            ["processing.peptide_batch_size"] = "performance only - parquet row-group flush size",
            ["processing.merge_memory_mb"] = "performance only - DuckDB buffer ceiling, spills instead",

            // The QC report is regenerated on every run (it is seconds) and caches nothing.
            ["qc_report.enabled"] = "the QC report is never cached",
            ["qc_report.save_plots"] = "the QC report is never cached",

            // Folded onto the flat transition_rollup.library_* keys by PrismConfig.Parse before any
            // stage reads it, so the flat keys above already carry its effect.
            ["transition_rollup.library_assist"] = "folded onto the flat library_* keys at parse time",
        };

    /// <summary>
    /// External files a stage reads whose CONTENT matters - fingerprinted by path, size and
    /// last-write-time, because their config key holds only a path and a path can be rewritten.
    /// </summary>
    public static IReadOnlyList<string> ExternalFiles(string stageId, PrismConfig config) => stageId switch
    {
        TransitionRollup => Paths(config.TransitionRollup.LibraryPath),
        ProteinRollup => Paths(config.ProteinRollup.Ibaq.FastaPath, config.Parsimony.FastaPath),
        _ => Array.Empty<string>(),
    };

    private static string[] Paths(params string?[] paths) =>
        paths.Where(p => !string.IsNullOrWhiteSpace(p)).Select(p => p!).ToArray();

    /// <summary>
    /// The values of a stage's declared keys, rendered as a stable string for hashing. A key naming a
    /// whole section expands to every leaf under it, so a section listed here cannot gain a key that
    /// silently escapes the fingerprint.
    /// </summary>
    public static string Values(string stageId, PrismConfig config)
    {
        if (!ByStage.TryGetValue(stageId, out var keys))
            throw new ArgumentException($"Unknown pipeline stage '{stageId}'.", nameof(stageId));

        var sb = new StringBuilder();
        foreach (var key in keys.SelectMany(k => ExpandToLeaves(k)).Distinct(StringComparer.Ordinal)
                     .OrderBy(k => k, StringComparer.Ordinal))
            sb.Append(key).Append('=').Append(Render(ReadPath(config, key))).Append('\n');
        return sb.ToString();
    }

    /// <summary>Every leaf key path of <see cref="PrismConfig"/>, in snake_case dotted form.</summary>
    public static IReadOnlyList<string> AllLeafKeys() => Leaves(typeof(PrismConfig), "").ToList();

    /// <summary>Leaf key paths under one section (or the key itself when it is already a leaf).</summary>
    public static IEnumerable<string> ExpandToLeaves(string key)
    {
        var type = TypeAtPath(typeof(PrismConfig), key);
        if (type is null || IsLeaf(type))
            return new[] { key };
        return Leaves(type, key);
    }

    private static IEnumerable<string> Leaves(Type type, string prefix)
    {
        foreach (var p in type.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (!p.CanRead || p.GetIndexParameters().Length > 0)
                continue;
            var path = prefix.Length == 0 ? ToSnake(p.Name) : prefix + "." + ToSnake(p.Name);
            if (IsLeaf(p.PropertyType))
                yield return path;
            else
                foreach (var child in Leaves(p.PropertyType, path))
                    yield return child;
        }
    }

    private static bool IsLeaf(Type t)
    {
        var u = Nullable.GetUnderlyingType(t) ?? t;
        return u.IsPrimitive || u.IsEnum || u == typeof(string) || u == typeof(decimal)
            || typeof(System.Collections.IEnumerable).IsAssignableFrom(u);
    }

    private static Type? TypeAtPath(Type root, string path)
    {
        var current = root;
        foreach (var segment in path.Split('.'))
        {
            var prop = FindProperty(current, segment);
            if (prop is null)
                return null;
            current = prop.PropertyType;
        }
        return current;
    }

    /// <summary>Read a snake_case dotted path off a config instance; null when any segment is null.</summary>
    public static object? ReadPath(object? instance, string path)
    {
        var current = instance;
        foreach (var segment in path.Split('.'))
        {
            if (current is null)
                return null;
            var prop = FindProperty(current.GetType(), segment);
            if (prop is null)
                throw new ArgumentException(
                    $"Config path '{path}' does not exist on {instance?.GetType().Name}.", nameof(path));
            current = prop.GetValue(current);
        }
        return current;
    }

    private static PropertyInfo? FindProperty(Type type, string snakeSegment)
    {
        var pascal = ToPascal(snakeSegment);
        return type.GetProperty(pascal, BindingFlags.Public | BindingFlags.Instance)
            ?? type.GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .FirstOrDefault(p => string.Equals(ToSnake(p.Name), snakeSegment, StringComparison.Ordinal));
    }

    private static string Render(object? value) => value switch
    {
        null => "(null)",
        string s => s,
        bool b => b ? "true" : "false",
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        System.Collections.IEnumerable list =>
            "[" + string.Join("|", list.Cast<object?>().Select(Render)) + "]",
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => value.ToString() ?? "",
    };

    internal static string ToSnake(string pascal)
    {
        var sb = new StringBuilder();
        for (var i = 0; i < pascal.Length; i++)
        {
            if (char.IsUpper(pascal[i]) && i > 0)
                sb.Append('_');
            sb.Append(char.ToLowerInvariant(pascal[i]));
        }
        return sb.ToString();
    }

    internal static string ToPascal(string snake) => string.Concat(
        snake.Split('_', StringSplitOptions.RemoveEmptyEntries)
            .Select(part => char.ToUpperInvariant(part[0]) + part[1..]));
}
