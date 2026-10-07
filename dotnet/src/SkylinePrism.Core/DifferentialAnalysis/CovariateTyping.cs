using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>How a covariate enters the design.</summary>
public enum CovariateKind
{
    /// <summary>One mean-centered column: a straight-line effect of the value.</summary>
    Numeric,

    /// <summary>One indicator per level after the first: a separate shift for each level.</summary>
    Categorical,
}

/// <summary>The type <see cref="CovariateTyping.Infer"/> gives a column, and why.</summary>
/// <param name="Kind">The inferred type.</param>
/// <param name="AllNumeric">Whether every non-blank value parses as a number, i.e. whether
/// <see cref="CovariateKind.Numeric"/> is possible at all.</param>
/// <param name="Reason">The rule that decided it, in words a reader can check against the data.</param>
public sealed record CovariateTypeGuess(CovariateKind Kind, bool AllNumeric, string Reason);

/// <summary>
/// The default type of a covariate column. Every front end shows it and lets it be overridden
/// (<c>--covariate-type</c>, the pane's Adjust-for list), so this decides only what happens when
/// nobody says.
/// </summary>
/// <remarks>
/// <para>Numbers are not enough to make a column numeric. An integer-coded subject ID fitted as one
/// centered column is a straight-line effect of an arbitrary ID number: it changes the estimates and
/// the residual variance and models nothing. An integer-coded batch with three or more levels has the
/// same problem - nothing makes batch 3 differ from batch 1 by twice what batch 2 does. Before this
/// rule both were fitted numeric because their values parsed, with no word to the user.</para>
/// <para>So a column of numbers is categorical when its name says it labels groups (a whole word
/// from <see cref="CategoryWords"/>) AND its values look like labels - whole numbers, some of which
/// repeat across samples, as a subject's ID repeats over its visits - or when its values are whole
/// numbers taking at most <see cref="MaxCategoricalLevels"/> distinct values. Otherwise it stays
/// numeric: age, a clinical score, years from diagnosis. A two-level 0/1 column comes out
/// categorical, which fits exactly the same model as the centered numeric column it used to be.</para>
/// <para>The name alone is not enough: "Patient age", "Run order" and "Cycle time" contain a listed
/// word and are measurements. Decimals, or whole numbers that never repeat (an injection order),
/// keep such a column numeric. A whole-number measurement that does repeat - an age in years, one
/// per patient - still comes out categorical, and its row in the pane says why and switches it.</para>
/// <para>Words are matched WHOLE, after splitting the name on punctuation, spaces, case changes and
/// letter-digit boundaries: "File set" and "PatientID" match, "Onset site" does not.</para>
/// </remarks>
public static class CovariateTyping
{
    /// <summary>Whole numbers with at most this many distinct values are taken as categories.</summary>
    public const int MaxCategoricalLevels = 10;

    /// <summary>Name words that mark a column as labelling groups rather than measuring a quantity.</summary>
    public static readonly IReadOnlyList<string> CategoryWords =
        new[] { "patient", "subject", "donor", "id", "batch", "plate", "cycle", "set", "run" };

    /// <summary>The default type of <paramref name="name"/>, given its per-sample values (null = missing).</summary>
    public static CovariateTypeGuess Infer(string name, IReadOnlyList<string?> values)
    {
        var present = values.Where(v => !IsMissing(v)).Select(v => v!.Trim()).ToList();
        if (present.Count == 0)
            return new CovariateTypeGuess(CovariateKind.Categorical, false, "it has no values");

        var finite = new List<double>(present.Count);
        foreach (var v in present)
        {
            if (!TryNumber(v, out var x))
                return new CovariateTypeGuess(CovariateKind.Categorical, false,
                    $"it has values that are not numbers ('{v}')");
            finite.Add(x);
        }

        var levels = finite.Distinct().Count();
        var whole = finite.All(x => x == Math.Floor(x));
        if (CategoryWord(name) is { } word && whole && levels < finite.Count)
            return new CovariateTypeGuess(CovariateKind.Categorical, true,
                $"its name contains '{word}' and its values are whole numbers that repeat across "
                + "samples, as labels do");

        if (whole && levels <= MaxCategoricalLevels)
            return new CovariateTypeGuess(CovariateKind.Categorical, true,
                $"its values are whole numbers with only {levels} distinct value{(levels == 1 ? "" : "s")}");

        return new CovariateTypeGuess(CovariateKind.Numeric, true,
            $"its values are numbers with {levels} distinct values");
    }

    /// <summary>The first <see cref="CategoryWords"/> entry that is a whole word of the name, or null.</summary>
    public static string? CategoryWord(string name)
    {
        // Split camelCase and letter/digit boundaries first, then on anything that is not a letter or digit.
        var spaced = Regex.Replace(name, @"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ");
        var words = Regex.Split(spaced, @"[^A-Za-z0-9]+")
            .Where(w => w.Length > 0)
            .Select(w => w.ToLowerInvariant())
            .ToHashSet();
        return CategoryWords.FirstOrDefault(words.Contains);
    }

    /// <summary>The spellings of "no value" a metadata cell arrives in.</summary>
    /// <remarks>
    /// <c>#N/A</c> is what Skyline and Excel write into an empty cell - a QC pool has no age - and
    /// left as a value it made the whole column text, so a numeric covariate became categories and,
    /// under a contrast between subjects, was dropped as nested. Not "None": that is a real level
    /// often enough (a treatment of none) to be left alone.
    /// </remarks>
    private static readonly HashSet<string> MissingTokens =
        new(StringComparer.OrdinalIgnoreCase) { "#N/A", "N/A", "NA", "#NA", "NULL" };

    /// <summary>
    /// Whether a metadata value means "no value": blank, a missing-value token (<c>#N/A</c>, <c>NA</c>,
    /// <c>NULL</c>), or a number that is not finite (<c>NaN</c>, <c>Infinity</c>).
    /// </summary>
    public static bool IsMissing(string? v)
    {
        if (string.IsNullOrWhiteSpace(v))
            return true;
        var t = v.Trim();
        return MissingTokens.Contains(t) || (TryNumber(t, out var x) && !double.IsFinite(x));
    }

    /// <summary>An invariant-culture number, the one parse every covariate path uses.</summary>
    internal static bool TryNumber(string v, out double x) =>
        double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out x);

    /// <summary>The word used for a kind in output and on the command line.</summary>
    public static string Token(CovariateKind kind) =>
        kind == CovariateKind.Numeric ? "numeric" : "categorical";

    /// <summary>Parse <see cref="Token"/>'s words (and their first letters); null for anything else.</summary>
    public static CovariateKind? ParseToken(string s) => s.Trim().ToLowerInvariant() switch
    {
        "numeric" or "n" or "continuous" => CovariateKind.Numeric,
        "categorical" or "c" or "factor" => CovariateKind.Categorical,
        _ => null,
    };
}
