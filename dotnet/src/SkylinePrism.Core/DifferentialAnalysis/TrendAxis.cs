using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// One way of reading a metadata column as the numeric x a trend is fitted against.
/// </summary>
/// <param name="Column">The metadata column it reads.</param>
/// <param name="Position">
/// Which number in each value it takes, 1-based; 0 when the whole value is itself a number.
/// </param>
/// <param name="Label">How the axis names itself in a picker.</param>
/// <param name="Preview">
/// Distinct value -> number, in increasing number order, for showing what the parse did. At most a
/// handful of entries: it exists to be read, not to be complete.
/// </param>
/// <param name="Distinct">How many distinct numbers it yields.</param>
/// <param name="Covered">How many samples get a number from it.</param>
public sealed record TrendAxisOption(
    string Column,
    int Position,
    string Label,
    IReadOnlyList<(string Value, double Number)> Preview,
    int Distinct,
    int Covered);

/// <summary>
/// Reading a trend's x axis out of a metadata column, including one whose values are text with a
/// number inside them.
/// </summary>
/// <remarks>
/// <para>A study's timepoint is very often written for a human rather than for an axis -
/// <c>V2_Week 8</c>, <c>Day 14</c>, <c>T0</c>. Offering only columns that parse WHOLLY as a number
/// left those studies unable to fit a trend at all: the picker simply did not list the column, so
/// there was nothing to select and no message saying why.</para>
///
/// <para><b>The hazard is picking the wrong number, not failing to find one.</b> <c>V2_Week 8</c>
/// holds two, and they are different quantities: the visit index and the elapsed weeks. On the
/// cohort this was built for they disagree - <c>V1_Week 2</c> and <c>V1_Week 4</c> share a visit
/// number, <c>V2_Week 4</c> and <c>V1_Week 4</c> share a week - so a rule like "take the first
/// number" produces a plausible axis that is not time, and every slope read off it would be wrong
/// with nothing to show for it. So no rule guesses: each POSITION that can be read is offered as its
/// own option, carrying a preview of what it parsed, and the choice is the user's.</para>
///
/// <para>A value holding no number at all (<c>na</c>, an empty cell) yields NaN, which
/// <see cref="TrendSamples.Resolve"/> already drops and reports. A value holding FEWER numbers than
/// the position disqualifies that position outright rather than yielding NaN for those samples -
/// a column read one way for most samples and another way for the rest is not one axis.</para>
/// </remarks>
public static class TrendAxis
{
    /// <summary>A trend needs at least this many distinct x values to have a slope at all.</summary>
    private const int MinDistinct = 2;

    // Signed integers and decimals. Deliberately NOT exponent notation: "1e5" inside a label is far
    // more often an identifier fragment than a number, and a trend axis read from a sample name is
    // the wrong kind of surprise.
    private static readonly Regex Number = new(@"-?\d+(?:\.\d+)?", RegexOptions.Compiled);

    /// <summary>
    /// Every way <paramref name="values"/> can be read as a trend axis, best first.
    /// </summary>
    /// <remarks>
    /// A column that is wholly numeric is returned as a single option with
    /// <see cref="TrendAxisOption.Position"/> 0 and no preview - there is nothing to show, the value
    /// IS the number. Otherwise one option per readable position, ordered by how many distinct
    /// numbers it yields, because the position carrying the most resolution is the one most likely
    /// to be the measurement rather than a group index. That is an ORDERING, not a choice: every
    /// readable position is offered.
    /// </remarks>
    public static IReadOnlyList<TrendAxisOption> OptionsFor(string column, IReadOnlyList<string?> values)
    {
        var present = values.Where(v => !string.IsNullOrWhiteSpace(v)).Select(v => v!.Trim()).ToList();
        if (present.Count == 0)
            return Array.Empty<TrendAxisOption>();

        // Wholly numeric: the historical case, and the only one with nothing to explain.
        if (present.All(v => double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out _)))
        {
            var numbers = present
                .Select(v => double.Parse(v, NumberStyles.Float, CultureInfo.InvariantCulture))
                .ToList();
            var distinctWhole = numbers.Distinct().Count();
            return distinctWhole < MinDistinct
                ? Array.Empty<TrendAxisOption>()
                : new[]
                {
                    new TrendAxisOption(column, 0, column, Array.Empty<(string, double)>(),
                        distinctWhole, numbers.Count),
                };
        }

        // Keyed by DISTINCT value: a timepoint is shared by every sample at it - ten subjects all
        // read "V1_Week 2" - so keying by sample would throw on the first real cohort.
        var matches = present
            .Distinct(StringComparer.Ordinal)
            .ToDictionary(v => v, v => Number.Matches(v), StringComparer.Ordinal);
        var maxNumbers = matches.Values.Max(m => m.Count);
        if (maxNumbers == 0)
            return Array.Empty<TrendAxisOption>();

        var options = new List<TrendAxisOption>();
        for (var pos = 1; pos <= maxNumbers; pos++)
        {
            // A value with SOME numbers but fewer than this position cannot be read here. Skipping
            // the whole position is the point: yielding NaN for those samples would mix two readings
            // of one column, which is not an axis.
            if (matches.Values.Any(m => m.Count > 0 && m.Count < pos))
                continue;

            var byValue = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (var (text, m) in matches)
                if (m.Count >= pos
                    && double.TryParse(m[pos - 1].Value, NumberStyles.Float, CultureInfo.InvariantCulture,
                        out var parsed))
                    byValue[text] = parsed;

            var distinct = byValue.Values.Distinct().Count();
            if (distinct < MinDistinct)
                continue;

            var covered = present.Count(v => byValue.ContainsKey(v));
            options.Add(new TrendAxisOption(
                column, pos, LabelFor(column, pos, maxNumbers, HintFor(matches, pos)),
                PreviewOf(byValue), distinct, covered));
        }

        return options.OrderByDescending(o => o.Distinct).ThenBy(o => o.Position).ToList();
    }

    /// <summary>
    /// Every axis <paramref name="columns"/> can offer, best first within each column.
    /// </summary>
    /// <remarks>
    /// The one list both front ends build their picker from, so a label offered in the pane is a
    /// label <c>prism differential --trend-over</c> accepts.
    /// </remarks>
    public static IReadOnlyList<TrendAxisOption> AllFor(
        IEnumerable<string> columns, Func<string, string?[]> valuesOf)
    {
        var all = new List<TrendAxisOption>();
        foreach (var column in columns)
            all.AddRange(OptionsFor(column, valuesOf(column)));

        // Ordered, never filtered. A number can be read out of almost any identifier - a sample id,
        // a plate barcode - so most of this list is noise on a real study, and the temptation is to
        // drop what does not look like a measurement. There is no rule that separates them: a time
        // course on one subject has one sample per timepoint and so looks exactly like an id, and a
        // dropped axis is invisible with no way to ask for it. Ordering costs a reader nothing and
        // hides nothing.
        //
        // Ordered so the UNAMBIGUOUS axes come first - a column that is already a number, which is
        // the only kind that existed before and the only kind nothing had to interpret - and the
        // interpreted ones after, in the metadata's own column order.
        //
        // Deliberately NOT ranked by how timepoint-like a column looks. Two such rules were tried
        // against the cohort this was built for and both put an identifier first: "values repeat"
        // ranked a 19-value patient id above a 7-value timepoint, and ranking by HOW HEAVILY they
        // repeat promoted a near-constant column instead, because two distinct values repeat hardest
        // of all. There is no signal here that separates a measurement from an identifier, and a
        // ranking that is wrong is worse than no ranking: it chooses on the user's behalf, in a
        // picker whose whole purpose is that the choice cannot be made for them.
        //
        // So callers default to a wholly-numeric axis and require an interpreted one to be picked
        // (see the pane's PopulateTrendColumns), and the preview says what the pick parsed.
        return all
            .OrderByDescending(o => o.Position == 0)
            .ThenBy(o => o.Column, StringComparer.Ordinal)
            .ThenBy(o => o.Position)
            .ToList();
    }

    /// <summary>
    /// The axis <paramref name="label"/> names, or null when no column offers one.
    /// </summary>
    /// <remarks>
    /// A bare column name matches that column's best axis, so <c>--trend-over Week</c> keeps working
    /// on a numeric column and a user need not learn the parenthesised form to use an obvious one.
    /// The full label is what the pane shows and what a recorded command carries, because on a column
    /// offering two it is the only form that says which.
    /// </remarks>
    public static TrendAxisOption? Find(
        string label, IEnumerable<string> columns, Func<string, string?[]> valuesOf)
    {
        var all = AllFor(columns, valuesOf);
        return all.FirstOrDefault(o => string.Equals(o.Label, label, StringComparison.Ordinal))
            ?? all.FirstOrDefault(o => string.Equals(o.Column, label, StringComparison.Ordinal));
    }

    /// <summary>
    /// <paramref name="values"/> read through <paramref name="option"/>, aligned to the samples:
    /// NaN where the sample has no value, or none at that position.
    /// </summary>
    public static double[] Read(IReadOnlyList<string?> values, TrendAxisOption option)
    {
        var x = new double[values.Count];
        for (var i = 0; i < values.Count; i++)
        {
            x[i] = double.NaN;
            var v = values[i]?.Trim();
            if (string.IsNullOrEmpty(v))
                continue;

            if (option.Position == 0)
            {
                if (double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var whole))
                    x[i] = whole;
                continue;
            }

            var m = Number.Matches(v);
            if (m.Count >= option.Position
                && double.TryParse(m[option.Position - 1].Value, NumberStyles.Float,
                    CultureInfo.InvariantCulture, out var parsed))
                x[i] = parsed;
        }

        return x;
    }

    /// <summary>
    /// The axis's name in a picker. A column read wholly as a number keeps its own name; one read at
    /// a position says WHICH number it takes, because two options from one column are otherwise
    /// indistinguishable in a list.
    /// </summary>
    /// <remarks>
    /// Named by the word in front of the number wherever the values agree on one, so the two ways to
    /// read <c>V2_Week 8</c> offer themselves as "(V)" and "(Week)" rather than as "(1st number)" and
    /// "(2nd number)". Which of those is elapsed time is obvious in the first form and a guess in the
    /// second - and guessing is the whole hazard here, since both parse and only one is the axis.
    /// </remarks>
    private static string LabelFor(string column, int position, int maxNumbers, string? hint)
    {
        if (hint is not null)
            return $"{column} ({hint})";
        return maxNumbers == 1 ? $"{column} (number)" : $"{column} ({Ordinal(position)} number)";
    }

    private static string Ordinal(int n) => n switch
    {
        1 => "1st", 2 => "2nd", 3 => "3rd", _ => $"{n}th",
    };

    // Letters only: a separator run ("_", " ", "-") in front of the number names nothing.
    private static readonly Regex TrailingWord = new(@"([A-Za-z]+)[^A-Za-z0-9]*$", RegexOptions.Compiled);

    /// <summary>
    /// The word immediately before this position's number, when every value agrees on it; null
    /// otherwise, including when some value has nothing but punctuation in front of the number.
    /// </summary>
    /// <remarks>
    /// Required to agree across ALL values, not merely to be common: a label drawn from most of a
    /// column would name the axis after a word some of its samples do not carry, which is the kind of
    /// thing a reader trusts precisely because it looks specific.
    /// </remarks>
    private static string? HintFor(IReadOnlyDictionary<string, MatchCollection> matches, int position)
    {
        string? agreed = null;
        foreach (var (text, m) in matches)
        {
            if (m.Count < position)
                continue; // a value with no numbers at all says nothing about the label

            var before = text[..m[position - 1].Index];
            var word = TrailingWord.Match(before);
            if (!word.Success)
                return null;

            var candidate = word.Groups[1].Value;
            if (agreed is null)
                agreed = candidate;
            else if (!string.Equals(agreed, candidate, StringComparison.OrdinalIgnoreCase))
                return null;
        }

        return agreed;
    }

    /// <summary>
    /// A few distinct value -> number pairs in increasing number order. Both ends are shown when
    /// there are many, because what a reader checks is that the axis runs the right way and starts
    /// where they expect.
    /// </summary>
    private static IReadOnlyList<(string Value, double Number)> PreviewOf(
        IReadOnlyDictionary<string, double> byValue)
    {
        var ordered = byValue
            .GroupBy(kv => kv.Value)
            .OrderBy(g => g.Key)
            .Select(g => (Value: g.OrderBy(kv => kv.Key, StringComparer.Ordinal).First().Key, Number: g.Key))
            .ToList();

        const int shown = 6;
        if (ordered.Count <= shown)
            return ordered;

        var head = ordered.Take(shown - 2).ToList();
        head.AddRange(ordered.Skip(ordered.Count - 2));
        return head;
    }

    /// <summary>
    /// The preview as one line: <c>V0_Week 0 -> 0, V1_Week 2 -> 2, ...</c>, or empty for a column
    /// that needed no parsing.
    /// </summary>
    public static string DescribePreview(TrendAxisOption option)
    {
        if (option.Preview.Count == 0)
            return string.Empty;

        var inv = CultureInfo.InvariantCulture;
        var parts = option.Preview.Select(p => $"{p.Value} -> {p.Number.ToString("0.###", inv)}");
        var line = string.Join(", ", parts);
        return option.Distinct > option.Preview.Count ? line + ", ..." : line;
    }
}
