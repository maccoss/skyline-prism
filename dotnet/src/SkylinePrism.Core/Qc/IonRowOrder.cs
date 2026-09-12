using System;
using System.Collections.Generic;
using System.Linq;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// The order replicates are drawn in on the ion accounting bar plot.
/// </summary>
/// <remarks>
/// <para>The plot has one bar per replicate and no room to label them, so the order IS the axis:
/// it is the only thing that says which bar is which, and it decides what the plot can show. Sorted
/// by acquisition it shows drift over a batch; grouped by sample type it shows whether the controls
/// sit where the experimental samples do.</para>
/// </remarks>
public static class IonRowOrder
{
    /// <summary>How to order the replicates.</summary>
    public enum By
    {
        /// <summary>Acquisition order, from each data file's own start timestamp.</summary>
        RunOrder,

        /// <summary>The instrument file's name, compared so that 10 sorts after 2.</summary>
        FileName,

        /// <summary>Grouped by sample type, then by file name within each group.</summary>
        SampleType,
    }

    /// <summary>
    /// Sample types in the order they are grouped, most-to-least numerous in a normal cohort. A type
    /// not named here sorts after these, alphabetically, rather than being dropped or merged.
    /// </summary>
    private static readonly string[] TypeOrder = { "experimental", "reference", "qc" };

    /// <summary>
    /// Whether every replicate that COULD state when it was acquired does. Run order is only
    /// offered when they all can: a cohort where some carry a timestamp and some do not would
    /// silently interleave the ones that cannot, which reads as an acquisition order and is not one.
    /// </summary>
    /// <remarks>
    /// Rows with no data file are excluded from the question rather than answering no. A replicate
    /// whose file could not be paired - the normal fate of reference and QC injections named
    /// identically in every plate - has no file to read a timestamp from and never will, so
    /// counting its silence would disable run order for the whole cohort forever, and the fallback
    /// message would blame a cache that a re-measure cannot fix.
    /// </remarks>
    public static bool CanOrderByRun(IEnumerable<IonAccountingRow> rows)
    {
        var answerable = rows?.Where(r => !string.IsNullOrWhiteSpace(r.DataFile)).ToList();
        return answerable is { Count: > 0 } && answerable.All(r => r.AcquiredUtc is not null);
    }

    /// <summary>Re-order the rows. Stable, and never drops or duplicates one.</summary>
    public static IReadOnlyList<IonAccountingRow> Sort(IReadOnlyList<IonAccountingRow> rows, By by)
    {
        if (rows is null || rows.Count <= 1)
            return rows ?? Array.Empty<IonAccountingRow>();

        return by switch
        {
            // Falls back to file name when the cache predates the timestamp, so the plot is still
            // ordered by something a reader can follow rather than by measurement order.
            // Rows with no timestamp sort last rather than at the epoch - an unpaired replicate
            // has no place in an acquisition order, and putting it first would read as one.
            By.RunOrder when CanOrderByRun(rows) =>
                rows.OrderBy(r => r.AcquiredUtc ?? DateTime.MaxValue)
                    .ThenBy(FileKey, NaturalOrder.Comparer).ToList(),
            By.RunOrder => Sort(rows, By.FileName),

            By.SampleType => rows
                .OrderBy(TypeRank)
                .ThenBy(r => r.SampleType ?? "", StringComparer.OrdinalIgnoreCase)
                .ThenBy(FileKey, NaturalOrder.Comparer)
                .ToList(),

            _ => rows.OrderBy(FileKey, NaturalOrder.Comparer).ToList(),
        };
    }

    /// <summary>
    /// The instrument file's name, falling back to the replicate when a row has no file - a replicate
    /// whose file could not be paired still has to land somewhere predictable.
    /// </summary>
    private static string FileKey(IonAccountingRow row) =>
        string.IsNullOrWhiteSpace(row.DataFile)
            ? row.Sample ?? ""
            : System.IO.Path.GetFileName(row.DataFile);

    private static int TypeRank(IonAccountingRow row)
    {
        var index = Array.FindIndex(
            TypeOrder, t => string.Equals(t, row.SampleType, StringComparison.OrdinalIgnoreCase));
        return index >= 0 ? index : TypeOrder.Length;
    }
}

/// <summary>
/// Comparing names the way a person reads them: the digits inside a name compare as numbers, so
/// <c>A2</c> sorts before <c>A10</c>.
/// </summary>
/// <remarks>
/// Plain ordinal comparison puts a 48-well plate in the order A1, A10, A11, A12, A2 - which looks
/// like a defect rather than an ordering, and on a plot whose whole axis is the order it is one.
/// </remarks>
public static class NaturalOrder
{
    public static IComparer<string> Comparer { get; } = new NaturalComparer();

    private sealed class NaturalComparer : IComparer<string>
    {
        public int Compare(string? a, string? b)
        {
            if (ReferenceEquals(a, b))
                return 0;
            if (a is null)
                return -1;
            if (b is null)
                return 1;

            int i = 0, j = 0;
            while (i < a.Length && j < b.Length)
            {
                if (char.IsDigit(a[i]) && char.IsDigit(b[j]))
                {
                    // Compare the whole runs of digits as numbers, by length first and then by
                    // value, so arbitrarily long runs need no parsing and cannot overflow. Leading
                    // zeros are skipped so 007 and 7 are the same number.
                    var startA = i;
                    var startB = j;
                    while (i < a.Length && a[i] == '0') i++;
                    while (j < b.Length && b[j] == '0') j++;
                    var digitsA = i;
                    var digitsB = j;
                    while (digitsA < a.Length && char.IsDigit(a[digitsA])) digitsA++;
                    while (digitsB < b.Length && char.IsDigit(b[digitsB])) digitsB++;

                    var lengthA = digitsA - i;
                    var lengthB = digitsB - j;
                    if (lengthA != lengthB)
                        return lengthA - lengthB;
                    for (var k = 0; k < lengthA; k++)
                    {
                        if (a[i + k] != b[j + k])
                            return a[i + k] - b[j + k];
                    }
                    // Equal in value; a difference in zero padding breaks the tie so the order is
                    // total and the sort stays deterministic.
                    if (i - startA != j - startB)
                        return (i - startA) - (j - startB);
                    i = digitsA;
                    j = digitsB;
                    continue;
                }

                var compared = char.ToUpperInvariant(a[i]).CompareTo(char.ToUpperInvariant(b[j]));
                if (compared != 0)
                    return compared;
                i++;
                j++;
            }
            return (a.Length - i) - (b.Length - j);
        }
    }
}
