using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// Reads a Thermo Method Editor scheduled inclusion list into an <see cref="IsolationScheme"/>.
/// </summary>
/// <remarks>
/// <para>For a targeted acquisition this file IS the isolation scheme, and the only complete record of
/// it: PRM/MTM windows are scheduled rather than cyclic, so Skyline cannot import them from the data
/// (it looks for a repeating cycle and reports "No repeating isolation scheme found"), and the document
/// stores nothing. The inclusion list is what was loaded onto the instrument.</para>
///
/// <para>Columns, as written by Skyline-Cadenza's <c>ThermoCsvWriter</c> and by the Method Editor's own
/// mass-list export:</para>
/// <code>
/// Compound, Formula, Adduct, m/z, z, t start (min), t stop (min), Isolation Window (m/z), HCD Collision Energy
/// </code>
/// <para>One row is one <i>slot</i>: <c>m/z</c> is the window centre, <c>Isolation Window (m/z)</c> its
/// full width, and <c>t start</c>/<c>t stop</c> the interval it fires in. PRM writes one row per
/// precursor; MTM writes one row per slot, with the members joined in <c>Compound</c> - which is why an
/// MTM cell can legitimately count several precursors while a PRM cell counts one.</para>
///
/// <para>Header matching ignores case, spaces, underscores and units, so the older
/// <c>t (min)</c> / <c>Window (min)</c> spellings and vendor variants still resolve.</para>
/// </remarks>
public static class ThermoInclusionList
{
    /// <summary>Read an inclusion list from disk. Throws with a specific reason if it is not one.</summary>
    public static IsolationScheme Load(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Inclusion list not found: {path}", path);
        return Parse(File.ReadAllLines(path), Path.GetFileNameWithoutExtension(path));
    }

    /// <summary>Same as <see cref="Load"/> but returns null instead of throwing.</summary>
    public static IsolationScheme? TryLoad(string path, Action<string>? log = null)
    {
        try
        {
            return Load(path);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or FormatException
                                       or UnauthorizedAccessException)
        {
            log?.Invoke($"Could not read the inclusion list {Path.GetFileName(path)}: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Parse inclusion-list lines. Every row must yield an m/z and an isolation width; the RT columns are
    /// optional, and a row without them becomes an always-on window (an unscheduled inclusion list).
    /// </summary>
    public static IsolationScheme Parse(IReadOnlyList<string> lines, string name = "Inclusion list")
    {
        var header = lines.FirstOrDefault(l => !string.IsNullOrWhiteSpace(l))
            ?? throw new InvalidDataException("The inclusion list is empty.");
        var headerIndex = lines.ToList().IndexOf(header);
        var columns = SplitCsv(header).Select(Normalize).ToList();

        var mzCol = Find(columns, "mz", "precursormz", "massmz", "mass");
        var widthCol = Find(columns, "isolationwindowmz", "isolationwindow", "windowmz", "isolationwidth");
        var startCol = Find(columns, "tstartmin", "tstart", "starttimemin", "starttime", "rtstartmin");
        var stopCol = Find(columns, "tstopmin", "tstop", "endtimemin", "endtime", "rtstopmin", "tendmin");

        if (mzCol < 0)
            throw new InvalidDataException(
                "No m/z column found. Expected a Thermo inclusion list with columns like "
                + "\"m/z, z, t start (min), t stop (min), Isolation Window (m/z)\".");
        if (widthCol < 0)
            throw new InvalidDataException(
                "No \"Isolation Window (m/z)\" column found - without the window width the isolation "
                + "windows cannot be reconstructed.");

        var windows = new List<IsolationWindow>();
        for (var i = headerIndex + 1; i < lines.Count; i++)
        {
            if (string.IsNullOrWhiteSpace(lines[i]))
                continue;
            var fields = SplitCsv(lines[i]);
            var mz = Num(fields, mzCol);
            var width = Num(fields, widthCol);
            if (mz is null || width is null || width <= 0)
                continue; // a comment or a malformed row - skip rather than fail the whole list

            var start = Num(fields, startCol);
            var stop = Num(fields, stopCol);
            var scheduled = start is not null && stop is not null && stop > start;
            windows.Add(new IsolationWindow(
                mz.Value - width.Value / 2, mz.Value + width.Value / 2,
                Margin: 0,
                RtStart: scheduled ? start!.Value : double.NaN,
                RtStop: scheduled ? stop!.Value : double.NaN));
        }

        if (windows.Count == 0)
            throw new InvalidDataException("The inclusion list has a header but no usable rows.");

        return new IsolationScheme(name, windows.OrderBy(w => w.Start).ToList());
    }

    private static int Find(IReadOnlyList<string> columns, params string[] candidates)
    {
        foreach (var candidate in candidates)
            for (var i = 0; i < columns.Count; i++)
                if (string.Equals(columns[i], candidate, StringComparison.Ordinal))
                    return i;
        return -1;
    }

    // Strip case, spaces, underscores and the parenthesised units the vendor headers carry, so
    // "t start (min)", "t_start", "T Start" and "tstartmin" all resolve to one key.
    private static string Normalize(string column)
    {
        var sb = new StringBuilder(column.Length);
        foreach (var c in column)
            if (char.IsLetterOrDigit(c))
                sb.Append(char.ToLowerInvariant(c));
        return sb.ToString();
    }

    private static double? Num(IReadOnlyList<string> fields, int index)
    {
        if (index < 0 || index >= fields.Count)
            return null;
        // Instrument/method files are invariant; a locale-aware parse here would silently read "3,0014"
        // as 30014 on a comma-decimal machine.
        return double.TryParse(fields[index].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var v)
            ? v
            : null;
    }

    // Minimal CSV split honouring quoted fields - the Compound column joins MTM members with " | " and
    // is quoted when it contains a comma.
    private static List<string> SplitCsv(string line)
    {
        var fields = new List<string>();
        var sb = new StringBuilder();
        var inQuotes = false;
        for (var i = 0; i < line.Length; i++)
        {
            var c = line[i];
            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"')
                    {
                        sb.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = false;
                    }
                }
                else
                {
                    sb.Append(c);
                }
            }
            else if (c == '"')
            {
                inQuotes = true;
            }
            else if (c == ',' || c == '\t')
            {
                fields.Add(sb.ToString());
                sb.Clear();
            }
            else
            {
                sb.Append(c);
            }
        }
        fields.Add(sb.ToString());
        return fields;
    }
}
