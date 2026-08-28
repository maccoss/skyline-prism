using System.Collections.Generic;
using System.Text;

namespace SkylinePrism.Core.IO;

/// <summary>
/// One-line CSV splitting and quoting, shared by everything that reads or writes PRISM's small
/// metadata CSVs.
/// <para>
/// It exists because <c>line.Split(',')</c> was copied into three readers of
/// <c>sample_metadata.csv</c>, and that file's values are not comma-free: a batch label, a sample name
/// or a replicate annotation carried over from Skyline can contain one, and the writer has always
/// quoted them. A naive split then shifts every field after it - which does not throw, it just keys a
/// sample by the wrong string and silently drops it from the QC report's control groups.
/// </para>
/// </summary>
public static class CsvLine
{
    /// <summary>Split one CSV line, honoring double-quoted fields and doubled ("") escapes.</summary>
    public static string[] Split(string line)
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
                    if (i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i++; }
                    else inQuotes = false;
                }
                else sb.Append(c);
            }
            else if (c == '"') inQuotes = true;
            else if (c == ',') { fields.Add(sb.ToString()); sb.Clear(); }
            else sb.Append(c);
        }
        fields.Add(sb.ToString());
        return fields.ToArray();
    }

    /// <summary>Quote a value for CSV output, only when it needs it.</summary>
    public static string Quote(string? value)
    {
        var v = value ?? "";
        return v.Contains(',') || v.Contains('"') || v.Contains('\n') || v.Contains('\r')
            ? "\"" + v.Replace("\"", "\"\"") + "\""
            : v;
    }

    /// <summary>
    /// Index of <paramref name="name"/> in a header row, ignoring case and surrounding whitespace, or
    /// -1. Tolerant because these headers are round-tripped through Skyline exports and spreadsheets.
    /// </summary>
    public static int IndexOf(string[] header, string name)
    {
        for (var i = 0; i < header.Length; i++)
            if (header[i].Trim().Equals(name, System.StringComparison.OrdinalIgnoreCase))
                return i;
        return -1;
    }
}
