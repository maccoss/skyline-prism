using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace SkylinePrism.Core.Parsimony;

/// <summary>
/// Reads/writes protein_groups.csv (Stage 3 output). Columns:
/// GroupID, LeadingProtein, LeadingUniProtID, LeadingGeneName, LeadingName,
/// LeadingDescription, MemberProteins, SubsumedProteins, NPeptides, NUniquePeptides,
/// NRazorPeptides, NAllMappedPeptides, UniquePeptides, RazorPeptides, AllPeptides.
/// Peptide/protein lists are ';'-separated inside a field.
///
/// Note: the CSV's AllPeptides column is the parsimony-assigned set (unique ∪ razor), not
/// the full all-mapped set (only its COUNT is stored). When reading, AllMappedPeptides is
/// set to that same list as a best-effort proxy (exact when there are no shared peptides).
/// The pipeline passes in-memory groups (with true all-mapped peptides) to the rollup.
/// </summary>
public static class ProteinGroupsCsv
{
    public static List<ProteinGroup> Read(string path)
    {
        var lines = File.ReadAllLines(path);
        if (lines.Length == 0)
            return new List<ProteinGroup>();

        var header = ParseLine(lines[0]);
        var col = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < header.Count; i++)
            col[header[i]] = i;

        var groups = new List<ProteinGroup>();
        for (var r = 1; r < lines.Length; r++)
        {
            if (string.IsNullOrWhiteSpace(lines[r]))
                continue;
            var f = ParseLine(lines[r]);

            string Get(string name) => col.TryGetValue(name, out var idx) && idx < f.Count ? f[idx] : "";
            List<string> Split(string name)
            {
                var v = Get(name);
                return string.IsNullOrEmpty(v)
                    ? new List<string>()
                    : v.Split(';', StringSplitOptions.RemoveEmptyEntries).ToList();
            }

            var peptides = Split("AllPeptides");
            groups.Add(new ProteinGroup
            {
                GroupId = Get("GroupID"),
                LeadingProtein = Get("LeadingProtein"),
                LeadingUniProtId = Get("LeadingUniProtID"),
                LeadingGeneName = Get("LeadingGeneName"),
                LeadingName = Get("LeadingName"),
                LeadingDescription = Get("LeadingDescription"),
                MemberProteins = Split("MemberProteins"),
                SubsumedProteins = Split("SubsumedProteins"),
                Peptides = peptides,
                UniquePeptides = Split("UniquePeptides"),
                RazorPeptides = Split("RazorPeptides"),
                AllMappedPeptides = peptides,
            });
        }
        return groups;
    }

    public static void Write(IReadOnlyList<ProteinGroup> groups, string path)
    {
        var sb = new StringBuilder();
        sb.Append("GroupID,LeadingProtein,LeadingUniProtID,LeadingGeneName,LeadingName,")
          .Append("LeadingDescription,MemberProteins,SubsumedProteins,NPeptides,NUniquePeptides,")
          .Append("NRazorPeptides,NAllMappedPeptides,UniquePeptides,RazorPeptides,AllPeptides\n");

        foreach (var g in groups)
        {
            var fields = new[]
            {
                g.GroupId,
                g.LeadingProtein,
                g.LeadingUniProtId,
                g.LeadingGeneName,
                g.LeadingName,
                g.LeadingDescription,
                string.Join(";", g.MemberProteins),
                string.Join(";", g.SubsumedProteins),
                g.Peptides.Count.ToString(CultureInfo.InvariantCulture),
                g.UniquePeptides.Count.ToString(CultureInfo.InvariantCulture),
                g.RazorPeptides.Count.ToString(CultureInfo.InvariantCulture),
                g.AllMappedPeptides.Count.ToString(CultureInfo.InvariantCulture),
                string.Join(";", g.UniquePeptides.OrderBy(x => x, StringComparer.Ordinal)),
                string.Join(";", g.RazorPeptides.OrderBy(x => x, StringComparer.Ordinal)),
                string.Join(";", g.Peptides.OrderBy(x => x, StringComparer.Ordinal)),
            };
            sb.Append(string.Join(",", fields.Select(Escape))).Append('\n');
        }
        File.WriteAllText(path, sb.ToString());
    }

    private static string Escape(string field)
    {
        if (field.Contains(',') || field.Contains('"') || field.Contains('\n'))
            return "\"" + field.Replace("\"", "\"\"") + "\"";
        return field;
    }

    /// <summary>Minimal RFC4180 line parser (handles double-quoted fields with embedded commas).</summary>
    private static List<string> ParseLine(string line)
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
            else if (c == ',')
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
