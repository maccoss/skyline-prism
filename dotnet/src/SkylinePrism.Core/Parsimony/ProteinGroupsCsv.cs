using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace SkylinePrism.Core.Parsimony;

/// <summary>
/// Reads/writes the protein_groups.csv produced by Stage 3, whose columns are:
/// GroupID, LeadingProtein, LeadingUniProtID, LeadingGeneName, LeadingName,
/// LeadingDescription, MemberProteins, SubsumedProteins, NPeptides, NUniquePeptides,
/// NRazorPeptides, NAllMappedPeptides, UniquePeptides, RazorPeptides, AllPeptides.
/// Peptide lists are ';'-separated inside a field.
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
            List<string> Peptides(string name)
            {
                var v = Get(name);
                return string.IsNullOrEmpty(v)
                    ? new List<string>()
                    : v.Split(';', StringSplitOptions.RemoveEmptyEntries).ToList();
            }

            groups.Add(new ProteinGroup
            {
                GroupId = Get("GroupID"),
                LeadingProtein = Get("LeadingProtein"),
                LeadingUniProtId = Get("LeadingUniProtID"),
                LeadingGeneName = Get("LeadingGeneName"),
                LeadingName = Get("LeadingName"),
                LeadingDescription = Get("LeadingDescription"),
                UniquePeptides = Peptides("UniquePeptides"),
                RazorPeptides = Peptides("RazorPeptides"),
                AllMappedPeptides = Peptides("AllPeptides"),
            });
        }
        return groups;
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
