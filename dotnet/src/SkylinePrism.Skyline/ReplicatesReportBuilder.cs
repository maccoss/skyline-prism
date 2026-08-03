#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SkylinePrism.Skyline;

/// <summary>
/// Generates the <c>PRISM-Replicates</c> report definition (<c>.skyr</c>).
///
/// Skyline's built-in "Replicates" document grid is not a named, exportable report. When PRISM is
/// attached to a RUNNING Skyline it reconstructs that grid over the RPC
/// (<see cref="SkylineReportDriver"/>), but for a document that is NOT open the only way to get the
/// same columns out is to install a saved report and export it with <c>SkylineCmd</c>. Both paths build
/// the view here so they stay identical: the standard replicate columns plus one
/// <c>annotation_&lt;Name&gt;</c> column per replicate-targeted Document Annotation.
///
/// Annotation names come from the RPC replicate-column list (live) or from
/// <see cref="SkyDocumentInfo.ReplicateAnnotationNames"/> (closed document).
/// </summary>
public static class ReplicatesReportBuilder
{
    /// <summary>The report/view name PRISM installs and exports.</summary>
    public const string ViewName = "PRISM-Replicates";

    private const string RowSource = "pwiz.Skyline.Model.Databinding.Entities.Replicate";

    /// <summary>
    /// The standard columns of the built-in Replicates view. The empty name is the row label (the
    /// replicate name itself) - that is how Skyline's view XML addresses it.
    /// </summary>
    private static readonly string[] StandardColumns =
        { "", "SampleType", "AnalyteConcentration" };

    /// <summary>
    /// Build the <c>.skyr</c> XML for a Replicates report carrying <paramref name="annotationNames"/> as
    /// annotation columns. Names already prefixed with <c>annotation_</c> are passed through, so a caller
    /// may supply either form.
    /// </summary>
    public static string BuildXml(IEnumerable<string>? annotationNames = null)
    {
        var columns = new List<string>(StandardColumns);
        foreach (var col in NormalizeAnnotationColumns(annotationNames))
            if (!columns.Contains(col, StringComparer.Ordinal))
                columns.Add(col);

        var sb = new System.Text.StringBuilder();
        sb.Append("<?xml version=\"1.0\"?>\n<views>\n");
        sb.Append($"  <view name=\"{ViewName}\" rowsource=\"{RowSource}\" uimode=\"proteomic\">\n");
        foreach (var c in columns)
            sb.Append($"    <column name=\"{XmlEscape(c)}\" />\n");
        sb.Append("  </view>\n</views>\n");
        return sb.ToString();
    }

    /// <summary>Write <see cref="BuildXml"/> to <paramref name="path"/> and return the path.</summary>
    public static string WriteSkyr(string path, IEnumerable<string>? annotationNames = null)
    {
        var dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);
        File.WriteAllText(path, BuildXml(annotationNames));
        return path;
    }

    /// <summary>
    /// Map annotation names to the column names a Skyline view definition accepts, dropping blanks and
    /// duplicates while preserving order.
    ///
    /// <para><b>The result is QUOTED</b> - <c>"annotation_Plate"</c>, not <c>annotation_Plate</c>. Skyline
    /// parses a view's <c>column/@name</c> as a databinding PropertyPath, whose bare-identifier syntax does
    /// not allow <c>_</c>; the unquoted form is rejected at export time with
    /// <i>"Error parsing annotation_Plate at location 10: Invalid character _"</i> and NO report is written.
    /// Since the <c>annotation_</c> prefix itself contains an underscore, every annotation column needs the
    /// quotes. Verified against SkylineCmd: quoted exports a column headed with the plain annotation name
    /// ("Plate"), which is what the metadata parser reads.</para>
    /// </summary>
    public static IReadOnlyList<string> NormalizeAnnotationColumns(IEnumerable<string>? annotationNames)
    {
        var result = new List<string>();
        if (annotationNames is null)
            return result;
        foreach (var raw in annotationNames)
        {
            if (string.IsNullOrWhiteSpace(raw))
                continue;
            var name = raw.Trim().Trim('"');
            var property = name.StartsWith("annotation_", StringComparison.OrdinalIgnoreCase)
                ? name
                : "annotation_" + name;
            var col = "\"" + property + "\"";
            if (!result.Contains(col, StringComparer.Ordinal))
                result.Add(col);
        }
        return result;
    }

    private static string XmlEscape(string s) => s
        .Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");
}
