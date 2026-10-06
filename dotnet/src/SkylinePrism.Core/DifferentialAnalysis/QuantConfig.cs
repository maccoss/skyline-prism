using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>The two arms of a contrast, or a trend column - what the analysis compares.</summary>
/// <param name="GroupBy">The grouping column (two-arm contrasts).</param>
/// <param name="GroupA">Arm A's levels - a list, as <c>-a</c> takes one, so a union arm records as one.</param>
/// <param name="GroupB">Arm B's levels.</param>
/// <param name="TrendOver">The axis a trend is fitted against (trend designs): a column, or one
/// reading of a text column (<see cref="TrendAxis"/>).</param>
/// <param name="RestrictTo">The samples a trend was fitted on, when not all of them.</param>
public sealed record QuantContrast(
    string? GroupBy, IReadOnlyList<string>? GroupA, IReadOnlyList<string>? GroupB, string? TrendOver,
    IReadOnlyList<QuantRestriction>? RestrictTo = null)
{
    /// <summary>A one-line description for the report and the status line.</summary>
    public string Describe() => TrendOver is not null
        ? $"trend over {TrendOver}"
          + (RestrictTo is { Count: > 0 } r ? $" (restricted to {QuantRestriction.Describe(r)})" : string.Empty)
        : $"{ContrastArms.Describe(GroupB ?? Array.Empty<string>())} vs "
          + $"{ContrastArms.Describe(GroupA ?? Array.Empty<string>())} by {GroupBy}";
}

/// <summary>
/// The parameters of one quantification analysis - what the quant report was produced from, and enough
/// to reproduce it. Serializes to YAML (shown in the report and written beside its outputs) and JSON
/// (machine-readable provenance), mirroring how the pipeline records its own config in
/// <c>parameters.json</c>. This is a DESCRIPTION built by the caller from its settings, not the
/// execution engine - the engine is <see cref="DifferentialOptions"/>.
/// </summary>
/// <param name="Prior">The <c>--prior</c> value requested - what a re-run passes.</param>
/// <param name="Command">The <c>prism differential ... --report</c> command that reproduces the analysis (<see cref="QuantCommand"/>).</param>
/// <param name="CommandUnavailable">Why no command could be written, when <paramref name="Command"/> is null.</param>
/// <param name="DetectionTest">The detection test that ran (<see cref="Detection.DetectionAnalysis.Describe"/>), if any.</param>
/// <param name="ClinicalCsvs">Clinical CSVs joined to the samples (<see cref="DifferentialDataset.AttachedClinicalCsvs"/>), if any.</param>
/// <param name="PriorUsed">The prior that actually ran, with its source (<see cref="DifferentialResult.VariancePrior"/>,
/// e.g. "intensity-trend from controls"); it differs from the request when a requested prior had to fall back.</param>
/// <param name="Subject">The subject column, for a design that reads one (paired, blocked, trend within subject).</param>
/// <param name="BlockCorrelation">The intra-subject correlation a blocked contrast was fitted at.</param>
public sealed record QuantConfig(
    string Level,
    QuantContrast Contrast,
    string Design,
    string Test,
    string Prior,
    string PriorUsed,
    string Correction,
    IReadOnlyList<string> Covariates,
    string HitRule,
    bool DetectionEnabled,
    double DetectionQ,
    bool EnrichmentEnabled,
    IReadOnlyList<string> EnrichmentSources,
    string EnrichmentDirection,
    IReadOnlyList<string> MarkerPanels,
    IReadOnlyList<string>? ClinicalCsvs = null,
    string? Command = null,
    string? CommandUnavailable = null,
    string? DetectionTest = null,
    string? Subject = null,
    double? BlockCorrelation = null)
{
    private static readonly JsonSerializerOptions Json = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public string ToJson() => JsonSerializer.Serialize(this, Json);

    /// <summary>The parameters as a compact, human-readable YAML block - the "YAML like this" the
    /// report shows and writes to <c>quant_parameters.yaml</c>.</summary>
    public string ToYaml()
    {
        var inv = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();
        sb.Append("level: ").Append(Level).Append('\n');
        // Only when one was joined: a contrast on a clinical column is not reproducible from the output
        // directory alone, and this is the file `--clinical` needs.
        if (ClinicalCsvs is { Count: > 0 })
            sb.Append("clinical_csv: ").Append(YamlList(ClinicalCsvs)).Append('\n');
        sb.Append("contrast:\n");
        if (Contrast.TrendOver is not null)
            // Through Yaml() like group_by below: a trend column named "1.0", or one holding a colon,
            // would otherwise be emitted bare and read back as something other than its name.
        {
            sb.Append("  trend_over: ").Append(Yaml(Contrast.TrendOver)).Append('\n');
            if (Contrast.RestrictTo is { Count: > 0 } restrictions)
            {
                sb.Append("  restrict_to:\n");
                foreach (var r in restrictions)
                    sb.Append("    - column: ").Append(Yaml(r.Column)).Append('\n')
                      .Append("      values: ").Append(YamlList(r.Values)).Append('\n');
            }
        }
        else
        {
            sb.Append("  group_by: ").Append(Yaml(Contrast.GroupBy)).Append('\n');
            sb.Append("  group_a: ").Append(YamlList(Contrast.GroupA ?? Array.Empty<string>())).Append('\n');
            sb.Append("  group_b: ").Append(YamlList(Contrast.GroupB ?? Array.Empty<string>())).Append('\n');
        }

        sb.Append("design: ").Append(Design).Append('\n');
        // The subject column changes every p-value under the designs that read one, so a record
        // without it cannot be re-run - and the free-text command below is absent whenever the
        // request holds something no command line can carry.
        if (Subject is not null)
            sb.Append("subject: ").Append(Yaml(Subject)).Append('\n');
        if (BlockCorrelation is { } rho)
            sb.Append("block_correlation: ").Append(rho.ToString("R", inv)).Append('\n');
        sb.Append("test: ").Append(Test).Append('\n');
        sb.Append("prior: ").Append(Prior).Append('\n');
        sb.Append("prior_used: ").Append(Yaml(PriorUsed)).Append('\n');
        sb.Append("correction: ").Append(Correction).Append('\n');
        sb.Append("covariates: ").Append(YamlList(Covariates)).Append('\n');
        sb.Append("hit_rule: ").Append(Yaml(HitRule)).Append('\n');
        sb.Append("detection:\n");
        sb.Append("  enabled: ").Append(DetectionEnabled ? "true" : "false").Append('\n');
        if (DetectionTest is not null)
            sb.Append("  test: ").Append(Yaml(DetectionTest)).Append('\n');
        sb.Append("  q_threshold: ").Append(DetectionQ.ToString("0.####", inv)).Append('\n');
        sb.Append("enrichment:\n");
        sb.Append("  enabled: ").Append(EnrichmentEnabled ? "true" : "false").Append('\n');
        sb.Append("  sources: ").Append(YamlList(EnrichmentSources)).Append('\n');
        sb.Append("  direction: ").Append(EnrichmentDirection).Append('\n');
        sb.Append("markers:\n");
        sb.Append("  panels: ").Append(YamlList(MarkerPanels)).Append('\n');
        // A literal block, not a quoted scalar: the command carries quotes and Windows backslashes,
        // and escaping them would leave a line that no longer pastes into a shell as-is.
        if (!string.IsNullOrEmpty(Command))
            sb.Append("command: |-\n  ").Append(Command).Append('\n');
        else if (!string.IsNullOrEmpty(CommandUnavailable))
            sb.Append("command_unavailable: ").Append(Yaml(CommandUnavailable)).Append('\n');
        return sb.ToString();
    }

    // Characters that, appearing anywhere, force quoting so the value cannot be read as structure.
    private static readonly char[] YamlSpecials =
        { ':', '#', '\'', '"', '\\', '\n', '\r', '\t', ',', '[', ']', '{', '}' };

    // Characters that only force quoting when they LEAD the value (a YAML indicator in first position
    // starts a flow collection, an alias/anchor, a tag, a block scalar, etc.).
    private const string YamlLeadIndicators = "-?:,[]{}#&*!|>'\"%@`";

    // Plain scalars YAML 1.1 reads as a boolean or a null rather than as the string they look like.
    // A clinical group column really is often spelled y/n or yes/no, so this is the common case, not
    // the exotic one.
    private static readonly HashSet<string> YamlKeywords = new(StringComparer.OrdinalIgnoreCase)
    {
        "true", "false", "yes", "no", "on", "off", "y", "n",
        "null", "~", ".inf", "-.inf", "+.inf", ".nan",
    };

    /// <summary>
    /// A scalar, quoted whenever the plain form would come back as something other than this exact
    /// string.
    /// </summary>
    /// <remarks>
    /// Two ways a plain scalar lies, and both reach this from real metadata. It can RE-TYPE: a
    /// case/control column spelled <c>0</c>/<c>1</c> emits <c>group_a: 0</c> and reads back as an
    /// integer, and <c>NO</c>, <c>yes</c>, <c>on</c> and <c>null</c> read back as booleans and nulls.
    /// And the escape set has to match the quoting style: inside a YAML double-quoted scalar the
    /// BACKSLASH is an escape introducer, so escaping only the quote turned a value holding
    /// <c>\run</c> into a carriage return, and a value holding a newline was emitted raw and broke the
    /// document. Backslash is escaped first, or it would double-escape what the later replacements add.
    /// </remarks>
    private static string Yaml(string? v)
    {
        if (string.IsNullOrEmpty(v))
            return "\"\"";

        var needsQuote = v.IndexOfAny(YamlSpecials) >= 0
            || YamlLeadIndicators.IndexOf(v[0]) >= 0
            || char.IsWhiteSpace(v[0]) || char.IsWhiteSpace(v[^1])
            || YamlKeywords.Contains(v)
            || LooksNumeric(v);

        if (!needsQuote)
            return v;

        var escaped = v.Replace("\\", "\\\\")
            .Replace("\"", "\\\"")
            .Replace("\n", "\\n")
            .Replace("\r", "\\r")
            .Replace("\t", "\\t");
        return "\"" + escaped + "\"";
    }

    /// <summary>Whether the plain form would be read as a number rather than as a string.</summary>
    private static bool LooksNumeric(string v) =>
        double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out _)
        || long.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out _)
        // 0x1f / 0o17 are ints in YAML but not to TryParse, and a leading-zero run like 0755 is octal
        // in YAML 1.1 while TryParse reads it as decimal 755 - either way the plain form is not this
        // string.
        || v.StartsWith("0x", StringComparison.OrdinalIgnoreCase)
        || v.StartsWith("0o", StringComparison.OrdinalIgnoreCase)
        || (v.Length > 1 && v[0] == '0' && v.All(char.IsDigit));

    private static string YamlList(IReadOnlyList<string> items) =>
        items.Count == 0 ? "[]" : "[" + string.Join(", ", items.Select(Yaml)) + "]";
}
