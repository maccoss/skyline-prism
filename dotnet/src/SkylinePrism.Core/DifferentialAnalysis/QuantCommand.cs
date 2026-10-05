using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace SkylinePrism.Core.DifferentialAnalysis;

/// <summary>
/// The <c>prism differential ... --report</c> command that reproduces a <see cref="QuantRequest"/> -
/// written into every quant report, whichever front end produced it.
/// </summary>
/// <remarks>
/// <para>The quant report's counterpart to the pipeline's "Show Command Line": a report clicked in the
/// pane names the exact command that regenerates it headless, and a report typed at the CLI names its
/// own command back in canonical form. Every statistical choice is spelled out explicitly rather than
/// left to a default, so the command still means the same thing if a default later changes.</para>
/// <para>It only NAMES what it cannot carry - a clinical CSV, a user's own marker list - so those
/// must exist wherever the command is run. And where a value cannot survive the CLI's parsing at all
/// (a level containing a comma, which <c>-a</c> splits on), no command is offered rather than one
/// that would silently select something else: see <see cref="TryArguments"/>.</para>
/// </remarks>
public static class QuantCommand
{
    /// <summary>
    /// The arguments after <c>prism</c>, one per element, unquoted - or false, with the reason, when
    /// the request holds a value no <c>prism differential</c> command can express faithfully.
    /// </summary>
    /// <param name="request">What was run.</param>
    /// <param name="options">The options it RAN with, when those differ from
    /// <see cref="QuantRequest.Options"/> (see <see cref="QuantAnalysis.EffectiveOptions"/>).</param>
    public static bool TryArguments(QuantRequest request, DifferentialOptions options,
        out IReadOnlyList<string> arguments, out string? reason)
    {
        var ds = request.Dataset;
        var o = options;
        var rule = request.Rule;
        var isTrend = o.Design is DifferentialDesign.LinearTrend or DifferentialDesign.LinearTrendWithinSubject;
        var inv = CultureInfo.InvariantCulture;
        var problems = new List<string>();

        // A single-valued flag reads the next argument unless it starts with '-'.
        void Single(List<string> into, string flag, string value, string what)
        {
            Check(value, what, splitsOnComma: false);
            into.AddRange(new[] { flag, value });
        }

        // A list-valued flag also splits on ',' and trims each piece.
        void Listed(List<string> into, string flag, IEnumerable<string> values, string what)
        {
            var list = values.ToList();
            foreach (var v in list)
                Check(v, what, splitsOnComma: true);
            into.AddRange(new[] { flag, string.Join(",", list) });
        }

        // leadingDashOk: a value that travels inside a larger argument (COLUMN=VALUE) never starts it,
        // so its own first character cannot be read as a flag.
        void Check(string value, string what, bool splitsOnComma, bool leadingDashOk = false)
        {
            if (value.Length == 0)
                problems.Add($"{what} is empty");
            else if (value[0] == '-' && !leadingDashOk)
                problems.Add($"{what} '{value}' starts with '-', which the command line reads as a flag");
            else if (splitsOnComma && value.Contains(','))
                problems.Add($"{what} '{value}' contains a comma, which the command line splits on");
            else if (splitsOnComma && value.Trim().Length != value.Length)
                problems.Add($"{what} '{value}' has leading or trailing spaces, which the command line trims");
            else if (value.IndexOfAny(new[] { '"', '$', '`' }) >= 0)
                problems.Add($"{what} '{value}' contains a character no shell quotes the same way");
            // Backslash, but only where the two shells actually disagree. Inside bash double quotes a
            // backslash is an escape ONLY before $ ` " \ or a newline; the first three are already
            // refused above, which leaves the doubled backslash - bash collapses "\\" to one, PowerShell
            // keeps both - and a trailing one, which escapes the closing quote. A single backslash
            // before an ordinary character is literal in both, so an ordinary Windows path (C:\data\run)
            // still gets a command; refusing every backslash would mean never offering one on Windows.
            else if (value.Contains(@"\\", StringComparison.Ordinal))
                problems.Add($"{what} '{value}' contains a doubled backslash, which bash collapses to "
                    + "one inside quotes and PowerShell does not (a UNC path reaches the program with "
                    + "its leading separator lost)");
            else if (value[^1] == '\\')
                problems.Add($"{what} '{value}' ends with a backslash, which escapes the closing quote");
        }

        var args = new List<string> { "differential" };
        Single(args, "-d", TrimSeparator(request.OutputDir), "the output directory");
        foreach (var clinical in ds.AttachedClinicalCsvs)
            Single(args, "--clinical", clinical, "a clinical CSV path");
        args.AddRange(new[] { "--level", DifferentialTokens.Level(ds.Level) });
        args.AddRange(new[] { "--design", DifferentialTokens.Design(o.Design) });

        if (isTrend)
        {
            Single(args, "--trend-over", o.TrendColumn ?? string.Empty, "the trend column");
            // Recorded, or the command reproduces the analysis over a DIFFERENT set of samples -
            // every sample with a value on the axis rather than the subset that was fitted - and
            // nothing in its output would say so.
            foreach (var r in request.Restrictions.Where(r => r.Values.Count > 0))
            {
                Check(r.Column, "the restrict column", splitsOnComma: false);
                foreach (var v in r.Values)
                    Check(v, "a restricted value", splitsOnComma: true, leadingDashOk: true);
                // One flag per restriction, the form the CLI intersects - a single flag carrying two
                // columns would have to invent a separator the parser does not take.
                args.AddRange(new[] { "--restrict-to", r.Column + "=" + string.Join(",", r.Values) });
            }
        }
        else
        {
            Single(args, "--group-by", request.GroupBy ?? string.Empty, "the group-by column");
            // One comma-separated argument per arm - the form -a/-b split on.
            Listed(args, "-a", request.ALevels, "an arm A level");
            Listed(args, "-b", request.BLevels, "an arm B level");
        }

        if ((o.Design is DifferentialDesign.Paired or DifferentialDesign.LinearTrendWithinSubject)
            && request.SubjectColumn is not null)
            Single(args, "--subject", request.SubjectColumn, "the subject column");

        args.AddRange(new[] { "--test", DifferentialTokens.Test(o.Test) });
        args.AddRange(new[] { "--prior", DifferentialTokens.Prior(o.Prior) });
        // The command's default IS to fit the prior on the controls whenever the run has any, so only
        // the opposite choice - on a run that has controls - needs saying.
        if (o.PriorGroupColumns is null && ds.MetadataColumns.Contains("sample_type")
            && ControlSampleTypes.PriorGroups(ds.MetadataValues("sample_type")) is not null)
            args.Add("--prior-from-groups");
        if (o.Covariates is { Count: > 0 } covariates)
            Listed(args, "--adjust-for", covariates.Select(c => c.Name), "a covariate column");
        args.AddRange(new[] { "--correction", DifferentialTokens.Correction(o.Correction) });
        args.AddRange(new[] { "--alpha", rule.PThreshold.ToString("R", inv) });
        if (!rule.UseAdjusted)
            args.Add("--raw-p");
        args.AddRange(new[] { "--min-log2fc", rule.Log2FcThreshold.ToString("R", inv) });
        // Against the option's OWN default, not a literal 2 repeated here: the class's promise is that
        // the command still means the same thing if a default later changes, and a second copy of the
        // number is the one way it could stop being true.
        if (o.MinPerGroup != DifferentialOptions.DefaultMinPerGroup)
            args.AddRange(new[] { "--min-per-group", o.MinPerGroup.ToString(inv) });

        args.Add("--report");
        // Only when the panels could be grouped - which is when the report has a marker section at all.
        // A trend with no marker column wrote none, and --markers on a trend without one is refused.
        var markerColumn = request.MarkerGroupBy ?? (isTrend ? null : request.GroupBy);
        if (request.MarkerPanels.Count > 0 && markerColumn is not null && ds.MetadataColumns.Contains(markerColumn))
        {
            Listed(args, "--markers", request.MarkerPanels.Select(p => p.Name), "a marker panel name");
            Single(args, "--markers-group-by", markerColumn, "the markers group-by column");
        }

        if (request.EnrichmentPoster is null)
            args.Add("--no-enrichment");

        arguments = args;
        reason = problems.Count == 0 ? null : string.Join("; ", problems.Distinct(StringComparer.Ordinal));
        return reason is null;
    }

    /// <summary><see cref="TryArguments(QuantRequest, DifferentialOptions, out IReadOnlyList{string}, out string?)"/> over the request's own options.</summary>
    public static bool TryArguments(QuantRequest request, out IReadOnlyList<string> arguments, out string? reason) =>
        TryArguments(request, request.Options, out arguments, out reason);

    /// <summary>
    /// The whole command as one line to paste into a shell, or null (with <paramref name="reason"/>)
    /// when no command can express the request.
    /// </summary>
    /// <remarks>
    /// Anything that is not plainly a word is double-quoted - the convention the pipeline's own Show
    /// Command Line uses. A path is ALWAYS quoted: unquoted, bash reads each backslash as an escape and
    /// C:\Users\... arrives as C:Users.... Characters that bash and PowerShell treat differently even
    /// inside double quotes are refused by <see cref="TryArguments(QuantRequest, DifferentialOptions, out IReadOnlyList{string}, out string?)"/>,
    /// so what is printed here means the same in either shell.
    /// </remarks>
    public static string? For(QuantRequest request, DifferentialOptions options, out string? reason) =>
        TryArguments(request, options, out var args, out reason) ? Line(args) : null;

    /// <summary>
    /// Arguments already resolved by <see cref="TryArguments(QuantRequest, DifferentialOptions, out IReadOnlyList{string}, out string?)"/>,
    /// as the one line to paste into a shell.
    /// </summary>
    /// <remarks>
    /// Separate from <see cref="For"/> so a caller holding the argument list does not resolve the
    /// request a second time to print it. The report records both, and two resolutions could in
    /// principle disagree - the line and the list it is supposed to be a rendering of.
    /// </remarks>
    public static string Line(IReadOnlyList<string> arguments) =>
        "prism " + string.Join(" ", arguments.Select(Quote));

    // Not ',' or '@': unquoted, PowerShell reads a,b as an array and a leading @ as splatting.
    private static string Quote(string arg) =>
        arg.Length > 0 && arg.All(c => char.IsLetterOrDigit(c) || "-_.=+".Contains(c))
            ? arg
            : "\"" + arg + "\"";

    /// <summary>
    /// A trailing separator would sit against the closing quote as \" - an escaped quote to bash - so
    /// it is dropped; the directory is the same one.
    /// </summary>
    private static string TrimSeparator(string path) => Path.TrimEndingDirectorySeparator(path);
}
