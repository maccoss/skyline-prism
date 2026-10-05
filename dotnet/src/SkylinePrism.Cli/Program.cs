using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.DifferentialAnalysis.Enrichment;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;

namespace SkylinePrism.Cli;

/// <summary>
/// Entry point for the cross-platform `prism` CLI. Mirrors the Python subcommands
/// (run / merge / config-template / version). QC report generation (qc) arrives with Layer 8.
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "--help" or "-h" or "help")
            return PrintUsage();
        // Note: `-v` is intentionally NOT a version alias - Python's CLI uses -v for verbose. C# has no
        // verbosity levels (it always logs fully to console + prism_run_<ts>.log). Use --version / version.
        if (args[0] is "--version" or "version")
            return PrintVersion();

        var rest = args[1..];
        if (Array.Exists(rest, a => a is "--help" or "-h"))
        {
            Console.WriteLine(CommandHelp(args[0]));
            return 0;
        }

        try
        {
            return args[0] switch
            {
                "run" => CmdRun(rest),
                "merge" => CmdMerge(rest),
                "qc" => CmdQc(rest),
                "differential" => CmdDifferential(rest),
                "ion-accounting" => CmdIonAccounting(rest),
                "isolation-scheme" => CmdIsolationScheme(rest),
                "compare" => CmdCompare(rest),
                "config-template" => CmdConfigTemplate(rest),
                _ => Unknown(args[0]),
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int CmdRun(string[] args)
    {
        var opts = ParseOptions(args, multiValue: new HashSet<string> { "-i", "--input", "-m", "--metadata" });
        var inputs = new List<string>(opts.GetList("-i", "--input"));
        var metadataFiles = opts.GetList("-m", "--metadata");
        var outputDir = opts.GetSingle("-o", "--output-dir");
        var configPath = opts.GetSingleOrNull("-c", "--config");
        var provenancePath = opts.GetSingleOrNull("--from-provenance");
        var forceReprocess = opts.GetSingleOrNull("--force-reprocess") is not null;

        PrismConfig config;
        var provenanceLoaded = false;
        if (provenancePath is not null)
        {
            config = Provenance.LoadConfig(provenancePath, out var redirectedFasta);
            // A substituted database must never be silent: it changes protein grouping and iBAQ counts.
            foreach (var key in redirectedFasta)
                Console.Error.WriteLine(
                    $"NOTE: {key} no longer exists; using the copy this run archived beside its outputs.");
            if (inputs.Count == 0)
                foreach (var s in Provenance.SourceFiles(provenancePath))
                    inputs.Add(s);
            provenanceLoaded = true;
            config.Validate();
        }
        else
        {
            config = configPath is not null
                ? PrismConfig.LoadValidated(configPath, w => Console.Error.WriteLine($"WARNING: {w}"))
                : new PrismConfig();
        }

        if (opts.GetSingleOrNull("--no-save-plots") is not null)
            config.QcReport.SavePlots = false; // override: skip writing qc_plots/*.png

        if (inputs.Count == 0 || outputDir is null)
        {
            Console.Error.WriteLine(
                "Usage: prism run -i <input...> -o <output-dir> [-c <config.yaml>] [-m <metadata...>] "
                + "[--from-provenance <parameters.json>] [--no-save-plots]");
            return 2;
        }

        Directory.CreateDirectory(outputDir);

        // Timestamped run log in the output dir (mirrors the Python CLI), tee'd with the console.
        var logPath = Path.Combine(outputDir, $"prism_run_{DateTime.Now:yyyyMMdd_HHmmss}.log");
        using var logFile = new StreamWriter(logPath, append: false) { AutoFlush = true };
        void Log(string m)
        {
            Console.WriteLine(m);
            logFile.WriteLine(m);
        }

        if (provenanceLoaded)
            Log($"PRISM: loaded settings from provenance {provenancePath}");

        // Warned about, not refused. A CLI run is usually scripted, and something that stops a
        // pipeline to ask a question is worse than the surprise it prevents - so this says what is
        // about to go and carries on. It is silent when the previous run used the same version and
        // the same settings, which is how a QC report gets regenerated and a partial ion accounting
        // topped up; a warning that fires on the ordinary case stops being read.
        // With the inputs, so the merge and the transition rollup are answered by the same
        // CanReuse the pipeline will call rather than predicted - which is what catches an input
        // file edited in place under settings that did not move.
        if (ExistingResults.Inspect(outputDir, config, inputs).Warning() is { } warning)
            Log("WARNING: " + warning);

        Log($"PRISM: merging {inputs.Count} input(s) -> {outputDir}");
        var result = PrismPipeline.Run(
            inputs, outputDir, config, metadataFiles.Count > 0 ? metadataFiles : null, Log, forceReprocess);
        Log($"Done: {result.NPeptides} peptides, {result.NProteins} proteins, {result.NSamples} samples, "
            + $"{result.Batches.Count} batch(es). Outputs in {outputDir}");
        Console.WriteLine($"Run log: {logPath}");
        return 0;
    }

    private static int CmdMerge(string[] args)
    {
        var opts = ParseOptions(args, multiValue: new HashSet<string>(), positional: true);
        var output = opts.GetSingle("-o", "--output");
        var inputs = opts.Positional;
        if (inputs.Count == 0 || output is null)
        {
            Console.Error.WriteLine("Usage: prism merge <input...> -o <output.parquet>");
            return 2;
        }
        // The merge writes peptide-hash partitions, so the output is a DIRECTORY of parquet files, not
        // one file. Accept a "-o something.parquet" anyway - it is the natural thing to type, and it was
        // the contract until this release - but drop the extension rather than create a directory called
        // "x.parquet", and say plainly where the data actually went.
        var root = output.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase)
            ? output[..^".parquet".Length]
            : output;

        var result = DuckDbMerge.Merge(inputs, root);
        Console.WriteLine(
            $"Merged {inputs.Count} file(s) -> {result.OutputPath}{Path.DirectorySeparatorChar} "
            + $"({result.TotalRows} rows in {result.Partitions} peptide partition(s))");
        // hive_partitioning=false matters: without it DuckDB reads the _pep_bucket=N directory names
        // back as an extra column, so the caller gets a schema PRISM itself never sees. Every read
        // inside PRISM passes it (MergedParquetReader.Scan) and so must the advice we print.
        Console.WriteLine(
            $"  Read it as a single table with: read_parquet('{result.OutputPath.Replace('\\', '/')}"
            + "/**/*.parquet', hive_partitioning=false)");
        return 0;
    }

    private static int CmdQc(string[] args)
    {
        var opts = ParseOptions(args, multiValue: new HashSet<string>());
        var dir = opts.GetSingle("-d", "--dir") ?? opts.GetSingle("-o", "--output-dir");
        var configPath = opts.GetSingleOrNull("-c", "--config");
        if (dir is null)
        {
            Console.Error.WriteLine("Usage: prism qc -d <output-dir> [-c config]");
            return 2;
        }
        // -c carries report options only. The processing settings the report PRINTS come from the
        // directory's own parameters.json (QcReport.ReadRunConfig), never from this config - filling
        // the sections a QC-only config omits with defaults would attribute the numbers to settings
        // that never ran.
        var config = configPath is not null
            ? PrismConfig.LoadValidated(configPath, w => Console.Error.WriteLine($"WARNING: {w}"))
            : new PrismConfig();
        if (opts.GetSingleOrNull("--no-save-plots") is not null)
            config.QcReport.SavePlots = false;
        var path = QcReport.Generate(
            dir, config, savePlots: config.QcReport.SavePlots, log: Console.WriteLine);
        Console.WriteLine($"QC report written to: {path}");
        return 0;
    }

    /// <summary>
    /// Two-group differential abundance against a finished output directory - the whole statistical
    /// menu the Differential pane offers, from a headless shell.
    /// </summary>
    /// <remarks>
    /// Its own command, and never part of <c>prism run</c>: a contrast is a question asked OF a
    /// finished result, and the same result answers many of them. It reads
    /// <c>corrected_{peptides,proteins}.parquet</c> and <c>sample_metadata.csv</c> and writes
    /// nothing back, so it cannot disturb the run it reads.
    ///
    /// <para>Arms are resolved by <see cref="ContrastArms"/>, the same code the GUI uses, so a
    /// selection made in the pane and one typed here cannot mean different samples.</para>
    /// </remarks>
    private static int CmdDifferential(string[] args)
    {
        var opts = ParseOptions(args,
            multiValue: new HashSet<string>
                { "-a", "--group-a", "-b", "--group-b", "--adjust-for", "--markers", "--clinical",
                  "--restrict-to" });
        var dir = opts.GetSingleOrNull("-d", "--dir") ?? opts.GetSingleOrNull("--output-dir");
        RefuseReportFlagsWithoutReport(opts);
        // Resolved before anything runs, so a mistyped panel name refuses the command rather than
        // failing after differential.csv has already been written.
        var markerPanels = ResolveMarkerPanels(opts.GetList("--markers"));
        var groupBy = opts.GetSingleOrNull("-g", "--group-by");
        // Both spellings on both arms: a level with a space in it is one -a argument, several levels
        // are several, and a comma-separated list is what a reader reaches for first.
        var aLevels = SplitLevels(opts.GetList("-a", "--group-a"));
        var bLevels = SplitLevels(opts.GetList("-b", "--group-b"));
        var trendRequested = IsTrendRequested(opts);
        if (dir is null || (!trendRequested && (groupBy is null || aLevels.Count == 0 || bLevels.Count == 0)))
        {
            Console.Error.WriteLine(
                "Usage: prism differential -d <output-dir> --group-by <column> -a <level...> -b <level...>");
            Console.Error.WriteLine(
                "   or: prism differential -d <output-dir> --design trend --trend-over <column>");
            Console.Error.WriteLine("Run 'prism differential --help' for the full option list.");
            return 2;
        }

        var level = (opts.GetSingleOrNull("--level") ?? "protein").ToLowerInvariant() switch
        {
            "peptide" or "peptides" => FeatureLevel.Peptide,
            "protein" or "proteins" => FeatureLevel.Protein,
            var other => throw new ArgumentException($"--level must be protein or peptide, not '{other}'"),
        };

        var dataset = DifferentialDataset.Load(dir, level);
        // Before anything validates a column name, so --group-by, --adjust-for and the rest can name
        // a clinical column, as they can in the pane once a clinical CSV is attached there.
        AttachClinicalFrom(opts, dataset);
        RefuseUngroupableMarkers(opts, dataset, markerPanels, trendRequested);
        if (trendRequested)
            return RunDifferentialTrend(opts, dataset, level, dir, markerPanels);
        if (!dataset.MetadataColumns.Contains(groupBy!))
            throw new ArgumentException(
                $"No metadata column '{groupBy}'. Available: {string.Join(", ", dataset.MetadataColumns)}");

        var arms = ContrastArms.Resolve(dataset.MetadataValues(groupBy!), aLevels, bLevels);
        if (!arms.Ok)
        {
            Console.Error.WriteLine($"Error: {arms.Error}");
            var present = dataset.MetadataValues(groupBy!)
                .Where(v => !string.IsNullOrEmpty(v)).Distinct(StringComparer.Ordinal)
                .OrderBy(v => v, StringComparer.Ordinal);
            Console.Error.WriteLine($"Values of '{groupBy}': {string.Join(", ", present)}");
            return 2;
        }

        var options = DifferentialOptionsFrom(opts, dataset, groupBy!);
        var result = Differential.Run(dataset.ExprLog2, dataset.FeatureIds, arms.A, arms.B, options);

        var aLabel = ContrastArms.Describe(aLevels);
        var bLabel = ContrastArms.Describe(bLevels);
        Console.WriteLine($"{options.Describe(result.VariancePrior)}: {groupBy} = {bLabel} vs {aLabel}");
        // Named, not positional. The line above reads "B vs A" while NA/NB are A then B, so a bare
        // "n = 16 vs 160" under it invites reading the first number as the arm named first.
        Console.WriteLine(
            $"  n = {bLabel} {result.NB} vs {aLabel} {result.NA}; "
            + $"{result.NFeaturesTested} of {result.NFeaturesTotal} "
            + $"{(level == FeatureLevel.Peptide ? "peptides" : "proteins")} tested");
        foreach (var m in result.Messages.Concat(result.Warnings))
            Console.WriteLine($"  {m}");

        // The count is the headline, so say what it counts: the same list under BY and under BH is
        // two different claims, and a bare "41 significant" records neither.
        var rule = SignificanceRuleFrom(opts);
        var hits = result.Rows.Count(rule.IsSignificant);
        Console.WriteLine(
            $"  {hits} hit{(hits == 1 ? "" : "s")} ({rule.Describe()}, {DifferentialCsv.CorrectionName(options.Correction)})");

        var outPath = opts.GetSingleOrNull("-o", "--output") ?? Path.Combine(dir, "differential.csv");
        DifferentialCsv.Write(outPath, result, dataset, options, rule, groupBy!, aLabel, bLabel);
        Console.WriteLine($"Results written to: {outPath}");

        if (opts.GetSingleOrNull("--report") is not null)
            WriteQuantReport(opts, dataset, dir, options, rule, result, groupBy, arms.A, arms.B,
                aLevels, bLevels, markerPanels);
        return 0;
    }

    /// <summary>
    /// <c>--report</c>: the quant report the pane's Quant report button writes, for this contrast.
    /// </summary>
    /// <remarks>
    /// Runs the SAME <see cref="QuantAnalysis"/> as the button, from the options this command already
    /// built and the result it already computed (so the contrast is not run twice), and writes
    /// quant/differential.csv through the same writer - so it is byte-identical to the file just
    /// written, and a report typed here matches one clicked there. Every view that could not run is
    /// printed with the reason - a report is never refused for a missing optional view.
    /// </remarks>
    private static void WriteQuantReport(ParsedOptions opts, DifferentialDataset dataset, string dir,
        DifferentialOptions options, SignificanceRule rule, DifferentialResult result, string? groupBy,
        IReadOnlyList<int> groupA, IReadOnlyList<int> groupB,
        IReadOnlyList<string> aLevels, IReadOnlyList<string> bLevels, IReadOnlyList<ProteinList> panels)
    {
        // Enrichment runs by default, as it does from the button; --no-enrichment is for machines
        // with no route to g:Profiler, where waiting on the request timeout would be pointless.
        using var poster = opts.GetSingleOrNull("--no-enrichment") is null ? new HttpJsonPoster() : null;

        var report = QuantAnalysis.Run(new QuantRequest
        {
            OutputDir = dir,
            Dataset = dataset,
            Options = options,
            Rule = rule,
            Differential = result,
            SubjectColumn = opts.GetSingleOrNull("--subject", "--pair-by"),
            GroupBy = groupBy,
            GroupA = groupA,
            GroupB = groupB,
            ALevels = aLevels,
            BLevels = bLevels,
            MarkerPanels = panels,
            MarkerGroupBy = opts.GetSingleOrNull("--markers-group-by"),
            EnrichmentPoster = poster,
            Restrictions = ParseRestrictions(opts),
        });

        Console.WriteLine($"Quant report written to: {report.HtmlPath}");
        foreach (var note in report.Notes)
            Console.WriteLine($"  {note}");
    }

    /// <summary>
    /// The named marker panels, from the same set the Markers pane offers: the user's saved lists
    /// plus the shipped panels (a shipped one a user list shadows appears with a "(PRISM)" suffix).
    /// Matched ignoring case; an unknown name is refused with the available ones listed.
    /// </summary>
    private static List<ProteinList> ResolveMarkerPanels(IEnumerable<string> raw)
    {
        var names = SplitLevels(raw).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        if (names.Count == 0)
            return new List<ProteinList>();

        var available = ProteinListSet.Load().WithBuiltIns().Where(l => l.Members.Count > 0).ToList();
        return names.Select(name =>
                available.FirstOrDefault(l => string.Equals(l.Name, name, StringComparison.OrdinalIgnoreCase))
                ?? throw new ArgumentException(
                    $"No marker panel '{name}'. Available: {string.Join(", ", available.Select(l => l.Name))}"))
            .ToList();
    }

    /// <summary>
    /// <c>--clinical</c>: join an external clinical table to the samples - the same
    /// <see cref="DifferentialDataset.AttachClinical"/> the pane's Clinical CSV input runs, so the key
    /// column is detected the same way and the same columns are added.
    /// </summary>
    /// <remarks>
    /// <para>Refused when nothing matches: the pane can say so and carry on, but a command that named a
    /// clinical file almost certainly depends on one of its columns, and would otherwise fail later on
    /// "no metadata column" without saying the join was the cause.</para>
    /// <para>Repeatable, and joined in the order given - the pane lets a second file be attached on top
    /// of the first, and a report records both, so the command that regenerates it must take both.</para>
    /// </remarks>
    private static void AttachClinicalFrom(ParsedOptions opts, DifferentialDataset dataset)
    {
        foreach (var path in opts.GetList("--clinical"))
        {
            var joined = dataset.AttachClinical(path);
            var rate = (joined.MatchRate * 100).ToString("0", CultureInfo.InvariantCulture) + "%";
            if (joined.KeyColumn is null || joined.AddedColumns.Count == 0)
                throw new ArgumentException(
                    $"--clinical: no column of {Path.GetFileName(path)} matched the samples (best match {rate}); "
                    + "the key column needs to hold the replicate names for at least half the samples.");
            Console.WriteLine($"Clinical CSV: joined {Path.GetFileName(path)} on '{joined.KeyColumn}' "
                + $"({rate} of samples matched), added {string.Join(", ", joined.AddedColumns)}");
        }
    }

    /// <summary>
    /// Marker panels need a column to group by. An unknown one is refused, as an unknown
    /// <c>--group-by</c> is; and a trend has no contrast column to fall back to, so it must be named.
    /// The report itself would only note the omission - the CLI refuses instead, because a typo there
    /// would otherwise produce a report quietly missing the section the command asked for.
    /// </summary>
    private static void RefuseUngroupableMarkers(ParsedOptions opts, DifferentialDataset dataset,
        IReadOnlyList<ProteinList> panels, bool trend)
    {
        var groupBy = opts.GetSingleOrNull("--markers-group-by");
        if (groupBy is not null && !dataset.MetadataColumns.Contains(groupBy))
            throw new ArgumentException(
                $"No metadata column '{groupBy}' to group the marker panels by. "
                + $"Available: {string.Join(", ", dataset.MetadataColumns)}");
        if (panels.Count > 0 && trend && groupBy is null)
            throw new ArgumentException(
                "--markers under a trend design needs --markers-group-by <column>: a trend has no "
                + "contrast column to group the panels by.");
    }

    /// <summary>
    /// A report-only flag without <c>--report</c> is refused, as <c>--subject</c> is without a paired
    /// design: silently ignoring it would let a command read as if it produced a marker section it
    /// never wrote.
    /// </summary>
    private static void RefuseReportFlagsWithoutReport(ParsedOptions opts)
    {
        if (opts.GetSingleOrNull("--report") is not null)
            return;
        if (opts.GetList("--markers").Count > 0)
            throw new ArgumentException("--markers needs --report.");
        foreach (var flag in new[] { "--markers-group-by", "--no-enrichment" })
            if (opts.GetSingleOrNull(flag) is not null)
                throw new ArgumentException($"{flag} needs --report.");
    }

    /// <summary>Whether the flags ask for a trend design.</summary>
    private static bool IsTrendRequested(ParsedOptions opts) =>
        (opts.GetSingleOrNull("--design") ?? string.Empty).ToLowerInvariant()
            is "trend" or "trend-within-subject" or "trend-repeated";

    /// <summary>
    /// <c>prism differential --design trend</c>: fit a slope against a numeric column.
    /// </summary>
    /// <remarks>
    /// Separate from the two-arm path for the same reason the pane's is - there are no arms to
    /// resolve or name - while sharing the options builder, the hit rule and the CSV writer, so a
    /// trend result is the same file shape as any other.
    /// </remarks>
    /// <summary>
    /// The sample columns a trend runs over: every sample, less those excluded by each
    /// <c>--restrict-to COLUMN=VALUE[,VALUE...]</c>.
    /// </summary>
    /// <remarks>
    /// <para>A trend pools every sample that has an x, which silently merges studies that share a
    /// timepoint scale - two arms of one cohort both measured at week 4 become one slope through
    /// both. Restricting is the only way to fit the one that was meant.</para>
    /// <para>Expressed by dropping the COLUMN rather than by NaN-ing its x, because NaN arrives at
    /// <see cref="TrendSamples.Resolve"/> as "no value in the trend column" - so a deliberate subset
    /// was counted and reported as missing data. The pane drops columns for the same reason, so the
    /// two front ends agree on which samples a restriction keeps and on what they say about it.</para>
    /// </remarks>
    private static int[] RestrictTrend(ParsedOptions opts, DifferentialDataset dataset)
    {
        var request = new QuantRequest
        {
            OutputDir = string.Empty,
            Dataset = dataset,
            Options = new DifferentialOptions(),
            Rule = SignificanceRule.Default,
            Restrictions = ParseRestrictions(opts),
        };
        // Through the same resolver the quant report uses, so `prism differential` and the report it
        // can write never select different samples from one set of flags.
        return QuantAnalysis.TrendColumnsFor(request, dataset);
    }

    /// <summary>
    /// <c>--restrict-to COLUMN=VALUE[,VALUE...]</c>, validated against the run's own metadata.
    /// </summary>
    private static IReadOnlyList<QuantRestriction> ParseRestrictions(ParsedOptions opts)
        => opts.GetList("--restrict-to").Select(ParseOneRestriction).ToList();

    private static QuantRestriction ParseOneRestriction(string spec)
    {
        var eq = spec.IndexOf('=');
        if (eq <= 0 || eq == spec.Length - 1)
            throw new ArgumentException($"--restrict-to wants COLUMN=VALUE[,VALUE...], not '{spec}'.");

        var column = spec[..eq].Trim();
        var keep = spec[(eq + 1)..].Split(',')
            .Select(v => v.Trim()).Where(v => v.Length > 0).ToList();
        if (keep.Count == 0)
            throw new ArgumentException($"--restrict-to '{spec}' names no values to keep.");
        return new QuantRestriction(column, keep);
    }

    private static int RunDifferentialTrend(
        ParsedOptions opts, DifferentialDataset dataset, FeatureLevel level, string dir,
        IReadOnlyList<ProteinList> markerPanels)
    {
        var options = DifferentialOptionsFrom(opts, dataset, groupBy: string.Empty);
        var trendOver = options.TrendColumn!;
        var axis = TrendAxis.Find(trendOver, dataset.MetadataColumns, dataset.MetadataValues)!;
        var x = TrendAxis.Read(dataset.MetadataValues(axis.Column), axis);
        var columns = RestrictTrend(opts, dataset);
        var restricted = dataset.SampleIds.Length - columns.Length;
        if (restricted > 0)
            Console.WriteLine(
                $"--restrict-to: {restricted} sample(s) outside the kept values were left out.");
        if (columns.Length == 0)
            throw new ArgumentException("--restrict-to: no sample matches the kept values.");
        var result = Differential.RunTrend(dataset.ExprLog2, dataset.FeatureIds, columns, x, options);
        var rule = SignificanceRuleFrom(opts);

        // Over the samples the fit USED, not the whole column. The endpoints become the span in the
        // results header, which states that log2fc is the modeled change across it - so reading them
        // from excluded samples would describe the numbers by a range the model never saw. Before
        // --restrict-to the two were the same set and the distinction did not arise.
        var (xLow, xHigh) = DifferentialCsv.TrendEndpoints(columns.Select(c => x[c]).ToList());
        var span = $"{xLow} to {xHigh}";
        var n = result.NSubjects > 0
            ? $"{result.NA} samples in {result.NSubjects} subjects"
            : $"{result.NA} samples";

        Console.WriteLine($"{options.Describe(result.VariancePrior)}: {trendOver} {span}");
        Console.WriteLine(
            $"  n = {n}; {result.NFeaturesTested} of {result.NFeaturesTotal} "
            + $"{(level == FeatureLevel.Peptide ? "peptides" : "proteins")} tested");
        foreach (var m in result.Messages.Concat(result.Warnings))
            Console.WriteLine($"  {m}");

        var effectName = $"log2 change across {trendOver}";
        var hits = result.Rows.Count(rule.IsSignificant);
        Console.WriteLine($"  {hits} hit{(hits == 1 ? "" : "s")} ({rule.Describe(effectName)}, "
            + $"{DifferentialCsv.CorrectionName(options.Correction)})");

        var outPath = opts.GetSingleOrNull("-o", "--output") ?? Path.Combine(dir, "differential.csv");
        DifferentialCsv.Write(outPath, result, dataset, options, rule, trendOver,
            aLabel: xLow, bLabel: xHigh, effectName: effectName);
        Console.WriteLine($"Results written to: {outPath}");

        // A trend has no arms; the report takes its column and range from the options and the data.
        if (opts.GetSingleOrNull("--report") is not null)
            WriteQuantReport(opts, dataset, dir, options, rule, result, groupBy: null,
                Array.Empty<int>(), Array.Empty<int>(), Array.Empty<string>(), Array.Empty<string>(),
                markerPanels);
        return 0;
    }

    /// <summary>
    /// <c>--min-per-group</c> as a whole number. It used to be parsed as a double and cast, so 2.7 ran
    /// as 2 with nothing said.
    /// </summary>
    private static int MinPerGroupFrom(string? text)
    {
        if (text is null)
            return DifferentialOptions.DefaultMinPerGroup;
        if (!int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var n) || n < 1)
            throw new ArgumentException($"--min-per-group must be a whole number of at least 1, not '{text}'.");
        return n;
    }

    /// <summary>Levels from repeated flags and/or comma-separated lists, in the order given.</summary>
    private static List<string> SplitLevels(IEnumerable<string> raw) =>
        raw.SelectMany(v => v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
           .ToList();

    private static double ParseDouble(string? text, double fallback) =>
        text is not null && double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)
            ? v : fallback;

    /// <summary>
    /// The hit rule, from the same flags the pane offers as controls.
    /// </summary>
    /// <remarks>
    /// <c>--alpha</c> keeps its old meaning as the p-value cut. It used to be applied to the
    /// adjusted p unconditionally; <c>--raw-p</c> now moves it to the uncorrected one, and the
    /// printed rule says which, so a logged run cannot be mistaken for a corrected one.
    /// </remarks>
    private static SignificanceRule SignificanceRuleFrom(ParsedOptions opts) => new()
    {
        PThreshold = ParseDouble(opts.GetSingleOrNull("--alpha"), 0.05),
        UseAdjusted = opts.GetSingleOrNull("--raw-p") is null,
        Log2FcThreshold = ParseDouble(opts.GetSingleOrNull("--min-log2fc"), 1.0),
    };

    /// <summary>The statistical selections, read from the flags the same way the pane reads combos.</summary>
    private static DifferentialOptions DifferentialOptionsFrom(
        ParsedOptions opts, DifferentialDataset dataset, string groupBy)
    {
        var design = (opts.GetSingleOrNull("--design") ?? "unpaired").ToLowerInvariant() switch
        {
            "unpaired" => DifferentialDesign.Unpaired,
            "paired" => DifferentialDesign.Paired,
            "trend" => DifferentialDesign.LinearTrend,
            "trend-within-subject" or "trend-repeated" => DifferentialDesign.LinearTrendWithinSubject,
            var other => throw new ArgumentException(
                $"--design must be unpaired, paired, trend or trend-within-subject, not '{other}'"),
        };
        var test = (opts.GetSingleOrNull("--test") ?? "moderated").ToLowerInvariant() switch
        {
            "moderated" or "moderated-t" => DifferentialTest.ModeratedT,
            "welch" or "welch-t" => DifferentialTest.WelchT,
            "student" or "student-t" => DifferentialTest.StudentT,
            "paired-t" => DifferentialTest.PairedT,
            "wilcoxon" => DifferentialTest.Wilcoxon,
            "mann-whitney" or "mannwhitney" => DifferentialTest.MannWhitney,
            var other => throw new ArgumentException($"Unknown --test '{other}'"),
        };
        var prior = (opts.GetSingleOrNull("--prior") ?? "intensity-trend").ToLowerInvariant() switch
        {
            "intensity-trend" => VariancePrior.IntensityTrend,
            "global" => VariancePrior.Global,
            "limma-trend" => VariancePrior.LimmaTrend,
            "peptide-count" => VariancePrior.PeptideCount,
            var other => throw new ArgumentException($"Unknown --prior '{other}'"),
        };
        var correction = (opts.GetSingleOrNull("--correction") ?? "bh").ToLowerInvariant() switch
        {
            "bh" or "benjamini-hochberg" => MultipleTesting.BenjaminiHochberg,
            "by" or "benjamini-yekutieli" => MultipleTesting.BenjaminiYekutieli,
            "holm" => MultipleTesting.Holm,
            "bonferroni" => MultipleTesting.Bonferroni,
            "none" => MultipleTesting.None,
            var other => throw new ArgumentException($"Unknown --correction '{other}'"),
        };

        string?[]? subjects = null;
        // One column, two designs: it matches each subject's two samples under --design paired, and
        // gives each subject its own level under --design trend-within-subject. --subject is the
        // name that reads correctly for both; --pair-by stays as its alias.
        var pairBy = opts.GetSingleOrNull("--subject", "--pair-by");
        var needsSubject = design is DifferentialDesign.Paired
            or DifferentialDesign.LinearTrendWithinSubject;
        if (needsSubject)
        {
            if (pairBy is null)
                throw new ArgumentException(
                    $"--design {DifferentialTokens.Design(design)} needs --subject <column> to group samples by.");
            if (!dataset.MetadataColumns.Contains(pairBy))
                throw new ArgumentException($"No metadata column '{pairBy}' to group subjects by.");
            subjects = dataset.MetadataValues(pairBy);
        }
        else if (pairBy is not null)
        {
            // Silently ignoring it would report an unpaired result for a command that reads paired.
            throw new ArgumentException(
                "--subject needs --design paired or --design trend-within-subject.");
        }

        var isTrend = design is DifferentialDesign.LinearTrend
            or DifferentialDesign.LinearTrendWithinSubject;
        var trendOver = opts.GetSingleOrNull("--trend-over");
        // Kept, not just checked: the tested COLUMN is what --adjust-for is compared with below, and
        // trendOver may be a label naming one reading of a column rather than the column itself.
        TrendAxisOption? axis = null;
        if (isTrend)
        {
            if (trendOver is null)
                throw new ArgumentException(
                    "A trend design needs --trend-over <axis>, the column to fit the slope against.");
            // Resolved through TrendAxis, not by a bare column check, so the CLI accepts exactly the
            // axes the pane offers - including one read out of a text column, named for the number it
            // takes ("Longitudinal Draw Description (Week)").
            axis = TrendAxis.Find(trendOver, dataset.MetadataColumns, dataset.MetadataValues);
            if (axis is null)
            {
                var offered = TrendAxis.AllFor(dataset.MetadataColumns, dataset.MetadataValues);
                var readings = offered.Where(a => string.Equals(a.Column, trendOver, StringComparison.Ordinal)).ToList();
                // Every axis, not the first few: a cut list can leave out exactly the one meant, and
                // nothing else on the command line shows the rest. A column offering two readings is
                // named for what it is - a choice the command has to make - rather than "not found".
                throw new ArgumentException(readings.Count > 1
                    ? $"'{trendOver}' can be read as more than one axis: "
                      + string.Join(", ", readings.Select(a => $"'{a.Label}'"))
                      + ". They are different quantities; name the one you mean with --trend-over."
                    : $"No trend axis '{trendOver}'. "
                      + (offered.Count == 0
                          ? "This run has no column a slope can be fitted against."
                          : "Available: " + string.Join(", ", offered.Select(o => $"'{o.Label}'")) + "."));
            }
        }
        else if (trendOver is not null)
        {
            throw new ArgumentException("--trend-over needs --design trend or --design trend-within-subject.");
        }

        // Refused rather than ignored, for the reason every other misplaced flag here is: a command
        // that quietly drops a restriction fits the slope through samples the user excluded and
        // reports it as if they had not.
        if (!isTrend && opts.GetList("--restrict-to").Count > 0)
        {
            throw new ArgumentException(
                "--restrict-to needs --design trend or --design trend-within-subject. On a two-arm "
                + "contrast, choose the samples with -a and -b.");
        }

        if (isTrend)
        {
            QuantAnalysis.ValidateRestrictions(ParseRestrictions(opts), dataset, "--restrict-to");
        }

        var covariates = new List<Covariate>();
        foreach (var name in SplitLevels(opts.GetList("--adjust-for")))
        {
            if (!dataset.MetadataColumns.Contains(name))
                throw new ArgumentException($"No metadata column '{name}' to adjust for.");
            // The tested term, whichever it is. On a trend the guard used to compare against an
            // EMPTY groupBy, so --trend-over week --adjust-for week built [1, week, week] - exactly
            // singular - and died on a rank check naming neither flag. And it is the axis's COLUMN:
            // the label of a reading ("Visit (Week)") names no column, so comparing with it let
            // --adjust-for Visit through, whose categories encode the very timepoints being tested.
            var tested = axis?.Column ?? groupBy;
            if (!string.IsNullOrEmpty(tested) && string.Equals(name, tested, StringComparison.Ordinal))
                throw new ArgumentException(
                    $"'{name}' is the term being tested; adjusting for it would leave nothing to test.");
            covariates.Add(Covariate.FromMetadata(name, dataset.MetadataValues(name)));
        }
        if (covariates.Count > 0 && test != DifferentialTest.ModeratedT)
            throw new ArgumentException(
                "--adjust-for needs --test moderated; the other tests have no design matrix to put a "
                + "covariate in.");

        // A test that the chosen design cannot run is REFUSED, never quietly swapped. The pane hides
        // the inapplicable entries so the question cannot arise there; the CLI has no such filter,
        // and without this it accepted --design trend --test mann-whitney, ran the moderated t, and
        // then printed "Mann-Whitney U" over the result because Describe() reads what was ASKED for.
        var allowed = design switch
        {
            DifferentialDesign.Unpaired => new[]
            {
                DifferentialTest.ModeratedT, DifferentialTest.WelchT, DifferentialTest.StudentT,
                DifferentialTest.MannWhitney,
            },
            DifferentialDesign.Paired => new[]
            {
                DifferentialTest.ModeratedT, DifferentialTest.PairedT, DifferentialTest.Wilcoxon,
            },
            // A trend has no two samples to compare, only a slope.
            _ => new[] { DifferentialTest.ModeratedT },
        };
        if (Array.IndexOf(allowed, test) < 0)
            throw new ArgumentException(
                $"--test {DifferentialTokens.Test(test)} does not apply to --design {DifferentialTokens.Design(design)}. "
                + $"That design runs: {string.Join(", ", allowed.Select(DifferentialTokens.Test))}.");

        // The prior's per-feature SCALE comes from the run's QC and reference replicates whenever
        // it has any. That is the default, not an option, because the design groups of a real study
        // contain the biology the analysis exists to find: a prior fitted on them describes
        // measurement noise plus that biology, and shrinks genuine effects toward nothing. Control
        // injections are nominal replicates, so their spread IS the measurement variance the prior
        // is meant to describe. The prior degrees of freedom stay global either way, so the amount
        // of shrinkage remains calibrated to the study samples.
        IReadOnlyList<IReadOnlyList<int>>? priorGroups = null;
        var fromGroups = opts.GetSingleOrNull("--prior-from-groups") is not null;
        var fromControls = opts.GetSingleOrNull("--prior-from-controls") is not null;
        if (fromGroups && fromControls)
            throw new ArgumentException(
                "--prior-from-controls and --prior-from-groups ask for opposite things.");

        if (!fromGroups && dataset.MetadataColumns.Contains("sample_type"))
            priorGroups = ControlSampleTypes.PriorGroups(dataset.MetadataValues("sample_type"));

        // Asked for explicitly, an absent source is an error rather than a silent fallback.
        if (fromControls && priorGroups is null)
            throw new ArgumentException(
                "--prior-from-controls found no control type with two or more replicates.");

        return new DifferentialOptions
        {
            Design = design,
            Test = test,
            Prior = prior,
            Correction = correction,
            SubjectLabels = subjects,
            TrendColumn = trendOver,
            PeptideCounts = dataset.PeptideCounts,
            PriorGroupColumns = priorGroups,
            Covariates = covariates.Count > 0 ? covariates : null,
            MinPerGroup = MinPerGroupFrom(opts.GetSingleOrNull("--min-per-group")),
        };
    }

    /// <summary>
    /// Measure acquired and assigned ions from the instrument files, and cache the result.
    /// </summary>
    /// <remarks>
    /// Its own command, and never part of <c>prism run</c>, because of what it costs: the cohort
    /// this was written against is 192 files at about 6 GB each, roughly 1.1 TB, normally over a
    /// network share, against a pipeline that otherwise reads one exported report.
    ///
    /// <para>BOTH halves of the fraction are measured the same way - intensity times ion injection
    /// time in seconds, summed from the same peak arrays - so the ratio is dimensionless.</para>
    /// </remarks>
    private static int CmdIonAccounting(string[] args)
    {
        var opts = ParseOptions(args, multiValue: new HashSet<string>());
        var dir = opts.GetSingle("-d", "--dir") ?? opts.GetSingle("-o", "--output-dir");
        var rawDir = opts.GetSingleOrNull("-r", "--raw-dir");
        var productText = opts.GetSingleOrNull("--product-tolerance");
        var precursorText = opts.GetSingleOrNull("--precursor-tolerance");
        var schemeName = opts.GetSingleOrNull("--scheme");
        var force = Array.Exists(args, a => a is "--force");
        var max = int.TryParse(opts.GetSingleOrNull("--max"), out var m) && m > 0 ? m : 0;
        var lanes = int.TryParse(opts.GetSingleOrNull("--lanes"), out var l) && l > 0 ? l : 0;
        var probeLanes = Array.Exists(args, a => a is "--probe-lanes");

        // The probe measures reading, so it needs no tolerances, no scheme and no output directory
        // content - only files to read. Checking it before the usage block keeps that honest.
        if (probeLanes && rawDir is not null)
        {
            var slice = int.TryParse(opts.GetSingleOrNull("--probe-spectra"), out var ps) && ps > 0
                ? ps
                : LaneProbe.DefaultSliceSpectra;
            return CmdProbeLanes(rawDir, slice);
        }

        if (dir is null || rawDir is null || productText is null)
        {
            Console.Error.WriteLine(
                "Usage: prism ion-accounting -d <output-dir> -r <raw-dir> "
                + "--product-tolerance \"10 ppm\" [--precursor-tolerance \"10 ppm\"] "
                + "[--scheme <name>] [--max N] [--lanes N] [--force]");
            Console.Error.WriteLine(
                "       prism ion-accounting -r <raw-dir> --probe-lanes    "
                + "(how many files this storage is worth reading at once)");
            return 2;
        }
        if (!Directory.Exists(dir))
        {
            Console.Error.WriteLine($"Error: no such output directory: {dir}");
            return 2;
        }
        if (!Directory.Exists(rawDir))
        {
            Console.Error.WriteLine($"Error: no such raw directory: {rawDir}");
            return 2;
        }

        var product = ProductMassTolerance.ParseSetting(productText);
        if (product is null)
        {
            Console.Error.WriteLine(
                $"Error: could not read --product-tolerance \"{productText}\". Write it as the +/- "
                + "tolerance the document states, e.g. \"10 ppm\" or \"0.4 m/z\".");
            return 2;
        }

        ProductMassTolerance? precursor = null;
        if (precursorText is not null)
        {
            precursor = ProductMassTolerance.ParseSetting(precursorText);
            if (precursor is null)
            {
                Console.Error.WriteLine(
                    $"Error: could not read --precursor-tolerance \"{precursorText}\".");
                return 2;
            }
        }

        // Recorded where they are known, for the same reason the isolation windows are: an
        // archived directory has to be able to say what its numbers were extracted with. Non-fatal
        // and additive - a directory with no parameters.json simply gets nothing.
        if (Provenance.RecordExtraction(dir, product, precursor, "command line"))
        {
            Console.WriteLine(
                "Recorded the extraction windows in parameters.json, so the numbers stay "
                + "interpretable once the document has moved on.");
        }

        OptionalReaders.Register(Console.WriteLine);
        if (!IonAccountingReaders.Available)
        {
            Console.Error.WriteLine(
                "Error: this build has no instrument-file reader, so ions cannot be counted. The "
                + "Windows Skyline tool package carries one.");
            return 1;
        }

        var scheme = IsolationSchemeResolver.Resolve(dir, rawDir, Console.WriteLine, schemeName);
        if (scheme is null)
            return 1;

        var result = IonAccountingRun.Compute(
            dir, rawDir, scheme, product, precursor,
            Array.Empty<ProteinList>(), sampleTypes: null, log: Console.WriteLine, force: force,
            maxReplicates: max, lanes: lanes);

        if (result is null)
        {
            Console.Error.WriteLine(
                "No ion accounting was produced; the reason is above. The QC report will continue "
                + "to plot assigned signal without a fraction.");
            return 1;
        }

        Console.WriteLine(
            $"Wrote {Path.Combine(dir, IonAccountingStore.FileName)} and "
            + $"{IonAccountingStore.CyclesFile}. Re-run 'prism qc -d' to plot them.");
        return 0;
    }

    /// <summary>
    /// Measure how many instrument files this storage is worth reading at once, and say so.
    /// </summary>
    /// <remarks>
    /// Separate from the measurement run because the answer belongs to the STORAGE. Baking a number
    /// into the binary means one lab's share decides everybody's default, which is how that constant
    /// came to say 2 and then 4 on no better evidence than which share had been measured last.
    /// </remarks>
    private static int CmdProbeLanes(string rawDir, int sliceSpectra)
    {
        if (!Directory.Exists(rawDir))
        {
            Console.Error.WriteLine($"Error: no such raw directory: {rawDir}");
            return 2;
        }

        OptionalReaders.Register(Console.WriteLine);
        var files = ReplicateDataFiles.Enumerate(rawDir);
        if (files.Count == 0)
        {
            Console.Error.WriteLine($"Error: no instrument data files in {rawDir}.");
            return 2;
        }

        Console.WriteLine(
            $"Probing {rawDir} with the first {sliceSpectra:N0} spectra of up to "
            + $"{Math.Min(8, files.Count):N0} of its {files.Count:N0} file(s).");
        Console.WriteLine(
            "  Each arm re-reads the same files, so later arms benefit from the page cache - the "
            + "comparison is biased slightly IN FAVOUR of more lanes.");

        // No scheme is needed: with no claims there is nothing to place in a window.
        var scheme = new IsolationScheme("probe", Array.Empty<IsolationWindow>());
        var arms = LaneProbe.Run(
            files, scheme, arms: null, sliceSpectra: sliceSpectra, log: Console.WriteLine);
        if (arms.Count == 0)
        {
            Console.Error.WriteLine("Nothing was measured.");
            return 1;
        }

        var recommended = LaneProbe.Recommend(arms);
        var fastest = arms.OrderByDescending(a => a.SpectraPerSecond).First();

        Console.WriteLine();
        Console.WriteLine($"Recommended: --lanes {recommended}");
        if (recommended != fastest.Lanes)
        {
            Console.WriteLine(
                $"  {fastest.Lanes} lanes was faster ({fastest.SpectraPerSecond:N0} vs "
                + $"{arms.First(a => a.Lanes == recommended).SpectraPerSecond:N0} spectra/s) by less "
                + "than 10%, which does not pay for the extra memory: every lane holds another "
                + "file's decode buffers, and running out here does not degrade, it faults.");
        }
        Console.WriteLine();
        Console.WriteLine("  What this measured, and what it did not:");
        Console.WriteLine(
            "  - It RANKS lane counts for this storage. It does not predict runtime: the slice is "
            + "read from the middle of each file, where seeking costs more than streaming, so the "
            + "rates above run about 3x below a whole-file read on the share this was built for.");
        Console.WriteLine(
            "  - The GB column is the probe's own memory, which is NOT the run's. The probe carries "
            + "no claim sets; a real run at 8 lanes reached 32 GB of 64 where the probe showed 6. "
            + "Memory is the reason to stop short of the fastest arm, and this cannot tell you "
            + "about it - budget roughly 4 GB per lane for the run itself.");
        Console.WriteLine(
            "  - It measures THIS directory, now. A share busy with someone else's run answers for "
            + "that load, so re-probe if the answer looks unlike the storage you think you have.");
        return 0;
    }

    private static int CmdCompare(string[] args)
    {
        var opts = ParseOptions(args, multiValue: new HashSet<string>());
        var run1 = opts.GetSingleOrNull("-1", "--run1");
        var run2 = opts.GetSingleOrNull("-2", "--run2");
        var output = opts.GetSingleOrNull("-o", "--output") ?? "rollup_comparison.html";
        var sampleType = (opts.GetSingleOrNull("-s", "--sample-type") ?? "qc").ToLowerInvariant();
        var topN = int.TryParse(opts.GetSingleOrNull("-n", "--top-n"), out var t) && t > 0 ? t : 20;
        if (run1 is null || run2 is null)
        {
            Console.Error.WriteLine("Usage: prism compare -1 <run1-dir> -2 <run2-dir> [-o report.html] "
                + "[-s reference|qc|all] [-n topN]");
            return 2;
        }
        var path = RollupComparison.Generate(run1, run2, output, sampleType, topN);
        Console.WriteLine($"Comparison report written to: {path}");
        return 0;
    }

    private static int CmdConfigTemplate(string[] args)
    {
        var opts = ParseOptions(args, multiValue: new HashSet<string>());
        var outPath = opts.GetSingleOrNull("-o", "--output");
        var minimal = opts.GetSingleOrNull("--minimal") is not null;
        var yaml = minimal ? ConfigTemplate.Minimal() : ConfigTemplate.Default();
        if (outPath is not null)
        {
            File.WriteAllText(outPath, yaml);
            Console.WriteLine($"Configuration template written to: {outPath}");
        }
        else
        {
            Console.WriteLine(yaml);
        }
        return 0;
    }

    private static int PrintVersion()
    {
        Console.WriteLine($"prism {PrismVersion.Current}");
        return 0;
    }

    /// <summary>
    /// Read the acquisition's isolation windows and write them down beside the outputs.
    /// </summary>
    /// <remarks>
    /// <para>The headless half of what the Spectrum density tab does when it opens. Worth its own
    /// command because the windows are needed for a plot and cost one file open, where
    /// <c>ion-accounting</c> - the only other thing that resolves them - reads every file in the
    /// cohort.</para>
    ///
    /// <para><b>Why write them down at all.</b> A DIA analysis document stores
    /// <c>isolation_scheme name="Results only"</c> and no windows: Skyline reads them from the data at
    /// import and does not record them. So once the instrument files are moved off a share - which is
    /// the normal end of an analysis - nothing says what the data was acquired with, and a density map
    /// falls back to a built-in layout that looks exactly as plausible as the right one. Run this
    /// while the files are still reachable.</para>
    /// </remarks>
    private static int CmdIsolationScheme(string[] args)
    {
        var opts = ParseOptions(args, multiValue: new HashSet<string>());
        var dir = opts.GetSingle("-d", "--dir") ?? opts.GetSingle("-o", "--output-dir");
        var rawDir = opts.GetSingleOrNull("-r", "--raw-dir");
        var schemeName = opts.GetSingleOrNull("--scheme");
        var force = Array.Exists(args, a => a is "--force");

        if (dir is null)
        {
            Console.Error.WriteLine(
                "Usage: prism isolation-scheme -d <output-dir> [-r <raw-dir>] [--scheme <name>] "
                + "[--force]");
            return 2;
        }
        if (!Directory.Exists(dir))
        {
            Console.Error.WriteLine($"Error: no such output directory: {dir}");
            return 2;
        }
        if (rawDir is not null && !Directory.Exists(rawDir))
        {
            Console.Error.WriteLine($"Error: no such raw directory: {rawDir}");
            return 2;
        }
        if (force && rawDir is null)
        {
            Console.Error.WriteLine(
                "Error: --force re-reads the windows from the data, so it needs -r <raw-dir>.");
            return 2;
        }

        OptionalReaders.Register(Console.WriteLine);

        var scheme = force
            ? IsolationSchemeResolver.FromData(dir, rawDir!, Console.WriteLine)
            : IsolationSchemeResolver.Resolve(dir, rawDir, Console.WriteLine, schemeName);
        if (scheme is null)
        {
            Console.Error.WriteLine(
                "No isolation scheme could be resolved; the reason is above."
                + (rawDir is null ? " Pass -r <raw-dir> to read it from the data files." : ""));
            return 1;
        }

        // Also true of a scheme that came from the cache: a directory written before provenance
        // carried the windows has them in the XML and not in parameters.json, and this is how that
        // gets put right without re-reading anything.
        var catalog = IsolationSchemeCatalog.Load(
            Path.Combine(dir, IsolationSchemeCatalog.FileName));
        if (catalog is not null && Provenance.RecordIsolationSchemes(dir, catalog))
            Console.WriteLine($"Recorded in {Provenance.FileName}.");

        Console.WriteLine(scheme.Describe());
        return 0;
    }

    private static int PrintUsage()
    {
        Console.WriteLine(UsageText);
        return 0;
    }

    private static int Unknown(string command)
    {
        Console.Error.WriteLine($"error: unrecognized command '{command}'");
        Console.Error.WriteLine();
        Console.Error.WriteLine(UsageText);
        return 2;
    }

    private static string CommandHelp(string command) => command switch
    {
        "run" => RunHelp,
        "merge" => MergeHelp,
        "qc" => QcHelp,
        "differential" => DifferentialHelp,
        "ion-accounting" => IonAccountingHelp,
        "isolation-scheme" => IsolationSchemeHelp,
        "compare" => CompareHelp,
        "config-template" => ConfigTemplateHelp,
        _ => UsageText,
    };

    private const string UsageText = """
        Skyline-PRISM: Proteomics Reference-Integrated Signal Modeling

        Retention-time-aware normalization of transition-level LC-MS proteomics data
        exported from Skyline, with robust protein quantification (Tukey median polish)
        and ComBat batch correction.

        EXAMPLES:
            # Run the full pipeline
            prism run -i report.csv -o output/ -c config.yaml

            # Merge several Skyline reports into one parquet
            prism merge plate1.csv plate2.csv -o data.parquet

            # Test disease against control at protein level
            prism differential -d output/ --group-by condition -a Control -b Disease

            # Emit an annotated configuration template
            prism config-template -o config.yaml

        Usage: prism <command> [options]

        Commands:
            run                Run the full PRISM pipeline (rollup, normalize, batch-correct, QC)
            merge              Merge Skyline transition reports into one parquet
            qc                 (Re)generate the QC report from an existing output directory
            differential       Two-group differential abundance on a finished output directory
            ion-accounting     Count acquired ions and the fraction assigned to a peptide
            isolation-scheme   Read the acquisition's DIA isolation windows and record them
            compare            Compare control-sample CVs between two runs
            config-template    Emit an annotated configuration template
            version            Print the version

        Run 'prism <command> --help' for the options of a specific command.
        """;

    private const string RunHelp = """
        prism run - Run the full PRISM pipeline

        Rolls transitions up to peptides, normalizes, applies ComBat batch correction,
        performs protein parsimony and rollup, and writes the corrected peptide/protein
        matrices (linear) plus a QC report. Reads transition-level Skyline reports.

        Usage: prism run -i <input...> -o <output-dir> [options]

        Options:
            -i, --input <FILE...>         Skyline transition report(s), CSV/TSV/parquet; repeatable
            -o, --output-dir <DIR>        Output directory
            -c, --config <FILE>           YAML configuration (see 'prism config-template')
            -m, --metadata <FILE...>      Replicate metadata / Replicates report(s), merged; repeatable
                --from-provenance <FILE>  Re-run with the settings from a prior parameters.json
                --force-reprocess         Ignore the merge cache and re-read the inputs
            -h, --help                    Show this help

        EXAMPLES:
            prism run -i report.csv -o out/ -c config.yaml
            prism run -i plate1.csv plate2.csv -o out/ -m replicates.csv
            prism run -i new.csv -o out2/ --from-provenance out/parameters.json
        """;

    private const string MergeHelp = """
        prism merge - Merge Skyline transition reports into one parquet

        Streams and concatenates several Skyline transition reports (CSV/TSV/parquet)
        into a single sorted parquet - the same merge step 'prism run' performs.

        Usage: prism merge <input...> -o <output.parquet>

        Options:
            -o, --output <FILE>    Merged parquet path
            -h, --help             Show this help

        EXAMPLES:
            prism merge plate1.csv plate2.csv -o merged.parquet
        """;

    private const string QcHelp = """
        prism qc - (Re)generate the QC report

        Rebuilds qc_report.html (and the plot PNGs) from the parquet outputs already in
        an output directory, without re-running the pipeline. The processing settings it
        reports are read from that directory's parameters.json.

        Usage: prism qc -d <output-dir> [options]

        Options:
            -d, --dir <DIR>       Output directory from a prior 'prism run'
            -c, --config <FILE>   YAML configuration (optional; QC report settings only)
                --no-save-plots   Embed the plots only; do not write qc_plots/*.png
            -h, --help            Show this help
        """;

    private const string DifferentialHelp = """
        prism differential - Two-group differential abundance on a finished output directory

        Tests one contrast against the corrected peptide or protein matrix a `prism run`
        already produced, and writes a results table. Reads the output directory and
        writes nothing back into it except the results file (and, with --report, the
        quant/ folder), so it cannot disturb the run.

        This is the same engine, and the same statistical menu, as the Skyline tool's
        Differential pane; the arms are resolved by the same code, so a contrast set up
        in the pane and one typed here mean the same samples.

        Usage: prism differential -d <output-dir> --group-by <column> -a <level...> -b <level...>

        Required:
            -d, --dir DIR          Output directory from `prism run`
            -g, --group-by COL     Metadata column defining the groups
            -a, --group-a LEVEL... Level(s) forming arm A (the reference arm)
            -b, --group-b LEVEL... Level(s) forming arm B (the treatment arm)

        Each arm takes several levels - repeat the flag, list them space-separated, or
        comma-separate them - and the arm is their union. A level cannot be in both arms.
        A positive log2FC means higher in B.

        A TREND design has no arms: it fits a slope against --trend-over instead, and
        reports the modeled change across that column's observed range rather than the
        raw slope, so --min-log2fc means the same thing as it does on a two-arm contrast
        whatever units the column is in. Use trend-within-subject whenever the same
        subjects are followed across that column - treating one subject's repeated
        samples as independent understates the standard error.

        Options:
            --clinical CSV...      Join an external clinical table to the samples first, exactly as
                                   the pane's Clinical CSV input does (key column detected by value);
                                   its columns can then be named by --group-by, --adjust-for,
                                   --subject, --trend-over and --markers-group-by. Repeatable: several
                                   tables are joined in the order given
            --level LEVEL          protein (default) or peptide
            --design DESIGN        unpaired (default), paired, trend, trend-within-subject
            --subject COL          Metadata column identifying the subject (alias: --pair-by).
                                   Required by --design paired, which matches each subject's two
                                   samples, and by --design trend-within-subject, which gives each
                                   subject its own level
            --trend-over AXIS      The column to fit a slope against; required by, and only valid
                                   with, a trend design. A column of plain numbers is named directly
                                   ("Week"). A column whose values EMBED a number is named for the
                                   number it takes - "Longitudinal Draw Description (Week)" reads 8
                                   out of "V2_Week 8", "(V)" reads 2 - because those are different
                                   quantities and only one is the axis. Pass an unknown name to see
                                   the list.
            --restrict-to COL=V[,V...]
                                   Fit the trend on only the samples whose COL is one of these
                                   values; repeatable. Without it a trend pools every sample that
                                   has a value on the axis, which merges two studies that share a
                                   timepoint scale into one slope. Trend designs only - on a two-arm
                                   contrast, -a and -b already choose the samples.
            --test TEST            moderated (default), welch, student, paired-t,
                                   wilcoxon, mann-whitney
            --prior PRIOR          Variance prior for the moderated t: intensity-trend
                                   (default, matches the lab's proteomics-toolkit),
                                   global, limma-trend, peptide-count
            --prior-from-controls  Fit the variance prior on the QC and reference replicates.
                                   This is the DEFAULT whenever the run has two or more replicates
                                   of a control type; passing it makes that explicit and turns a
                                   missing control set into an error rather than a fallback
            --prior-from-groups    Fit the variance prior on the contrast groups instead. Their
                                   spread includes the biological variation the analysis is looking
                                   for, so the prior describes measurement noise plus that biology
                                   and shrinks genuine effects toward nothing - use only to
                                   reproduce an older result
            --adjust-for COL...    Covariates to adjust the contrast for (moderated only)
            --correction METHOD    bh (default), by, holm, bonferroni, none
            --alpha A              p-value threshold for the printed hit count (default 0.05)
            --raw-p                Apply --alpha to the RAW p rather than the adjusted one. For
                                   judging a pilot too small for anything to survive correction;
                                   the printed rule always says which p was used
            --min-log2fc X         Minimum |log2 fold change| for a hit (default 1, i.e. two-fold).
                                   0 turns the effect-size filter off and lets p alone decide
            --min-per-group N      Minimum samples per arm (default 2)
            -o, --output FILE      Results CSV (default <output-dir>/differential.csv)

        Quant report:
            --report               Also write the quant report - quant_report.html and its CSVs,
                                   under <output-dir>/quant/ - exactly as the Differential pane's
                                   Quant report button does: this contrast's volcano and hits, the
                                   detection test the design calls for (Fisher; McNemar when paired;
                                   the Firth GLM when adjusted), g:Profiler enrichment, and any
                                   marker panels. A view that cannot run is skipped and printed with
                                   the reason; it never stops the report
            --markers PANEL...     Marker panels to include: your saved lists and the shipped ones,
                                   e.g. "EV markers (core)". Repeat, space- or comma-separate
            --markers-group-by COL Column to group the panels by (default: --group-by)
            --no-enrichment        Skip g:Profiler, for machines with no internet access

        EXAMPLES:
            # Disease against control, protein level, default moderated t
            prism differential -d output/ --group-by condition -a Control -b Disease

            # Two severity levels pooled into one arm, adjusted for sex and age
            prism differential -d output/ --group-by stage -a Control Mild -b Severe \
                --adjust-for sex,age

            # A within-subject design: each subject's pre and post sample
            prism differential -d output/ --group-by timepoint -a Pre -b Post \
                --design paired --pair-by subject --test paired-t

            # A dose-response, one sample per subject
            prism differential -d output/ --design trend --trend-over dose_mg

            # A time course following the same subjects
            prism differential -d output/ --design trend-within-subject \
                --trend-over week --subject patient_id

            # Peptide level, where BY is the correction to reach for
            prism differential -d output/ --level peptide --group-by condition \
                -a Control -b Disease --correction by

            # The full quant report for a contrast, with the shipped EV marker panel
            prism differential -d output/ --group-by condition -a Control -b Disease \
                --report --markers "EV markers (core)"
        """;

    private const string IonAccountingHelp = """
        prism ion-accounting - Count the ions acquired, and the fraction assigned to a peptide

        Usage: prism ion-accounting -d <output-dir> -r <raw-dir> --product-tolerance "10 ppm" [options]

        Reads each replicate's instrument file once and reports, at MS1 and at MS2, how many
        ions reached the detector and what fraction of them fall inside a region some peptide
        of this analysis claims. Both halves are measured the same way - a scan's intensity
        times its ion injection time - so the ratio is a genuine fraction.

        Options:
          -d, --dir <dir>                 Output directory of a finished run (required)
          -r, --raw-dir <dir>             Directory holding the instrument files (required)
              --product-tolerance <tol>   The document's product extraction window, as the +/-
                                          tolerance it states: "10 ppm" or "0.4 m/z" (required)
              --precursor-tolerance <tol> The document's precursor extraction window. Omitted,
                                          only the MS2 half is computed - a guessed tolerance
                                          would change the number with nothing to show it
              --scheme <name>             Which scheme from isolation_schemes.xml; required
                                          only when the file holds more than one
              --force                     Recompute even when the cache matches
              --max <n>                   Stop after n replicates, for a first look at a cohort
                                          whose files are a terabyte on a network share
              --lanes <n>                 Files to read at a time (default 2). Reading is 99.5%
                                          of the cost, but it is limited by the storage rather
                                          than the CPU: four lanes measured slightly WORSE than
                                          two on an SMB share, at twice the memory. Raise it only
                                          if you measure a gain

        Writes ion_accounting.parquet and ion_cycles.parquet into the output directory. The
        cache is keyed on both tolerances, the isolation scheme, the selected protein lists
        and a fingerprint of the instrument files, so a re-run with different settings
        recomputes rather than replotting the previous numbers under a new caption.

        Not part of `prism run`: a cohort is often a terabyte of instrument files on a
        network share, and everything else in the pipeline reads one exported report.
        """;

    private const string IsolationSchemeHelp = """
        prism isolation-scheme - Read the acquisition's DIA isolation windows and record them

        A DIA analysis document does not store its isolation windows: Skyline reads them from
        the data files at import and records only 'Results only'. The windows therefore live in
        the instrument files alone, and once those are moved or deleted nothing can say what the
        data was acquired with - the Spectrum density map then bins on a built-in layout that
        looks exactly as plausible as the right one.

        This reads them from one data file (it costs a file open; the windows are scan headers in
        the first two acquisition cycles) and writes them beside the outputs, in both
        isolation_schemes.xml and parameters.json. Run it while the data files are reachable.

        With no -r it only reports and records what the directory already knows.

        Usage: prism isolation-scheme -d <output-dir> [-r <raw-dir>] [options]

        Options:
            -d, --dir <dir>      Output directory of a PRISM run (required)
            -r, --raw-dir <dir>  Where the instrument files are, if the windows must be read
                --scheme <name>  Pick this scheme when several are cached
                --force          Re-read from the data even if a scheme is already cached

        Examples:
            prism isolation-scheme -d output/ -r /data/raw/
            prism isolation-scheme -d output/            # report what is already recorded
        """;

    private const string CompareHelp = """
        prism compare - Compare control-sample CVs between two runs

        Compares corrected_peptides from two run directories: reports the median
        control-sample CV per run and the peptides that improved or worsened most.

        Usage: prism compare -1 <run1-dir> -2 <run2-dir> [options]

        Options:
            -1, --run1 <DIR>          First run's output directory
            -2, --run2 <DIR>          Second run's output directory
            -o, --output <FILE>       Comparison report HTML (default: rollup_comparison.html)
            -s, --sample-type <TYPE>  Samples to compare: reference | qc | all (default: qc)
            -n, --top-n <N>           Peptides to list per direction (default: 20)
            -h, --help                Show this help

        EXAMPLES:
            prism compare -1 run_a/ -2 run_b/ -o compare.html -s qc
        """;

    private const string ConfigTemplateHelp = """
        prism config-template - Emit an annotated configuration template

        Writes a commented YAML configuration listing every option and its default.
        Use --minimal for just the common knobs.

        Usage: prism config-template [-o <file>] [--minimal]

        Options:
            -o, --output <FILE>   Write to a file (default: stdout)
                --minimal         Emit only the common options
            -h, --help            Show this help
        """;

    // Minimal option parser: -flag value (repeatable for multiValue flags); leftover tokens
    // are positional.
    private static ParsedOptions ParseOptions(string[] args, HashSet<string> multiValue, bool positional = false)
    {
        var single = new Dictionary<string, string>(StringComparer.Ordinal);
        var lists = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        var pos = new List<string>();

        var i = 0;
        while (i < args.Length)
        {
            var tok = args[i];
            if (tok.StartsWith('-'))
            {
                if (multiValue.Contains(tok))
                {
                    var list = lists.TryGetValue(tok, out var l) ? l : lists[tok] = new List<string>();
                    i++;
                    while (i < args.Length && !args[i].StartsWith('-'))
                        list.Add(args[i++]);
                }
                else if (i + 1 < args.Length && !args[i + 1].StartsWith('-'))
                {
                    single[tok] = args[i + 1];
                    i += 2;
                }
                else
                {
                    single[tok] = "true";
                    i++;
                }
            }
            else
            {
                pos.Add(tok);
                i++;
            }
        }
        return new ParsedOptions(single, lists, pos);
    }

    private sealed record ParsedOptions(
        Dictionary<string, string> Single,
        Dictionary<string, List<string>> Lists,
        List<string> Positional)
    {
        public string? GetSingleOrNull(params string[] keys)
        {
            foreach (var k in keys)
                if (Single.TryGetValue(k, out var v))
                    return v;
            return null;
        }

        public string? GetSingle(params string[] keys) => GetSingleOrNull(keys);

        public List<string> GetList(params string[] keys)
        {
            foreach (var k in keys)
                if (Lists.TryGetValue(k, out var v))
                    return v;
            return new List<string>();
        }
    }
}
