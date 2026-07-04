using System;
using System.Collections.Generic;
using System.IO;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Core.Qc;

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
            config = Provenance.LoadConfig(provenancePath);
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
        var result = DuckDbMerge.MergeAndSort(inputs, output);
        Console.WriteLine($"Merged {inputs.Count} file(s) -> {result.OutputPath} ({result.TotalRows} rows)");
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
        var config = configPath is not null
            ? PrismConfig.LoadValidated(configPath, w => Console.Error.WriteLine($"WARNING: {w}"))
            : new PrismConfig();
        if (opts.GetSingleOrNull("--no-save-plots") is not null)
            config.QcReport.SavePlots = false;
        var path = QcReport.Generate(dir, config, savePlots: config.QcReport.SavePlots);
        Console.WriteLine($"QC report written to: {path}");
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
        var version = typeof(Program).Assembly.GetName().Version?.ToString() ?? "unknown";
        Console.WriteLine($"prism {version}");
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

            # Emit an annotated configuration template
            prism config-template -o config.yaml

        Usage: prism <command> [options]

        Commands:
            run                Run the full PRISM pipeline (rollup, normalize, batch-correct, QC)
            merge              Merge Skyline transition reports into one parquet
            qc                 (Re)generate the QC report from an existing output directory
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
        an output directory, without re-running the pipeline.

        Usage: prism qc -d <output-dir> [options]

        Options:
            -d, --dir <DIR>       Output directory from a prior 'prism run'
            -c, --config <FILE>   YAML configuration (optional; QC settings only)
            -h, --help            Show this help
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
