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
        if (args.Length == 0)
            return PrintUsage();

        try
        {
            return args[0] switch
            {
                "run" => CmdRun(args[1..]),
                "merge" => CmdMerge(args[1..]),
                "qc" => CmdQc(args[1..]),
                "compare" => CmdCompare(args[1..]),
                "config-template" => CmdConfigTemplate(args[1..]),
                "--version" or "-v" or "version" => PrintVersion(),
                "--help" or "-h" or "help" => PrintUsage(),
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
        }
        else
        {
            config = configPath is not null ? PrismConfig.Load(configPath) : new PrismConfig();
        }

        if (inputs.Count == 0 || outputDir is null)
        {
            Console.Error.WriteLine(
                "Usage: prism run -i <input...> -o <output-dir> [-c <config.yaml>] [-m <metadata...>] "
                + "[--from-provenance <parameters.json>]");
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
        var config = configPath is not null ? PrismConfig.Load(configPath) : new PrismConfig();
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
        Console.WriteLine($"prism (Skyline-PRISM C#) {version}");
        return 0;
    }

    private static int PrintUsage()
    {
        Console.WriteLine("Skyline-PRISM CLI");
        Console.WriteLine();
        Console.WriteLine("Usage: prism <command> [options]");
        Console.WriteLine();
        Console.WriteLine("Commands:");
        Console.WriteLine("  run -i <in...> -o <dir> [-c config] [-m metadata...] [--from-provenance p.json] [--force-reprocess]");
        Console.WriteLine("                                           Run the full PRISM pipeline");
        Console.WriteLine("  merge <input...> -o <out.parquet>        Merge Skyline reports");
        Console.WriteLine("  qc -d <output-dir> [-c config]           (Re)generate the QC report");
        Console.WriteLine("  compare -1 <run1> -2 <run2> [-o rpt.html] [-s qc] [-n 20]  Compare two runs' CVs");
        Console.WriteLine("  config-template [-o file] [--minimal]    Emit a configuration template");
        Console.WriteLine("  version                                  Print the version");
        return 0;
    }

    private static int Unknown(string command)
    {
        Console.Error.WriteLine($"Unknown command '{command}'. Run 'prism help'.");
        return 2;
    }

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
