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
        var opts = ParseOptions(args, multiValue: new HashSet<string> { "-i", "--input" });
        var inputs = new List<string>(opts.GetList("-i", "--input"));
        var outputDir = opts.GetSingle("-o", "--output-dir");
        var configPath = opts.GetSingleOrNull("-c", "--config");
        var provenancePath = opts.GetSingleOrNull("--from-provenance");

        PrismConfig config;
        if (provenancePath is not null)
        {
            config = Provenance.LoadConfig(provenancePath);
            if (inputs.Count == 0)
                foreach (var s in Provenance.SourceFiles(provenancePath))
                    inputs.Add(s);
            Console.WriteLine($"PRISM: loaded settings from provenance {provenancePath}");
        }
        else
        {
            config = configPath is not null ? PrismConfig.Load(configPath) : new PrismConfig();
        }

        if (inputs.Count == 0 || outputDir is null)
        {
            Console.Error.WriteLine(
                "Usage: prism run -i <input...> -o <output-dir> [-c <config.yaml>] [--from-provenance <parameters.json>]");
            return 2;
        }

        Console.WriteLine($"PRISM: merging {inputs.Count} input(s) -> {outputDir}");
        var result = PrismPipeline.Run(inputs, outputDir, config, log: Console.WriteLine);
        Console.WriteLine(
            $"Done: {result.NPeptides} peptides, {result.NProteins} proteins, {result.NSamples} samples, "
            + $"{result.Batches.Count} batch(es). Outputs in {outputDir}");
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

    private static int CmdConfigTemplate(string[] args)
    {
        var opts = ParseOptions(args, multiValue: new HashSet<string>());
        var outPath = opts.GetSingleOrNull("-o", "--output");
        var yaml = ConfigTemplate.Default();
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
        Console.WriteLine("  run -i <input...> -o <dir> [-c config]   Run the full PRISM pipeline");
        Console.WriteLine("  merge <input...> -o <out.parquet>        Merge Skyline reports");
        Console.WriteLine("  config-template [-o file]                Emit a configuration template");
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
