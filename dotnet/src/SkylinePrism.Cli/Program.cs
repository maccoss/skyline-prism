using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
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
                "ms2-signal" => CmdMs2Signal(rest),
                "ion-accounting" => CmdIonAccounting(rest),
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
    /// Read the instrument data files for a finished run and record how much MS2 each acquired.
    /// </summary>
    /// <remarks>
    /// Its own command, and never part of <c>prism run</c>, because of what it costs: the cohort this
    /// was written against is 192 files at ~6 GB each - about 1.1 TB, normally over a network share -
    /// against a pipeline that otherwise reads one exported report. The accounting is perfectly
    /// useful without it; this is what turns "how much signal was assigned" into "what fraction of
    /// what the instrument acquired was assigned".
    /// </remarks>
    private static int CmdMs2Signal(string[] args)
    {
        var opts = ParseOptions(args, multiValue: new HashSet<string>());
        var dir = opts.GetSingle("-d", "--dir") ?? opts.GetSingle("-o", "--output-dir");
        var rawDir = opts.GetSingleOrNull("-r", "--raw-dir");
        if (dir is null || rawDir is null)
        {
            Console.Error.WriteLine(
                "Usage: prism ms2-signal -d <output-dir> -r <raw-dir> [--lanes N] [--max N]");
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

        var lanes = int.TryParse(opts.GetSingleOrNull("--lanes"), out var l)
            ? l
            : Ms2SignalReaders.DefaultLanes;
        var max = int.TryParse(opts.GetSingleOrNull("--max"), out var m) ? m : 0;

        if (Ms2SignalReaders.All.Count == 0)
        {
            // Said plainly rather than reported as "0 files readable": this build simply has no
            // reader, which is the normal state of the cross-platform CLI, and no raw directory can
            // fix it.
            Console.Error.WriteLine(
                "Error: this build has no instrument-file reader, so acquired MS2 signal cannot be "
                + "read. The Windows Skyline tool package carries one.");
            return 1;
        }

        // The replicates to resolve come from the accounting itself, so the two halves of the
        // fraction are keyed the same way and in the same order.
        var accounting = Ms2SignalAccounting.ReadCached(dir);
        if (accounting is null || accounting.IsEmpty)
        {
            Console.Error.WriteLine(
                $"Error: no {Ms2SignalAccounting.AccountingFile} in {dir}. Run the MS2 signal "
                + "accounting first (qc_report.ms2_signal.enabled), then this.");
            return 1;
        }

        var result = Ms2AcquiredSignal.Populate(
            dir, rawDir, accounting.Rows.Select(r => r.Sample),
            Console.WriteLine, lanes, max);

        if (!result.AnyUsable)
        {
            Console.Error.WriteLine(
                "No data file could be read, so no denominator was recorded. The QC report will "
                + "continue to plot assigned signal without a fraction.");
            return 1;
        }

        var joined = accounting.WithAcquired(Ms2AcquiredSignal.ReadTotals(dir));
        Console.WriteLine(
            $"Median assigned/acquired over {result.Usable:N0} replicate(s): "
            + $"{joined.MedianAcquiredFraction():P1}");
        Console.WriteLine(
            $"Wrote {Path.Combine(dir, Ms2AcquiredSignal.FileName)}. "
            + "Re-run 'prism qc -d' to put the acquired bar on the plot.");
        return 0;
    }

    /// <summary>
    /// Measure acquired and assigned ions from the instrument files, and cache the result.
    /// </summary>
    /// <remarks>
    /// Its own command, and never part of <c>prism run</c>, for the same reason as
    /// <c>ms2-signal</c>: the cohort this was written against is 192 files at about 6 GB each,
    /// roughly 1.1 TB, normally over a network share. What it buys over <c>ms2-signal</c> is that
    /// BOTH halves of the fraction are measured the same way - intensity times ion injection time,
    /// summed from the same peak arrays - so the ratio is dimensionless. The earlier command divided
    /// an intensity-time integral by an intensity, which on a real file is wrong by the mean
    /// injection time, measured at 7.0x.
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

        if (dir is null || rawDir is null || productText is null)
        {
            Console.Error.WriteLine(
                "Usage: prism ion-accounting -d <output-dir> -r <raw-dir> "
                + "--product-tolerance \"10 ppm\" [--precursor-tolerance \"10 ppm\"] "
                + "[--scheme <name>] [--max N] [--force]");
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

        OptionalReaders.Register(Console.WriteLine);
        if (!IonAccountingReaders.Available)
        {
            Console.Error.WriteLine(
                "Error: this build has no instrument-file reader, so ions cannot be counted. The "
                + "Windows Skyline tool package carries one.");
            return 1;
        }

        var scheme = ResolveScheme(dir, schemeName)
            ?? ImportSchemeFromData(dir, rawDir);
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
    /// The isolation scheme to account against, from the output directory's own
    /// <c>isolation_schemes.xml</c>. Never guessed: fragments in different isolation windows never
    /// share signal, so the wrong scheme silently changes every number.
    /// </summary>
    private static IsolationScheme? ResolveScheme(string dir, string? named)
    {
        var path = Path.Combine(dir, IsolationSchemeCatalog.FileName);
        var catalog = IsolationSchemeCatalog.Load(path);
        var usable = catalog?.UsableSchemes
            ?? (IReadOnlyList<IsolationScheme>)Array.Empty<IsolationScheme>();

        // No windows is the NORMAL state of a DIA analysis document: it stores
        // <isolation_scheme name="Results only" /> and Skyline keeps the windows in the data files.
        // Saying nothing here lets the caller import them from a file instead of failing.
        if (usable.Count == 0)
            return null;

        if (!string.IsNullOrWhiteSpace(named))
        {
            foreach (var candidate in usable)
            {
                if (string.Equals(candidate.Name, named, StringComparison.OrdinalIgnoreCase))
                    return candidate;
            }
            Console.Error.WriteLine(
                $"Error: no isolation scheme named '{named}'. Available: "
                + string.Join(", ", usable.Select(s => s.Name)));
            return null;
        }

        if (usable.Count > 1)
        {
            Console.Error.WriteLine(
                "Error: more than one isolation scheme is available, so --scheme must name one: "
                + string.Join(", ", usable.Select(s => s.Name)));
            return null;
        }

        Console.WriteLine($"Isolation scheme: {usable[0].Describe()}");
        return usable[0];
    }

    /// <summary>
    /// Read the isolation windows out of the first data file, for the usual case where the document
    /// does not carry them.
    /// </summary>
    /// <remarks>
    /// Uses the acquired-only read, which is headers only - the windows are a property of the
    /// ACQUISITION METHOD, so one file describes every replicate of the cohort and there is no
    /// reason to decode a peak to find them. The result is written to isolation_schemes.xml so the
    /// next run, and the QC report, both reuse it.
    /// </remarks>
    private static IsolationScheme? ImportSchemeFromData(string dir, string rawDir)
    {
        var files = ReplicateDataFiles.Enumerate(rawDir);
        if (files.Count == 0)
        {
            Console.Error.WriteLine($"Error: no instrument data files in {rawDir}.");
            return null;
        }

        var first = files[0];
        Console.WriteLine(
            "No isolation scheme with windows was cached, which is normal for a DIA analysis "
            + $"document. Reading the windows from {Path.GetFileName(first)}.");

        var record = Ms2SignalReaders.Read(first, Console.WriteLine);
        if (record.IsolationWindows.Count == 0)
        {
            Console.Error.WriteLine(
                "Error: that file reported no repeating isolation windows, so there is no scheme to "
                + "account against. A DDA acquisition has one window per spectrum and is not "
                + "supported here.");
            return null;
        }

        var name = $"Imported from {Path.GetFileNameWithoutExtension(first)}";
        var scheme = new IsolationScheme(name, record.IsolationWindows);
        Console.WriteLine($"Isolation scheme: {scheme.Describe()}");

        try
        {
            var path = Path.Combine(dir, IsolationSchemeCatalog.FileName);
            var catalog = IsolationSchemeCatalog.Load(path) ?? new IsolationSchemeCatalog();
            catalog.AddDocumentScheme(name, scheme);
            catalog.Save(path);
            Console.WriteLine($"Cached it in {IsolationSchemeCatalog.FileName}.");
        }
        catch (IOException ex)
        {
            // Not fatal: the scheme is in hand, and re-reading one file next time costs seconds.
            Console.WriteLine($"Could not cache the scheme: {ex.Message}");
        }

        return scheme;
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
        "ms2-signal" => Ms2SignalHelp,
        "ion-accounting" => IonAccountingHelp,
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

    private const string Ms2SignalHelp = """
        prism ms2-signal - Read acquired MS2 signal from the instrument data files

        Records how much MS2 each replicate's data file actually acquired, which is the
        denominator the MS2 signal accounting cannot get from any Skyline export. Writes
        ms2_signal.parquet into the output directory; re-run 'prism qc -d' afterwards and
        the accounting plot gains its acquired bar and reports assigned/acquired.

        Separate from 'prism run' on purpose: a cohort is hundreds of gigabytes of raw
        data, usually over a network share, where the rest of the pipeline reads one
        exported report. Run it when you want the fraction, not on every analysis.

        Replicates are matched to files on the file stem - an exact match first, then the
        longest stem ending in the replicate name, which is what handles the prefixes
        acquisition software adds.

        Usage: prism ms2-signal -d <output-dir> -r <raw-dir> [options]

        Options:
            -d, --dir <DIR>       Output directory from a prior 'prism run'
            -r, --raw-dir <DIR>   Directory holding that cohort's instrument data files
                --lanes <N>       Concurrent reads (default 8; measured to plateau there)
                --max <N>         Stop after N files - a spot check rather than the cohort
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
              --lanes <n>                 Files to read at a time (default 8). Reading is 99.5%
                                          of the cost and it is per file, so this is the one
                                          setting that changes how long a cohort takes

        Writes ion_accounting.parquet and ion_cycles.parquet into the output directory. The
        cache is keyed on both tolerances, the isolation scheme, the selected protein lists
        and a fingerprint of the instrument files, so a re-run with different settings
        recomputes rather than replotting the previous numbers under a new caption.

        Not part of `prism run`: a cohort is often a terabyte of instrument files on a
        network share, and everything else in the pipeline reads one exported report.
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
