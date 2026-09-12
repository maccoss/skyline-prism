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

            # Emit an annotated configuration template
            prism config-template -o config.yaml

        Usage: prism <command> [options]

        Commands:
            run                Run the full PRISM pipeline (rollup, normalize, batch-correct, QC)
            merge              Merge Skyline transition reports into one parquet
            qc                 (Re)generate the QC report from an existing output directory
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
