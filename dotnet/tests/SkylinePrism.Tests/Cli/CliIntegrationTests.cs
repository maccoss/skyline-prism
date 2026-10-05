using SkylinePrism.Core.IO;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using SkylinePrism.Cli;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;

namespace SkylinePrism.Tests.Cli;

/// <summary>
/// End-to-end CLI coverage: drives Program.Main in-process against the mini fixture and asserts exit
/// codes + output files. Covers Program.cs (arg parsing + dispatch), the run/merge/qc/compare/
/// config-template commands, and RollupComparison - none of which the unit suite otherwise touches.
/// </summary>
[Collection("cli")]
public class CliIntegrationTests
{
    private static readonly object ConsoleLock = new();

    private static (int Code, string Output) Invoke(params string[] args)
    {
        lock (ConsoleLock)
        {
            var origOut = Console.Out;
            var origErr = Console.Error;
            using var sw = new StringWriter();
            Console.SetOut(sw);
            Console.SetError(sw);
            try
            {
                return (Program.Main(args), sw.ToString());
            }
            finally
            {
                Console.SetOut(origOut);
                Console.SetError(origErr);
            }
        }
    }

    private static string Fixture(params string[] parts)
        => Path.Combine(new[] { AppContext.BaseDirectory, "fixtures" }.Concat(parts).ToArray());

    private static readonly string Input1 = Fixture("mini", "merge", "mini_plate1.csv");
    private static readonly string Input2 = Fixture("mini", "merge", "mini_plate2.csv");
    private static readonly string Config = Fixture("mini", "e2e-sum", "config.yaml");

    private static string TempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "prism_cli_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    private static void Cleanup(string dir)
    {
        try { Directory.Delete(dir, recursive: true); }
        catch (IOException) { /* best-effort */ }
    }

    private static int Run(string outDir)
        => Invoke("run", "-i", Input1, Input2, "-o", outDir, "-c", Config).Code;

    [Fact]
    public void Run_ProducesExpectedOutputs()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));
            // qc_report.html is gated by qc_report.enabled (off in this config); the qc command
            // test covers report generation.
            foreach (var f in new[]
            {
                "corrected_peptides.parquet", "corrected_proteins.parquet",
                "peptides_log2_internal.parquet", "proteins_raw.parquet",
                "protein_groups.csv", "parameters.json", "sample_metadata.csv",
            })
                Assert.True(File.Exists(Path.Combine(outDir, f)), $"missing output: {f}");
            Assert.Contains(Directory.GetFiles(outDir), p => Path.GetFileName(p).StartsWith("prism_run_"));
        }
        finally { Cleanup(outDir); }
    }

    [Fact]
    public void Merge_ProducesParquet()
    {
        var outDir = TempDir();
        // A ".parquet" -o is still accepted (it was the contract before the merge partitioned), but the
        // data lands in a directory of that name without the extension.
        var requested = Path.Combine(outDir, "merged.parquet");
        var actual = Path.Combine(outDir, "merged");
        try
        {
            var (code, output) = Invoke("merge", Input1, Input2, "-o", requested);
            Assert.Equal(0, code);
            Assert.False(File.Exists(requested));
            Assert.True(MergedDataset.Exists(actual), "merge produced no dataset");
            Assert.NotEmpty(MergedDataset.Open(actual).Partitions);
            Assert.Contains("rows", output);
        }
        finally { Cleanup(outDir); }
    }

    [Fact]
    public void Compare_ProducesReport()
    {
        var a = TempDir();
        var b = TempDir();
        var report = Path.Combine(b, "compare.html");
        try
        {
            Assert.Equal(0, Run(a));
            Assert.Equal(0, Run(b));
            var (code, _) = Invoke("compare", "-1", a, "-2", b, "-o", report, "-s", "all", "-n", "5");
            Assert.Equal(0, code);
            Assert.True(File.Exists(report));
            Assert.Contains("Rollup Comparison", File.ReadAllText(report));
        }
        finally { Cleanup(a); Cleanup(b); }
    }

    [Fact]
    public void Qc_RegeneratesReport()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));
            var html = Path.Combine(outDir, "qc_report.html");
            File.Delete(html);
            Assert.Equal(0, Invoke("qc", "-d", outDir).Code);
            Assert.True(File.Exists(html));
        }
        finally { Cleanup(outDir); }
    }

    [Fact]
    public void ConfigTemplate_FullAndMinimal_WriteFiles()
    {
        var outDir = TempDir();
        try
        {
            var full = Path.Combine(outDir, "full.yaml");
            var min = Path.Combine(outDir, "min.yaml");
            Assert.Equal(0, Invoke("config-template", "-o", full).Code);
            Assert.Equal(0, Invoke("config-template", "--minimal", "-o", min).Code);
            Assert.Contains("transition_rollup", File.ReadAllText(full));
            // Minimal is a strict subset of the full template.
            Assert.True(new FileInfo(min).Length < new FileInfo(full).Length);
        }
        finally { Cleanup(outDir); }
    }

    [Fact]
    public void Version_PrintsVersion()
    {
        var (code, output) = Invoke("--version");
        Assert.Equal(0, code);
        Assert.Contains("prism", output);
    }

    [Theory]
    [InlineData(new[] { "bogus" }, 2)]                       // unknown command
    [InlineData(new[] { "run", "-o", "x" }, 2)]              // run without -i
    [InlineData(new[] { "merge", "-o", "x.parquet" }, 2)]    // merge without inputs
    [InlineData(new[] { "qc" }, 2)]                          // qc without -d
    [InlineData(new[] { "compare", "-1", "a" }, 2)]          // compare without -2
    public void InvalidInvocations_ReturnUsageCode(string[] args, int expected)
        => Assert.Equal(expected, Invoke(args).Code);

    [Fact]
    public void NoArgs_PrintsUsage()
    {
        var (code, output) = Invoke();
        Assert.Equal(0, code);
        Assert.Contains("Usage", output);
    }

    [Theory]
    [InlineData("run")]
    [InlineData("merge")]
    [InlineData("qc")]
    [InlineData("compare")]
    [InlineData("config-template")]
    public void CommandHelp_ReturnsZeroWithCommandUsage(string cmd)
    {
        var (code, output) = Invoke(cmd, "--help");
        Assert.Equal(0, code);
        Assert.Contains($"Usage: prism {cmd}", output);
    }

    /// <summary>
    /// An input edited in place is caught, even though not one setting moved.
    /// </summary>
    /// <remarks>
    /// This is the direction a config comparison cannot see at all, and it is the one that matters
    /// most: the YAML is identical, so comparing it says "nothing will change" while the merge and
    /// everything below it is about to be recomputed from different data. The merge is checked the
    /// way the pipeline checks it - against its own sidecar beside <c>merged_data</c>, which stamps
    /// each input's path, size and write time.
    /// </remarks>
    [Fact]
    public void Run_WarnsWhenAnInputChangedUnderUnchangedSettings()
    {
        var outDir = TempDir();
        var inputDir = TempDir();
        try
        {
            var a = Path.Combine(inputDir, "plate1.csv");
            var b = Path.Combine(inputDir, "plate2.csv");
            File.Copy(Input1, a);
            File.Copy(Input2, b);

            Assert.Equal(0, Invoke("run", "-i", a, b, "-o", outDir, "-c", Config).Code);

            // Same settings, same inputs: silent.
            var repeat = Invoke("run", "-i", a, b, "-o", outDir, "-c", Config);
            Assert.Equal(0, repeat.Code);
            Assert.DoesNotContain("already holds results", repeat.Output, StringComparison.Ordinal);

            // The same bytes, written again - which is what an export re-run looks like. Nothing in
            // the config has moved; only the file's stamp has.
            File.WriteAllBytes(a, File.ReadAllBytes(a));
            File.SetLastWriteTimeUtc(a, DateTime.UtcNow.AddSeconds(5));

            var changed = Invoke("run", "-i", a, b, "-o", outDir, "-c", Config);
            Assert.Equal(0, changed.Code);
            Assert.Contains("already holds results", changed.Output, StringComparison.Ordinal);

            // Named as what it is. Not one setting moved, so reporting this as "different settings"
            // would send the reader to a config diff that shows nothing at all.
            Assert.Contains("input files that have changed", changed.Output, StringComparison.Ordinal);
            Assert.DoesNotContain("different settings", changed.Output, StringComparison.Ordinal);
        }
        finally
        {
            Cleanup(outDir);
            Cleanup(inputDir);
        }
    }

    /// <summary>
    /// `prism differential` end to end: run the pipeline, then ask a contrast of its output.
    /// </summary>
    /// <remarks>
    /// One pipeline run serves every assertion below - it is about a second on the mini fixture, but
    /// a contrast is microseconds, so paying for it once per fact would be most of the test time.
    /// </remarks>
    [Fact]
    public void Differential_TestsAContrastAgainstAFinishedRun()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));

            var csv = Path.Combine(outDir, "differential.csv");
            var (code, output) = Invoke(
                "differential", "-d", outDir, "--group-by", "sample_type", "-a", "qc", "-b", "experimental");

            Assert.Equal(0, code);
            Assert.True(File.Exists(csv), "the default results path is inside the output directory");

            // The status line names the method that produced the numbers - with a menu this size it
            // is the only thing that makes a saved console log interpretable.
            // The prior's SOURCE is part of the headline, and the mini fixture has QC and reference
            // replicates, so the default is to fit on them rather than on the contrast groups.
            Assert.Contains("moderated t (intensity-trend prior from controls), unpaired", output,
                StringComparison.Ordinal);
            Assert.Contains("sample_type = experimental vs qc", output, StringComparison.Ordinal);
            Assert.Contains("Benjamini-Hochberg", output, StringComparison.Ordinal);

            var lines = File.ReadAllLines(csv);

            // The header outlives the shell it was produced in, so it carries the contrast direction.
            Assert.StartsWith("# contrast: sample_type = experimental vs qc", lines[0], StringComparison.Ordinal);
            Assert.Contains("positive log2FC is higher in experimental", lines[0], StringComparison.Ordinal);
            Assert.StartsWith("# method:", lines[1], StringComparison.Ordinal);
            // The rule is recorded, and recorded as NOT having filtered the rows - every tested
            // feature is in the file, so a reader must not take the header as a description of
            // which rows survived.
            Assert.Contains(lines, l => l.StartsWith("# hit rule", StringComparison.Ordinal)
                                        && l.Contains("NOT filtered", StringComparison.Ordinal));

            // Found by its content rather than its index, so adding another comment line is not a
            // test failure - the index is what broke when the hit rule was added.
            var headerIndex = Array.FindIndex(lines, l => l.StartsWith("feature_id,", StringComparison.Ordinal));
            Assert.Equal(
                "feature_id,label,gene,protein,accession,log2fc,fc,ave_expr,statistic,"
                + "p_value,adj_p_value,mean_a,mean_b",
                lines[headerIndex]);

            var rows = lines.Skip(headerIndex + 1).Where(l => l.Length > 0).ToList();
            Assert.NotEmpty(rows);

            // Rows come out most significant first, and the p-values are real numbers in invariant
            // culture - a decimal comma would silently shift every column one to the right.
            var pValues = rows
                .Select(r => double.Parse(r.Split(',')[9], CultureInfo.InvariantCulture))
                .ToList();
            Assert.Equal(pValues.OrderBy(v => v), pValues);
            Assert.All(pValues, v => Assert.InRange(v, 0.0, 1.0));
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    /// <summary>
    /// Each arm is the UNION of its levels, so a level added to an arm can only grow it.
    /// </summary>
    [Fact]
    public void Differential_PoolsSeveralLevelsIntoOneArm()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));

            var (oneCode, one) = Invoke(
                "differential", "-d", outDir, "-g", "sample_type", "-a", "qc", "-b", "experimental",
                "-o", Path.Combine(outDir, "one.csv"));
            var (bothCode, both) = Invoke(
                "differential", "-d", outDir, "-g", "sample_type", "-a", "qc,reference",
                "-b", "experimental", "-o", Path.Combine(outDir, "both.csv"));

            Assert.Equal(0, oneCode);
            Assert.Equal(0, bothCode);
            Assert.Contains("= experimental vs qc", one, StringComparison.Ordinal);
            Assert.Contains("= experimental vs qc + reference", both, StringComparison.Ordinal);

            // Same arm B, larger arm A: the A count must rise and B must not move.
            var (aOne, bOne) = ArmCounts(one);
            var (aBoth, bBoth) = ArmCounts(both);
            Assert.True(aBoth > aOne, $"pooling should grow arm A ({aOne} -> {aBoth})");
            Assert.Equal(bOne, bBoth);
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    /// <summary>
    /// The arm sizes the status line reports, as (A, B).
    /// </summary>
    /// <remarks>
    /// The line is "n = &lt;B label&gt; &lt;nB&gt; vs &lt;A label&gt; &lt;nA&gt;" - B first, matching
    /// the contrast line above it, with each count next to the arm it belongs to. It used to be a
    /// bare "n = nA vs nB" under a line reading "B vs A", which invited reading the first number as
    /// the arm named first.
    /// </remarks>
    private static (int A, int B) ArmCounts(string output)
    {
        // The labels can contain spaces ("qc + reference"), so they are matched lazily and the
        // pattern is anchored on the semicolon that ends the counts.
        var m = Regex.Match(output, @"n = .+? (\d+) vs .+? (\d+);");
        Assert.True(m.Success, $"the status line should report both arm sizes; got: {output}");
        // Group 1 is B (printed first), group 2 is A.
        return (int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture),
                int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture));
    }

    /// <summary>
    /// The refusals. Each one is a thing a user can type that would otherwise produce a plausible
    /// answer to a different question than the one asked.
    /// </summary>
    [Theory]
    // A level on both sides puts the same samples on both sides of the contrast.
    [InlineData("both arms", "-g", "sample_type", "-a", "qc,experimental", "-b", "experimental")]
    // A typo in a level name, named by the arm it emptied.
    [InlineData("Arm B matched no samples", "-g", "sample_type", "-a", "qc", "-b", "nosuchlevel")]
    // A column that is not in the metadata at all.
    [InlineData("No metadata column", "-g", "nosuchcolumn", "-a", "qc", "-b", "experimental")]
    // Paired without the column that says which samples are a pair.
    [InlineData("needs --subject", "-g", "sample_type", "-a", "qc", "-b", "experimental", "--design", "paired")]
    // ... and the reverse, which would otherwise report an unpaired result for a paired-looking command.
    [InlineData("--subject needs --design paired", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--subject", "batch")]
    // --pair-by is kept as an alias, so it must reach the same refusal rather than being ignored.
    [InlineData("--subject needs --design paired", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--pair-by", "batch")]
    // A trend needs its numeric column...
    [InlineData("needs --trend-over", "--design", "trend")]
    // ...and a within-subject trend needs a subject column as well.
    [InlineData("needs --subject", "--design", "trend-within-subject", "--trend-over", "batch")]
    // A trend column without a trend design would otherwise be silently ignored.
    [InlineData("--trend-over needs --design trend", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--trend-over", "batch")]
    // A covariate handed to a test with no design matrix to put it in.
    [InlineData("--adjust-for needs --test moderated", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--test", "welch", "--adjust-for", "batch")]
    // Adjusting for the contrast itself leaves nothing to test.
    [InlineData("is the term being tested", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--adjust-for", "sample_type")]
    // The same rule on a trend, where the tested term is the trend column rather than the group-by.
    // Unguarded this built [1, x, x] and died on a rank check naming neither flag.
    [InlineData("is the term being tested", "--design", "trend", "--trend-over", "batch",
        "--adjust-for", "batch")]
    // A test the design cannot run is refused rather than silently swapped for the moderated t.
    [InlineData("does not apply to --design trend", "--design", "trend", "--trend-over", "batch",
        "--test", "mann-whitney")]
    [InlineData("does not apply to --design paired", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--design", "paired", "--subject", "batch", "--test", "welch")]
    [InlineData("Unknown --test", "-g", "sample_type", "-a", "qc", "-b", "experimental", "--test", "ttest")]
    [InlineData("Unknown --correction", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--correction", "fdr")]
    [InlineData("--level must be", "-g", "sample_type", "-a", "qc", "-b", "experimental", "--level", "gene")]
    // Report-only flags without --report would read as if the command wrote a section it never did.
    [InlineData("--markers needs --report", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--markers", "EV markers (core)")]
    [InlineData("--markers-group-by needs --report", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--markers-group-by", "batch")]
    [InlineData("--no-enrichment needs --report", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--no-enrichment")]
    // A mistyped panel is refused before anything is written, not after differential.csv exists.
    [InlineData("No marker panel 'nosuchpanel'", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--report", "--markers", "nosuchpanel")]
    [InlineData("to group the marker panels by", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--report", "--markers-group-by", "nosuchcolumn")]
    // A trend has no contrast column for the panels to fall back to.
    [InlineData("needs --markers-group-by", "--design", "trend", "--trend-over", "batch",
        "--report", "--markers", "EV markers (core)")]
    // A fractional minimum used to be truncated to an integer with nothing said.
    [InlineData("--min-per-group must be a whole number", "-g", "sample_type", "-a", "qc", "-b", "experimental",
        "--min-per-group", "2.7")]
    public void Differential_RefusesAndSaysWhy(string expected, params string[] args)
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));

            var (code, output) = Invoke(new[] { "differential", "-d", outDir }.Concat(args).ToArray());

            Assert.NotEqual(0, code);
            Assert.Contains(expected, output, StringComparison.Ordinal);
            Assert.False(File.Exists(Path.Combine(outDir, "differential.csv")),
                "a refused contrast must not leave a results file behind");
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    /// <summary>
    /// <c>--report</c> writes the quant report the pane's button writes, from the same contrast - and its
    /// differential.csv is the very file the command just wrote, byte for byte, wherever -o put it.
    /// </summary>
    [Fact]
    public void Differential_Report_WritesTheQuantReport_WithTheSameDifferentialCsv()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));
            var custom = Path.Combine(outDir, "elsewhere", "my_contrast.csv");

            var (code, output) = Invoke("differential", "-d", outDir, "-g", "sample_type", "-a", "qc",
                "-b", "experimental", "-o", custom, "--report", "--no-enrichment",
                "--markers", "EV markers (core)");

            Assert.Equal(0, code);
            var quant = Path.Combine(outDir, "quant");
            Assert.True(File.Exists(Path.Combine(quant, "quant_report.html")));
            Assert.Contains("Quant report written to:", output, StringComparison.Ordinal);
            Assert.Contains("Enrichment skipped: not requested.", output, StringComparison.Ordinal);

            // One writer, one file: the report's copy IS the command's results table.
            Assert.Equal(File.ReadAllBytes(custom), File.ReadAllBytes(Path.Combine(quant, "differential.csv")));

            // Detection ran from the run's own merged_data, and names its test.
            var detection = File.ReadAllLines(Path.Combine(quant, "detection.csv"));
            Assert.StartsWith("# test: Fisher exact", detection[0], StringComparison.Ordinal);

            // The recorded parameters are this command's own flag values.
            var yaml = File.ReadAllText(Path.Combine(quant, "quant_parameters.yaml"));
            Assert.Contains("design: unpaired", yaml, StringComparison.Ordinal);
            Assert.Contains("test: moderated", yaml, StringComparison.Ordinal);
            Assert.Contains("correction: bh", yaml, StringComparison.Ordinal);

            var html = File.ReadAllText(Path.Combine(quant, "quant_report.html"));
            Assert.Contains("EV markers (core)", html, StringComparison.Ordinal);
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    /// <summary>
    /// <c>--clinical</c> joins an external table the way the pane's Clinical CSV input does, so a
    /// contrast on a clinical column - the usual disease-vs-control case - runs headless, and the
    /// report records which file it needed.
    /// </summary>
    [Fact]
    public void Differential_Clinical_ContrastsOnAClinicalColumn_AndTheReportRecordsTheFile()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));
            var names = DifferentialDataset.Load(outDir, FeatureLevel.Protein).MetadataValues("sample");
            var clinical = Path.Combine(outDir, "clinical.csv");
            File.WriteAllLines(clinical, new[] { "PatientName,Diagnosis" }
                .Concat(names.Select((n, i) => $"{n},{(i % 2 == 0 ? "AD" : "Control")}")));

            var (code, output) = Invoke("differential", "-d", outDir, "--clinical", clinical,
                "-g", "Diagnosis", "-a", "Control", "-b", "AD", "--report", "--no-enrichment");

            Assert.Equal(0, code);
            // Named per file, since --clinical is repeatable.
            Assert.Contains("Clinical CSV: joined clinical.csv on 'PatientName'", output, StringComparison.Ordinal);
            Assert.StartsWith("# contrast: Diagnosis = AD vs Control",
                File.ReadLines(Path.Combine(outDir, "differential.csv")).First(), StringComparison.Ordinal);
            var yaml = File.ReadAllText(Path.Combine(outDir, "quant", "quant_parameters.yaml"));
            Assert.Contains("clinical_csv:", yaml, StringComparison.Ordinal);
            Assert.Contains("clinical.csv", yaml, StringComparison.Ordinal);
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    /// <summary>
    /// GUI-to-CLI parity, end to end. A report is built the way the pane's Quant report button builds
    /// it - a <see cref="QuantRequest"/> straight into <see cref="QuantAnalysis"/> - and then the
    /// command that report records is run through the real CLI. Every file the two write must be
    /// byte-identical (the HTML apart from its generated-at time), so a report clicked and the command
    /// it names cannot drift apart without this failing.
    /// </summary>
    /// <remarks>
    /// <para>The one part of the pane this cannot execute is reading its WPF controls into the options;
    /// that mapping is DiffOptions / DiffRule / TryGetGroups, which build the same objects this test
    /// does - including the pane's quirks, which is what the scenarios are for:</para>
    /// <list type="bullet">
    /// <item><c>adjusted</c>: a clinical covariate under the moderated t (Firth GLM detection).</item>
    /// <item><c>paired</c>: a subject column, Holm correction (McNemar detection).</item>
    /// <item><c>welch-stale-covariate</c>: the pane greys Adjust-for out under Welch but keeps the tick,
    /// so the request still carries it. The report must run - and record - the unadjusted analysis,
    /// since the CLI refuses --adjust-for without the moderated t.</item>
    /// <item><c>two-clinical</c>: a second clinical CSV attached on top of the first; the recorded
    /// command must carry both.</item>
    /// </list>
    /// <para>Enrichment is off on both sides: g:Profiler answers from its current database, so its
    /// output is not something a byte comparison could hold still.</para>
    /// </remarks>
    [Theory]
    [InlineData("adjusted")]
    [InlineData("paired")]
    [InlineData("welch-stale-covariate")]
    [InlineData("two-clinical")]
    public void Differential_Report_FromTheButtonsRequest_IsReproducedByItsRecordedCommand(string scenario)
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));
            var paired = scenario == "paired";

            // Clinical tables carrying the diagnosis, a numeric covariate and a subject key - in one file,
            // or split across two for the two-clinical scenario.
            var names = DifferentialDataset.Load(outDir, FeatureLevel.Protein).MetadataValues("sample");
            string Diagnosis(int i) => i % 2 == 0 ? "AD" : "Control";
            var files = new List<string>();
            if (scenario == "two-clinical")
            {
                files.Add(WriteLines(outDir, "clinical_dx.csv", "PatientName,Diagnosis,Subject",
                    names.Select((n, i) => $"{n},{Diagnosis(i)},S{i / 2}")));
                files.Add(WriteLines(outDir, "clinical_age.csv", "PatientName,Age",
                    names.Select((n, i) => $"{n},{30 + i % 40}")));
            }
            else
            {
                files.Add(WriteLines(outDir, "clinical.csv", "PatientName,Diagnosis,Age,Subject",
                    names.Select((n, i) => $"{n},{Diagnosis(i)},{30 + i % 40},S{i / 2}")));
            }

            // --- the button's path ---
            var ds = DifferentialDataset.Load(outDir, FeatureLevel.Protein);
            foreach (var f in files)
                ds.AttachClinical(f);
            var arms = ContrastArms.Resolve(ds.MetadataValues("Diagnosis"), new[] { "Control" }, new[] { "AD" });
            Assert.True(arms.Ok);
            var age = new[] { Covariate.FromMetadata("Age", ds.MetadataValues("Age")) };
            var options = new DifferentialOptions
            {
                Design = paired ? DifferentialDesign.Paired : DifferentialDesign.Unpaired,
                Test = scenario == "welch-stale-covariate" ? DifferentialTest.WelchT : DifferentialTest.ModeratedT,
                Prior = VariancePrior.IntensityTrend,
                Correction = paired ? MultipleTesting.Holm : MultipleTesting.BenjaminiHochberg,
                SubjectLabels = paired ? ds.MetadataValues("Subject") : null,
                Covariates = paired ? null : age,
                PeptideCounts = ds.PeptideCounts,
                PriorGroupColumns = ControlSampleTypes.PriorGroups(ds.MetadataValues("sample_type")),
                MinPerGroup = 2,
            };
            var panel = SkylinePrism.Core.Qc.ProteinListSet.Load().WithBuiltIns()
                .First(l => l.Name == SkylinePrism.Core.Qc.ProteinList.EvMarkersName);
            var request = new QuantRequest
            {
                OutputDir = outDir,
                Dataset = ds,
                Options = options,
                // Non-default on purpose, so the command has to carry every part of the rule.
                Rule = new SignificanceRule { PThreshold = 0.1, UseAdjusted = false, Log2FcThreshold = 0.5 },
                SubjectColumn = paired ? "Subject" : null,
                GroupBy = "Diagnosis",
                GroupA = arms.A,
                GroupB = arms.B,
                ALevels = new[] { "Control" },
                BLevels = new[] { "AD" },
                MarkerPanels = new[] { panel },
                MarkerGroupBy = "sample_type",
            };
            var guiReport = QuantAnalysis.Run(request);
            var guiDir = Path.Combine(outDir, "quant_gui");
            Directory.Move(Path.Combine(outDir, "quant"), guiDir);

            // --- exactly the command that report recorded, through the real CLI ---
            Assert.NotNull(guiReport.CommandArguments);
            var recorded = guiReport.CommandArguments!.ToArray();
            if (scenario == "welch-stale-covariate")
            {
                Assert.DoesNotContain("--adjust-for", recorded);
                Assert.Contains(guiReport.Notes, n => n.StartsWith("Not adjusted for Age", StringComparison.Ordinal));
            }
            if (scenario == "two-clinical")
                Assert.Equal(2, recorded.Count(a => a == "--clinical"));

            var (code, output) = Invoke(recorded);
            Assert.True(code == 0, output);
            var cliDir = Path.Combine(outDir, "quant");

            var guiFiles = Directory.GetFiles(guiDir).Select(Path.GetFileName).OrderBy(f => f).ToList();
            var cliFiles = Directory.GetFiles(cliDir).Select(Path.GetFileName).OrderBy(f => f).ToList();
            Assert.Equal(guiFiles, cliFiles);
            Assert.Contains("detection.csv", guiFiles);

            static string Untimed(string html) =>
                Regex.Replace(html, @"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", "<time>");
            foreach (var file in guiFiles)
            {
                var gui = Path.Combine(guiDir, file!);
                var cli = Path.Combine(cliDir, file!);
                if (file!.EndsWith(".html", StringComparison.Ordinal))
                    Assert.True(Untimed(File.ReadAllText(gui)) == Untimed(File.ReadAllText(cli)),
                        $"{file} differs between the button's report and its recorded command's");
                else
                    Assert.True(File.ReadAllBytes(gui).AsSpan().SequenceEqual(File.ReadAllBytes(cli)),
                        $"{file} differs between the button's report and its recorded command's");
            }

            // And the detection that ran is the one the (effective) design calls for.
            var expectedTest = scenario switch
            {
                "paired" => "# test: McNemar",
                "welch-stale-covariate" => "# test: Fisher",
                _ => "# test: Firth",
            };
            Assert.StartsWith(expectedTest, File.ReadLines(Path.Combine(cliDir, "detection.csv")).First(),
                StringComparison.Ordinal);
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    private static string WriteLines(string dir, string name, string header, IEnumerable<string> rows)
    {
        var path = Path.Combine(dir, name);
        File.WriteAllLines(path, new[] { header }.Concat(rows));
        return path;
    }

    /// <summary>
    /// A level the command line cannot carry - here one containing a comma, which <c>-a</c>/<c>-b</c>
    /// split on - gets NO command rather than a wrong one: the report says why, and nothing is recorded
    /// that would regenerate a different contrast.
    /// </summary>
    [Fact]
    public void Differential_Report_WithALevelNoCommandCanCarry_SaysSoInsteadOfRecordingOne()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));
            var names = DifferentialDataset.Load(outDir, FeatureLevel.Protein).MetadataValues("sample");
            var clinical = WriteLines(outDir, "clinical.csv", "PatientName,Diagnosis",
                names.Select((n, i) => $"{n},{(i % 2 == 0 ? "\"AD, early\"" : "Control")}"));

            var ds = DifferentialDataset.Load(outDir, FeatureLevel.Protein);
            ds.AttachClinical(clinical);
            var arms = ContrastArms.Resolve(ds.MetadataValues("Diagnosis"), new[] { "Control" }, new[] { "AD, early" });
            Assert.True(arms.Ok);

            var r = QuantAnalysis.Run(new QuantRequest
            {
                OutputDir = outDir,
                Dataset = ds,
                Options = new DifferentialOptions { Prior = VariancePrior.Global },
                Rule = SignificanceRule.Default,
                GroupBy = "Diagnosis",
                GroupA = arms.A,
                GroupB = arms.B,
                ALevels = new[] { "Control" },
                BLevels = new[] { "AD, early" },
            });

            Assert.Null(r.CommandArguments);
            Assert.Contains(r.Notes, n => n.StartsWith("No command-line equivalent", StringComparison.Ordinal)
                                          && n.Contains("contains a comma", StringComparison.Ordinal));
            var yaml = File.ReadAllText(Path.Combine(outDir, "quant", "quant_parameters.yaml"));
            Assert.Contains("command_unavailable:", yaml, StringComparison.Ordinal);
            Assert.DoesNotContain("command: |-", yaml, StringComparison.Ordinal);
            Assert.Contains("No command line can regenerate this report",
                File.ReadAllText(r.HtmlPath), StringComparison.Ordinal);
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    [Fact]
    public void Differential_Clinical_ThatMatchesNothing_IsRefusedBeforeAnythingIsWritten()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));
            var clinical = Path.Combine(outDir, "unrelated.csv");
            File.WriteAllText(clinical, "Id,Diagnosis\nZZZ-1,AD\nZZZ-2,Control\n");

            var (code, output) = Invoke("differential", "-d", outDir, "--clinical", clinical,
                "-g", "Diagnosis", "-a", "Control", "-b", "AD");

            Assert.NotEqual(0, code);
            Assert.Contains("--clinical: no column of unrelated.csv matched the samples", output,
                StringComparison.Ordinal);
            Assert.False(File.Exists(Path.Combine(outDir, "differential.csv")));
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    /// <summary>
    /// Every word quant_parameters.yaml can record is one this command accepts, so a saved report's
    /// parameters always read as a runnable command. The table itself is pinned in DifferentialTokensTests.
    /// </summary>
    [Fact]
    public void Differential_AcceptsEveryRecordedToken()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));
            // A subject key and a numeric column, so the paired and trend designs have what they need.
            var names = DifferentialDataset.Load(outDir, FeatureLevel.Protein).MetadataValues("sample");
            var clinical = WriteLines(outDir, "clinical.csv", "PatientName,Diagnosis,Week,Subject",
                names.Select((n, i) => $"{n},{(i % 2 == 0 ? "AD" : "Control")},{i % 4},S{i / 2}"));
            string[] With(string[] contrast, params string[] extra) =>
                new[] { "differential", "-d", outDir, "--clinical", clinical }.Concat(contrast).Concat(extra).ToArray();
            var twoArm = new[] { "-g", "Diagnosis", "-a", "Control", "-b", "AD" };
            void Accepted(string[] args)
            {
                var (code, output) = Invoke(args);
                Assert.True(code == 0, $"refused: {string.Join(" ", args)}\n{output}");
            }

            foreach (var c in Enum.GetValues<MultipleTesting>())
                Accepted(With(twoArm, "--correction", DifferentialTokens.Correction(c)));
            foreach (var p in Enum.GetValues<VariancePrior>())
                Accepted(With(twoArm, "--prior", DifferentialTokens.Prior(p)));
            foreach (var l in Enum.GetValues<FeatureLevel>())
                Accepted(With(twoArm, "--level", DifferentialTokens.Level(l)));

            // Every design, each with the tests it runs - between them, every test word.
            foreach (var t in new[] { DifferentialTest.ModeratedT, DifferentialTest.WelchT,
                         DifferentialTest.StudentT, DifferentialTest.MannWhitney })
                Accepted(With(twoArm, "--design", DifferentialTokens.Design(DifferentialDesign.Unpaired),
                    "--test", DifferentialTokens.Test(t)));
            foreach (var t in new[] { DifferentialTest.ModeratedT, DifferentialTest.PairedT, DifferentialTest.Wilcoxon })
                Accepted(With(twoArm, "--design", DifferentialTokens.Design(DifferentialDesign.Paired),
                    "--subject", "Subject", "--test", DifferentialTokens.Test(t)));
            Accepted(With(Array.Empty<string>(), "--design", DifferentialTokens.Design(DifferentialDesign.LinearTrend),
                "--trend-over", "Week"));
            Accepted(With(Array.Empty<string>(), "--design",
                DifferentialTokens.Design(DifferentialDesign.LinearTrendWithinSubject),
                "--trend-over", "Week", "--subject", "Subject"));

            // Every enum value was exercised above.
            Assert.Equal(6, Enum.GetValues<DifferentialTest>().Length);
            Assert.Equal(4, Enum.GetValues<DifferentialDesign>().Length);
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    /// <summary>
    /// A text timepoint offering two readings: the bare column name is refused with both named, and
    /// adjusting for the column a reading comes from is refused as the tested term.
    /// </summary>
    [Fact]
    public void Differential_ATextTimepoint_IsNotGuessed_AndItsColumnIsTheTestedTerm()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));
            var metaPath = Path.Combine(outDir, "sample_metadata.csv");
            var meta = File.ReadAllLines(metaPath);
            File.WriteAllLines(metaPath, meta.Select((line, i) =>
                i == 0 ? line + ",visit" : line.Length == 0 ? line : $"{line},V{(i - 1) % 4}_Week {2 * ((i - 1) % 4)}"));

            var (bare, bareOut) = Invoke("differential", "-d", outDir, "--design", "trend", "--trend-over", "visit");
            Assert.NotEqual(0, bare);
            Assert.Contains("'visit' can be read as more than one axis: 'visit (", bareOut, StringComparison.Ordinal);

            // "visit (Week)" names a READING; the column is still visit, whose categories encode the
            // very timepoints tested. Compared with the label, this used to get through.
            var (adjusted, adjustedOut) = Invoke("differential", "-d", outDir, "--design", "trend",
                "--trend-over", "visit (Week)", "--test", "moderated", "--adjust-for", "visit");
            Assert.NotEqual(0, adjusted);
            Assert.Contains("'visit' is the term being tested", adjustedOut, StringComparison.Ordinal);
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    /// <summary>
    /// A trend report: no arms, so no detection and no raw-value table - both said on the console -
    /// and the panels grouped by the column named for them.
    /// </summary>
    [Fact]
    public void Differential_Report_OnATrend_SaysWhatItLeftOut()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));

            // Give the run a numeric column to fit against: append one to its sample metadata.
            var metaPath = Path.Combine(outDir, "sample_metadata.csv");
            var meta = File.ReadAllLines(metaPath);
            File.WriteAllLines(metaPath, meta.Select((line, i) =>
                i == 0 ? line + ",week" : line.Length == 0 ? line : $"{line},{(i - 1) % 4}"));

            var (code, output) = Invoke("differential", "-d", outDir, "--design", "trend",
                "--trend-over", "week", "--report", "--no-enrichment",
                "--markers", "EV markers (core)", "--markers-group-by", "sample_type");

            Assert.Equal(0, code);
            Assert.Contains("Detection skipped: it compares two groups", output, StringComparison.Ordinal);
            Assert.Contains("No differential_values.csv", output, StringComparison.Ordinal);
            var quant = Path.Combine(outDir, "quant");
            Assert.False(File.Exists(Path.Combine(quant, "differential_values.csv")));
            Assert.StartsWith("# trend: week from 0 to 3",
                File.ReadLines(Path.Combine(quant, "differential.csv")).First(), StringComparison.Ordinal);
            Assert.Contains("trend_over: week",
                File.ReadAllText(Path.Combine(quant, "quant_parameters.yaml")), StringComparison.Ordinal);
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    /// <summary>
    /// The intensity trend takes its shape from the control replicates by default, and says so -
    /// and --prior-from-groups takes it from the contrast groups instead.
    /// </summary>
    /// <remarks>
    /// The source sets the trend's shape and its level is fitted to the residuals either way, so the
    /// two give different numbers and must be distinguishable from the output alone. This pins that
    /// the default is the control-based one and that the source is named.
    /// </remarks>
    [Fact]
    public void Differential_FitsThePriorOnControlsByDefault_AndNamesTheSource()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));
            string[] Args(params string[] extra) => new[]
            {
                "differential", "-d", outDir, "-g", "sample_type", "-a", "qc", "-b", "experimental",
                "-o", Path.Combine(outDir, "d.csv"),
            }.Concat(extra).ToArray();

            var byDefault = Invoke(Args()).Output;
            var byGroups = Invoke(Args("--prior-from-groups")).Output;
            var byControls = Invoke(Args("--prior-from-controls")).Output;

            Assert.Contains("intensity-trend prior from controls", byDefault, StringComparison.Ordinal);
            Assert.Contains("intensity-trend prior from controls", byControls, StringComparison.Ordinal);
            Assert.Contains("intensity-trend prior from design groups", byGroups, StringComparison.Ordinal);

            // Asking for both at once is a contradiction, not a precedence puzzle.
            var both = Invoke(Args("--prior-from-controls", "--prior-from-groups"));
            Assert.NotEqual(0, both.Code);
            Assert.Contains("opposite things", both.Output, StringComparison.Ordinal);
        }
        finally
        {
            Cleanup(outDir);
        }
    }

    [Fact]
    public void Differential_WithoutTheRequiredFlags_PrintsUsage()
    {
        var (code, output) = Invoke("differential", "-d", "nowhere");

        Assert.Equal(2, code);
        Assert.Contains("Usage: prism differential", output, StringComparison.Ordinal);
    }

    [Fact]
    public void Differential_IsListedAndDocumented()
    {
        var usage = Invoke("--help").Output;
        Assert.Contains("differential", usage, StringComparison.Ordinal);

        var help = Invoke("differential", "--help").Output;
        Assert.Contains("--group-by", help, StringComparison.Ordinal);
        Assert.Contains("--prior-from-controls", help, StringComparison.Ordinal);
        Assert.Contains("--correction", help, StringComparison.Ordinal);
    }

    /// <summary>
    /// Running onto a directory that already holds results says so - and only when the results would
    /// actually differ, which is what keeps the warning worth reading.
    /// </summary>
    /// <remarks>
    /// The decision itself is covered by <c>ExistingResultsTests</c>; this drives the real command
    /// three times so the wiring is proven end to end rather than by inspection. A run is about a
    /// second on the mini fixture.
    /// </remarks>
    [Fact]
    public void Run_WarnsOnlyWhenItWouldReplaceDifferentResults()
    {
        var outDir = TempDir();
        try
        {
            Assert.Equal(0, Run(outDir));

            var (repeatCode, repeatOutput) = Invoke("run", "-i", Input1, Input2, "-o", outDir, "-c", Config);
            Assert.Equal(0, repeatCode);
            Assert.DoesNotContain("already holds results", repeatOutput, StringComparison.Ordinal);

            // One setting changed, which moves the transition rollup and everything downstream of it.
            var changed = Path.Combine(outDir, "changed-config.yaml");
            File.WriteAllText(
                changed, File.ReadAllText(Config).Replace("min_transitions: 1", "min_transitions: 2"));

            var (changedCode, changedOutput) =
                Invoke("run", "-i", Input1, Input2, "-o", outDir, "-c", changed);
            Assert.Equal(0, changedCode);
            Assert.Contains("WARNING:", changedOutput, StringComparison.Ordinal);
            Assert.Contains("already holds results", changedOutput, StringComparison.Ordinal);
            Assert.Contains("corrected_peptides.parquet", changedOutput, StringComparison.Ordinal);
        }
        finally
        {
            Cleanup(outDir);
        }
    }
}
