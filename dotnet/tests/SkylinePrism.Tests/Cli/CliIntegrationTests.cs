using SkylinePrism.Core.IO;
using System;
using System.IO;
using System.Linq;
using SkylinePrism.Cli;
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
