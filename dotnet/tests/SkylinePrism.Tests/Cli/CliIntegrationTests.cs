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
        var merged = Path.Combine(outDir, "merged.parquet");
        try
        {
            var (code, output) = Invoke("merge", Input1, Input2, "-o", merged);
            Assert.Equal(0, code);
            Assert.True(File.Exists(merged));
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
}
