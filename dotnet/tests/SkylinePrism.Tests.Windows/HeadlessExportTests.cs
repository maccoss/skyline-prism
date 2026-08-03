using System;
using System.IO;
using System.Linq;
using SkylinePrism.Skyline;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The headless (SkylineCmd) export path: the command line built for a closed document, the file naming
/// that carries each document's batch label into the merge, and SkylineCmd discovery. These run without
/// Skyline installed - the actual process launch is covered by manual verification against a real
/// document, not by CI.
/// </summary>
public class HeadlessExportTests
{
    [Fact]
    public void BuildArgs_OpensTheDocumentAndExportsInvariantCsv()
    {
        var args = HeadlessSkylineExporter.BuildArgs(
            @"C:\data\plate1.sky", @"C:\tool\Reports\Skyline-PRISM.skyr", "PRISM", @"C:\out\plate1.csv");

        Assert.Equal(@"--in=C:\data\plate1.sky", args[0]);
        Assert.Contains(@"--report-add=C:\tool\Reports\Skyline-PRISM.skyr", args);
        Assert.Contains("--report-conflict-resolution=overwrite", args);
        Assert.Contains("--report-name=PRISM", args);
        Assert.Contains(@"--report-file=C:\out\plate1.csv", args);
        // Skyline's command line has no parquet writer, and numbers must be invariant.
        Assert.Contains("--report-format=csv", args);
        Assert.Contains("--report-invariant", args);
    }

    [Fact]
    public void BuildArgs_OmitsReportFormat_WhenTheExtensionShouldChoose()
    {
        var args = HeadlessSkylineExporter.BuildArgs(
            @"C:\data\p.sky", null, "PRISM", @"C:\out\p.parquet", format: null);

        // There is no --report-format=parquet (the flag only accepts csv|tsv); SkylineCmd picks parquet
        // from the .parquet extension, so passing any format would force a text writer instead.
        Assert.DoesNotContain(args, a => a.StartsWith("--report-format", StringComparison.Ordinal));
        Assert.Contains(@"--report-file=C:\out\p.parquet", args);
        Assert.Contains("--report-invariant", args);
    }

    [Fact]
    public void BuildArgs_NeverSavesTheDocument()
    {
        var args = HeadlessSkylineExporter.BuildArgs(@"C:\data\plate1.sky", null, "PRISM", @"C:\out\p.csv");

        // The user's document is opened read-only; a stray --save would rewrite it.
        Assert.DoesNotContain(args, a => a.StartsWith("--save", StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(args, a => a.StartsWith("--out", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void BuildArgs_OmitsReportAdd_WhenNoDefinitionIsBundled()
    {
        var args = HeadlessSkylineExporter.BuildArgs(@"C:\data\p.sky", null, "PRISM", @"C:\out\p.csv");

        // Falls back to whatever "PRISM" report is already in the user's Skyline settings; passing an
        // empty --report-add would abort the whole invocation.
        Assert.DoesNotContain(args, a => a.StartsWith("--report-add", StringComparison.Ordinal));
        Assert.DoesNotContain(args, a => a.StartsWith("--report-conflict-resolution", StringComparison.Ordinal));
        Assert.Contains("--report-name=PRISM", args);
    }

    [Fact]
    public void MetadataFileName_PairsWithTheReportStem()
    {
        // The report stem IS the batch/Source Document label the merge derives, so the metadata file has
        // to be derivable from it to stay paired when several documents share one directory.
        Assert.Equal("plate1.metadata.csv", SkylineReportDriver.MetadataFileName("plate1"));
        Assert.NotEqual(
            SkylineReportDriver.MetadataFileName("plate1"),
            SkylineReportDriver.MetadataFileName("plate2"));
    }

    [Fact]
    public void Create_ThrowsAnActionableMessage_WhenSkylineCmdCannotBeFound()
    {
        // A path that does not exist forces discovery, which fails on a machine without Skyline. On a
        // machine WITH Skyline this returns an exporter - both outcomes are valid, so assert on whichever
        // happens rather than requiring an install.
        try
        {
            var exporter = HeadlessSkylineExporter.Create(@"Z:\nope\SkylineCmd.exe");
            Assert.NotNull(exporter); // Skyline is installed here; discovery succeeded
        }
        catch (InvalidOperationException ex)
        {
            Assert.Contains("SkylineCmd.exe", ex.Message);
            Assert.Contains(SkylineCmdLocator.OverrideEnvVar, ex.Message); // tells the user how to fix it
        }
    }

    [Fact]
    public void FindAll_OnlyReturnsCandidatesWithASkylineExeBesideThem()
    {
        // The decisive rule: the SkylineCmd.exe in the ClickOnce "...exe_..." folders has no Skyline
        // executable beside it and fails with "Unable to find Skyline.exe".
        foreach (var candidate in SkylineCmdLocator.FindAll())
        {
            var dir = Path.GetDirectoryName(candidate.CmdPath)!;
            Assert.True(File.Exists(candidate.SkylineExePath));
            Assert.Equal(dir, Path.GetDirectoryName(candidate.SkylineExePath));
        }
    }

    [Fact]
    public void FindAll_ReturnsNewestFirst()
    {
        var all = SkylineCmdLocator.FindAll();
        var times = all.Select(c => c.LastWriteUtc).ToList();
        Assert.Equal(times.OrderByDescending(t => t).ToList(), times);
    }

    [Fact]
    public void Find_PrefersAnExplicitPathThatExists()
    {
        var temp = Path.Combine(Path.GetTempPath(), "prism_cmd_" + Guid.NewGuid().ToString("N") + ".exe");
        File.WriteAllBytes(temp, new byte[] { 0x4D, 0x5A });
        try
        {
            Assert.Equal(Path.GetFullPath(temp), SkylineCmdLocator.Find(temp));
        }
        finally
        {
            File.Delete(temp);
        }
    }
}
