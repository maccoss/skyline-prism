using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// The stamp that decides whether a merge can be reused - and, through it, every stage below the
/// merge, since it is their upstream ingredient.
/// </summary>
/// <remarks>
/// It used to name each input by its full path, so the same cohort on a share stamped differently from
/// two machines: one mapping it as <c>Z:</c> and another as <c>Y:</c> re-merged and recomputed
/// everything for results that would have come out identical. Inputs under the output directory - which
/// is where the tool exports every report - are now named relative to it.
/// </remarks>
public class SourceFingerprintTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "prism-stamp-" + Guid.NewGuid().ToString("N"));

    public SourceFingerprintTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    [Fact]
    public void StableForSameInputs_ChangesOnEdit()
    {
        var f = Path.Combine(Path.GetTempPath(), "prism_fp_" + Guid.NewGuid().ToString("N") + ".csv");
        File.WriteAllText(f, "a,b\n1,2\n");
        try
        {
            var fp1 = SourceFingerprint.Compute(new[] { f });
            Assert.Equal(fp1, SourceFingerprint.Compute(new[] { f })); // stable

            Thread.Sleep(10);
            File.WriteAllText(f, "a,b\n1,2\n3,4\n"); // size + mtime change
            Assert.NotEqual(fp1, SourceFingerprint.Compute(new[] { f }));
        }
        finally { File.Delete(f); }
    }

    [Fact]
    public void CacheEntry_RoundTrips()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_cache_" + Guid.NewGuid().ToString("N") + ".json");
        try
        {
            SourceFingerprint.Write(path, new SourceFingerprint.CacheEntry("ABC", 123, "Peptide"));
            var e = SourceFingerprint.TryRead(path);
            Assert.NotNull(e);
            Assert.Equal("ABC", e!.Fingerprint);
            Assert.Equal(123, e.TotalRows);
            Assert.Equal("Peptide", e.SortColumn);
        }
        finally { File.Delete(path); }
    }

    /// <summary>Two machines, two drive letters, one cohort.</summary>
    [Fact]
    public void TheSameCohortUnderTwoOutputPathsStampsTheSame()
    {
        var asZ = Cohort("as-Z");
        var asY = Cohort("as-Y");

        Assert.NotEqual(
            SourceFingerprint.Compute(asZ.Inputs),
            SourceFingerprint.Compute(asY.Inputs));
        Assert.Equal(
            SourceFingerprint.Compute(asZ.Inputs, asZ.OutputDir),
            SourceFingerprint.Compute(asY.Inputs, asY.OutputDir));
    }

    /// <summary>An input that moved or changed still stamps differently.</summary>
    [Fact]
    public void AChangedInputStampsDifferently()
    {
        var cohort = Cohort("changed");
        var before = SourceFingerprint.Compute(cohort.Inputs, cohort.OutputDir);

        File.WriteAllText(cohort.Inputs[0], "more rows than it had");
        File.SetLastWriteTimeUtc(cohort.Inputs[0], new DateTime(2026, 3, 3, 0, 0, 0, DateTimeKind.Utc));

        Assert.NotEqual(before, SourceFingerprint.Compute(cohort.Inputs, cohort.OutputDir));
    }

    /// <summary>
    /// An input OUTSIDE the output directory keeps its full path: there is nothing to make it relative
    /// to, and two unrelated files of the same size and time must not look alike.
    /// </summary>
    [Fact]
    public void AnInputOutsideTheOutputDirectoryKeepsItsPath()
    {
        var one = Cohort("outside-a");
        var two = Cohort("outside-b");

        // Stamped against an unrelated output directory, neither input is under it.
        var elsewhere = Path.Combine(_dir, "somewhere-else");
        Assert.NotEqual(
            SourceFingerprint.Compute(one.Inputs, elsewhere),
            SourceFingerprint.Compute(two.Inputs, elsewhere));
    }

    /// <summary>
    /// A sibling directory that merely shares a prefix is outside, not under.
    /// </summary>
    /// <remarks>
    /// A prefix test is not a boundary test: <c>.../outside/report.csv</c> starts with
    /// <c>.../out</c>, and stamping it relative as <c>../outside/report.csv</c> would let two unrelated
    /// files of the same size and time collide when their directories are mounted differently.
    /// </remarks>
    [Fact]
    public void ASiblingSharingAPrefixIsNotUnderTheOutputDirectory()
    {
        var outDir = Path.Combine(_dir, "out");
        Directory.CreateDirectory(outDir);
        var sibling = Path.Combine(_dir, "outside");
        Directory.CreateDirectory(sibling);
        var report = Path.Combine(sibling, "report.csv");
        File.WriteAllText(report, "rows");
        File.SetLastWriteTimeUtc(report, new DateTime(2026, 1, 1, 12, 0, 0, DateTimeKind.Utc));
        var inputs = new[] { report };

        // Outside means outside: the stamp is the one it has with no output directory at all.
        Assert.Equal(
            SourceFingerprint.Compute(inputs),
            SourceFingerprint.Compute(inputs, outDir));
    }

    /// <summary>
    /// A directory already stamped the old way keeps that stamp while it still describes the inputs,
    /// so its whole chain of stage fingerprints survives this change rather than recomputing once.
    /// </summary>
    [Fact]
    public void ADirectoryStampedTheOldWayKeepsIt()
    {
        var cohort = Cohort("legacy");
        var legacy = SourceFingerprint.Compute(cohort.Inputs);
        var machineIndependent = SourceFingerprint.Compute(cohort.Inputs, cohort.OutputDir);
        Assert.NotEqual(legacy, machineIndependent);

        // What such a directory recorded is what it keeps.
        Assert.Equal(legacy, SourceFingerprint.Preferred(legacy, cohort.Inputs, cohort.OutputDir));

        // A directory with nothing recorded, or one whose inputs have moved on, takes the new form.
        Assert.Equal(
            machineIndependent, SourceFingerprint.Preferred(null, cohort.Inputs, cohort.OutputDir));
        Assert.Equal(
            machineIndependent,
            SourceFingerprint.Preferred("stale", cohort.Inputs, cohort.OutputDir));
    }

    /// <summary>The suffix the callers append - the merge's own settings - travels with both forms.</summary>
    [Fact]
    public void ThePreferredStampCarriesTheCallersSuffix()
    {
        var cohort = Cohort("suffix");
        var legacy = SourceFingerprint.Compute(cohort.Inputs) + "|settings";

        Assert.Equal(legacy, SourceFingerprint.Preferred(legacy, cohort.Inputs, cohort.OutputDir, "|settings"));
        Assert.Equal(
            SourceFingerprint.Compute(cohort.Inputs, cohort.OutputDir) + "|settings",
            SourceFingerprint.Preferred(null, cohort.Inputs, cohort.OutputDir, "|settings"));
    }

    /// <summary>
    /// One output directory holding one exported report, as the tool lays it out. The two cohorts
    /// differ only in the directory they sit under, which is the drive-letter case.
    /// </summary>
    private (string OutputDir, IReadOnlyList<string> Inputs) Cohort(string name)
    {
        var outputDir = Path.Combine(_dir, name, "PRISM-Output");
        var reports = Path.Combine(outputDir, "skyline-reports");
        Directory.CreateDirectory(reports);
        var report = Path.Combine(reports, "Plate1.parquet");
        File.WriteAllText(report, "rows");
        File.SetLastWriteTimeUtc(report, new DateTime(2026, 1, 1, 12, 0, 0, DateTimeKind.Utc));
        return (outputDir, new[] { report });
    }
}
