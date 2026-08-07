using System;
using System.IO;
using System.Runtime.InteropServices;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// Where the merge spills its sort. PRISM output routinely lives on a mapped network drive - the
/// Skyline tool defaults to a folder beside the document - and DuckDB was spilling a multi-gigabyte
/// sort there over SMB, which is slow enough to look like a hang.
/// </summary>
public class MergeTempDirectoryTests
{
    [Fact]
    public void LocalOutput_SpillsBesideTheOutput()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_td_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var resolved = DuckDbMerge.ResolveTempDirectory(Path.Combine(dir, "merged_data.parquet"));

            // Same volume as the output: nothing crosses a filesystem, and the scratch is obvious.
            Assert.Equal(Path.Combine(dir, ".duckdb_temp"), resolved);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// A UNC output path must not take the spill with it. Checked by path shape rather than by
    /// mounting a share, so it runs anywhere; the drive-letter case (V:\...) goes through the same
    /// branch via <c>DriveInfo.DriveType</c>.
    /// </summary>
    [Fact]
    public void UncOutput_SpillsToTheLocalTempDirectory()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return; // UNC is a Windows path form; the Unix mount case is covered by DriveInfo.

        var resolved = DuckDbMerge.ResolveTempDirectory(@"\\server\share\PRISM-Output\merged_data.parquet");

        Assert.StartsWith(Path.GetTempPath(), resolved, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(@"\\server\share", resolved, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void EnvironmentVariable_OverridesEverything()
    {
        var custom = Path.Combine(Path.GetTempPath(), "prism_td_env_" + Guid.NewGuid().ToString("N"));
        var previous = Environment.GetEnvironmentVariable(DuckDbMerge.TempDirEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(DuckDbMerge.TempDirEnvVar, custom);

            // Even for a local output, which would otherwise spill beside itself.
            var resolved = DuckDbMerge.ResolveTempDirectory(
                Path.Combine(Path.GetTempPath(), "out", "merged_data.parquet"));

            Assert.StartsWith(custom, resolved, StringComparison.Ordinal);
        }
        finally
        {
            Environment.SetEnvironmentVariable(DuckDbMerge.TempDirEnvVar, previous);
        }
    }
}
