using System;
using System.IO;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// What a run is about to write over, and whether it would be writing anything different.
/// </summary>
/// <remarks>
/// Pointing PRISM at a directory that already held a cohort's results said nothing at all: it
/// recomputed what the settings changed and replaced the rest without a word. The hard part is not
/// noticing the files - it is staying quiet on the ordinary case, because re-running to regenerate a
/// report or top up a partial ion accounting is normal, and a warning that fires every time stops
/// being read.
/// </remarks>
public class ExistingResultsTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "prism-existing-" + Guid.NewGuid().ToString("N"));

    public ExistingResultsTests() => Directory.CreateDirectory(_dir);

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
    public void AnEmptyDirectoryHasNothingToSay()
    {
        var existing = ExistingResults.Inspect(_dir, new PrismConfig());
        Assert.False(existing.Any);
        Assert.False(existing.WouldReplace);
        Assert.Null(existing.Warning());
    }

    [Fact]
    public void ADirectoryThatDoesNotExistHasNothingToSay()
    {
        var existing = ExistingResults.Inspect(Path.Combine(_dir, "nope"), new PrismConfig());
        Assert.False(existing.Any);
        Assert.Null(existing.Warning());
    }

    /// <summary>
    /// The case that has to stay silent, or the warning becomes noise: same version, same settings.
    /// </summary>
    [Fact]
    public void TheSameVersionAndSettingsIsSilent()
    {
        var config = new PrismConfig();
        Results(config);

        var existing = ExistingResults.Inspect(_dir, config);
        Assert.True(existing.Any);
        Assert.True(existing.SameVersion);
        Assert.True(existing.SameSettings);
        Assert.False(existing.WouldReplace);
        Assert.Null(existing.Warning());
    }

    [Fact]
    public void DifferentSettingsAreReported()
    {
        Results(new PrismConfig());

        var changed = new PrismConfig();
        changed.TransitionRollup.Method = "median_polish";

        var existing = ExistingResults.Inspect(_dir, changed);
        Assert.True(existing.SameVersion);
        Assert.False(existing.SameSettings);
        Assert.True(existing.WouldReplace);

        var warning = existing.Warning()!;
        Assert.Contains("different settings", warning, StringComparison.Ordinal);
        Assert.Contains("corrected_peptides.parquet", warning, StringComparison.Ordinal);
        // It has to say that unchanged stages are reused, or it reads as "everything is lost".
        Assert.Contains("reused", warning, StringComparison.Ordinal);
    }

    /// <summary>
    /// Results with no provenance beside them cannot be compared, so they are reported rather than
    /// assumed to match - the safe direction when the question is whether a cohort survives.
    /// </summary>
    [Fact]
    public void ResultsWithNoProvenanceAreReported()
    {
        File.WriteAllText(Path.Combine(_dir, "corrected_peptides.parquet"), "x");

        var existing = ExistingResults.Inspect(_dir, new PrismConfig());
        Assert.True(existing.WouldReplace);
        Assert.Contains("a previous run", existing.Warning()!, StringComparison.Ordinal);
    }

    [Fact]
    public void AnUnreadableProvenanceIsReportedRatherThanTrusted()
    {
        Results(new PrismConfig());
        File.WriteAllText(Path.Combine(_dir, "parameters.json"), "{ not json");

        Assert.True(ExistingResults.Inspect(_dir, new PrismConfig()).WouldReplace);
    }

    /// <summary>
    /// Working files are not results. A directory holding only a merge and a stage cache is one a
    /// previous run left partway, and warning about it would train people to ignore the dialog.
    /// </summary>
    [Fact]
    public void IntermediatesAloneAreNotResults()
    {
        File.WriteAllText(Path.Combine(_dir, "peptides_rollup.parquet"), "x");
        File.WriteAllText(Path.Combine(_dir, "stage_cache.json"), "{}");
        Directory.CreateDirectory(Path.Combine(_dir, "merged_data"));

        Assert.False(ExistingResults.Inspect(_dir, new PrismConfig()).Any);
    }

    /// <summary>Writes what a completed run leaves: the reported outputs plus its provenance.</summary>
    private void Results(PrismConfig config)
    {
        File.WriteAllText(Path.Combine(_dir, "corrected_peptides.parquet"), "x");
        File.WriteAllText(Path.Combine(_dir, "corrected_proteins.parquet"), "x");
        File.WriteAllText(Path.Combine(_dir, "qc_report.html"), "<html></html>");
        Provenance.Write(
            Path.Combine(_dir, Provenance.FileName), config, new[] { "report.csv" },
            new Provenance.Stats(1, 10, 5, 5), "2026-01-01T00:00:00.0000000Z");
    }
}
