using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// "N of M replicates measured so far" - the one sentence that says the ion accounting plots cover
/// part of the cohort, now that they are drawn over the replicates that HAVE numbers.
/// </summary>
/// <remarks>
/// The denominator has to come from the ANALYSIS, not from the cache. Progress is saved after every
/// replicate and the placeholder rows for replicates with no instrument file of their own are
/// appended only once the scan loop ends, so a cache read while a measurement is running holds
/// exactly the replicates measured so far - against itself it could only ever say "N of N", which is
/// to say nothing at all, in precisely the case the sentence exists for.
/// </remarks>
public class IonCoverageCaptionTests : IDisposable
{
    /// <summary>The committed <c>mini</c> cohort's replicate count - the analysis's own denominator.</summary>
    private const int CohortReplicates = 166;

    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "prism-coverage-" + Guid.NewGuid().ToString("N"));

    public IonCoverageCaptionTests()
    {
        Directory.CreateDirectory(_dir);
        foreach (var file in Directory.GetFiles(
                     Path.Combine(AppContext.BaseDirectory, "fixtures", "mini", "e2e-sum", "output")))
        {
            File.Copy(file, Path.Combine(_dir, Path.GetFileName(file)));
        }
    }

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

    /// <summary>
    /// A spot check of three replicates says three of the cohort, not three of three.
    /// </summary>
    [Fact]
    public void APartialCacheCountsAgainstTheCohortAndNotAgainstItself()
    {
        IonAccountingStore.Write(_dir, Measured(3));

        var coverage = Regex.Match(Render(), @"(\d+) of ([\d,]+) replicates measured so far");

        Assert.True(coverage.Success, "The partial cache said nothing about how much it covers.");
        Assert.Equal("3", coverage.Groups[1].Value);
        Assert.Equal(CohortReplicates, int.Parse(coverage.Groups[2].Value.Replace(",", "")));
    }

    /// <summary>
    /// A finished measurement says nothing - the plots are the whole cohort, and a sentence saying so
    /// under every run would be noise.
    /// </summary>
    [Fact]
    public void AFullyMeasuredCohortSaysNothingAboutCoverage()
    {
        IonAccountingStore.Write(_dir, Measured(CohortReplicates));

        Assert.DoesNotContain("measured so far", Render(), StringComparison.Ordinal);
    }

    private string Render()
    {
        QcReport.Generate(_dir, new PrismConfig(), savePlots: false);
        return File.ReadAllText(Path.Combine(_dir, "qc_report.html"));
    }

    private static IonAccountingResult Measured(int replicates)
    {
        var rows = new List<IonAccountingRow>(replicates);
        for (var i = 0; i < replicates; i++)
        {
            var sample = $"r{i + 1}";
            rows.Add(new IonAccountingRow(
                sample, "experimental", sample + ".raw", Ms2ReadStatus.Ok, "test", 1, 167,
                Ms1Acquired: 1000, Ms2Acquired: 1000,
                Ms1Assigned: 100, Ms2Assigned: 100,
                Ms2Explained: 0, HasExplained: false,
                0, 60, 1000, 0, 0, 1, Array.Empty<double>(), Array.Empty<double>()));
        }

        return new IonAccountingResult(
            "key", "+/-10 ppm (centroided)", "+/-10 ppm (centroided)", "167 windows",
            Array.Empty<string>(), 4321, true, rows, Array.Empty<IonCycleRow>());
    }
}
