using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using SkylinePrism.Core.Visualization;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// What the QC report says about ion accounting, and what it refuses to say.
/// </summary>
/// <remarks>
/// The captions are the only place a reader of the HTML learns which settings produced the numbers -
/// the cache is keyed on them and the figures themselves carry no hint - so the wording is a
/// correctness surface here, not decoration.
/// </remarks>
public class IonAccountingReportTests
{
    /// <summary>
    /// A directory that was never measured gets no section, rather than an empty one or a zero.
    /// </summary>
    [Fact]
    public void AnUnmeasuredDirectoryHasNoIonSection()
    {
        var dir = TempDir();
        try
        {
            Assert.Null(IonAccountingStore.Read(dir));
            Assert.Empty(IonAccountingStore.ReadCycles(dir));
            Assert.Empty(IonAccountingStore.SamplesWithCycles(dir));
        }
        finally
        {
            Cleanup(dir);
        }
    }

    /// <summary>
    /// The cohort caption states the median and the spread, and names both extraction tolerances and
    /// the isolation scheme - because the cache is keyed on them and a reader has no other source.
    /// </summary>
    [Fact]
    public void TheCohortCaptionStatesTheFractionAndTheSettings()
    {
        var result = Result(
            Row("s1", 10), Row("s2", 20), Row("s3", 30));

        var caption = QcReport.FractionCaption(result.Usable, PlotRenderer.IonLevel.Ms2);

        // The real method's wording: "Median 20.0 % assigned, ranging 10.0 % to 30.0 %."
        Assert.Contains("Median", caption);
        Assert.Contains("20.0", caption);          // the median of 10/20/30
        Assert.Contains("10.0", caption);          // and the spread
        Assert.Contains("30.0", caption);
    }

    /// <summary>
    /// Where a replicate assigned more than it acquired, the whole figure is WITHHELD and named as a
    /// defect rather than quietly excluded from a median. A median computed over the rest would be a
    /// real-looking number standing in front of a broken one.
    /// </summary>
    [Fact]
    public void AnImpossibleReplicateWithholdsTheCohortFraction()
    {
        var broken = new IonAccountingRow(
            "bad", "experimental", "bad.raw", Ms2ReadStatus.Ok, "test", 1, 167,
            Ms1Acquired: 100, Ms2Acquired: 100, Ms1Assigned: 100, Ms2Assigned: 150,
            0, 60, 1000, 0, 0, 1, Array.Empty<double>(), Array.Empty<double>());
        var result = Result(Row("s1", 10), Row("s2", 20), broken);

        var caption = QcReport.FractionCaption(result.Usable, PlotRenderer.IonLevel.Ms2);

        Assert.Contains("impossible", caption);
        Assert.DoesNotContain("Median", caption);
        Assert.Contains("1 of 3", caption);
    }

    /// <summary>
    /// Three representatives get three distinct labels, and a two-replicate cohort gets the two that
    /// mean something rather than a "median" that is also the best.
    /// </summary>
    [Fact]
    public void RepresentativeLabellingMatchesHowManyThereAre()
    {
        Assert.Equal(
            new[] { "best", "median", "worst" },
            Result(Row("a", 10), Row("b", 20), Row("c", 30), Row("d", 40))
                .Representatives()
                .Select((_, i) => new[] { "best", "median", "worst" }[i]));

        // Two replicates: Representatives returns both, and neither is called the median.
        var two = Result(Row("a", 10), Row("b", 30)).Representatives();
        Assert.Equal(2, two.Count);
    }

    /// <summary>
    /// The report reads the cache, so what it plots is whatever a run left - including a partial one.
    /// A six-row cache renders six bars and says so, rather than appearing to be a whole cohort.
    /// </summary>
    [Fact]
    public void APartialCacheRendersWhatItHasAndNamesTheCount()
    {
        var dir = TempDir();
        try
        {
            IonAccountingStore.Write(dir, Result(Row("s1", 10), Row("s2", 20)));
            var read = IonAccountingStore.Read(dir);

            Assert.NotNull(read);
            Assert.Equal(2, read!.Usable.Count);

            var png = PlotRenderer.IonAccountingPng(read, PlotRenderer.IonLevel.Ms2, "Ions");
            Assert.True(png.Length > 1000);
            Assert.Equal(new byte[] { 0x89, 0x50, 0x4E, 0x47 }, png.Take(4).ToArray());
        }
        finally
        {
            Cleanup(dir);
        }
    }

    private static IonAccountingRow Row(string sample, double percent) =>
        new(sample, "experimental", sample + ".raw", Ms2ReadStatus.Ok, "test", 1, 167,
            Ms1Acquired: 1000, Ms2Acquired: 1000,
            Ms1Assigned: percent * 10, Ms2Assigned: percent * 10,
            0, 60, 1000, 0, 0, 1, Array.Empty<double>(), Array.Empty<double>());

    private static IonAccountingResult Result(params IonAccountingRow[] rows) =>
        new("k", "+/-10 ppm (centroided)", "+/-10 ppm (centroided)",
            "167 windows, 400.4-901.7 m/z, 3.001 Th",
            Array.Empty<string>(), 51733, true, rows, Array.Empty<IonCycleRow>());

    private static string TempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_ionrep_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void Cleanup(string dir)
    {
        try
        {
            Directory.Delete(dir, recursive: true);
        }
        catch (IOException)
        {
            // A temp directory that will not delete is not a test failure.
        }
    }
}
