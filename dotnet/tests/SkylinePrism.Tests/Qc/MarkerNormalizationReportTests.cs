using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Normalization;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Reading back what a run recorded about its marker normalization. This is what the QC tab's Marker
/// score and Marker loadings plots are built from, and the point of reading rather than recomputing is
/// that the plot cannot disagree with what was actually subtracted - so the round trip through the
/// writer is the test that matters.
/// </summary>
public class MarkerNormalizationReportTests
{
    private static string TempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism-marker-report-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static MarkerNormalization.MarkerScore Score() => new(
        Score: new[] { -1.5, 0.25, 2.0 },
        Loadings: new[] { 0.6, -0.3, 0.1 },
        MarkerNames: new[] { "CD9", "SDCBP", "ANXA2" },
        VarianceExplained: 0.704,
        CorrelationWithMean: 0.951);

    /// <summary>
    /// The writer and the reader are one contract. The loadings are written as '#' comments so the file
    /// still opens as a plain two-column CSV in a spreadsheet, which is exactly the detail a naive
    /// reader would drop.
    /// </summary>
    [Fact]
    public void ItRoundTripsWhatTheStageWrote()
    {
        var dir = TempDir();
        try
        {
            var samples = new[] { "A", "B", "C" };
            MarkerNormalizeStage.WriteScoreCsv(
                Path.Combine(dir, MarkerNormalizationReport.FileName), samples, Score());

            var report = MarkerNormalizationReport.Read(dir);

            Assert.NotNull(report);
            Assert.Equal(samples, report!.Samples);
            Assert.Equal(new[] { -1.5, 0.25, 2.0 }, report.Scores);
            Assert.Equal(new[] { "CD9", "SDCBP", "ANXA2" }, report.MarkerNames);
            Assert.Equal(new[] { 0.6, -0.3, 0.1 }, report.Loadings);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// A replicate name containing a comma survives, because the writer quotes it. Skyline replicate
    /// names are user-typed and do contain commas.
    /// </summary>
    [Fact]
    public void AQuotedReplicateNameSurvives()
    {
        var dir = TempDir();
        try
        {
            MarkerNormalizeStage.WriteScoreCsv(
                Path.Combine(dir, MarkerNormalizationReport.FileName),
                new[] { "Plate 1, well A1", "B", "C" }, Score());

            var report = MarkerNormalizationReport.Read(dir);

            Assert.Equal("Plate 1, well A1", report!.Samples[0]);
            Assert.Equal(-1.5, report.Scores[0]);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// A run that did not use marker normalization has no file, and that is not an error - the plot says
    /// so rather than the window failing to open.
    /// </summary>
    [Fact]
    public void AnAbsentFileReadsAsNullRatherThanThrowing()
    {
        var dir = TempDir();
        try
        {
            Assert.Null(MarkerNormalizationReport.Read(dir));
            Assert.Null(MarkerNormalizationReport.Read(Path.Combine(dir, "does", "not", "exist")));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>The loadings block's own "# marker,loading" header is a comment too, and is not a marker.</summary>
    [Fact]
    public void TheCommentHeaderIsNotReadAsAMarker()
    {
        var report = MarkerNormalizationReport.Parse(new[]
        {
            "sample_id,marker_score",
            "A,1.0",
            "",
            "# marker,loading",
            "# CD9,0.6",
        });

        Assert.Equal(new[] { "CD9" }, report!.MarkerNames);
        Assert.Equal(new[] { "A" }, report.Samples);
    }

    /// <summary>
    /// Largest-first ordering, and the concentration figure the plot titles with. A panel whose axis is
    /// carried by one protein is a single-protein normalization wearing a panel's clothes, and that is
    /// only visible if the loadings are ranked.
    /// </summary>
    [Fact]
    public void LoadingsRankByMagnitude_AndTheDominanceShareIsReported()
    {
        var report = MarkerNormalizationReport.Parse(new[]
        {
            "sample_id,marker_score", "A,1.0", "",
            "# SMALL,0.1", "# BIG,-0.8", "# MID,0.3",
        })!;

        Assert.Equal(
            new[] { "BIG", "MID", "SMALL" },
            report.LoadingsByMagnitude().Select(t => t.Marker));
        // Magnitude, so the sign does not let a strongly opposing marker hide.
        Assert.Equal(0.8 / 1.2, report.LargestLoadingShare(), 6);
    }

    /// <summary>An empty or content-free file is "no report", not a report with nothing in it.</summary>
    [Fact]
    public void AFileWithNoUsableRowsReadsAsNull()
    {
        Assert.Null(MarkerNormalizationReport.Parse(Array.Empty<string>()));
        Assert.Null(MarkerNormalizationReport.Parse(new[] { "sample_id,marker_score", "" }));
    }
}
