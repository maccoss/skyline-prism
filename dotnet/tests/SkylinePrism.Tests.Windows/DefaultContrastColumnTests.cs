using System.Collections.Generic;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The column the Differential and Markers panes group by when they first open a run.
/// </summary>
/// <remarks>
/// <c>sample_metadata.csv</c> gained <c>acquired_time</c> and <c>run_order</c> straight after the
/// resolved columns, and both are unique per run. The default was the first non-reserved column with
/// two values, so on any document with Skyline's Acquired Time it became <c>acquired_time</c>: the
/// Differential pane opened on "Each group needs at least 2 samples (A has 1, B has 1)", reset to
/// it on every level change, and the Markers pane grouped by run ("across 90 groups"). Found by
/// launching the tool on a 90-run BioFIND plate, where the default is Skyline's SampleType.
/// </remarks>
public class DefaultContrastColumnTests
{
    private static string? Pick(params (string Column, string?[] Values)[] cols)
    {
        var byName = new Dictionary<string, string?[]>();
        var names = new List<string>();
        foreach (var (c, v) in cols)
        {
            names.Add(c);
            byName[c] = v;
        }
        return MainWindow.DefaultContrastColumn(names, c => byName[c]);
    }

    [Fact]
    public void SkipsTheAcquisitionColumns_ForTheFirstAnnotation()
    {
        var col = Pick(
            ("sample", new[] { "a", "b", "c", "d" }),
            ("sample_type", new[] { "qc", "qc", "experimental", "experimental" }),
            ("batch", new[] { "p1", "p1", "p2", "p2" }),
            ("acquired_time", new[] { "2026-09-23 01:34:37", "2026-09-23 02:03:36", "2026-09-23 02:32:37", "2026-09-23 03:01:58" }),
            ("run_order", new[] { "1", "2", "3", "4" }),
            ("Sample_Category", new[] { "BioFIND", "BioFIND", "Pool", "Pool" }));

        Assert.Equal("Sample_Category", col);
    }

    /// <summary>A column of identifiers - a well, a barcode - cannot form an arm of two.</summary>
    [Fact]
    public void SkipsAColumnWithNoValueSharedByTwoSamples()
    {
        var col = Pick(
            ("Well", new[] { "A1", "A2", "A3", "A4" }),
            ("Visit", new[] { "V01", "V01", "V02", "V03" }));

        Assert.Equal("Visit", col);
    }

    /// <summary>Two blanks are two samples with no value, not a group.</summary>
    [Fact]
    public void BlanksAreNotASharedValue()
    {
        var col = Pick(
            ("Tube_Note", new[] { "", "", null, "cracked", "relabeled" }),
            ("Visit", new[] { "V01", "V01", "V02", "V02", "V02" }));

        Assert.Equal("Visit", col);
    }

    [Fact]
    public void FallsBackToSampleType_WhenNoAnnotationQualifies()
    {
        var col = Pick(
            ("sample", new[] { "a", "b", "c" }),
            ("sample_type", new[] { "qc", "qc", "experimental" }),
            ("run_order", new[] { "1", "2", "3" }),
            ("Plate", new[] { "1", "1", "1" }),
            ("Well", new[] { "A1", "A2", "A3" }));

        Assert.Equal("sample_type", col);
    }
}
