using System;
using System.IO;
using System.Linq;
using System.Threading;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The input-list logic behind the tool's Inputs tab. Batch labels are the load-bearing part: each one
/// becomes the exported report's FILE STEM, and <c>DuckDbMerge</c> reads that stem back as the input's
/// Source Document / Batch. So a label that is duplicated, empty, or not file-name safe silently merges
/// two documents into one batch (skipping ComBat) or breaks the export outright.
/// </summary>
public class PrismInputTests
{
    private static string TempFile(string name, string content = "a,b\n1,2\n")
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_input_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, name);
        File.WriteAllText(path, content);
        return path;
    }

    [Fact]
    public void SanitizeLabel_ReplacesCharactersThatAreIllegalInAFileName()
    {
        var sanitized = PrismInput.SanitizeLabel("Plate:01/A*B?");

        Assert.DoesNotContain(sanitized, ch => Path.GetInvalidFileNameChars().Contains(ch));
        Assert.Equal("Plate_01_A_B_", sanitized);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void SanitizeLabel_FallsBackForBlankInput(string? input)
        => Assert.Equal("batch", PrismInput.SanitizeLabel(input));

    [Fact]
    public void SanitizeLabel_TrimsButOtherwiseLeavesGoodLabelsAlone()
        => Assert.Equal("Plate_07", PrismInput.SanitizeLabel("  Plate_07  "));

    [Fact]
    public void EnsureUniqueLabels_DisambiguatesDuplicates()
    {
        // Two documents of the same name in different folders is completely normal.
        var inputs = new[]
        {
            PrismInput.FromReportFile(TempFile("PRISM.csv")),
            PrismInput.FromReportFile(TempFile("PRISM.csv")),
            PrismInput.FromReportFile(TempFile("PRISM.csv")),
        };

        PrismInput.EnsureUniqueLabels(inputs);

        Assert.Equal(new[] { "PRISM", "PRISM_2", "PRISM_3" }, inputs.Select(i => i.BatchLabel));
        Assert.Equal(3, inputs.Select(i => i.BatchLabel).Distinct().Count());
    }

    [Fact]
    public void EnsureUniqueLabels_TreatsCaseInsensitiveCollisionsAsCollisions()
    {
        // The labels become file names, and Windows file names are case-insensitive.
        var inputs = new[]
        {
            PrismInput.FromReportFile(TempFile("plate.csv")),
            PrismInput.FromReportFile(TempFile("PLATE.csv")),
        };

        PrismInput.EnsureUniqueLabels(inputs);

        Assert.NotEqual(
            inputs[0].BatchLabel.ToLowerInvariant(), inputs[1].BatchLabel.ToLowerInvariant());
    }

    [Fact]
    public void EnsureUniqueLabels_SanitizesUserEditedLabels()
    {
        var inputs = new[] { PrismInput.FromReportFile(TempFile("a.csv")) };
        inputs[0].BatchLabel = "Plate 1: run/2"; // typed into the grid

        PrismInput.EnsureUniqueLabels(inputs);

        Assert.DoesNotContain(inputs[0].BatchLabel, ch => Path.GetInvalidFileNameChars().Contains(ch));
    }

    [Fact]
    public void EnsureUniqueLabels_IsIdempotent()
    {
        var inputs = new[]
        {
            PrismInput.FromReportFile(TempFile("PRISM.csv")),
            PrismInput.FromReportFile(TempFile("PRISM.csv")),
        };

        PrismInput.EnsureUniqueLabels(inputs);
        var first = inputs.Select(i => i.BatchLabel).ToArray();
        PrismInput.EnsureUniqueLabels(inputs); // e.g. a second Run click

        Assert.Equal(first, inputs.Select(i => i.BatchLabel));
    }

    [Fact]
    public void FromReportFile_DefaultsTheLabelToTheFileStem()
    {
        var input = PrismInput.FromReportFile(TempFile("Plate_07.parquet"));

        Assert.Equal("Plate_07", input.BatchLabel);
        Assert.Equal(PrismInputKind.ReportFile, input.Kind);
    }

    [Fact]
    public void FromClosedDocument_UsesTheDocumentNameAndKeepsTheFullPath()
    {
        var sky = TempFile("MyStudy.sky", "<srm_settings/>");

        var input = PrismInput.FromClosedDocument(sky);

        Assert.Equal(PrismInputKind.ClosedDocument, input.Kind);
        Assert.Equal("MyStudy", input.BatchLabel);
        Assert.Equal(Path.GetFullPath(sky), input.Path);
    }

    [Fact]
    public void Prepare_ForAReportFile_UsesItInPlaceWithoutCopying()
    {
        var report = TempFile("Plate_07.parquet");
        var metadata = TempFile("Plate_07.metadata.csv", "Replicate,SampleType\nR1,Standard\n");
        var input = PrismInput.FromReportFile(report, metadata);
        var workDir = Path.Combine(Path.GetTempPath(), "prism_prep_" + Guid.NewGuid().ToString("N"));

        var result = input.Prepare(workDir, null, null, null, _ => { }, CancellationToken.None);

        Assert.Equal(report, result.InputPath); // read in place - no re-export, no copy
        Assert.Equal(metadata, result.ReplicatesCsv);
        Assert.True(result.InputIsParquet); // .parquet extension recognized
        Assert.Equal("Plate_07", result.DocumentLabel);
    }

    [Fact]
    public void Prepare_ForACsvReportFile_ReportsItAsNotParquet()
    {
        var input = PrismInput.FromReportFile(TempFile("Plate.csv"));

        var result = input.Prepare(
            Path.GetTempPath(), null, null, null, _ => { }, CancellationToken.None);

        Assert.False(result.InputIsParquet);
    }

    [Fact]
    public void Prepare_ForAMissingReportFile_Throws()
    {
        var report = TempFile("gone.csv");
        var input = PrismInput.FromReportFile(report);
        File.Delete(report);

        Assert.Throws<FileNotFoundException>(() => input.Prepare(
            Path.GetTempPath(), null, null, null, _ => { }, CancellationToken.None));
    }

    [Fact]
    public void Prepare_HonoursACancelledToken()
    {
        var input = PrismInput.FromReportFile(TempFile("a.csv"));
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        Assert.Throws<OperationCanceledException>(() => input.Prepare(
            Path.GetTempPath(), null, null, null, _ => { }, cts.Token));
    }

    [Fact]
    public void Prepare_UsesTheEditedLabel_NotTheOriginalFileName()
    {
        var input = PrismInput.FromReportFile(TempFile("PRISM.parquet"));
        input.BatchLabel = "Plate_B"; // user renamed it in the grid

        var result = input.Prepare(
            Path.GetTempPath(), null, null, null, _ => { }, CancellationToken.None);

        Assert.Equal("Plate_B", result.DocumentLabel);
    }

    [Fact]
    public void TryGetDigestionEnzyme_ReadsItFromAClosedDocument()
    {
        var sky = TempFile("Doc.sky", """
<?xml version="1.0" encoding="utf-8"?>
<srm_settings format_version="25.1">
  <settings_summary name="Default">
    <peptide_settings><enzyme name="LysC" cut="K" no_cut="" sense="C" /></peptide_settings>
  </settings_summary>
</srm_settings>
""");

        Assert.Equal("lysc", PrismInput.FromClosedDocument(sky).TryGetDigestionEnzyme(_ => { }));
    }

    [Fact]
    public void TryGetDigestionEnzyme_ReturnsNullForAPlainReportFile()
        => Assert.Null(PrismInput.FromReportFile(TempFile("a.csv")).TryGetDigestionEnzyme(_ => { }));

    [Fact]
    public void KindLabel_IsHumanReadableForEachSource()
    {
        Assert.Equal("Report file", PrismInput.FromReportFile(TempFile("a.csv")).KindLabel);
        Assert.Equal("Skyline document", PrismInput.FromClosedDocument(TempFile("a.sky", "<x/>")).KindLabel);
    }
}
