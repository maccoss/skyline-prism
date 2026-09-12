using System;
using System.IO;
using System.Text.Json;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Writing down what only a Skyline document can say, so a later run never has to ask for it.
///
/// <para>The extraction tolerances live in <c>&lt;transition_full_scan&gt;</c> and the instrument
/// file paths in <c>&lt;sample_file&gt;</c>. A pre-exported report carries neither - including the
/// report PRISM itself writes into <c>skyline-reports/</c>, so re-running ion accounting against a
/// previous run's own export had to ask the user for a tolerance the first run already knew and
/// threw away.</para>
/// </summary>
public class AcquisitionProvenanceTests
{
    private static string RunDirectory()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_acq_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        Provenance.Write(
            Path.Combine(dir, Provenance.FileName), new PrismConfig(), Array.Empty<string>(),
            new Provenance.Stats(4, 100, 20, 20), "2026-09-12T04:00:00Z");
        return dir;
    }

    /// <summary>THE case: what one run learned, the next run reads back.</summary>
    [Fact]
    public void WhatTheDocumentSaidSurvivesForTheNextRun()
    {
        var dir = RunDirectory();
        try
        {
            Assert.True(Provenance.RecordAcquisition(
                dir, "+/-10 ppm (centroided)", "+/-10 ppm (centroided)", @"R:\cohort"));

            var (product, precursor, files) = Provenance.ReadAcquisition(dir);
            Assert.Equal("+/-10 ppm (centroided)", product);
            Assert.Equal("+/-10 ppm (centroided)", precursor);
            Assert.Equal(@"R:\cohort", files);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// Recorded BESIDE the processing parameters, never inside them. A tolerance is a fact about the
    /// acquisition, not a setting of this pipeline, and --from-provenance must not replay it as one.
    /// </summary>
    [Fact]
    public void ItDoesNotLeakIntoTheReplayableConfig()
    {
        var dir = RunDirectory();
        try
        {
            var path = Path.Combine(dir, Provenance.FileName);
            var before = Provenance.LoadConfig(path);
            Provenance.RecordAcquisition(dir, "+/-10 ppm", "+/-10 ppm", @"R:\cohort");

            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            Assert.True(doc.RootElement.TryGetProperty("acquisition", out _));
            var parameters = doc.RootElement.GetProperty("processing_parameters");
            Assert.False(parameters.TryGetProperty("acquisition", out _));
            Assert.False(parameters.TryGetProperty("product_tolerance", out _));

            var after = Provenance.LoadConfig(path);
            Assert.Equal(before.TransitionRollup.Method, after.TransitionRollup.Method);
            Assert.Equal(before.Parsimony.Enzyme, after.Parsimony.Enzyme);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>Partial knowledge is still worth keeping - a tolerance with no reachable folder.</summary>
    [Fact]
    public void WhatIsKnownIsRecordedEvenWhenTheRestIsNot()
    {
        var dir = RunDirectory();
        try
        {
            Assert.True(Provenance.RecordAcquisition(dir, "+/-10 ppm", null, null));

            var (product, precursor, files) = Provenance.ReadAcquisition(dir);
            Assert.Equal("+/-10 ppm", product);
            Assert.Null(precursor);
            Assert.Null(files);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>Nothing to record writes nothing - an empty block would read as "asked and unknown".</summary>
    [Fact]
    public void NothingKnownWritesNothing()
    {
        var dir = RunDirectory();
        try
        {
            Assert.False(Provenance.RecordAcquisition(dir, null, null, null));
            Assert.False(
                JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, Provenance.FileName)))
                    .RootElement.TryGetProperty("acquisition", out _));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>Re-recording the same facts is not a change, so a re-run does not churn the file.</summary>
    [Fact]
    public void RecordingTheSameFactsTwiceIsNotAChange()
    {
        var dir = RunDirectory();
        try
        {
            Assert.True(Provenance.RecordAcquisition(dir, "+/-10 ppm", "+/-10 ppm", @"R:\cohort"));
            Assert.False(Provenance.RecordAcquisition(dir, "+/-10 ppm", "+/-10 ppm", @"R:\cohort"));
            Assert.True(Provenance.RecordAcquisition(dir, "+/-20 ppm", "+/-10 ppm", @"R:\cohort"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// A directory that is not a finished run records nothing and is not turned into one - inventing
    /// a parameters.json there would make it look like a run to every reader that checks.
    /// </summary>
    [Fact]
    public void ADirectoryWithNoRunIsLeftAlone()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_acq_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            Assert.False(Provenance.RecordAcquisition(dir, "+/-10 ppm", null, @"R:\cohort"));
            Assert.False(File.Exists(Path.Combine(dir, Provenance.FileName)));
            Assert.Equal((null, null, null), Provenance.ReadAcquisition(dir));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
