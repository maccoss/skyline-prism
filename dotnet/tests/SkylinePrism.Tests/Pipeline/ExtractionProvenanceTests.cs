using System;
using System.IO;
using System.Text.Json;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// The extraction windows a run used, recorded so an archived result can still say what produced it.
/// </summary>
/// <remarks>
/// <para>The tolerances live in the Skyline document, and a result outlives the document as surely
/// as it outlives the instrument files. The extraction window decides how much fragment sharing is
/// found between co-isolated peptides, so every assigned figure moves with it - a directory that
/// cannot say which window it used cannot have its numbers interpreted.</para>
///
/// <para>An earlier attempt at this recorded <c>ToSetting()</c> and was inert on half the
/// instruments in the building. These tests exist mostly to pin that: the resolving-power analyzers
/// round-trip, because they are the ones a string cannot hold.</para>
/// </remarks>
public class ExtractionProvenanceTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "prism-extract-prov-" + Guid.NewGuid().ToString("N"));

    public ExtractionProvenanceTests() => Directory.CreateDirectory(_dir);

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
    /// Every analyzer round-trips, including the three <c>ToSetting()</c> returns null for.
    /// </summary>
    /// <remarks>
    /// tof, orbitrap and ft_icr have no single-number setting string, and QIT loses its selective
    /// extraction flag in one. Those are exactly the cases the previous version recorded as nothing
    /// while reading as though it worked, because the centroided case - the common one - was fine.
    /// </remarks>
    [Theory]
    [InlineData("centroided", 10.0, null, false)]
    [InlineData("qit", 0.7, null, false)]
    [InlineData("qit", 0.7, null, true)]          // ToSetting(): null
    [InlineData("tof", 30000.0, null, false)]     // ToSetting(): null
    [InlineData("orbitrap", 60000.0, 400.0, false)]   // ToSetting(): null
    [InlineData("ft_icr", 100000.0, 400.0, true)]     // ToSetting(): null
    public void EveryAnalyzerRoundTrips(
        string analyzer, double resolution, double? resolutionMz, bool selective)
    {
        var tolerance = new ProductMassTolerance(analyzer, resolution, resolutionMz, selective);
        WriteProvenance();

        Assert.True(Provenance.RecordExtraction(_dir, tolerance, tolerance, "document"));

        var (product, precursor) = Provenance.ReadExtraction(_dir);
        Assert.Equal(tolerance, product);
        Assert.Equal(tolerance, precursor);

        // The window itself is what all this is for, so compare the geometry rather than trusting
        // record equality alone.
        Assert.Equal(tolerance.WindowAt(500).Start, product!.WindowAt(500).Start, 12);
        Assert.Equal(tolerance.WindowAt(500).End, product.WindowAt(500).End, 12);
    }

    [Fact]
    public void TheCaptionIsRecordedBesideTheValuesForAHumanToRead()
    {
        WriteProvenance();
        Provenance.RecordExtraction(
            _dir, new ProductMassTolerance("orbitrap", 60000, 400), null, "document");

        using var doc = JsonDocument.Parse(
            File.ReadAllText(Path.Combine(_dir, "parameters.json")));
        var product = doc.RootElement.GetProperty("extraction").GetProperty("product");

        Assert.Equal("orbitrap", product.GetProperty("analyzer").GetString());
        Assert.Equal(60000, product.GetProperty("resolution").GetDouble());
        Assert.Equal(400, product.GetProperty("resolution_mz").GetDouble());
        Assert.False(product.GetProperty("selective_extraction").GetBoolean());
        Assert.Equal(
            "resolving power 60000 at m/z 400 (Orbitrap)", product.GetProperty("summary").GetString());
    }

    /// <summary>
    /// Additive: what <c>--from-provenance</c> reads back must not move.
    /// </summary>
    [Fact]
    public void RecordingLeavesTheProcessingParametersAlone()
    {
        var config = new PrismConfig();
        config.TransitionRollup.MinTransitions = 7;
        WriteProvenance(config);

        var before = File.ReadAllText(Path.Combine(_dir, "parameters.json"));
        Provenance.RecordExtraction(
            _dir, new ProductMassTolerance("centroided", 10), null, "document");
        var after = File.ReadAllText(Path.Combine(_dir, "parameters.json"));

        Assert.NotEqual(before, after);
        Assert.Equal(7, Provenance.LoadConfig(
            Path.Combine(_dir, "parameters.json"))!.TransitionRollup.MinTransitions);
    }

    [Fact]
    public void RecordingTheSameSettingsTwiceIsNotAChange()
    {
        WriteProvenance();
        var tolerance = new ProductMassTolerance("centroided", 10);

        Assert.True(Provenance.RecordExtraction(_dir, tolerance, tolerance, "document"));
        Assert.False(Provenance.RecordExtraction(_dir, tolerance, tolerance, "document"));

        // A genuinely different window is a change, or a re-run at a new tolerance would go
        // unrecorded while every figure in the directory moved.
        Assert.True(Provenance.RecordExtraction(
            _dir, new ProductMassTolerance("centroided", 20), tolerance, "document"));
    }

    /// <summary>Nothing to say, nowhere to say it, and an unreadable file are all non-fatal.</summary>
    [Fact]
    public void RecordingIsNeverFatal()
    {
        WriteProvenance();
        Assert.False(Provenance.RecordExtraction(_dir, null, null, "document"));

        var empty = Path.Combine(_dir, "empty");
        Directory.CreateDirectory(empty);
        Assert.False(Provenance.RecordExtraction(
            empty, new ProductMassTolerance("centroided", 10), null));
        Assert.Equal((null, null), Provenance.ReadExtraction(empty));

        File.WriteAllText(Path.Combine(empty, "parameters.json"), "{ not json");
        Assert.False(Provenance.RecordExtraction(
            empty, new ProductMassTolerance("centroided", 10), null));
        Assert.Equal((null, null), Provenance.ReadExtraction(empty));
    }

    [Fact]
    public void ADirectoryThatRecordedNothingReadsBackAsNothing()
    {
        WriteProvenance();
        Assert.Equal((null, null), Provenance.ReadExtraction(_dir));
    }

    private void WriteProvenance(PrismConfig? config = null) =>
        Provenance.Write(
            Path.Combine(_dir, "parameters.json"), config ?? new PrismConfig(),
            new[] { "report.csv" }, new Provenance.Stats(1, 10, 5, 5),
            "2026-01-01T00:00:00.0000000Z");
}
