using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using SkylinePrism.Core.Library;
using SkylinePrism.Core.Rollup;
using SkylinePrism.Tests.DifferentialAnalysis;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Library;

/// <summary>
/// Library assist's noise-gated ratio fit, its fragment-noise curve, and the per-peptide precision
/// check, against <c>fixtures/library_assist/noise_gated.json</c>.
///
/// <para>The golden is a first-principles construction - there is no outside implementation of this
/// method - written from the definitions in <c>generate.py</c>'s docstring with plain Python loops.
/// Tolerance 1e-12 relative: every quantity is closed-form arithmetic summed in the same order, so
/// agreement is to rounding, and a looser number would hide a real difference.</para>
/// </summary>
public class NoiseGatedRatioGoldenTests
{
    private const double Rtol = 1e-12;
    private const string File = "noise_gated.json";

    private static JsonElement Load()
    {
        var path = Fixtures.Path2("library_assist", File);
        Assert.True(System.IO.File.Exists(path),
            $"Missing golden fixture {path}. Regenerate with "
            + "'uv run dotnet/tests/fixtures/library_assist/generate.py' from the repository root.");
        using var doc = JsonDocument.Parse(System.IO.File.ReadAllText(path));
        return doc.RootElement.Clone();
    }

    private static JsonElement Case(string section, string name) =>
        Load().GetProperty(section).EnumerateArray().Single(c => c.GetProperty("name").GetString() == name);

    private static IEnumerable<object[]> Names(string section) =>
        Load().GetProperty(section).EnumerateArray().Select(c => new object[] { c.GetProperty("name").GetString()! });

    public static IEnumerable<object[]> FitCases() => Names("fit_cases");
    public static IEnumerable<object[]> NoiseCases() => Names("noise_cases");
    public static IEnumerable<object[]> PrecisionCases() => Names("precision_cases");

    [Fact]
    public void Golden_NamesItsReference()
    {
        var reference = Load().GetProperty("reference").GetString();
        Assert.Contains("first-principles", reference);
    }

    [Theory]
    [MemberData(nameof(FitCases))]
    public void NoiseGatedRatio_MatchesGolden(string name)
    {
        var c = Case("fit_cases", name);
        var observed = Golden.Mat(c, "observed");
        var lib = Golden.Vec(c, "library");
        double[,]? retainedSource = c.GetProperty("retained_source").ValueKind == JsonValueKind.Null
            ? null
            : Golden.Mat(c, "retained_source");
        FragmentNoiseCurve? curve = null;
        if (c.GetProperty("curve").ValueKind != JsonValueKind.Null)
        {
            var cv = c.GetProperty("curve");
            curve = FragmentNoiseCurve.FromPoints(Golden.Vec(cv, "level"), Golden.Vec(cv, "sd"));
        }

        var fit = LibraryRollup.NoiseGatedRatio(
            observed, lib,
            minFragments: c.GetProperty("min_fragments").GetInt32(),
            outlierThreshold: Golden.Num(c, "threshold"),
            noiseSigma: Golden.Num(c, "sigma"),
            fragmentSd: curve is null ? null : curve.SdAt,
            removeOutliers: c.GetProperty("remove").GetBoolean(),
            retainedSource: retainedSource);

        var e = c.GetProperty("expected");
        Golden.CloseAll(Golden.Vec(e, "abundance"), fit.Abundance, Rtol, $"{name} abundance");
        Golden.CloseAll(Golden.Vec(e, "retained"), fit.Retained, Rtol, $"{name} retained");
        Assert.Equal(e.GetProperty("kept").EnumerateArray().Select(x => x.GetInt32()).ToArray(), fit.Kept);
    }

    [Theory]
    [MemberData(nameof(NoiseCases))]
    public void FragmentNoiseEstimate_MatchesGolden(string name)
    {
        var c = Case("noise_cases", name);
        var blocks = new List<PeptideBlock>();
        PeptideBlock? current = null;
        foreach (var row in c.GetProperty("rows").EnumerateArray())
        {
            var r = row.EnumerateArray().ToArray();
            var pep = r[0].GetString()!;
            if (current is null || current.Peptide != pep)
            {
                current = new PeptideBlock { Peptide = pep };
                blocks.Add(current);
            }
            current.TransitionId.Add(r[1].GetString()!);
            current.PrecursorCharge.Add(r[2].GetInt32());
            current.SampleIndex.Add(r[3].GetInt32());
            current.Area.Add(Golden.Num(r[4]));
            current.IsPrecursor.Add(r[5].GetBoolean());
            current.RetentionTime.Add(0.0);
        }
        var groups = c.GetProperty("groups").EnumerateArray()
            .Select(g => new NoiseGroup(
                g.GetProperty("name").GetString()!,
                g.GetProperty("indices").EnumerateArray().Select(i => i.GetInt32()).ToArray()))
            .ToList();
        var o = c.GetProperty("options");
        var options = new FragmentNoiseOptions(
            FineWidthLog10: Golden.Num(o, "fine_width"),
            TargetBins: o.GetProperty("target_bins").GetInt32(),
            MinCellsPerBin: o.GetProperty("min_cells_per_bin").GetInt32(),
            MinSignalPointsPerBin: o.GetProperty("min_signal_per_bin").GetInt32());

        var curve = FragmentNoise.Estimate(
            blocks, c.GetProperty("n_samples").GetInt32(), groups,
            c.GetProperty("source").GetString()!, c.GetProperty("exclude_precursor").GetBoolean(), options);

        var e = c.GetProperty("expected");
        Assert.Equal(e.GetProperty("cells").GetInt64(), curve.Cells);
        Golden.CloseAll(Golden.Vec(e, "noise_level"), curve.NoiseLevel, Rtol, $"{name} noise level");
        Golden.CloseAll(Golden.Vec(e, "noise_sd"), curve.NoiseSd, Rtol, $"{name} noise sd");
        Golden.CloseAll(Golden.Vec(e, "signal_level"), curve.SignalLevel, Rtol, $"{name} signal level");
        Golden.CloseAll(Golden.Vec(e, "signal_cv"), curve.SignalCv, Rtol, $"{name} signal cv");
        Assert.Equal(c.GetProperty("source").GetString(), curve.Source);
    }

    [Fact]
    public void CurveLookups_MatchGolden()
    {
        var c = Load().GetProperty("interp_cases").EnumerateArray().Single();
        var noise = c.GetProperty("noise");
        var signal = c.GetProperty("signal");
        var curve = FragmentNoiseCurve.FromPoints(
            Golden.Vec(noise, "level"), Golden.Vec(noise, "sd"),
            Golden.Vec(signal, "level"), Golden.Vec(signal, "cv"));

        var sdQ = Golden.Vec(c, "sd_queries");
        Golden.CloseAll(Golden.Vec(c, "sd_expected"), sdQ.Select(curve.SdAt).ToArray(), Rtol, "sd");
        var cvQ = Golden.Vec(c, "cv_queries");
        Golden.CloseAll(Golden.Vec(c, "cv_expected"), cvQ.Select(curve.ExpectedCvAt).ToArray(), Rtol, "cv");
    }

    [Theory]
    [MemberData(nameof(PrecisionCases))]
    public void PrecisionRow_MatchesGolden(string name)
    {
        var c = Case("precision_cases", name);
        var cv = c.GetProperty("curve");
        var curve = FragmentNoiseCurve.FromPoints(
            Array.Empty<double>(), Array.Empty<double>(), Golden.Vec(cv, "level"), Golden.Vec(cv, "cv"));
        var indices = c.GetProperty("indices").EnumerateArray().Select(i => i.GetInt32()).ToArray();

        var row = LibraryAssistPrecision.Row(
            "group", indices,
            Golden.Vec(c, "abundance"), Golden.Vec(c, "retained"),
            c.GetProperty("kept").EnumerateArray().Select(k => (double)k.GetInt32()).ToArray(),
            Golden.Vec(c, "signal"), curve);

        var e = c.GetProperty("expected");
        Golden.Close(Golden.Num(e, "observed_cv"), row.ObservedCv, Rtol, $"{name} observed");
        Golden.Close(Golden.Num(e, "expected_cv"), row.ExpectedCv, Rtol, $"{name} expected");
        Golden.Close(Golden.Num(e, "ratio"), row.Ratio, Rtol, $"{name} ratio");
        Golden.Close(Golden.Num(e, "retained_mean"), row.RetainedMean, Rtol, $"{name} retained");
        Golden.Close(Golden.Num(e, "kept_mean"), row.KeptMean, Rtol, $"{name} kept");
        Golden.Close(Golden.Num(e, "signal_mean"), row.SignalMean, Rtol, $"{name} signal");
        Assert.Equal(indices.Length, row.Injections);
    }
}
