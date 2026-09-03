using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The exact m/z range Skyline extracted each product ion over, which is what decides whether two
/// fragments are the same detector signal.
///
/// <para>The arithmetic is transcribed from Skyline - <c>TransitionSettings.GetDenominator</c> /
/// <c>GetFilterWindow</c>, applied as <c>targetMz +/- filterWindow/2</c> in
/// <c>SpectrumFilterPair.cs</c> - so these tests check it against windows worked out independently
/// from what each setting MEANS: <c>product_res</c> is the +/- tolerance, so 10 ppm extracts
/// +/-10 ppm and 0.7 m/z extracts +/-0.7 m/z. A transcription slip of a factor of two would change
/// how much fragment sharing is found, in a direction nobody would notice by looking at a plot.</para>
/// </summary>
public class ProductMassToleranceTests
{
    /// <summary>
    /// The case the Levitt documents use: <c>product_mass_analyzer="centroided" product_res="10"</c>.
    /// 10 ppm at m/z 1000 is +/-0.01 m/z.
    /// </summary>
    [Fact]
    public void CentroidedResIsThePlusMinusPpm()
    {
        var t = ProductMassTolerance.Parse("centroided", "10");
        Assert.NotNull(t);

        var w = t!.WindowAt(1000);
        Assert.Equal(999.99, w.Start, 9);
        Assert.Equal(1000.01, w.End, 9);
        Assert.Equal(1000.0, w.Center, 9);

        // ...and it scales with m/z, which is what "ppm" means.
        Assert.Equal(10.0, t.WindowAt(1000).Width / 2 / 1000 * 1e6, 9);
        Assert.Equal(10.0, t.WindowAt(200).Width / 2 / 200 * 1e6, 9);
    }

    /// <summary>
    /// QIT states a fixed tolerance in m/z, so <c>product_res="0.7"</c> extracts anything within
    /// 0.7 m/z of the predicted product m/z - a 1.4 m/z range in total. Pinned because the number in
    /// the document is 0.7 and it is natural to read that as the whole window, which would halve the
    /// sharing found. A 0.7 m/z total window is written as <c>product_res="0.35"</c>.
    /// </summary>
    [Fact]
    public void QitResIsThePlusMinusMz()
    {
        var t = ProductMassTolerance.Parse("qit", "0.7")!;

        var w = t.WindowAt(1000);
        Assert.Equal(999.3, w.Start, 9);
        Assert.Equal(1000.7, w.End, 9);

        // ...and it does not vary with mass, which is what distinguishes it from TOF.
        Assert.Equal(t.WindowAt(200).Width, t.WindowAt(2000).Width, 12);

        // The settings a QIT product-ion method would actually use, stated as the +/- tolerance.
        Assert.Equal(0.8, ProductMassTolerance.Parse("qit", "0.4")!.WindowAt(500).Width, 12);
        Assert.Equal(0.7, ProductMassTolerance.Parse("qit", "0.35")!.WindowAt(500).Width, 12);
    }

    /// <summary>
    /// TOF states a resolving power, so the peak at m/z 1000 and R = 30,000 is 1000/30000 wide and
    /// Skyline extracts +/- one of those (RES_PER_FILTER = 2 makes the total range two peak widths).
    /// </summary>
    [Fact]
    public void TofResIsResolvingPowerAndExtractsPlusMinusOnePeakWidth()
    {
        var t = ProductMassTolerance.Parse("tof", "30000");
        Assert.NotNull(t);

        var peakWidth = 1000.0 / 30000.0;
        var w = t!.WindowAt(1000);
        Assert.Equal(1000 - peakWidth, w.Start, 12);
        Assert.Equal(1000 + peakWidth, w.End, 12);
    }

    /// <summary>Selective extraction halves it - Skyline's RES_PER_FILTER_SELECTIVE = 1.</summary>
    [Fact]
    public void SelectiveExtractionHalvesTheResolvingPowerWindow()
    {
        var normal = ProductMassTolerance.Parse("tof", "30000", null, "false")!;
        var selective = ProductMassTolerance.Parse("tof", "30000", null, "true")!;

        Assert.Equal(normal.WindowAt(1000).Width / 2, selective.WindowAt(1000).Width, 12);

        // ...but centroided ppm is unaffected: its denominator does not use ResPerFilter.
        var ppm = ProductMassTolerance.Parse("centroided", "10", null, "true")!;
        Assert.Equal(0.02, ppm.WindowAt(1000).Width, 12);
    }

    /// <summary>
    /// Orbitrap resolving power is calibrated at a stated m/z and the window widens as m/z^1.5 - so it
    /// is NOT the same as TOF at the same nominal R, and treating them alike would be wrong by a
    /// growing factor across the mass range.
    /// </summary>
    [Fact]
    public void OrbitrapUsesTheCalibrationMzAndWidensFasterThanTof()
    {
        var orbi = ProductMassTolerance.Parse("orbitrap", "30000", "400")!;
        var tof = ProductMassTolerance.Parse("tof", "30000")!;

        Assert.True(orbi.WindowAt(1000).Width > 0);
        Assert.NotEqual(tof.WindowAt(1000).Width, orbi.WindowAt(1000).Width, 6);

        // m/z^1.5 scaling: doubling m/z multiplies the window by 2^1.5.
        var ratio = orbi.WindowAt(1000).Width / orbi.WindowAt(500).Width;
        Assert.Equal(System.Math.Pow(2, 1.5), ratio, 9);
    }

    /// <summary>
    /// The overlap test is what the union accounting is built on. Strict at the boundary, matching the
    /// retention-time rule: ranges that merely touch share no counts.
    /// </summary>
    [Fact]
    public void WindowsOverlapOnlyWhenTheyShareMz()
    {
        var t = ProductMassTolerance.Parse("centroided", "10")!;   // +/-0.01 at m/z 1000

        Assert.True(t.WindowAt(1000.000).Overlaps(t.WindowAt(1000.015)));   // ranges intersect
        Assert.False(t.WindowAt(1000.000).Overlaps(t.WindowAt(1000.030)));  // clear of each other

        // Exactly touching, so nothing is shared. QIT because its width does not vary with m/z:
        // [999.5, 1000.5] against [1000.5, 1001.5]. A ppm window cannot touch exactly - it widens
        // with mass, so the upper of two adjacent windows always starts just inside the lower one.
        var qit = ProductMassTolerance.Parse("qit", "0.5")!;
        Assert.False(qit.WindowAt(1000.0).Overlaps(qit.WindowAt(1001.0)));
        Assert.True(qit.WindowAt(1000.0).Overlaps(qit.WindowAt(1000.9)));

        // Symmetric, and every window overlaps itself.
        Assert.True(t.WindowAt(1000.0).Overlaps(t.WindowAt(1000.0)));
        Assert.Equal(
            t.WindowAt(1000.000).Overlaps(t.WindowAt(1000.015)),
            t.WindowAt(1000.015).Overlaps(t.WindowAt(1000.000)));
    }

    /// <summary>
    /// A document that does not say enough yields null rather than a plausible wrong number: orbitrap
    /// and ft_icr are meaningless without the calibration m/z, and Skyline itself refuses them.
    /// </summary>
    [Fact]
    public void UninterpretableSettingsGiveNull()
    {
        Assert.Null(ProductMassTolerance.Parse(null, "10"));
        Assert.Null(ProductMassTolerance.Parse("centroided", null));
        Assert.Null(ProductMassTolerance.Parse("centroided", "not a number"));
        Assert.Null(ProductMassTolerance.Parse("centroided", "0"));
        Assert.Null(ProductMassTolerance.Parse("orbitrap", "30000"));          // no res_mz
        Assert.Null(ProductMassTolerance.Parse("ft_icr", "30000"));            // no res_mz
        Assert.NotNull(ProductMassTolerance.Parse("orbitrap", "30000", "400"));
    }

    /// <summary>
    /// These are XML attributes, not display text. Parsing them under a comma-decimal culture would
    /// read "0.7" as 7 and silently widen the window tenfold.
    /// </summary>
    [Fact]
    public void ParsesInvariantCultureRegardlessOfTheCurrentOne()
    {
        var previous = System.Globalization.CultureInfo.CurrentCulture;
        try
        {
            System.Globalization.CultureInfo.CurrentCulture =
                new System.Globalization.CultureInfo("de-DE");
            var t = ProductMassTolerance.Parse("qit", "0.7")!;
            // +/-0.7, not +/-7: a comma-decimal parse would read "0.7" as 7 and widen this tenfold.
            Assert.Equal(999.3, t.WindowAt(1000).Start, 9);
            Assert.Equal(1000.7, t.WindowAt(1000).End, 9);
        }
        finally
        {
            System.Globalization.CultureInfo.CurrentCulture = previous;
        }
    }

    /// <summary>
    /// The description reaches a log line and a plot caption, so it names the unit - and says +/-
    /// where that is a single number, because that is what the document's setting means.
    /// </summary>
    [Fact]
    public void DescribeNamesTheUnitAndThePlusMinus()
    {
        Assert.Equal("+/-10 ppm (centroided)", ProductMassTolerance.Parse("centroided", "10")!.Describe());
        Assert.Equal("+/-0.7 m/z (QIT)", ProductMassTolerance.Parse("qit", "0.7")!.Describe());
        Assert.Contains("resolving power", ProductMassTolerance.Parse("tof", "30000")!.Describe());
    }
    /// <summary>
    /// The Skyline tool hands a document's extraction setting to the config as text, so the text must
    /// mean exactly the tolerance it came from - a rounding in the spelling would change how much
    /// fragment sharing the accounting finds, with the plot naming a tolerance that was never in force.
    /// </summary>
    [Fact]
    public void ToSettingRoundTripsThroughParseSetting()
    {
        foreach (var (analyzer, res) in new[] { ("centroided", "10"), ("qit", "0.7"), ("centroided", "12.5"), ("qit", "0.3") })
        {
            var t = ProductMassTolerance.Parse(analyzer, res)!;
            var setting = t.ToSetting();
            Assert.NotNull(setting);
            Assert.Equal(t, ProductMassTolerance.ParseSetting(setting));
        }

        Assert.Equal("10 ppm", ProductMassTolerance.Parse("centroided", "10")!.ToSetting());
        Assert.Equal("0.7 m/z", ProductMassTolerance.Parse("qit", "0.7")!.ToSetting());
    }

    /// <summary>
    /// A resolving-power window is not one +/- number, so there is no setting to write. Null, not a
    /// nearest-ppm guess that would be right at a single m/z.
    /// </summary>
    [Fact]
    public void ToSettingIsNullForResolvingPowerAnalyzers()
    {
        Assert.Null(ProductMassTolerance.Parse("tof", "30000")!.ToSetting());
        Assert.Null(ProductMassTolerance.Parse("orbitrap", "60000", "400")!.ToSetting());
        Assert.Null(ProductMassTolerance.Parse("ft_icr", "100000", "400")!.ToSetting());
    }

    /// <summary>
    /// Selective extraction HALVES a QIT window (it divides Skyline's default arm, which is the one
    /// QIT takes), and the setting has no way to say so - "0.7 m/z" would come back meaning +/-0.7
    /// where the document extracted +/-0.35. Twice too wide merges fragments that never shared
    /// detector counts, while the caption names the document's own number, so there is nothing on the
    /// plot to notice. Null instead, which the caller already handles by keeping the configured value
    /// and saying why.
    /// </summary>
    [Fact]
    public void ToSettingRefusesASelectiveExtractionQitWindow()
    {
        var selective = ProductMassTolerance.Parse("qit", "0.7", selectiveExtraction: "true")!;
        var plain = ProductMassTolerance.Parse("qit", "0.7")!;

        // The premise: the two really are different windows, so one string cannot serve both.
        Assert.Equal(0.35, selective.WindowAt(1000).Width / 2, 9);
        Assert.Equal(0.70, plain.WindowAt(1000).Width / 2, 9);

        Assert.Null(selective.ToSetting());
        Assert.Equal("0.7 m/z", plain.ToSetting());
    }

    /// <summary>
    /// Centroided is the one analyzer selective extraction does not move (its denominator has no
    /// ResPerFilter), so the setting still expresses it - and what matters is that the round trip
    /// preserves the WINDOW, not that it preserves the flag.
    /// </summary>
    [Fact]
    public void ToSettingKeepsTheWindowForASelectiveCentroidedTolerance()
    {
        var selective = ProductMassTolerance.Parse("centroided", "10", selectiveExtraction: "true")!;

        var setting = selective.ToSetting();
        Assert.Equal("10 ppm", setting);

        var reparsed = ProductMassTolerance.ParseSetting(setting)!;
        foreach (var mz in new[] { 200.0, 500.0, 1000.0 })
        {
            Assert.Equal(selective.WindowAt(mz).Start, reparsed.WindowAt(mz).Start, 9);
            Assert.Equal(selective.WindowAt(mz).End, reparsed.WindowAt(mz).End, 9);
        }
    }
}
