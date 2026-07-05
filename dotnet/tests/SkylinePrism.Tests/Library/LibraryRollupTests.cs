using SkylinePrism.Core.Library;
using Xunit;

namespace SkylinePrism.Tests.Library;

/// <summary>
/// Library-assisted rollup: median-polish scale estimation with the library as prior, interference
/// (high positive residual) removal, and product-m/z matching.
/// </summary>
public class LibraryRollupTests
{
    [Fact]
    public void MedianPolish_ExactScale_RecoversScaleTimesLibrarySum()
    {
        // observed = 1000 * library exactly -> scale=1000, abundance = 1000 * sum(lib) = 1750.
        var observed = new double[,] { { 1000 }, { 500 }, { 250 } };
        var lib = new[] { 1.0, 0.5, 0.25 };
        var abund = LibraryRollup.MedianPolish(observed, lib, minFragments: 2);
        Assert.Equal(1750.0, abund[0], 6);
    }

    [Fact]
    public void MedianPolish_RemovesInterferedFragment()
    {
        // Fragment 4 is interfered (obs 5000 vs ~100 expected). It is excluded from the scale
        // estimate but its library intensity still counts toward sum(lib): 1000 * 1.85 = 1850.
        var observed = new double[,] { { 1000 }, { 500 }, { 250 }, { 5000 } };
        var lib = new[] { 1.0, 0.5, 0.25, 0.1 };
        var abund = LibraryRollup.MedianPolish(observed, lib, minFragments: 2, outlierThreshold: 1.0);
        Assert.Equal(1850.0, abund[0], 6);
    }

    [Fact]
    public void MedianPolish_TooFewMatches_ReturnsNaN()
    {
        var observed = new double[,] { { 1000 }, { 500 } };
        var lib = new[] { 1.0, double.NaN }; // only one valid library fragment
        var abund = LibraryRollup.MedianPolish(observed, lib, minFragments: 2);
        Assert.True(double.IsNaN(abund[0]));
    }

    [Fact]
    public void MatchByMz_RespectsTolerance()
    {
        var spectrum = new FragmentSpectrum { ModifiedSequence = "PEPTIDEK", PrecursorCharge = 2 };
        spectrum.FragmentsByMz[100.02] = 0.8;

        Assert.Equal(0.8, SpectralLibrary.MatchByMz(spectrum, 100.03)); // diff 0.01 <= 0.02
        Assert.Null(SpectralLibrary.MatchByMz(spectrum, 100.06));       // diff 0.04 > 0.02
    }

    [Fact]
    public void StripModifications_HandlesUnimodAndMassDelta()
    {
        Assert.Equal("PEPTCIDEK", SpectralLibrary.StripModifications("PEPTC(unimod:4)IDEK"));
        Assert.Equal("PEPTMIDEK", SpectralLibrary.StripModifications("PEPTM[+15.99491]IDEK"));
    }
}
