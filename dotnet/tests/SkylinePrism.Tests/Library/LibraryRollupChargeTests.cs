using System;
using System.IO;
using Microsoft.Data.Sqlite;
using SkylinePrism.Core.Library;
using Xunit;

namespace SkylinePrism.Tests.Library;

/// <summary>
/// Covers LibraryRollup.RollupCharge (library m/z match -> median-polish, the no-spectrum sum
/// fallback, and the too-few-matches NaN path) and MedianPolish edge cases that the algorithm-level
/// tests don't reach.
/// </summary>
public class LibraryRollupChargeTests
{
    // One spectrum "PEPTIDEK"/2 with fragments at m/z 100/200/300, intensities 1000/500/250
    // (base-peak normalized on load to 1.0 / 0.5 / 0.25).
    private static string WriteLib()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_librollup_" + Guid.NewGuid().ToString("N") + ".blib");
        using var conn = new SqliteConnection($"Data Source={path};Pooling=False");
        conn.Open();
        void Exec(string sql) { using var c = conn.CreateCommand(); c.CommandText = sql; c.ExecuteNonQuery(); }
        Exec("CREATE TABLE RefSpectra (id INTEGER PRIMARY KEY, peptideSeq TEXT, precursorMZ REAL, "
            + "precursorCharge INTEGER, peptideModSeq TEXT)");
        Exec("CREATE TABLE RefSpectraPeaks (RefSpectraID INTEGER, peakMZ BLOB, peakIntensity BLOB)");
        Exec("INSERT INTO RefSpectra VALUES (1, 'PEPTIDEK', 500.0, 2, 'PEPTIDEK')");

        var mz = new byte[3 * 8];
        Buffer.BlockCopy(new[] { 100.0, 200.0, 300.0 }, 0, mz, 0, mz.Length);
        var it = new byte[3 * 4];
        Buffer.BlockCopy(new[] { 1000f, 500f, 250f }, 0, it, 0, it.Length);
        using var p = conn.CreateCommand();
        p.CommandText = "INSERT INTO RefSpectraPeaks VALUES (1, $mz, $it)";
        p.Parameters.AddWithValue("$mz", mz);
        p.Parameters.AddWithValue("$it", it);
        p.ExecuteNonQuery();
        return path;
    }

    [Fact]
    public void RollupCharge_MatchesLibrary_MedianPolishScale()
    {
        var path = WriteLib();
        try
        {
            var lib = SpectralLibrary.LoadBlib(path);
            // observed = scale_s * (normalized library) exactly: sample0 scale 2, sample1 scale 4.
            var observed = new double[,] { { 2.0, 4.0 }, { 1.0, 2.0 }, { 0.5, 1.0 } };
            var productMz = new[] { 100.0, 200.0, 300.0 };

            var abund = LibraryRollup.RollupCharge(
                lib, "PEPTIDEK", 2, productMz, observed, minFragments: 2, mzTolerance: 0.02, outlierThreshold: 1.0);

            // result = exp(beta_s) * sum(lib) = scale * (1.0 + 0.5 + 0.25) = scale * 1.75.
            Assert.Equal(3.5, abund[0], 6);
            Assert.Equal(7.0, abund[1], 6);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void RollupCharge_NoSpectrum_FallsBackToSum()
    {
        var path = WriteLib();
        try
        {
            var lib = SpectralLibrary.LoadBlib(path);
            var observed = new double[,] { { 2.0, 4.0 }, { 3.0, 5.0 } };
            // Peptide absent from the library -> per-sample sum of observed.
            var abund = LibraryRollup.RollupCharge(
                lib, "MISSINGPEP", 2, new[] { 100.0, 200.0 }, observed, 2, 0.02, 1.0);
            Assert.Equal(5.0, abund[0], 9); // 2 + 3
            Assert.Equal(9.0, abund[1], 9); // 4 + 5
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void RollupCharge_TooFewMzMatches_ReturnsNaN()
    {
        var path = WriteLib();
        try
        {
            var lib = SpectralLibrary.LoadBlib(path);
            var observed = new double[,] { { 2.0, 4.0 }, { 1.0, 2.0 }, { 0.5, 1.0 } };
            // product m/z far from any library fragment -> 0 matches < minFragments.
            var abund = LibraryRollup.RollupCharge(
                lib, "PEPTIDEK", 2, new[] { 900.0, 901.0, 902.0 }, observed, 2, 0.02, 1.0);
            Assert.All(abund, v => Assert.True(double.IsNaN(v)));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void MedianPolish_TooFewValidLibraryRows_ReturnsNaN()
    {
        var observed = new double[,] { { 2.0, 4.0 }, { 1.0, 2.0 } };
        var lib = new[] { 1.0, double.NaN }; // only 1 valid row < minFragments=2
        var abund = LibraryRollup.MedianPolish(observed, lib, minFragments: 2);
        Assert.All(abund, v => Assert.True(double.IsNaN(v)));
    }

    [Fact]
    public void MedianPolish_RemoveOutliersFalse_SkipsInterferenceRemoval()
    {
        // Row 2 in sample 0 is interfered (100x), but with removeOutliers=false the median scale
        // still absorbs it (median of 3 log-ratios), so sample 0 stays near the true scale.
        var observed = new double[,] { { 2.0, 4.0 }, { 1.0, 2.0 }, { 50.0, 1.0 } };
        var lib = new[] { 1.0, 0.5, 0.25 };
        var kept = LibraryRollup.MedianPolish(observed, lib, minFragments: 2, removeOutliers: false);
        var removed = LibraryRollup.MedianPolish(observed, lib, minFragments: 2, removeOutliers: true);
        // sample 1 has no interference -> identical either way; sample 0 differs once the outlier is removed.
        Assert.Equal(removed[1], kept[1], 6);
        Assert.False(double.IsNaN(kept[0]));
    }
}
