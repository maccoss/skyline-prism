using System.IO;
using SkylinePrism.Core.Library;
using Xunit;

namespace SkylinePrism.Tests.Library;

/// <summary>
/// Carafe/DIA-NN TSV spectral-library loader: required-column parsing, underscore unwrapping,
/// decoy skipping, per-(peptide,charge) fragment aggregation, and format auto-detection.
/// </summary>
public class CarafeTsvLoaderTests
{
    private static string WriteTsv(string content)
    {
        var path = Path.Combine(Path.GetTempPath(), $"carafe_{System.Guid.NewGuid():N}.tsv");
        File.WriteAllText(path, content);
        return path;
    }

    [Fact]
    public void LoadCarafeTsv_ParsesFragments_UnwrapsAndSkipsDecoys()
    {
        // Two target precursors (PEPTIDEK/2 with 3 fragments, ELVISK/2 with 2) + one decoy row.
        // DIA-NN wraps sequences in underscores; Decoy=1 must be skipped.
        var tsv = string.Join("\n", new[]
        {
            "ModifiedPeptide\tStrippedPeptide\tPrecursorMz\tPrecursorCharge\tFragmentMz\tRelativeIntensity\tFragmentType\tFragmentNumber\tFragmentCharge\tDecoy",
            "_PEPTIDEK_\tPEPTIDEK\t500.25\t2\t246.12\t1.0\ty\t2\t1\t0",
            "_PEPTIDEK_\tPEPTIDEK\t500.25\t2\t359.20\t0.6\ty\t3\t1\t0",
            "_PEPTIDEK_\tPEPTIDEK\t500.25\t2\t147.11\t0.3\ty\t1\t1\t0",
            "_ELVISK_\tELVISK\t400.10\t2\t303.18\t1.0\ty\t2\t1\t0",
            "_ELVISK_\tELVISK\t400.10\t2\t204.10\t0.4\tb\t2\t1\t0",
            "_DECOYK_\tDECOYK\t333.00\t2\t111.00\t1.0\ty\t1\t1\t1",
            "",
        });
        var path = WriteTsv(tsv);
        try
        {
            var lib = SpectralLibrary.LoadCarafeTsv(path);

            Assert.Equal(2, lib.Count); // decoy excluded

            var pep = lib.GetSpectrum("PEPTIDEK", 2);
            Assert.NotNull(pep);
            Assert.Equal("PEPTIDEK", pep!.ModifiedSequence); // underscores stripped
            Assert.Equal(2, pep.PrecursorCharge);
            Assert.Equal(500.25, pep.PrecursorMz, 3);
            Assert.Equal(3, pep.FragmentsByMz.Count);
            Assert.Equal(1.0, SpectralLibrary.MatchByMz(pep, 246.12)!.Value, 6);
            Assert.Equal(0.6, SpectralLibrary.MatchByMz(pep, 359.20)!.Value, 6);

            Assert.Equal(2, lib.GetSpectrum("ELVISK", 2)!.FragmentsByMz.Count);
            Assert.Null(lib.GetSpectrum("DECOYK", 2)); // decoy not loaded
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_AutoDetectsTsvByExtension()
    {
        var path = WriteTsv(
            "ModifiedPeptide\tPrecursorCharge\tFragmentMz\tRelativeIntensity\n_AAAK_\t2\t200.10\t1.0\n");
        try
        {
            var lib = SpectralLibrary.Load(path); // .tsv -> Carafe
            Assert.Equal(1, lib.Count);
            Assert.NotNull(lib.GetSpectrum("AAAK", 2));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void LoadCarafeTsv_MissingRequiredColumn_Throws()
    {
        var path = WriteTsv("ModifiedPeptide\tPrecursorCharge\tFragmentMz\n_AAAK_\t2\t200.1\n"); // no RelativeIntensity
        try
        {
            var ex = Assert.Throws<InvalidDataException>(() => SpectralLibrary.LoadCarafeTsv(path));
            Assert.Contains("RelativeIntensity", ex.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
