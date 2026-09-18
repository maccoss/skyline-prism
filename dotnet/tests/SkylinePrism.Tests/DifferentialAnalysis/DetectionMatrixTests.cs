using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis.Detection;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Tests for <see cref="DetectionMatrix"/> against the committed partitioned merged_data fixture,
/// cross-checked with the same DuckDB detection query in the Python reference (cryptic_detection_matrix
/// with term=None over the cohort fixture).
/// </summary>
public class DetectionMatrixTests
{
    private static string CohortRoot => Fixtures.Path2("cohort");

    [Fact]
    public void Load_AllPeptides_MatchesReferenceQuery()
    {
        var data = DetectionMatrix.Load(CohortRoot, qThreshold: 0.01, term: null);

        Assert.Equal(327, data.PeptideIds.Length);
        Assert.Equal(192, data.SampleIds.Length);
        Assert.Equal("AAVETLGVPC(unimod:4)FLGGMAR", data.PeptideIds[0]);
        Assert.Equal("S001__@__Batch1", data.SampleIds[0]);
        Assert.Equal(0.0, data.Matrix[0, 0]);

        var nPep = data.PeptideIds.Length;
        var nSamp = data.SampleIds.Length;

        double total = 0, row0 = 0, col0 = 0;
        for (var p = 0; p < nPep; p++)
        for (var s = 0; s < nSamp; s++)
            total += data.Matrix[p, s];
        for (var s = 0; s < nSamp; s++)
            row0 += data.Matrix[0, s];
        for (var p = 0; p < nPep; p++)
            col0 += data.Matrix[p, 0];

        Assert.Equal(37566, (int)total);
        Assert.Equal(164, (int)row0);
        Assert.Equal(134, (int)col0);
        // Every cell is 0 or 1.
        Assert.True(data.Matrix.Cast<double>().All(v => v == 0.0 || v == 1.0));
    }

    [Fact]
    public void CrypticPeptideMap_FiltersByProteinTerm()
    {
        // Most fixture proteins are Swiss-Prot human entries (..._HUMAN); 321 of the 327 peptides map to
        // a HUMAN-containing protein (the other 6 have non-human/contaminant protein strings).
        var map = DetectionMatrix.CrypticPeptideMap(CohortRoot, "HUMAN");
        Assert.Equal(321, map.Count);
        Assert.All(map.Values, v => Assert.Contains("HUMAN", v));

        // A term matching nothing yields an empty map.
        Assert.Empty(DetectionMatrix.CrypticPeptideMap(CohortRoot, "no-such-cryptic-term"));
    }

    [Fact]
    public void Load_LegacySpacedColumnNames_Resolves()
    {
        // The mini fixture's merged_data uses the older spaced Skyline headers ("Peptide Modified
        // Sequence Unimod Ids", "Detection Q Value"); the loader must resolve them like the current
        // camelCase form.
        var data = DetectionMatrix.Load(Fixtures.Path2("mini", "e2e-sum", "output"), qThreshold: 0.01, term: null);
        Assert.True(data.PeptideIds.Length > 0);
        Assert.True(data.SampleIds.Length > 0);
        Assert.All(data.Matrix.Cast<double>(), v => Assert.True(v == 0.0 || v == 1.0));
    }

    [Theory]
    [InlineData("sp|P0DP02|HVC33_HUMAN", "HVC33_HUMAN")]
    [InlineData("CRYPTIC_UNIQUE|Q9Y2|S35U4_HUMAN", "S35U4_HUMAN")]
    [InlineData("first|second", "second")]
    [InlineData("solo", "solo")]
    public void CrypticShortLabel_PicksHumanTokenOrFallback(string input, string expected)
    {
        Assert.Equal(expected, DetectionMatrix.CrypticShortLabel(input));
    }
}
