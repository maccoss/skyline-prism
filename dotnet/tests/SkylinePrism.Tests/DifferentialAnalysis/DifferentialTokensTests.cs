using System;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;

namespace SkylinePrism.Tests.DifferentialAnalysis;

/// <summary>
/// Pins <see cref="DifferentialTokens"/> to the exact words <c>prism differential</c> takes. These are
/// what quant_parameters.yaml records, so a changed word silently makes every saved report's
/// parameters unreadable as a command. The CLI side of the round trip - each word IS accepted - is
/// <c>CliIntegrationTests.Differential_AcceptsEveryRecordedToken</c>.
/// </summary>
public class DifferentialTokensTests
{
    [Theory]
    [InlineData(DifferentialDesign.Unpaired, "unpaired")]
    [InlineData(DifferentialDesign.Paired, "paired")]
    [InlineData(DifferentialDesign.LinearTrend, "trend")]
    [InlineData(DifferentialDesign.LinearTrendWithinSubject, "trend-within-subject")]
    [InlineData(DifferentialDesign.BlockedBySubject, "blocked")]
    public void Design(DifferentialDesign design, string token) =>
        Assert.Equal(token, DifferentialTokens.Design(design));

    [Theory]
    [InlineData(DifferentialTest.ModeratedT, "moderated")]
    [InlineData(DifferentialTest.WelchT, "welch")]
    [InlineData(DifferentialTest.StudentT, "student")]
    [InlineData(DifferentialTest.MannWhitney, "mann-whitney")]
    [InlineData(DifferentialTest.PairedT, "paired-t")]
    [InlineData(DifferentialTest.Wilcoxon, "wilcoxon")]
    public void Test(DifferentialTest test, string token) =>
        Assert.Equal(token, DifferentialTokens.Test(test));

    [Theory]
    [InlineData(VariancePrior.IntensityTrend, "intensity-trend")]
    [InlineData(VariancePrior.Global, "global")]
    [InlineData(VariancePrior.LimmaTrend, "limma-trend")]
    [InlineData(VariancePrior.PeptideCount, "peptide-count")]
    public void Prior(VariancePrior prior, string token) =>
        Assert.Equal(token, DifferentialTokens.Prior(prior));

    [Theory]
    [InlineData(MultipleTesting.BenjaminiHochberg, "bh")]
    [InlineData(MultipleTesting.BenjaminiYekutieli, "by")]
    [InlineData(MultipleTesting.Holm, "holm")]
    [InlineData(MultipleTesting.Bonferroni, "bonferroni")]
    [InlineData(MultipleTesting.None, "none")]
    public void Correction(MultipleTesting correction, string token) =>
        Assert.Equal(token, DifferentialTokens.Correction(correction));

    [Fact]
    public void Level()
    {
        Assert.Equal("protein", DifferentialTokens.Level(FeatureLevel.Protein));
        Assert.Equal("peptide", DifferentialTokens.Level(FeatureLevel.Peptide));
    }

    /// <summary>A new enum value must get its own word, not fall through to another's default.</summary>
    [Fact]
    public void EveryValueHasADistinctWord()
    {
        static void Distinct<T>(Func<T, string> token) where T : struct, Enum
        {
            var words = Enum.GetValues<T>().Select(token).ToList();
            Assert.Equal(words.Count, words.Distinct(StringComparer.Ordinal).Count());
        }

        Distinct<DifferentialDesign>(DifferentialTokens.Design);
        Distinct<DifferentialTest>(DifferentialTokens.Test);
        Distinct<VariancePrior>(DifferentialTokens.Prior);
        Distinct<MultipleTesting>(DifferentialTokens.Correction);
        // FeatureLevel too: Level is the one mapping written as a ternary rather than a switch, so a
        // third level would silently record as "protein" - the exact fall-through this test is for.
        Distinct<FeatureLevel>(DifferentialTokens.Level);
    }
}
