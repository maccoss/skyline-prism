using SkylinePrism.App;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Which extraction tolerance the ion accounting runs against.
/// </summary>
public class IonToleranceChoiceTests
{
    private static ProductMassTolerance Ppm(double value) =>
        ProductMassTolerance.ParseSetting($"{value} ppm")!;

    /// <summary>
    /// THE case this exists for. With a pre-exported report as the input there is no document to
    /// read - a Skyline report carries no Full-Scan settings - so the run was skipped and the user
    /// had no way to proceed. What they type is the only source there is.
    /// </summary>
    [Fact]
    public void WithNoDocumentTheEnteredValueIsWhatRuns()
    {
        var (product, precursor, source) = IonToleranceChoice.Pick(
            typedProduct: Ppm(10), typedPrecursor: Ppm(10),
            documentProduct: null, documentPrecursor: null);

        Assert.Equal(Ppm(10), product);
        Assert.Equal(Ppm(10), precursor);
        Assert.Equal(IonToleranceChoice.AsEntered, source);
    }

    /// <summary>
    /// Nothing entered and no document that can speak means NO tolerance, never a plausible default.
    /// A guessed window changes how much fragment sharing is found between co-isolated peptides, so
    /// every figure on the pane would move with nothing there to say it had.
    /// </summary>
    [Fact]
    public void NeitherSourceMeansNoToleranceAtAll()
    {
        var (product, precursor, source) = IonToleranceChoice.Pick(null, null, null, null);

        Assert.Null(product);
        Assert.Null(precursor);
        Assert.Equal(IonToleranceChoice.NotStated, source);
    }

    /// <summary>With the box blank, the document is authoritative and says so in the log.</summary>
    [Fact]
    public void AnEmptyBoxLeavesTheDocumentInCharge()
    {
        var (product, precursor, source) = IonToleranceChoice.Pick(
            typedProduct: null, typedPrecursor: null,
            documentProduct: Ppm(10), documentPrecursor: Ppm(5));

        Assert.Equal(Ppm(10), product);
        Assert.Equal(Ppm(5), precursor);
        Assert.Equal(IonToleranceChoice.FromDocument, source);
    }

    /// <summary>
    /// A typed value overrides a document that states something else - overriding is the point of
    /// typing it - and the source phrase changes so the log can never present one as the other.
    /// </summary>
    [Fact]
    public void WhatIsTypedOverridesWhatTheDocumentStates()
    {
        var (product, _, source) = IonToleranceChoice.Pick(
            typedProduct: Ppm(20), typedPrecursor: null,
            documentProduct: Ppm(10), documentPrecursor: null);

        Assert.Equal(Ppm(20), product);
        Assert.Equal(IonToleranceChoice.AsEntered, source);
    }

    /// <summary>
    /// The two halves are independent settings. Entering only a product tolerance must not throw
    /// away the document's precursor one - that would silently drop the MS1 half of the accounting,
    /// and the plot would simply not have an MS1 fraction with nothing to explain why.
    /// </summary>
    [Fact]
    public void EnteringOnlyTheProductHalfKeepsTheDocumentsPrecursorWindow()
    {
        var (product, precursor, _) = IonToleranceChoice.Pick(
            typedProduct: Ppm(20), typedPrecursor: null,
            documentProduct: Ppm(10), documentPrecursor: Ppm(5));

        Assert.Equal(Ppm(20), product);
        Assert.Equal(Ppm(5), precursor);
    }

    /// <summary>
    /// A precursor tolerance on its own does not stand in for the product one. The product window is
    /// what the MS2 accounting cannot proceed without, and inferring it from the other half would be
    /// the guess this whole rule exists to avoid.
    /// </summary>
    [Fact]
    public void APrecursorWindowAloneDoesNotStandInForTheProductOne()
    {
        var (product, _, source) = IonToleranceChoice.Pick(
            typedProduct: null, typedPrecursor: Ppm(10),
            documentProduct: null, documentPrecursor: null);

        Assert.Null(product);
        Assert.Equal(IonToleranceChoice.NotStated, source);
    }
}
