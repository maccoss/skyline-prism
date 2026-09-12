using SkylinePrism.Core.Qc;

namespace SkylinePrism.App;

/// <summary>
/// Which extraction tolerance the ion accounting runs against, as a rule that can be tested without
/// a window.
/// </summary>
/// <remarks>
/// <para>Extracted from <c>MainWindow</c> for the same reason <see cref="BatchCorrectionDefault"/>
/// and <see cref="DensitySchemeDefault"/> were: it decides a number, it lives in a file at zero
/// coverage, and the case that matters is a precedence.</para>
///
/// <para><b>There is deliberately no default.</b> The tolerance decides how much fragment sharing is
/// found between co-isolated peptides, so a guessed one moves every figure on the pane with nothing
/// there to say it moved. When neither source can state it the caller skips the measurement; it must
/// never fall back to a plausible 10 ppm.</para>
/// </remarks>
internal static class IonToleranceChoice
{
    public const string FromDocument = "from the document's Full-Scan settings";
    public const string AsEntered = "as entered";
    public const string NotStated = "not stated";

    /// <summary>
    /// The tolerances to use, and a phrase naming where they came from for the run log.
    /// </summary>
    /// <param name="typedProduct">What the user entered, or null if the box was blank or unparseable.</param>
    /// <param name="documentProduct">What an input document declares, or null if none could say.</param>
    /// <remarks>
    /// <para><b>Typed wins.</b> It is the only thing a user can say when the input is a PRE-EXPORTED
    /// REPORT - that carries no Full-Scan settings at all, so the document route has nothing to read
    /// and the measurement was simply skipped with no way for the user to proceed. It wins over a
    /// document that states something else too, because overriding is the point of typing it; the
    /// caller logs the source either way, so the two can never be confused for each other.</para>
    ///
    /// <para>The two halves are independent settings, so entering only a product tolerance keeps the
    /// document's precursor one rather than discarding it. Nothing is inferred from the other half -
    /// a precursor entered alone does not stand in for the product tolerance, which is the one the
    /// run cannot proceed without.</para>
    /// </remarks>
    public static (ProductMassTolerance? Product, ProductMassTolerance? Precursor, string Source) Pick(
        ProductMassTolerance? typedProduct, ProductMassTolerance? typedPrecursor,
        ProductMassTolerance? documentProduct, ProductMassTolerance? documentPrecursor)
    {
        if (typedProduct is null)
        {
            return (documentProduct, documentPrecursor,
                documentProduct is null ? NotStated : FromDocument);
        }
        return (typedProduct, typedPrecursor ?? documentPrecursor, AsEntered);
    }
}
