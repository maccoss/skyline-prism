using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>Column auto-detection: find_column name-variant fallbacks + Detect priority/fallbacks.</summary>
public class SkylineColumnsTests
{
    [Fact]
    public void FindColumn_MatchesEachNameVariant()
    {
        Assert.Equal("Area", SkylineColumns.FindColumn(new[] { "Area" }, "Area"));                       // exact
        Assert.Equal("Retention_Time", SkylineColumns.FindColumn(new[] { "Retention_Time" }, "Retention Time")); // space->underscore
        Assert.Equal("Replicate Name", SkylineColumns.FindColumn(new[] { "Replicate Name" }, "Replicate_Name")); // underscore->space
        Assert.Equal("ProductMz", SkylineColumns.FindColumn(new[] { "ProductMz" }, "Product Mz"));        // spaces removed
        Assert.Null(SkylineColumns.FindColumn(new[] { "Something" }, "Nope"));                            // no match
    }

    [Fact]
    public void Detect_CsvHeaders_ResolvesAll()
    {
        var headers = new[]
        {
            "Peptide Modified Sequence Unimod Ids", "Sample ID", "Area", "Fragment Ion",
            "Precursor Charge", "Product Charge", "Retention Time", "Shape Correlation",
            "Product Mz", "Acquired Time", "Protein Accession", "Protein", "Protein Gene", "Batch",
        };
        var c = SkylineColumns.Detect(headers);
        Assert.Equal("Peptide Modified Sequence Unimod Ids", c.Peptide);
        Assert.Equal("Sample ID", c.Sample);
        Assert.Equal("Area", c.Abundance);
        Assert.NotNull(c.ShapeCorrelation);
        Assert.NotNull(c.ProductMz);
        Assert.NotNull(c.AcquiredTime);
        Assert.Equal("Protein Accession", c.Protein);
        Assert.Equal("Batch", c.Batch);
    }

    [Fact]
    public void Detect_ParquetUnderscoreHeaders_ResolveViaVariants()
    {
        var headers = new[]
        {
            "Peptide_Modified_Sequence", "Replicate_Name", "Area", "Fragment_Ion",
            "Precursor_Charge", "Product_Charge", "Retention_Time",
        };
        var c = SkylineColumns.Detect(headers);
        Assert.Equal("Peptide_Modified_Sequence", c.Peptide);
        Assert.Equal("Replicate_Name", c.Sample); // no "Sample ID" -> Replicate Name (underscore variant)
        Assert.Equal("Retention_Time", c.RetentionTime);
    }

    [Fact]
    public void Detect_MinimalHeaders_FallsBackAndNullsOptionals()
    {
        var c = SkylineColumns.Detect(new[] { "Replicate Name" });
        Assert.Equal("Peptide Modified Sequence", c.Peptide); // none of the 3 candidates -> fallback
        Assert.Equal("Replicate Name", c.Sample);
        Assert.Equal("Area", c.Abundance);                    // Require falls back to the last candidate
        Assert.Null(c.ShapeCorrelation);                      // optional columns absent
        Assert.Null(c.ProductMz);
        Assert.Null(c.Batch);
    }
}
