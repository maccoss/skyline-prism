using System.Collections.Generic;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// Column auto-detection + the data.* overrides: works for both the invariant/parquet (no-space) and
/// English/CSV (spaced) exports, and an explicit override wins if it resolves, else falls back to
/// auto-detect (matching Python's data.peptide_column behavior).
/// </summary>
public class ColumnDetectionTests
{
    private static HashSet<string> Invariant() => new()
    {
        "PeptideModifiedSequenceUnimodIds", "Peptide", "ProteinAccession", "Protein", "Area",
        "FragmentIon", "PrecursorCharge", "ProductCharge", "RetentionTime", "ReplicateName",
    };

    private static HashSet<string> English() => new()
    {
        "Peptide Modified Sequence Unimod Ids", "Peptide", "Protein Accession", "Protein", "Area",
        "Fragment Ion", "Precursor Charge", "Product Charge", "Retention Time", "Replicate Name",
    };

    [Fact]
    public void Detect_InvariantNoSpaceColumns()
    {
        var d = SkylineColumns.Detect(Invariant());
        Assert.Equal("PeptideModifiedSequenceUnimodIds", d.Peptide);
        Assert.Equal("ProteinAccession", d.Protein);
        Assert.Equal("Area", d.Abundance);
        Assert.Equal("FragmentIon", d.Transition);
        Assert.Equal("ReplicateName", d.Sample);
    }

    [Fact]
    public void Detect_EnglishSpacedColumns()
    {
        var d = SkylineColumns.Detect(English());
        Assert.Equal("Peptide Modified Sequence Unimod Ids", d.Peptide);
        Assert.Equal("Protein Accession", d.Protein);
        Assert.Equal("Fragment Ion", d.Transition);
        Assert.Equal("Replicate Name", d.Sample);
    }

    [Fact]
    public void Override_WinsOverAutoDetect()
    {
        var d = SkylineColumns.Detect(Invariant(), new ColumnOverrides(Peptide: "Peptide"));
        Assert.Equal("Peptide", d.Peptide); // forced the plain Peptide column
    }

    [Fact]
    public void Override_ResolvesAcrossNamingConvention()
    {
        // Config gives an English name but the data is invariant (no-space) - must still match.
        var d = SkylineColumns.Detect(Invariant(),
            new ColumnOverrides(Peptide: "Peptide Modified Sequence Unimod Ids"));
        Assert.Equal("PeptideModifiedSequenceUnimodIds", d.Peptide);
    }

    [Fact]
    public void Override_UnmatchedFallsBackToAutoDetect()
    {
        // The exact SEA-AD case: config data.peptide_column: "Peptide Modified Sequence" (absent) -> C#
        // auto-detects the Unimod-Ids column, matching what Python's output actually used.
        var cols = Invariant();
        cols.Remove("PeptideModifiedSequenceUnimodIds");
        cols.Add("PeptideModifiedSequenceUnimodIds"); // ensure present
        var d = SkylineColumns.Detect(Invariant(),
            new ColumnOverrides(Peptide: "Peptide Modified Sequence"));
        Assert.Equal("PeptideModifiedSequenceUnimodIds", d.Peptide);
    }

    [Fact]
    public void SampleId_WinsOverSampleColumnOverride()
    {
        // The merge synthesizes a batch-disambiguated "Sample ID" (<replicate>__@__<batch>); it must win
        // over data.sample_column (which names the INPUT replicate column), so output columns keep the
        // __@__ suffix and identical replicate names across batches stay distinct.
        var cols = new HashSet<string>
        {
            "PeptideModifiedSequenceUnimodIds", "ReplicateName", "Sample ID", "Area", "FragmentIon",
            "PrecursorCharge", "ProductCharge", "RetentionTime",
        };
        var d = SkylineColumns.Detect(cols, new ColumnOverrides(Sample: "Replicate Name"));
        Assert.Equal("Sample ID", d.Sample);
    }

    [Fact]
    public void FindColumn_IgnoresCaseSpaceUnderscore()
    {
        var cols = new HashSet<string> { "peptide_modified_sequence" };
        Assert.Equal("peptide_modified_sequence",
            SkylineColumns.FindColumn(cols, "Peptide Modified Sequence"));
        Assert.Equal("peptide_modified_sequence",
            SkylineColumns.FindColumn(cols, "PeptideModifiedSequence"));
    }
}
