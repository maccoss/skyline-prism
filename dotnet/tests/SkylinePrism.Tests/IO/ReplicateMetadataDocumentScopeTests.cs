using System;
using System.IO;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// Document-scoped metadata: when several Skyline documents (one per batch/plate) are merged, the SAME
/// replicate name usually appears in every one of them - reference and QC injections are named identically
/// per plate. Keyed by bare replicate name the last file would win, collapsing two documents into one batch
/// and overwriting sample types. Each file's rows are therefore also stored under
/// "&lt;replicate&gt;__@__&lt;document&gt;" - exactly the sample ID DuckDbMerge synthesizes - and those win.
/// </summary>
public class ReplicateMetadataDocumentScopeTests
{
    private static string WriteCsv(string name, string content)
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_md_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, name);
        File.WriteAllText(path, content);
        return path;
    }

    // Two plates, each with a reference "Pool_A" and a QC "Carl_A", plus a plate-specific study sample.
    private static (string PlateA, string PlateB) TwoPlates()
    {
        var a = WriteCsv("plate_a.metadata.csv",
            "Replicate,Sample Type,Plate\nPool_A,Standard,P1\nCarl_A,Quality Control,P1\nStudy_01,Unknown,P1\n");
        var b = WriteCsv("plate_b.metadata.csv",
            "Replicate,Sample Type,Plate\nPool_A,Standard,P2\nCarl_A,Quality Control,P2\nStudy_02,Unknown,P2\n");
        return (a, b);
    }

    [Fact]
    public void SharedReplicateNames_KeepTheirOwnDocumentsBatch()
    {
        var (a, b) = TwoPlates();

        var md = ReplicateMetadata.TryLoad(
            new[] { a, b }, batchColumn: "Plate",
            documentLabels: new[] { "plate_a", "plate_b" });

        Assert.NotNull(md);
        // The decisive assertion: the same replicate name resolves to a DIFFERENT batch per document.
        Assert.Equal("P1", md!.BatchFor("Pool_A__@__plate_a", "Pool_A"));
        Assert.Equal("P2", md.BatchFor("Pool_A__@__plate_b", "Pool_A"));
        Assert.Equal("P1", md.BatchFor("Carl_A__@__plate_a", "Carl_A"));
        Assert.Equal("P2", md.BatchFor("Carl_A__@__plate_b", "Carl_A"));
    }

    [Fact]
    public void SharedReplicateNames_KeepTheirOwnDocumentsSampleType()
    {
        var a = WriteCsv("plate_a.metadata.csv", "Replicate,Sample Type\nShared,Standard\n");
        var b = WriteCsv("plate_b.metadata.csv", "Replicate,Sample Type\nShared,Quality Control\n");

        var md = ReplicateMetadata.TryLoad(new[] { a, b }, documentLabels: new[] { "plate_a", "plate_b" });

        Assert.Equal("reference", md!.TypeFor("Shared__@__plate_a", "Shared"));
        Assert.Equal("qc", md.TypeFor("Shared__@__plate_b", "Shared"));
    }

    [Fact]
    public void UnqualifiedLookupStillWorks_ForTheSingleDocumentCase()
    {
        var (a, _) = TwoPlates();

        var md = ReplicateMetadata.TryLoad(new[] { a }, batchColumn: "Plate");

        // No document labels supplied: bare replicate keys, and a sample ID with an unknown suffix still
        // resolves through the fallback.
        Assert.Equal("P1", md!.BatchFor("Pool_A__@__whatever", "Pool_A"));
        Assert.Equal("reference", md.TypeFor("Pool_A__@__whatever", "Pool_A"));
        Assert.Equal("P1", md.BatchByReplicate["Pool_A"]);
    }

    [Fact]
    public void QualifiedEntryWins_OverTheBareFallback()
    {
        var (a, b) = TwoPlates();

        var md = ReplicateMetadata.TryLoad(
            new[] { a, b }, batchColumn: "Plate", documentLabels: new[] { "plate_a", "plate_b" });

        // The bare map still holds the last file's value (back-compat), but the qualified entry overrides it.
        Assert.Equal("P2", md!.BatchByReplicate["Pool_A"]);
        Assert.Equal("P1", md.BatchFor("Pool_A__@__plate_a", "Pool_A"));
    }

    [Fact]
    public void HasBatchFor_SeesQualifiedEntries()
    {
        var (a, b) = TwoPlates();

        var md = ReplicateMetadata.TryLoad(
            new[] { a, b }, batchColumn: "Plate", documentLabels: new[] { "plate_a", "plate_b" });

        Assert.True(md!.HasBatchFor("Study_01__@__plate_a", "Study_01"));
        Assert.False(md.HasBatchFor("Never_Seen__@__plate_a", "Never_Seen"));
    }

    [Fact]
    public void MismatchedLabelCount_IsRejected()
    {
        var (a, b) = TwoPlates();

        Assert.Throws<ArgumentException>(() =>
            ReplicateMetadata.TryLoad(new[] { a, b }, documentLabels: new[] { "only_one" }));
    }

    [Fact]
    public void QualifiedKey_MatchesTheMergedSampleIdFormat()
        => Assert.Equal("Pool_A__@__plate_a", ReplicateMetadata.QualifiedKey("Pool_A", "plate_a"));
}
