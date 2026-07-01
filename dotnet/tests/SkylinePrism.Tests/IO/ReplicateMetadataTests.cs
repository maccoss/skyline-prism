using System;
using System.IO;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// The Skyline Replicates report has annotation-dependent columns, so metadata parsing must
/// auto-detect Sample Type, map Skyline sample types to PRISM types, and accept an explicit
/// (variable) batch column name.
/// </summary>
public class ReplicateMetadataTests
{
    [Fact]
    public void TryLoad_MapsSampleTypes_AndExplicitBatchColumn()
    {
        var path = Path.Combine(Path.GetTempPath(), "rep_" + Guid.NewGuid().ToString("N") + ".csv");
        File.WriteAllText(path,
            "Replicate,Sample Type,Plate\n" +
            "Pool_A,Standard,B1\n" +
            "Carl_A,Quality Control,B1\n" +
            "Study_01,Unknown,B2\n");
        try
        {
            var md = ReplicateMetadata.TryLoad(path, batchColumn: "Plate");
            Assert.NotNull(md);

            Assert.Equal("reference", md!.TypeByReplicate["Pool_A"]);
            Assert.Equal("qc", md.TypeByReplicate["Carl_A"]);
            Assert.Equal("experimental", md.TypeByReplicate["Study_01"]);

            Assert.Equal("B1", md.BatchByReplicate["Pool_A"]);
            Assert.Equal("B2", md.BatchByReplicate["Study_01"]);
            Assert.True(md.HasBatches);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TryLoad_AutoDetectsSampleType_WhenNoBatchColumn()
    {
        var path = Path.Combine(Path.GetTempPath(), "rep_" + Guid.NewGuid().ToString("N") + ".csv");
        File.WriteAllText(path,
            "Replicate Name,Sample Type\n" +
            "S1,Standard\n" +
            "S2,Blank\n");
        try
        {
            var md = ReplicateMetadata.TryLoad(path);
            Assert.NotNull(md);
            Assert.Equal("reference", md!.TypeByReplicate["S1"]);
            Assert.Equal("experimental", md.TypeByReplicate["S2"]); // Blank -> experimental
            Assert.False(md.HasBatches);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData("Standard", "reference")]
    [InlineData("Quality Control", "qc")]
    [InlineData("Unknown", "experimental")]
    [InlineData("", "experimental")]
    public void MapSampleType_MatchesPython(string skyline, string expected)
        => Assert.Equal(expected, ReplicateMetadata.MapSampleType(skyline));
}
