using System;
using System.IO;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>Multiple -m metadata files are merged (later files win on a duplicate replicate).</summary>
public class ReplicateMetadataMergeTests
{
    [Fact]
    public void MergesAcrossFiles_LaterWins()
    {
        var a = WriteCsv("Replicate,Sample Type,Batch\nR1,Standard,B1\nR2,Quality Control,B1\n");
        var b = WriteCsv("Replicate,Sample Type,Batch\nR3,Standard,B2\nR2,Standard,B2\n"); // R2 overridden
        try
        {
            var md = ReplicateMetadata.TryLoad(new[] { a, b });
            Assert.NotNull(md);
            Assert.Equal("reference", md!.TypeByReplicate["R1"]);
            Assert.Equal("reference", md.TypeByReplicate["R3"]);
            Assert.Equal("reference", md.TypeByReplicate["R2"]); // b overrode a's "qc"
            Assert.Equal("B2", md.BatchByReplicate["R2"]);
            Assert.Equal("B1", md.BatchByReplicate["R1"]);
        }
        finally
        {
            File.Delete(a);
            File.Delete(b);
        }
    }

    private static string WriteCsv(string content)
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_meta_" + Guid.NewGuid().ToString("N") + ".csv");
        File.WriteAllText(path, content);
        return path;
    }
}
