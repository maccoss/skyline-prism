using System;
using System.IO;
using System.Threading;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>Source fingerprint is stable for unchanged inputs and changes when a file changes.</summary>
public class SourceFingerprintTests
{
    [Fact]
    public void StableForSameInputs_ChangesOnEdit()
    {
        var f = Path.Combine(Path.GetTempPath(), "prism_fp_" + Guid.NewGuid().ToString("N") + ".csv");
        File.WriteAllText(f, "a,b\n1,2\n");
        try
        {
            var fp1 = SourceFingerprint.Compute(new[] { f });
            Assert.Equal(fp1, SourceFingerprint.Compute(new[] { f })); // stable

            Thread.Sleep(10);
            File.WriteAllText(f, "a,b\n1,2\n3,4\n"); // size + mtime change
            Assert.NotEqual(fp1, SourceFingerprint.Compute(new[] { f }));
        }
        finally { File.Delete(f); }
    }

    [Fact]
    public void CacheEntry_RoundTrips()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_cache_" + Guid.NewGuid().ToString("N") + ".json");
        try
        {
            SourceFingerprint.Write(path, new SourceFingerprint.CacheEntry("ABC", 123, "Peptide"));
            var e = SourceFingerprint.TryRead(path);
            Assert.NotNull(e);
            Assert.Equal("ABC", e!.Fingerprint);
            Assert.Equal(123, e.TotalRows);
            Assert.Equal("Peptide", e.SortColumn);
        }
        finally { File.Delete(path); }
    }
}
