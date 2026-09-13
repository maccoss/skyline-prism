using System;
using System.IO;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// "The file is locked by another process" is where a diagnosis stops. This is what continues it.
/// </summary>
/// <remarks>
/// Three investigations went into one locked cache file without ever naming what held it. The
/// reasonable answer from the person running it - "nothing other than PRISM reads or writes it" - is
/// true of every program a person chooses to run, and says nothing about the scanner that opens a
/// file the moment it is closed. The Restart Manager answers the question directly.
/// </remarks>
public class FileHoldersTests
{
    [Fact]
    public void AHeldFileNamesTheProcessHoldingIt()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism-holder-" + Guid.NewGuid().ToString("N"));
        File.WriteAllText(path, "held");
        try
        {
            using (new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None))
            {
                var held = FileHolders.Describe(path);
                if (!OperatingSystem.IsWindows())
                {
                    // No Restart Manager to ask. Saying nothing is the contract; the caller falls
                    // back to the message it had before.
                    Assert.Null(held);
                    return;
                }

                Assert.NotNull(held);
                Assert.Contains("testhost", held!, StringComparison.OrdinalIgnoreCase);
                Assert.Contains("pid", held, StringComparison.OrdinalIgnoreCase);
            }

            // Released. An answer that persisted after the handle closed would be worse than none,
            // because it would name an innocent process.
            Assert.Null(FileHolders.Describe(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void AFileNobodyHasOpenNamesNobody()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism-holder-" + Guid.NewGuid().ToString("N"));
        File.WriteAllText(path, "free");
        try
        {
            Assert.Null(FileHolders.Describe(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>
    /// A diagnostic must never be the thing that fails: it runs inside the handler for a failure
    /// that has already cost hours of instrument reads.
    /// </summary>
    [Fact]
    public void AMissingOrNonsensePathIsAnswerednotThrown()
    {
        Assert.Null(FileHolders.Describe(
            Path.Combine(Path.GetTempPath(), "prism-does-not-exist-" + Guid.NewGuid())));
        Assert.Null(FileHolders.Describe(""));
    }
}
