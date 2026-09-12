using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.RawData;
using Xunit;

namespace SkylinePrism.Tests.RawData;

/// <summary>
/// Matching a replicate to the data file it was acquired into. This is the join between the two
/// halves of the MS2 signal fraction - the numerator keyed by PRISM sample id, the denominator by
/// whatever the acquisition software named the file - and nothing records the correspondence.
/// </summary>
public class ReplicateDataFilesTests
{
    [Fact]
    public void TheReplicateIsTheSampleIdWithoutItsBatch()
    {
        Assert.Equal("EXP25033_A", ReplicateDataFiles.ReplicateOf("EXP25033_A__@__plate1"));
        // No separator: a bare replicate name is already the replicate.
        Assert.Equal("EXP25033_A", ReplicateDataFiles.ReplicateOf("EXP25033_A"));
        Assert.Equal("", ReplicateDataFiles.ReplicateOf(""));
    }

    [Fact]
    public void AnExactStemMatchWins()
    {
        var files = new[] { @"C:\raw\other.raw", @"C:\raw\EXP25033_A.raw" };
        Assert.Equal(@"C:\raw\EXP25033_A.raw", ReplicateDataFiles.Resolve("EXP25033_A__@__p1", files));
    }

    /// <summary>
    /// The case the rule exists for: acquisition software prefixes the run. On the cohort this was
    /// written against, replicate FLARE-001-1-B1-013 lives in 2026-extended-FLARE-001-1-B1-013.raw.
    /// </summary>
    [Fact]
    public void APrefixedFileStillMatches()
    {
        var files = new[] { @"C:\raw\2026-extended-FLARE-001-1-B1-013.raw" };
        Assert.Equal(files[0], ReplicateDataFiles.Resolve("FLARE-001-1-B1-013__@__p1", files));
    }

    /// <summary>
    /// Why the longest match wins rather than the first. Where one replicate name is a suffix of
    /// another - B1_1 inside B1_11 - the shorter would otherwise steal the longer one's file, and
    /// both replicates would then be measured against the same denominator.
    /// </summary>
    [Fact]
    public void TheLongestSuffixMatchWinsSoNeighboursDoNotStealFiles()
    {
        var files = new[] { @"C:\raw\run_B1_1.raw", @"C:\raw\run_B1_11.raw" };

        Assert.Equal(@"C:\raw\run_B1_11.raw", ReplicateDataFiles.Resolve("B1_11__@__p1", files));
        // And the short one still finds its own, by exact-suffix rather than by being first.
        Assert.Equal(@"C:\raw\run_B1_1.raw", ReplicateDataFiles.Resolve("B1_1__@__p1", files));
    }

    /// <summary>
    /// A SUFFIX, not a substring. A substring rule would let a replicate claim any file that merely
    /// contains its name somewhere in the middle - and prefixes are the pattern that actually occurs.
    /// </summary>
    [Fact]
    public void AMatchInTheMiddleOfAStemDoesNotCount()
    {
        var files = new[] { @"C:\raw\prefix_SAMPLE_A_suffix.raw" };
        Assert.Null(ReplicateDataFiles.Resolve("SAMPLE_A__@__p1", files));
    }

    [Fact]
    public void NoFilesAndNoMatchBothGiveNull()
    {
        Assert.Null(ReplicateDataFiles.Resolve("A__@__p1", Array.Empty<string>()));
        Assert.Null(ReplicateDataFiles.Resolve("A__@__p1", new[] { @"C:\raw\B.raw" }));
    }

    /// <summary>
    /// An unmatched replicate is a missing denominator, and the report has to be able to name which -
    /// so ResolveAll reports them rather than quietly returning a shorter dictionary.
    /// </summary>
    [Fact]
    public void ResolveAllReportsWhatItCouldNotMatch()
    {
        var files = new[] { @"C:\raw\run_A.raw" };
        var resolved = ReplicateDataFiles.ResolveAll(new[] { "A__@__p1", "B__@__p1" }, files);

        Assert.Single(resolved.Matched);
        Assert.Equal(@"C:\raw\run_A.raw", resolved.Matched["A__@__p1"]);
        Assert.Equal(new[] { "B__@__p1" }, resolved.Unmatched);
        Assert.Empty(resolved.Ambiguous);
    }

    /// <summary>
    /// Reference and QC injections are normally named identically in every plate's document, so the
    /// same replicate name appears in two batches and both resolve to the one file that is present.
    /// Assigning it to both would give two plates the same denominator; assigning it to whichever
    /// came first would make the answer depend on dictionary order. Neither gets it.
    /// </summary>
    [Fact]
    public void OneFileCannotServeTwoReplicates()
    {
        var files = new[] { @"C:\raw\QC_1.raw" };
        var resolved = ReplicateDataFiles.ResolveAll(
            new[] { "QC_1__@__plateA", "QC_1__@__plateB" }, files);

        Assert.Empty(resolved.Matched);
        Assert.Empty(resolved.Unmatched);
        Assert.Equal(new[] { "QC_1__@__plateA", "QC_1__@__plateB" }, resolved.Ambiguous);
    }

    /// <summary>
    /// The same collision resolves normally when each replicate has a file of its own - the rule is
    /// one file per sample, not "a repeated replicate name is always ambiguous".
    /// </summary>
    [Fact]
    public void TwoReplicatesWithTheirOwnFilesBothResolve()
    {
        var files = new[] { @"C:\raw\plateA_QC_1.raw", @"C:\raw\plateB_QC_1.raw" };
        var resolved = ReplicateDataFiles.ResolveAll(
            new[] { "plateA_QC_1__@__a", "plateB_QC_1__@__b" }, files);

        Assert.Equal(2, resolved.Matched.Count);
        Assert.Empty(resolved.Ambiguous);
    }

    /// <summary>
    /// Enumerate must find acquisitions that are DIRECTORIES - a Bruker or Agilent .d - not only
    /// files, or a whole vendor silently has no denominator.
    /// </summary>
    [Fact]
    public void EnumerateFindsDirectoryAcquisitionsToo()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism_rdf_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(Path.Combine(dir, "a.raw"), "");
            File.WriteAllText(Path.Combine(dir, "b.mzML"), "");
            File.WriteAllText(Path.Combine(dir, "notes.txt"), "");
            Directory.CreateDirectory(Path.Combine(dir, "c.d"));

            var found = ReplicateDataFiles.Enumerate(dir).Select(Path.GetFileName).ToList();

            Assert.Contains("a.raw", found);
            Assert.Contains("b.mzML", found);
            Assert.Contains("c.d", found);
            Assert.DoesNotContain("notes.txt", found);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void EnumerateOfAMissingDirectoryIsEmptyRatherThanAThrow()
    {
        Assert.Empty(ReplicateDataFiles.Enumerate(
            Path.Combine(Path.GetTempPath(), "prism_no_such_" + Guid.NewGuid().ToString("N"))));
    }
}
