using System;
using System.IO;
using SkylinePrism.Core.DifferentialAnalysis;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// Inputs a user may also have open in Excel must still be readable. Excel holds an open CSV for
/// writing and shares only reads, which <see cref="File.ReadAllLines(string)"/> refuses: in the
/// Differential pane that made a level switch fail and empty every picker, with sample_metadata.csv
/// merely open in another window.
/// </summary>
public class SharedFileTests
{
    /// <summary>Open the file the way Excel does: write access, sharing reads only.</summary>
    private static FileStream HoldLikeExcel(string path) =>
        new(path, FileMode.Open, FileAccess.ReadWrite, FileShare.Read);

    [Fact]
    public void ReadAllLines_ReadsAFileHeldOpenForWriting()
    {
        var path = Path.Combine(Directory.CreateTempSubdirectory("prism-shared-").FullName, "meta.csv");
        File.WriteAllLines(path, new[] { "sample,Purpose", "a,KI", "b,KI_control" });
        try
        {
            using var excel = HoldLikeExcel(path);

            // The control: the plain read is refused. Windows enforces FileShare between processes
            // and handles; elsewhere it is advisory, so this half only holds there.
            if (OperatingSystem.IsWindows())
                Assert.Throws<IOException>(() => File.ReadAllLines(path));

            Assert.Equal(new[] { "sample,Purpose", "a,KI", "b,KI_control" }, SharedFile.ReadAllLines(path));
        }
        finally
        {
            Directory.Delete(Path.GetDirectoryName(path)!, recursive: true);
        }
    }

    /// <summary>
    /// The pre-run check finds exactly the files the run will write that another program holds. A CSV
    /// the run only READS - an additional metadata file kept in the output directory - can be open in
    /// Excel without stopping anything. Windows only: elsewhere a hold is advisory and nothing is ever
    /// reported.
    /// </summary>
    [Fact]
    public void HeldOpen_NamesOnlyTheRunsOwnOutputsExcelHas()
    {
        if (!OperatingSystem.IsWindows())
            return;
        var dir = Directory.CreateTempSubdirectory("prism-shared-").FullName;
        try
        {
            foreach (var name in new[] { "sample_metadata.csv", "protein_groups.csv", "clinical.csv" })
                File.WriteAllText(Path.Combine(dir, name), "x");
            var outputs = PrismPipeline.UserOpenedOutputs(new PrismConfig());

            Assert.Empty(SharedFile.HeldOpen(dir, outputs));
            using (HoldLikeExcel(Path.Combine(dir, "sample_metadata.csv")))
            using (HoldLikeExcel(Path.Combine(dir, "clinical.csv")))
                Assert.Equal(new[] { "sample_metadata.csv" }, SharedFile.HeldOpen(dir, outputs));

            // Released: the retry goes ahead.
            Assert.Empty(SharedFile.HeldOpen(dir, outputs));
            // The probe opened for write but wrote nothing.
            Assert.Equal("x", File.ReadAllText(Path.Combine(dir, "sample_metadata.csv")));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>The list follows the config: the matrices' format, and marker scores only when they are made.</summary>
    [Fact]
    public void UserOpenedOutputs_FollowTheConfig()
    {
        var defaults = PrismPipeline.UserOpenedOutputs(new PrismConfig());
        Assert.Contains("sample_metadata.csv", defaults);
        Assert.Contains("corrected_proteins.parquet", defaults);
        Assert.DoesNotContain("marker_normalization.csv", defaults);

        var csv = new PrismConfig();
        csv.Output.Format = "csv";
        csv.MarkerNormalization.Enabled = true;
        var asCsv = PrismPipeline.UserOpenedOutputs(csv);
        Assert.Contains("corrected_peptides.csv", asCsv);
        Assert.Contains("marker_normalization.csv", asCsv);
        Assert.DoesNotContain("corrected_peptides.parquet", asCsv);
    }

    [Fact]
    public void HeldOpen_IsFalseForAMissingFileOrDirectory()
    {
        Assert.False(SharedFile.IsHeldOpen(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "x.csv")));
        Assert.Empty(SharedFile.HeldOpen(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")),
            new[] { "sample_metadata.csv" }));
    }

    /// <summary>
    /// What a read racing the pipeline's rewrite of sample_metadata.csv sees - the header and a prefix
    /// of the rows - is refused, not analyzed as a quietly smaller cohort.
    /// </summary>
    [Fact]
    public void DifferentialDatasetLoad_RefusesMetadataMissingSomeOfTheMatrixSamples()
    {
        var source = Fixtures.Path2("mini", "e2e-sum", "output");
        var dir = Directory.CreateTempSubdirectory("prism-shared-").FullName;
        try
        {
            File.Copy(Path.Combine(source, "corrected_proteins.parquet"), Path.Combine(dir, "corrected_proteins.parquet"));
            var lines = File.ReadAllLines(Path.Combine(source, "sample_metadata.csv"));
            File.WriteAllLines(Path.Combine(dir, "sample_metadata.csv"), lines[..100]); // header + 99 of 166

            var ex = Assert.Throws<InvalidOperationException>(() => DifferentialDataset.Load(dir, FeatureLevel.Protein));
            Assert.Contains("67 of the 166 samples", ex.Message, StringComparison.Ordinal);
            Assert.Contains("no row in sample_metadata.csv", ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>The run's failure path recognises the lock however deeply the pipeline wrapped it.</summary>
    [Fact]
    public void IsSharingViolation_SeesThroughWrappers()
    {
        if (!OperatingSystem.IsWindows())
            return;
        var path = Path.Combine(Directory.CreateTempSubdirectory("prism-shared-").FullName, "meta.csv");
        File.WriteAllText(path, "x");
        try
        {
            IOException locked;
            using (HoldLikeExcel(path))
                locked = Assert.Throws<IOException>(() => File.WriteAllText(path, "y"));

            Assert.True(SharedFile.IsSharingViolation(locked));
            Assert.True(SharedFile.IsSharingViolation(new InvalidOperationException("stage 5", locked)));
            Assert.True(SharedFile.IsSharingViolation(new AggregateException(locked)));
            Assert.False(SharedFile.IsSharingViolation(new IOException("disk full")));
            Assert.False(SharedFile.IsSharingViolation(new FileNotFoundException("gone", path)));
        }
        finally
        {
            Directory.Delete(Path.GetDirectoryName(path)!, recursive: true);
        }
    }

    /// <summary>
    /// The reported failure end to end: both levels load while sample_metadata.csv is held open.
    /// </summary>
    [Theory]
    [InlineData(FeatureLevel.Protein)]
    [InlineData(FeatureLevel.Peptide)]
    public void DifferentialDatasetLoad_SucceedsWithSampleMetadataOpenInExcel(FeatureLevel level)
    {
        var source = Fixtures.Path2("mini", "e2e-sum", "output");
        var dir = Directory.CreateTempSubdirectory("prism-shared-").FullName;
        try
        {
            foreach (var name in new[] { "sample_metadata.csv", "corrected_proteins.parquet", "corrected_peptides.parquet" })
                File.Copy(Path.Combine(source, name), Path.Combine(dir, name));

            using var excel = HoldLikeExcel(Path.Combine(dir, "sample_metadata.csv"));
            var ds = DifferentialDataset.Load(dir, level);

            Assert.Equal(166, ds.SampleIds.Length);
            Assert.Contains("sample_type", ds.MetadataColumns);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
