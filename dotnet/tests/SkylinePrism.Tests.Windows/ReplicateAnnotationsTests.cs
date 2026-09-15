using System;
using System.IO;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The per-directory annotation snapshot both Visualization panes group by. The parser moved here out
/// of the window so that it could be one immutable value per directory, which is what stops one pane's
/// read from installing another directory's annotations under the other pane's cached data.
/// </summary>
public class ReplicateAnnotationsTests
{
    private static string TempReports()
    {
        var dir = Path.Combine(Path.GetTempPath(), "prism-ann-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>
    /// Two documents naming the same QC injection: the merged Sample ID finds its own document's row,
    /// the bare name falls back to the first document read, and the columns are the union of both.
    /// </summary>
    [Fact]
    public void QualifiedSampleIdsKeepEachDocumentsOwnValues()
    {
        var dir = TempReports();
        try
        {
            File.WriteAllLines(Path.Combine(dir, "A.metadata.csv"), new[]
            {
                "Replicate,Sample Type,Plate",
                "QC1,Quality Control,P1",
                "S1,Unknown,P1",
            });
            File.WriteAllLines(Path.Combine(dir, "B.metadata.csv"), new[]
            {
                "Replicate,Sample Type,Condition",
                "QC1,Quality Control,\"control, pooled\"",
                "S2,Unknown,treated",
            });

            var ann = ReplicateAnnotations.Read(dir);

            Assert.False(ann.IsEmpty);
            Assert.Equal(new[] { "Sample Type", "Plate", "Condition" }, ann.Columns);
            Assert.Equal("P1", ann.ValueOf("QC1__@__A", "Plate"));
            Assert.Equal("", ann.ValueOf("QC1__@__B", "Plate"));          // B has no Plate column
            Assert.Equal("control, pooled", ann.ValueOf("QC1__@__B", "Condition")); // quoted field
            Assert.Equal("P1", ann.ValueOf("QC1", "Plate"));               // bare name: first document
            Assert.Equal("treated", ann.ValueOf("S2__@__B", "Condition"));
            Assert.Equal("", ann.ValueOf("nobody", "Plate"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>A directory with no reports is the Empty snapshot, and lookups on it are "".</summary>
    [Fact]
    public void MissingReportsGiveTheEmptySnapshot()
    {
        var dir = TempReports();
        try
        {
            Assert.Same(ReplicateAnnotations.Empty, ReplicateAnnotations.Read(dir));
            Assert.Same(ReplicateAnnotations.Empty, ReplicateAnnotations.Read(Path.Combine(dir, "nope")));
            Assert.True(ReplicateAnnotations.Empty.IsEmpty);
            Assert.Empty(ReplicateAnnotations.Empty.Columns);
            Assert.Equal("", ReplicateAnnotations.Empty.ValueOf("QC1__@__A", "Sample Type"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ReplicateOfStripsTheBatchSuffixOnly()
    {
        Assert.Equal("QC1", ReplicateAnnotations.ReplicateOf("QC1__@__Plate 3"));
        Assert.Equal("QC1", ReplicateAnnotations.ReplicateOf("QC1"));
        Assert.Equal("", ReplicateAnnotations.ReplicateOf("__@__X"));
    }
}
