using System;
using System.Collections.Generic;
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
    /// Two documents naming the same QC injection: the merged Sample ID finds its own document's row
    /// and nothing else's, a document that shipped no report borrows nothing, and the columns are the
    /// union of both.
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
            Assert.Equal("", ann.ValueOf("QC1", "Plate"));                 // labeled reports key no bare names
            Assert.Equal("", ann.ValueOf("QC1__@__C", "Plate"));           // a document with no report borrows nothing
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

    /// <summary>
    /// The legacy single-document Metadata.csv has no label, so its rows are the fallback for any
    /// qualified sample ID of its replicates - the one place a bare name is consulted.
    /// </summary>
    [Fact]
    public void ALegacyMetadataCsvAnswersForItsReplicatesUnderAnyBatch()
    {
        var dir = TempReports();
        try
        {
            File.WriteAllLines(Path.Combine(dir, "Metadata.csv"), new[]
            {
                "Replicate,Sample Type,Plate",
                "QC1,Quality Control,P9",
            });

            var ann = ReplicateAnnotations.Read(dir);

            Assert.Equal("P9", ann.ValueOf("QC1__@__OldRun", "Plate"));
            Assert.Equal("P9", ann.ValueOf("QC1", "Plate"));
            Assert.Equal("", ann.ValueOf("QC2__@__OldRun", "Plate"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// A report that cannot be opened is logged and skipped; the rest of the directory still counts.
    /// Before, the exception left the Ion pane on "Reading the ion accounting..." with the readable
    /// accounting never drawn.
    /// </summary>
    [Fact]
    public void AnUnreadableReportIsSkippedAndLoggedRatherThanThrown()
    {
        var dir = TempReports();
        try
        {
            File.WriteAllLines(Path.Combine(dir, "A.metadata.csv"), new[] { "Replicate,Plate", "QC1,P1" });
            var locked = Path.Combine(dir, "B.metadata.csv");
            File.WriteAllLines(locked, new[] { "Replicate,Plate", "QC2,P2" });

            var logged = new List<string>();
            using (new FileStream(locked, FileMode.Open, FileAccess.Read, FileShare.None))
            {
                var ann = ReplicateAnnotations.Read(dir, logged.Add);

                Assert.Equal("P1", ann.ValueOf("QC1__@__A", "Plate"));
                Assert.Equal("", ann.ValueOf("QC2__@__B", "Plate"));
            }
            var message = Assert.Single(logged);
            Assert.Contains("B.metadata.csv", message);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
