using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// The Replicates report's annotations are the study - Subject, Timepoint, responder status - and they
/// have to reach sample_metadata.csv, whatever they are called. PRISM interprets three of the report's
/// columns; it must not decide that the rest are uninteresting.
/// </summary>
public class ReplicateAnnotationPassthroughTests
{
    private static string WriteTemp(string name, string content, string dir)
    {
        var path = Path.Combine(dir, name);
        File.WriteAllText(path, content);
        return path;
    }

    [Fact]
    public void EveryColumnIsCarried_InReportOrder()
    {
        var dir = Directory.CreateTempSubdirectory("prism_ann_").FullName;
        try
        {
            // Shaped like a real export: the built-in columns, then this document's own annotations.
            var path = WriteTemp("plate1.metadata.csv", """
                ReplicateName,SampleType,AnalyteConcentration,Subject,Timepoint,N/NR,TimeBetweenSamples(days)
                FLARE-001-1,Unknown,,001,1,NR-2,0
                Pool-01,Standard,,,,,
                """, dir);

            var md = ReplicateMetadata.TryLoad(new[] { path }, null, null, null, new[] { "plate1" });

            Assert.NotNull(md);
            // The replicate-name column is the sample itself, so it is not repeated as an annotation.
            Assert.Equal(
                new[] { "SampleType", "AnalyteConcentration", "Subject", "Timepoint", "N/NR", "TimeBetweenSamples(days)" },
                md!.ColumnNames.ToArray());

            var values = md.ValuesFor("FLARE-001-1__@__plate1", "FLARE-001-1");
            Assert.NotNull(values);
            Assert.Equal("001", values!["Subject"]);
            Assert.Equal("1", values["Timepoint"]);
            Assert.Equal("NR-2", values["N/NR"]);
            Assert.Equal("0", values["TimeBetweenSamples(days)"]);
            // ReplicateMetadata keeps the raw value; it is sample_metadata.csv that does not write a
            // second column for it (see WriteSampleMetadata - "SampleType" restates "sample_type").
            Assert.Equal("Unknown", values["SampleType"]);
            Assert.Equal("experimental", md.TypeFor("FLARE-001-1__@__plate1", "FLARE-001-1"));
            Assert.Equal("reference", md.TypeFor("Pool-01__@__plate1", "Pool-01"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void DocumentsWithDifferentColumns_AreUnioned_AndEachKeepsItsOwn()
    {
        // Each document's Replicates grid is its own: the annotations are per-document settings, so a
        // cohort can have one plate carrying Subject/Timepoint and another carrying Plate/Condition.
        var dir = Directory.CreateTempSubdirectory("prism_ann_").FullName;
        try
        {
            var a = WriteTemp("plateA.metadata.csv", """
                ReplicateName,SampleType,Subject,Timepoint
                S1,Unknown,001,1
                """, dir);
            var b = WriteTemp("plateB.metadata.csv", """
                ReplicateName,SampleType,Plate,Condition
                S1,Unknown,P2,treated
                """, dir);

            var md = ReplicateMetadata.TryLoad(
                new[] { a, b }, null, null, null, new[] { "plateA", "plateB" });

            Assert.NotNull(md);
            // Union, first-seen order - nothing is dropped for being absent from the other document.
            Assert.Equal(
                new[] { "SampleType", "Subject", "Timepoint", "Plate", "Condition" },
                md!.ColumnNames.ToArray());

            // The SAME replicate name in both documents keeps each document's own annotations.
            var fromA = md.ValuesFor("S1__@__plateA", "S1");
            var fromB = md.ValuesFor("S1__@__plateB", "S1");
            Assert.Equal("001", fromA!["Subject"]);
            Assert.False(fromA.ContainsKey("Plate"));
            Assert.Equal("P2", fromB!["Plate"]);
            Assert.False(fromB.ContainsKey("Subject"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void AnnotationValuesContainingCommas_SurviveTheRoundTrip()
    {
        // The reason every reader of sample_metadata.csv had to become quote-aware: a free-text
        // annotation with a comma used to shift every field after it, silently.
        var dir = Directory.CreateTempSubdirectory("prism_ann_").FullName;
        try
        {
            var path = WriteTemp("plate1.metadata.csv", """
                ReplicateName,SampleType,Notes
                S1,Unknown,"re-injected, second vial"
                """, dir);

            var md = ReplicateMetadata.TryLoad(new[] { path }, null, null, null, new[] { "plate1" });
            var values = md!.ValuesFor("S1__@__plate1", "S1");

            Assert.Equal("re-injected, second vial", values!["Notes"]);

            // ...and it comes back out of a CSV line intact.
            var line = string.Join(",", new[] { "S1", CsvLine.Quote(values["Notes"]), "x" });
            Assert.Equal(new[] { "S1", "re-injected, second vial", "x" }, CsvLine.Split(line));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
