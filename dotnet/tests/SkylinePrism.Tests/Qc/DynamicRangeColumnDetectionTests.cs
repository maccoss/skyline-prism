using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// Telling replicate columns from metadata in a corrected matrix.
///
/// <para>This has now broken twice by the same mechanism: a hand-maintained list of metadata column
/// names drifting from what the pipeline writes, so an unlisted column was read as a replicate and
/// parsing it as an abundance threw. At protein level that produced a blank tab; at peptide level
/// it produced <c>The input string 'AAAAAGAGLK' was not in a correct format</c> - a peptide sequence
/// being parsed as a number.</para>
///
/// <para>The peptide case cannot be fixed by extending the list, because the peptide column keeps
/// whatever name the Skyline export used and that is auto-detected per document. So detection is by
/// TYPE: a text column is never a replicate, whatever it is called.</para>
/// </summary>
public class DynamicRangeColumnDetectionTests
{
    /// <summary>
    /// The exact shape that failed: a peptide identifier column whose name is not one anybody
    /// guessed, holding sequences like "AAAAAGAGLK".
    /// </summary>
    [Fact]
    public void AnUnanticipatedTextColumnIsNotMistakenForAReplicate()
    {
        var path = NewParquet();
        try
        {
            // "Peptide Sequence" is a real Skyline export column name, and was NOT among the four the
            // old list guessed at.
            WriteMatrix(path,
                meta: new (string, Type)[]
                {
                    ("Peptide Sequence", typeof(string)),
                    ("leading_gene_name", typeof(string)),
                    ("n_transitions", typeof(long)),
                    ("mean_rt", typeof(double)),
                },
                metaData: new Array[]
                {
                    new[] { "AAAAAGAGLK", "SAMPLERPEP" },
                    new[] { "ALB", "APOA1" },
                    new[] { 6L, 5L },
                    new[] { 21.5, 33.25 },
                },
                samples: new[] { "R1__@__p1", "R2__@__p1" },
                sampleData: new[] { new[] { 1e6, 2e6 }, new[] { 3e6, 4e6 } });

            var table = ParquetTable.Load(path);
            var samples = DynamicRange.SampleColumns(table, AbundanceLevel.Peptide);

            Assert.Equal(new[] { "R1__@__p1", "R2__@__p1" }, samples);

            // And the thing that actually threw for the user must now work.
            var entries = DynamicRange.Compute(table, AbundanceLevel.Peptide, samples);
            Assert.Equal(2, entries.Count);
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>
    /// Type alone is not enough: <c>n_transitions</c> and <c>mean_rt</c> are numbers but not
    /// abundances, so they still have to be excluded by name. Averaging a retention time into a
    /// dynamic-range curve would be silently wrong rather than an exception.
    /// </summary>
    [Fact]
    public void NumericMetadataIsStillExcludedByName()
    {
        var path = NewParquet();
        try
        {
            WriteMatrix(path,
                meta: new (string, Type)[]
                {
                    ("Some Odd Peptide Column", typeof(string)),
                    ("n_transitions", typeof(long)),
                    ("mean_rt", typeof(double)),
                },
                metaData: new Array[]
                {
                    new[] { "PEPTIDEK", "OTHERPEPK" },
                    new[] { 7L, 4L },
                    new[] { 12.0, 44.0 },
                },
                samples: new[] { "A__@__b" },
                sampleData: new[] { new[] { 5e5, 6e5 } });

            var table = ParquetTable.Load(path);
            var samples = DynamicRange.SampleColumns(table, AbundanceLevel.Peptide);

            Assert.Equal(new[] { "A__@__b" }, samples);
            Assert.DoesNotContain("mean_rt", samples);
            Assert.DoesNotContain("n_transitions", samples);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void IsNumericColumn_SeparatesTextFromNumbers()
    {
        var path = NewParquet();
        try
        {
            WriteMatrix(path,
                meta: new (string, Type)[] { ("text", typeof(string)), ("count", typeof(long)) },
                metaData: new Array[] { new[] { "a", "b" }, new[] { 1L, 2L } },
                samples: new[] { "s1" },
                sampleData: new[] { new[] { 1.5, 2.5 } });

            var table = ParquetTable.Load(path);

            Assert.False(table.IsNumericColumn("text"));
            Assert.True(table.IsNumericColumn("count"));
            Assert.True(table.IsNumericColumn("s1"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static string NewParquet() =>
        Path.Combine(Path.GetTempPath(), "prism_drcols_" + Guid.NewGuid().ToString("N") + ".parquet");

    private static void WriteMatrix(
        string path, IReadOnlyList<(string Name, Type ElementType)> meta, IReadOnlyList<Array> metaData,
        IReadOnlyList<string> samples, IReadOnlyList<double[]> sampleData)
    {
        using var writer = StreamingWideWriter.Create(path, meta, samples);
        writer.WriteRowGroup(metaData, sampleData);
    }
}
