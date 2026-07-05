using System;
using System.IO;
using DuckDB.NET.Data;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>
/// Exercises BatchEstimator.Estimate end-to-end through the DuckDB parquet read (ReadSampleTimes +
/// TryParseTime), which the AssignBatches unit tests don't cover. Writes a real parquet of
/// sample + acquired-time rows and checks the resulting batch assignment.
/// </summary>
public class BatchEstimatorParquetTests
{
    // Two acquisition clusters (~5 min apart) separated by a 3-hour gap.
    private const string TimestampRows =
        "('S1', TIMESTAMP '2024-01-01 10:00:00'), ('S2', TIMESTAMP '2024-01-01 10:05:00'), " +
        "('S3', TIMESTAMP '2024-01-01 10:10:00'), ('S4', TIMESTAMP '2024-01-01 13:10:00'), " +
        "('S5', TIMESTAMP '2024-01-01 13:15:00'), ('S6', TIMESTAMP '2024-01-01 13:20:00')";

    private static string WriteParquet(string valuesSql)
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_batch_" + Guid.NewGuid().ToString("N") + ".parquet");
        using var conn = new DuckDBConnection("Data Source=:memory:");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = $"COPY (SELECT * FROM (VALUES {valuesSql}) AS t(sample, acq)) TO "
            + $"'{path.Replace("'", "''")}' (FORMAT PARQUET)";
        cmd.ExecuteNonQuery();
        return path;
    }

    [Fact]
    public void Estimate_GapDetection_SplitsAtLargeGap()
    {
        var path = WriteParquet(TimestampRows);
        try
        {
            var map = BatchEstimator.Estimate(path, "sample", "acq", method: "auto");
            Assert.Equal("batch_1", map["S1"]);
            Assert.Equal("batch_1", map["S3"]);
            Assert.Equal("batch_2", map["S4"]);
            Assert.Equal("batch_2", map["S6"]);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Estimate_Fixed_DividesByAcquisitionOrder()
    {
        var path = WriteParquet(TimestampRows);
        try
        {
            var map = BatchEstimator.Estimate(path, "sample", "acq", method: "fixed", nBatches: 3);
            Assert.Equal("batch_1", map["S1"]);
            Assert.Equal("batch_1", map["S2"]);
            Assert.Equal("batch_2", map["S3"]);
            Assert.Equal("batch_3", map["S6"]);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Estimate_ParsesStringTimestamps()
    {
        // Same clusters but the acquired-time column is VARCHAR, exercising TryParseTime's string path.
        var stringRows =
            "('S1', '2024-01-01 10:00:00'), ('S2', '2024-01-01 10:05:00'), ('S3', '2024-01-01 10:10:00'), " +
            "('S4', '2024-01-01 13:10:00'), ('S5', '2024-01-01 13:15:00'), ('S6', '2024-01-01 13:20:00')";
        var path = WriteParquet(stringRows);
        try
        {
            var map = BatchEstimator.Estimate(path, "sample", "acq", method: "auto");
            Assert.Equal("batch_1", map["S1"]);
            Assert.Equal("batch_2", map["S4"]);
        }
        finally { File.Delete(path); }
    }
}
