using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.IO;
using Xunit;

namespace SkylinePrism.Tests.IO;

/// <summary>Acquisition-time batch estimation: gap detection and fixed division.</summary>
public class BatchEstimatorTests
{
    [Fact]
    public void Gap_SplitsAtLargeAcquisitionGap()
    {
        // Two clusters of 4 samples, ~5 min apart within, with a 600 min gap between them.
        var t0 = new DateTime(2026, 1, 1, 0, 0, 0);
        var rows = new List<(string, DateTime)>();
        for (var i = 0; i < 4; i++)
            rows.Add(($"a{i}", t0.AddMinutes(5 * i)));
        var t1 = t0.AddMinutes(5 * 3 + 600);
        for (var i = 0; i < 4; i++)
            rows.Add(($"b{i}", t1.AddMinutes(5 * i)));

        var map = BatchEstimator.AssignBatches(rows, "auto", nBatches: null, gapIqrMultiplier: 1.5);

        Assert.Equal(2, map.Values.Distinct().Count());
        Assert.All(new[] { "a0", "a1", "a2", "a3" }, s => Assert.Equal(map["a0"], map[s]));
        Assert.All(new[] { "b0", "b1", "b2", "b3" }, s => Assert.Equal(map["b0"], map[s]));
        Assert.NotEqual(map["a0"], map["b0"]);
    }

    [Fact]
    public void Gap_NoLargeGap_ReturnsEmpty()
    {
        var t0 = new DateTime(2026, 1, 1, 0, 0, 0);
        var rows = Enumerable.Range(0, 6).Select(i => ($"s{i}", t0.AddMinutes(5 * i))).ToList();
        var map = BatchEstimator.AssignBatches(rows, "auto", null, 1.5);
        Assert.Empty(map); // evenly spaced -> single batch -> no assignment
    }

    [Fact]
    public void Fixed_DividesByAcquisitionOrder()
    {
        var t0 = new DateTime(2026, 1, 1, 0, 0, 0);
        var rows = Enumerable.Range(0, 5).Select(i => ($"s{i}", t0.AddMinutes(i))).ToList();
        var map = BatchEstimator.AssignBatches(rows, "fixed", nBatches: 2, gapIqrMultiplier: 1.5);
        // 5 samples into 2 batches -> 3 + 2 (remainder to the first).
        Assert.Equal("batch_1", map["s0"]);
        Assert.Equal("batch_1", map["s2"]);
        Assert.Equal("batch_2", map["s3"]);
        Assert.Equal("batch_2", map["s4"]);
    }
}
