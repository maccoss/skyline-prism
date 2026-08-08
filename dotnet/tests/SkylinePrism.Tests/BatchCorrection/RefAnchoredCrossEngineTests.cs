using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using SkylinePrism.Core.BatchCorrection;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.BatchCorrection;

/// <summary>
/// Cross-engine parity for reference-anchored ComBat, against fixtures generated from the Python
/// engine (<c>dotnet/tests/fixtures/refanchored/generate.py</c>).
///
/// <para>Standard ComBat is held to R's <c>sva</c>; the end-to-end mini fixtures hold the two
/// engines to each other. Reference-anchored had neither - it is PRISM's own method, so there is no
/// external implementation to check it against, and no fixture reached it. The two engines could
/// drift apart on it indefinitely with every test still green, which is the same blind spot that
/// let the standard path's NaN bug survive.</para>
///
/// <para>With no third party to appeal to, these pin Python's output and hold C# to it. That is
/// weaker than the sva goldens - it proves agreement, not correctness - but it converts a silent
/// drift into a failing test, and the behavior itself is pinned separately by
/// <see cref="ReferenceAnchoredComBatTests"/>.</para>
/// </summary>
public class RefAnchoredCrossEngineTests
{
    private static string Dir => Fixtures.Path2("refanchored");

    [Theory]
    [InlineData("dense")]
    [InlineData("sparse")]
    [InlineData("constant_in_batch")]
    [InlineData("single_ref_and_fallback")]
    [InlineData("skip_unreferenced_batch")]
    public void CSharp_MatchesThePythonEngine(string name)
    {
        var input = ReadMatrix(Path.Combine(Dir, name + "_input.csv"));
        var expected = ReadMatrix(Path.Combine(Dir, name + "_expected.csv"));
        var batchLabels = File.ReadAllLines(Path.Combine(Dir, name + "_batches.csv"))
            .Where(l => l.Length > 0).ToList();
        var refMask = File.ReadAllLines(Path.Combine(Dir, name + "_refmask.csv"))
            .Where(l => l.Length > 0).Select(l => l.Trim() == "1").ToList();
        var policy = File.ReadAllText(Path.Combine(Dir, name + "_policy.txt")).Trim();

        var actual = ReferenceAnchoredComBat.Run(
            input, batchLabels, refMask, noReferenceBatch: policy);

        Assert.Equal(expected.GetLength(0), actual.GetLength(0));
        Assert.Equal(expected.GetLength(1), actual.GetLength(1));

        var worst = 0.0;
        var compared = 0;
        for (var f = 0; f < expected.GetLength(0); f++)
            for (var s = 0; s < expected.GetLength(1); s++)
            {
                // A missing value must stay exactly where it was, in both engines.
                if (double.IsNaN(expected[f, s]) || double.IsNaN(actual[f, s]))
                {
                    Assert.True(double.IsNaN(expected[f, s]) && double.IsNaN(actual[f, s]),
                        $"{name}[{f},{s}]: python {expected[f, s]} vs C# {actual[f, s]}");
                    continue;
                }
                worst = Math.Max(worst, RelativeDifference(expected[f, s], actual[f, s]));
                compared++;
            }

        Assert.True(compared > 0, "compared nothing");

        // Not bit-identity: numpy's reductions sum in a different order from ours, so the last few
        // digits legitimately differ. Anything above this is an algorithmic divergence, not rounding.
        Assert.True(worst < 1e-10,
            $"{name}: worst relative difference between engines {worst:E3} over {compared} cells");
    }

    /// <summary>
    /// The fixtures have to actually reach the paths they are named for, or they pass while testing
    /// nothing. Checked here rather than trusted from the generator.
    /// </summary>
    [Fact]
    public void FixturesCoverTheCasesTheyClaim()
    {
        Assert.True(Count("sparse", double.IsNaN) > 0, "the 'sparse' fixture has no missing values");
        Assert.Equal(0, Count("dense", double.IsNaN));

        // single_ref_and_fallback: one batch with exactly one reference, one with none.
        var counts = ReferencesPerBatch("single_ref_and_fallback");
        Assert.Contains(1, counts);
        Assert.Contains(0, counts);

        // skip_unreferenced_batch: a batch with no references, under the "skip" policy.
        Assert.Contains(0, ReferencesPerBatch("skip_unreferenced_batch"));
        Assert.Equal("skip",
            File.ReadAllText(Path.Combine(Dir, "skip_unreferenced_batch_policy.txt")).Trim());
    }

    private static List<int> ReferencesPerBatch(string name)
    {
        var batches = File.ReadAllLines(Path.Combine(Dir, name + "_batches.csv"))
            .Where(l => l.Length > 0).ToList();
        var mask = File.ReadAllLines(Path.Combine(Dir, name + "_refmask.csv"))
            .Where(l => l.Length > 0).Select(l => l.Trim() == "1").ToList();
        return batches.Select((b, i) => (b, i))
            .GroupBy(x => x.b, StringComparer.Ordinal)
            .Select(g => g.Count(x => mask[x.i]))
            .ToList();
    }

    private static int Count(string name, Func<double, bool> predicate)
    {
        var m = ReadMatrix(Path.Combine(Dir, name + "_input.csv"));
        var n = 0;
        foreach (var v in m)
            if (predicate(v))
                n++;
        return n;
    }

    /// <summary>Bare numeric CSV as written by numpy's savetxt - no header, no row names.</summary>
    private static double[,] ReadMatrix(string path)
    {
        var lines = File.ReadAllLines(path).Where(l => l.Length > 0).ToArray();
        var width = lines[0].Split(',').Length;
        var matrix = new double[lines.Length, width];
        for (var i = 0; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            Assert.Equal(width, parts.Length);
            for (var j = 0; j < width; j++)
                matrix[i, j] = ParseValue(parts[j]);
        }
        return matrix;
    }

    /// <summary>numpy writes a missing value as "nan", which double.Parse will not take.</summary>
    private static double ParseValue(string token)
    {
        token = token.Trim();
        if (token.Equals("nan", StringComparison.OrdinalIgnoreCase))
            return double.NaN;
        return double.Parse(token, CultureInfo.InvariantCulture);
    }

    private static double RelativeDifference(double expected, double actual)
    {
        if (expected == actual)
            return 0;
        return Math.Abs(expected - actual) / Math.Max(Math.Abs(expected), 1e-30);
    }
}
