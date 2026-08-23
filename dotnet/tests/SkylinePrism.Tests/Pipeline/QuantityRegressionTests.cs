using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// Bit-exact regression gate: every quantity the pipeline reports must be unchanged, to the last
/// bit, from the committed reference.
/// <para>
/// The cross-language parity tests (<see cref="PipelineParityTests"/>,
/// <see cref="PipelineMethodParityTests"/>) compare against the <b>Python</b> goldens with a
/// tolerance, which is right for two independent implementations but leaves a gap: a C# change
/// that shifts a value by less than 1e-9 passes them silently. These tests compare C# against its
/// own committed fingerprint, so any change to any value in any output column fails and names the
/// column that moved.
/// </para>
/// <para>
/// <b>When this fails, that is the test doing its job.</b> Do not regenerate the digests to make it
/// pass. Find out which quantity moved and why; if the change is intended and correct, regenerate
/// deliberately (see <c>dotnet/tests/fixtures/README.md</c>) and say so in the release notes,
/// because it means users' numbers change too.
/// </para>
/// </summary>
public class QuantityRegressionTests
{
    /// <summary>Set to regenerate the committed digests instead of asserting against them.</summary>
    private const string UpdateVar = "PRISM_UPDATE_DIGESTS";

    public static IEnumerable<object[]> Fixtures2() => new[]
    {
        new object[] { "e2e-sum" },
        new object[] { "e2e-medpolish" },
        new object[] { "e2e-maxlfq" },
        new object[] { "e2e-topn" },
        new object[] { "e2e-prot-topn" },
        new object[] { "e2e-consensus" },
    };

    [Theory]
    [MemberData(nameof(Fixtures2))]
    public void Quantities_AreBitIdenticalToCommittedReference(string fixture)
    {
        var mergeDir = TestSupport.Fixtures.Path2("mini", "merge");
        var dir = TestSupport.Fixtures.Path2("mini", fixture);
        var inputs = new[]
        {
            Path.Combine(mergeDir, "mini_plate1.csv"),
            Path.Combine(mergeDir, "mini_plate2.csv"),
        };
        var config = PrismConfig.Load(Path.Combine(dir, "config.yaml"));

        var tempOut = Path.Combine(Path.GetTempPath(), "prism_q_" + Guid.NewGuid().ToString("N"));
        try
        {
            PrismPipeline.Run(inputs, tempOut, config);
            var actual = QuantityDigest.Compute(tempOut);
            Assert.NotEmpty(actual);

            // The digests live beside the fixture's other goldens, in the source tree rather than
            // the copied output tree, so regeneration updates the committed file.
            var digestPath = Path.Combine(dir, "quantities.sha256");
            if (Environment.GetEnvironmentVariable(UpdateVar) == "1")
            {
                File.WriteAllLines(SourcePathFor(fixture), actual);
                return;
            }

            Assert.True(File.Exists(digestPath),
                $"No committed digest for {fixture}. Generate it with {UpdateVar}=1 "
                + "(see dotnet/tests/fixtures/README.md).");

            var expected = File.ReadAllLines(digestPath)
                .Where(l => l.Length > 0 && !l.StartsWith('#'))
                .ToList();
            AssertSame(fixture, expected, actual);
        }
        finally
        {
            if (Directory.Exists(tempOut))
                Directory.Delete(tempOut, recursive: true);
        }
    }

    /// <summary>
    /// Compare digests and report what actually moved - which columns changed, and any that
    /// appeared or vanished - rather than a bare "hashes differ".
    /// </summary>
    private static void AssertSame(string fixture, List<string> expected, List<string> actual)
    {
        static Dictionary<string, string> Index(IEnumerable<string> lines)
        {
            var d = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var line in lines)
            {
                var parts = line.Split('\t');
                if (parts.Length == 3)
                    d[parts[0] + "\t" + parts[1]] = parts[2];
            }
            return d;
        }

        var e = Index(expected);
        var a = Index(actual);
        var problems = new List<string>();

        foreach (var (key, hash) in e.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            if (!a.TryGetValue(key, out var got))
                problems.Add($"  MISSING  {key.Replace('\t', ' ')}");
            else if (got != hash)
                problems.Add($"  CHANGED  {key.Replace('\t', ' ')}");
        }
        foreach (var key in a.Keys.Except(e.Keys, StringComparer.Ordinal).OrderBy(k => k, StringComparer.Ordinal))
            problems.Add($"  NEW      {key.Replace('\t', ' ')}");

        Assert.True(problems.Count == 0,
            $"{fixture}: {problems.Count} column(s) no longer match the committed quantities.\n"
            + string.Join("\n", problems.Take(40))
            + (problems.Count > 40 ? $"\n  ... and {problems.Count - 40} more" : "")
            + $"\n\nA quantity changed. Do not regenerate to silence this - establish which value moved\n"
            + $"and why. If the change is intended, regenerate with {UpdateVar}=1 and note it in the\n"
            + "release notes, because users' numbers change too.");
    }

    /// <summary>
    /// Path to the digest in the SOURCE tree. Fixtures are copied next to the test assembly, so
    /// writing to the copy would be discarded on the next build.
    /// </summary>
    private static string SourcePathFor(string fixture)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "dotnet", "tests", "fixtures")))
            dir = dir.Parent;
        if (dir is null)
            throw new InvalidOperationException("Could not locate the repo root from the test assembly.");
        return Path.Combine(dir.FullName, "dotnet", "tests", "fixtures", "mini", fixture, "quantities.sha256");
    }
}
