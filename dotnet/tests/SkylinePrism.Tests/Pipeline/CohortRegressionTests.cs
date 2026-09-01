using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Pipeline;
using SkylinePrism.Tests.TestSupport;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// The bit-exact regression gate on a REAL cohort, run from the merge boundary so one fixture drives
/// every downstream method.
///
/// <para><b>Why this exists alongside <see cref="QuantityRegressionTests"/>.</b> Those run on
/// <c>mini</c> - 2000 rows sliced from two plates - which is enough to pin the arithmetic but not the
/// structure the algorithms actually respond to. This fixture is 1,129,728 rows: 327 peptides across
/// 52 selected proteins (123 distinct protein assignments once shared peptides are followed), 192
/// samples, 2 batches, 16 reference / 16 QC / 160 experimental. So ComBat estimates its priors from a
/// realistic feature set, protein rollup sees proteins with 1 to 30+ peptides rather than uniform
/// ones, parsimony has 47 peptides genuinely shared between two selected proteins, and rt_lowess fits
/// against a real retention-time distribution.</para>
///
/// <para><b>Why it starts at the merge.</b> The input is the merged parquet, not the Skyline exports,
/// which is what makes one fixture serve every method: Stage 1 is deterministic and already covered
/// (<c>MergeParityTests</c>, <c>ExportFormatParityTests</c>), it is the slowest stage, and it depends
/// on 2.7 GB of raw exports that cannot be committed. Everything the downstream methods need is
/// carried in the merged table - <c>ShapeCorrelation</c> for top-N by correlation, <c>ProductMz</c>
/// for library-assisted rollup, <c>Batch</c> for ComBat, <c>AcquiredTime</c> for batch estimation.</para>
///
/// <para><b>The fixture is anonymized.</b> Replicates are <c>S001..S189</c>, batches <c>Batch1</c> and
/// <c>Batch2</c>, filenames derived from the replicate, and acquisition timestamps shifted to a fixed
/// epoch - the GAPS are preserved because batch estimation reads them, but the absolute times are not
/// this repository's to publish. Peptide sequences and UniProt accessions are kept as they are.</para>
///
/// <para><b>Two structural properties are deliberate</b>, because they are what real cohorts do and
/// what has broken before: 189 replicate names produce 192 sample IDs, because three QC injections
/// carry the same name in both batches (see the caution in CLAUDE.md about replicate names colliding
/// across documents); and the two metadata files use DIFFERENT header names - <c>ReplicateName</c>
/// from the RPC export path, <c>Replicate</c> from the headless one - which is what a mixed cohort
/// really presents.</para>
///
/// <para><b>When this fails, that is the test doing its job.</b> Same rule as the mini digests: find
/// out which quantity moved and why before even considering regeneration, because a change here means
/// users' numbers change too.</para>
/// </summary>
public class CohortRegressionTests
{
    /// <summary>Set to regenerate the committed digests instead of asserting against them.</summary>
    private const string UpdateVar = "PRISM_UPDATE_DIGESTS";

    /// <summary>Rows in the committed fixture; the merge cache must agree or Stage 1 re-runs.</summary>
    private const long FixtureRows = 1_129_728;

    /// <summary>
    /// The partition key the merge recorded for this table. Taken verbatim from the cache the real run
    /// wrote beside this data rather than re-derived, so the fixture cannot disagree with the merge
    /// that produced it.
    /// </summary>
    private const string FixturePartitionKey = "Peptide";

    /// <summary>
    /// One case per method combination worth pinning. Each is a directory under
    /// <c>fixtures/cohort/</c> holding <c>config.yaml</c> and <c>quantities.sha256</c>.
    /// </summary>
    public static IEnumerable<object[]> Cases() => new[]
    {
        new object[] { "sum-rtlowess-combat" },
        new object[] { "medpolish" },
        new object[] { "topn-correlation" },
        new object[] { "topn-intensity" },
        new object[] { "consensus" },
        new object[] { "protein-maxlfq" },
        new object[] { "protein-topn" },
        new object[] { "norm-median" },
        new object[] { "norm-quantile" },
        new object[] { "no-combat" },
        new object[] { "library-assist" },
    };

    [ReferencePlatformTheory]
    [MemberData(nameof(Cases))]
    public void CohortQuantities_AreBitIdenticalToCommittedReference(string caseName)
    {
        var cohort = Fixtures.Path2("cohort");
        var caseDir = Path.Combine(cohort, caseName);
        var config = PrismConfig.Load(Path.Combine(caseDir, "config.yaml"));

        // The spectral library ships beside the fixture, and the config names it relatively because the
        // fixture's absolute location depends on where the test assembly was built. LibraryPath is used
        // verbatim by the rollup, so it has to be made absolute here.
        if (!string.IsNullOrWhiteSpace(config.TransitionRollup.LibraryPath)
            && !Path.IsPathRooted(config.TransitionRollup.LibraryPath))
        {
            config.TransitionRollup.LibraryPath =
                Path.Combine(cohort, config.TransitionRollup.LibraryPath);
        }

        var tempOut = Path.Combine(Path.GetTempPath(), "prism_cohort_" + Guid.NewGuid().ToString("N"));
        try
        {
            var inputs = SeedMergedData(cohort, tempOut, config);
            var metadata = new[]
            {
                Path.Combine(cohort, "Batch1.metadata.csv"),
                Path.Combine(cohort, "Batch2.metadata.csv"),
            };

            var result = PrismPipeline.Run(inputs, tempOut, config, metadata);

            // The seeded merge must have been REUSED, not redone. If Stage 1 re-ran it would have read
            // the empty placeholder inputs, and the digests below would be hashes of nothing - passing
            // or failing for reasons that have nothing to do with the algorithms. Assert the structure
            // rather than merely that merged_data still exists, which a re-run would also satisfy.
            // These three hold for every case; the per-case filters only move the peptide and protein
            // counts, so those are pinned by the digests themselves.
            Assert.Equal(192, result.NSamples);
            Assert.Equal(new[] { "Batch1", "Batch2" }, result.Batches.OrderBy(b => b, StringComparer.Ordinal));
            Assert.True(result.NPeptides > 0 && result.NProteins > 0,
                $"cohort/{caseName} produced {result.NPeptides} peptides and {result.NProteins} proteins");

            var actual = QuantityDigest.Compute(tempOut);
            Assert.NotEmpty(actual);

            var digestPath = Path.Combine(caseDir, "quantities.sha256");
            if (Environment.GetEnvironmentVariable(UpdateVar) == "1")
            {
                // Record what the case produced. The reader skips '#' lines, so this is only a comment -
                // but it puts the shape of each case in the committed file and therefore in the diff,
                // which is how a reviewer sees that (say) library_assist still matched most peptides
                // rather than silently dropping them and leaving a digest that still looks valid.
                var header = new[]
                {
                    $"# case={caseName} peptides={result.NPeptides} proteins={result.NProteins} "
                    + $"samples={result.NSamples} batches={result.Batches.Count}",
                };
                File.WriteAllLines(SourceDigestPath(caseName), header.Concat(actual));
                return;
            }

            Assert.True(File.Exists(digestPath),
                $"No committed digest for cohort/{caseName}. Generate it with {UpdateVar}=1 "
                + "(see dotnet/tests/fixtures/README.md).");

            var expected = File.ReadAllLines(digestPath)
                .Where(l => l.Length > 0 && !l.StartsWith('#'))
                .ToList();
            AssertSame(caseName, expected, actual);
        }
        finally
        {
            TryDelete(tempOut);
        }
    }

    /// <summary>
    /// Put the committed merged table where the pipeline expects it, and write a cache entry that
    /// matches, so Stage 1 is skipped and never opens an input file.
    ///
    /// <para>This works because <see cref="SourceFingerprint.Compute"/> hashes each input's path,
    /// length and last-write time - not its contents - so empty placeholders are enough to reproduce
    /// the fingerprint the cache records. The placeholders exist only to be fingerprinted; if Stage 1
    /// ever did run, the assertion in the caller catches it rather than letting an empty merge through.</para>
    /// </summary>
    private static string[] SeedMergedData(string cohort, string tempOut, PrismConfig config)
    {
        Directory.CreateDirectory(tempOut);

        var mergedDir = Path.Combine(tempOut, "merged_data");
        foreach (var bucket in Directory.GetDirectories(cohort, "_pep_bucket=*"))
        {
            var target = Path.Combine(mergedDir, Path.GetFileName(bucket));
            Directory.CreateDirectory(target);
            foreach (var f in Directory.GetFiles(bucket))
                File.Copy(f, Path.Combine(target, Path.GetFileName(f)), overwrite: true);
        }

        var inputs = new[] { "Batch1", "Batch2" }
            .Select(b => Path.Combine(tempOut, b + ".parquet"))
            .ToArray();
        foreach (var p in inputs)
            File.WriteAllBytes(p, Array.Empty<byte>());

        // Exactly how PrismPipeline builds it: the source fingerprint plus the merge stage's own
        // config values, so a changed data.* override still invalidates the cache.
        var fingerprint = SourceFingerprint.Compute(inputs)
            + "|" + StageDependencies.Values(StageDependencies.Merge, config);
        var entry = new SourceFingerprint.CacheEntry(fingerprint, FixtureRows, FixturePartitionKey);
        File.WriteAllText(Path.Combine(tempOut, "merged_data.cache.json"), JsonSerializer.Serialize(entry));

        return inputs;
    }

    /// <summary>Digest path in the SOURCE tree, so regeneration updates the committed file.</summary>
    private static string SourceDigestPath(string caseName)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "SkylinePrism.CrossPlatform.slnf")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        return Path.Combine(dir!.FullName, "tests", "fixtures", "cohort", caseName, "quantities.sha256");
    }

    /// <summary>
    /// Report the columns that moved, not just that something did - a bare "digests differ" on a
    /// thousand-line file tells the reader nothing about where to look.
    /// </summary>
    private static void AssertSame(string caseName, List<string> expected, List<string> actual)
    {
        var exp = expected.ToHashSet(StringComparer.Ordinal);
        var act = actual.ToHashSet(StringComparer.Ordinal);
        if (exp.SetEquals(act))
            return;

        var expKeys = expected.ToDictionary(Key, v => v, StringComparer.Ordinal);
        var actKeys = actual.ToDictionary(Key, v => v, StringComparer.Ordinal);
        var moved = actKeys.Where(kv => expKeys.TryGetValue(kv.Key, out var e) && e != kv.Value)
            .Select(kv => kv.Key).OrderBy(k => k, StringComparer.Ordinal).ToList();
        var added = actKeys.Keys.Except(expKeys.Keys).OrderBy(k => k, StringComparer.Ordinal).ToList();
        var gone = expKeys.Keys.Except(actKeys.Keys).OrderBy(k => k, StringComparer.Ordinal).ToList();

        Assert.Fail(
            $"cohort/{caseName}: quantities changed.\n"
            + $"  {moved.Count} column(s) moved: {string.Join(", ", moved.Take(8))}"
            + (moved.Count > 8 ? " ..." : "") + "\n"
            + $"  {added.Count} added: {string.Join(", ", added.Take(5))}\n"
            + $"  {gone.Count} missing: {string.Join(", ", gone.Take(5))}\n"
            + "This is the gate working. Find out which quantity moved and why before regenerating.");

        static string Key(string line)
        {
            var i = line.LastIndexOf('\t');
            return i < 0 ? line : line[..i];
        }
    }

    private static void TryDelete(string dir)
    {
        try
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
        catch (IOException)
        {
            // A leftover temp directory costs disk, not correctness.
        }
    }
}
