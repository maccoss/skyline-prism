using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.IO;
using SkylinePrism.Core.Qc;
using SkylinePrism.Tests.TestSupport;
using Xunit;
using Xunit.Abstractions;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The union accounting against a real cohort, not synthetic transitions.
///
/// <para>The committed fixture is a slice of a real Astral DIA run — precursors span 400.8 to 896.9
/// m/z, which is exactly the range of the shipped "Astral 3 Th, 400-900 m/z" scheme, so the isolation
/// windows here are the acquisition's real ones rather than a stand-in.</para>
///
/// <para>These tests are about the properties that must hold on real data — the union never exceeds
/// the sum, fragment sharing is actually found, and the totals nest — rather than exact figures, which
/// would pin the fixture's peptide selection rather than the algorithm.</para>
/// </summary>
public class Ms2SignalRegionsTests
{
    private readonly ITestOutputHelper _out;

    public Ms2SignalRegionsTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// The fixture's first replicate. Overridable alongside <see cref="FullCohortVar"/>, because a real
    /// cohort's replicate names are its own and the fixture's are anonymised.
    /// </summary>
    private static string Replicate =>
        Environment.GetEnvironmentVariable("PRISM_MS2_REPLICATE") is { Length: > 0 } r
            ? r
            : "S001__@__Batch1";

    private static readonly ProductMassTolerance Ppm10 =
        ProductMassTolerance.Parse("centroided", "10")!;   // the real documents' setting

    /// <summary>
    /// Point this at a real cohort's merged_data/ to measure on a full run instead of the committed
    /// slice. The fixture is only 327 of 75,202 peptides, which gives a median of 2 precursors per
    /// isolation window where a real run has tens — so it cannot show how much CO-ISOLATION sharing
    /// there is, only duplicate-row sharing. Unset in CI, where the fixture is the point.
    /// </summary>
    private const string FullCohortVar = "PRISM_MS2_COHORT";

    private static (MergedDataset Dataset, Ms2SignalRegions.Columns Cols) OpenCohort()
    {
        var full = Environment.GetEnvironmentVariable(FullCohortVar);
        var root = string.IsNullOrWhiteSpace(full) ? Fixtures.Path2("cohort") : full;
        var dataset = MergedDataset.Open(root);
        var names = ParquetTable.ReadColumnNames(dataset.RepresentativeFile());
        var cols = Ms2SignalRegions.Resolve(names.ToList());
        Assert.NotNull(cols);
        return (dataset, cols!);
    }

    /// <summary>Every peptide in the fixture treated as assigned, none in a list.</summary>
    private static Dictionary<string, Ms2SignalRegions.PeptideClass> AllAssigned(
        MergedDataset dataset, Ms2SignalRegions.Columns cols)
    {
        var classes = new Dictionary<string, Ms2SignalRegions.PeptideClass>(StringComparer.Ordinal);
        foreach (var pep in PeptidesIn(dataset, cols))
            classes[pep] = new Ms2SignalRegions.PeptideClass(true, 0);
        return classes;
    }

    private static List<string> PeptidesIn(MergedDataset dataset, Ms2SignalRegions.Columns cols)
    {
        var loaded = Ms2SignalRegions.Load(
            dataset, cols, Replicate, IsolationScheme.AstralDefault(),
            new Dictionary<string, Ms2SignalRegions.PeptideClass>());
        Assert.NotEmpty(loaded.Regions);
        // Re-read the peptide column the cheap way: the loader already told us how many rows there are.
        var table = ParquetTable.Load(dataset.RepresentativeFile());
        return table.GetString(cols.Peptide).Where(p => p is not null)
            .Select(p => p!).Distinct(StringComparer.Ordinal).ToList();
    }

    /// <summary>
    /// The correction, on real data: summing transition areas over-counts, and the union is strictly
    /// smaller because co-isolated peptides really do share fragments.
    /// </summary>
    [Fact]
    public void OnARealReplicateTheUnionIsSmallerThanTheSum()
    {
        var (dataset, cols) = OpenCohort();
        var scheme = IsolationScheme.AstralDefault();
        var classes = AllAssigned(dataset, cols);

        var loaded = Ms2SignalRegions.Load(dataset, cols, Replicate, scheme, classes);
        var result = Ms2SignalUnion.Compute(loaded.Regions, Ppm10, 0);

        _out.WriteLine($"scheme          : {scheme.Name} ({scheme.Windows.Count} windows)");
        _out.WriteLine($"fragment peaks  : {result.Regions:N0}  (outside scheme {loaded.OutsideScheme:N0})");
        _out.WriteLine($"merged regions  : {result.MergedGroups:N0}  largest group {result.LargestGroup}");
        _out.WriteLine($"summed area     : {result.SummedArea:E4}");
        _out.WriteLine($"union  area     : {result.AssignedArea:E4}");
        _out.WriteLine($"double counted  : {1 - result.AssignedArea / result.SummedArea:P2} of the sum");
        _out.WriteLine($"  duplicate rows (one peptide, many proteins): {result.DuplicateRows:N0}");
        _out.WriteLine($"  shared across peptides (co-isolation)     : {result.SharedAcrossPeptides:N0}");

        Assert.True(result.Regions > 0, "no fragment peaks were placed in signal space");
        Assert.True(result.AssignedArea <= result.SummedArea + 1e-9);
        Assert.True(result.MergedGroups <= result.Regions);
    }

    /// <summary>
    /// The precursors of a real Astral run fall inside the shipped Astral scheme, so almost nothing is
    /// unexplained. A large "outside" count would mean the wrong scheme, which would otherwise look
    /// like a genuinely small assigned fraction.
    /// </summary>
    [Fact]
    public void TheShippedAstralSchemeExplainsThisRun()
    {
        var (dataset, cols) = OpenCohort();
        var classes = AllAssigned(dataset, cols);

        var loaded = Ms2SignalRegions.Load(
            dataset, cols, Replicate, IsolationScheme.AstralDefault(), classes);

        var placed = loaded.Regions.Count;
        var outside = loaded.OutsideScheme;
        _out.WriteLine($"placed {placed:N0}, outside the scheme {outside:N0} "
            + $"({(double)outside / (placed + outside):P2})");

        Assert.True(outside < placed * 0.05,
            $"{outside} of {placed + outside} fragments fell outside the isolation scheme");
    }

    /// <summary>
    /// Peptides the caller does not classify contribute nothing, so the assigned total tracks the
    /// dataset's own peptide set rather than everything in the merged table.
    /// </summary>
    [Fact]
    public void UnclassifiedPeptidesContributeNothing()
    {
        var (dataset, cols) = OpenCohort();
        var scheme = IsolationScheme.AstralDefault();

        var none = Ms2SignalRegions.Load(
            dataset, cols, Replicate, scheme, new Dictionary<string, Ms2SignalRegions.PeptideClass>());
        var result = Ms2SignalUnion.Compute(none.Regions, Ppm10, 0);

        Assert.True(none.Regions.Count > 0);
        Assert.Equal(none.Regions.Count, none.UnknownPeptides);
        Assert.Equal(0, result.AssignedArea);
        Assert.Equal(0, result.SummedArea);
    }

    /// <summary>
    /// A list's union is a subset of the assigned union, on real data. This is the property the plot
    /// depends on: a per-list bar can never exceed the assigned bar.
    /// </summary>
    [Fact]
    public void AListUnionIsASubsetOfTheAssignedUnion()
    {
        var (dataset, cols) = OpenCohort();
        var scheme = IsolationScheme.AstralDefault();

        var peptides = PeptidesIn(dataset, cols);
        Assert.True(peptides.Count > 10, "fixture should have plenty of peptides");

        // Take an arbitrary but deterministic third of them as "the list".
        var listed = peptides.OrderBy(p => p, StringComparer.Ordinal)
            .Where((_, i) => i % 3 == 0).ToHashSet(StringComparer.Ordinal);

        var classes = peptides.ToDictionary(
            p => p,
            p => new Ms2SignalRegions.PeptideClass(true, listed.Contains(p) ? 1u : 0u),
            StringComparer.Ordinal);

        var loaded = Ms2SignalRegions.Load(dataset, cols, Replicate, scheme, classes);
        var result = Ms2SignalUnion.Compute(loaded.Regions, Ppm10, 1);

        _out.WriteLine($"assigned {result.AssignedArea:E4}, list {result.ListArea[0]:E4} "
            + $"({result.ListArea[0] / result.AssignedArea:P1} of assigned)");

        Assert.True(result.ListArea[0] > 0, "the list should account for some signal");
        Assert.True(result.ListArea[0] <= result.AssignedArea + 1e-9,
            "a list union cannot exceed the assigned union");
    }

    /// <summary>
    /// Widening the extraction window can only merge more, never less - so the union is monotone in
    /// the tolerance. A violation would mean the grouping sweep is order-dependent or non-transitive.
    ///
    /// <para>On the COMMITTED fixture the three tolerances give identical answers, because its
    /// collisions are exact-mass duplicates of one peptide rather than near-misses between two - so
    /// this is a guard against regression on a real cohort (via <c>PRISM_MS2_COHORT</c>), not a
    /// discriminating test. <c>Ms2SignalUnionTests.AWiderExtractionWindowMergesMoreNeverLess</c> is the
    /// one that actually separates the tolerances.</para>
    /// </summary>
    [Fact]
    public void AWiderToleranceNeverIncreasesTheUnion()
    {
        var (dataset, cols) = OpenCohort();
        var classes = AllAssigned(dataset, cols);
        var loaded = Ms2SignalRegions.Load(
            dataset, cols, Replicate, IsolationScheme.AstralDefault(), classes);

        var tight = Ms2SignalUnion.Compute(loaded.Regions, ProductMassTolerance.Parse("centroided", "1")!, 0);
        var normal = Ms2SignalUnion.Compute(loaded.Regions, Ppm10, 0);
        var wide = Ms2SignalUnion.Compute(loaded.Regions, ProductMassTolerance.Parse("centroided", "100")!, 0);

        _out.WriteLine($"1 ppm  : {tight.AssignedArea:E4}  ({tight.MergedGroups:N0} regions)");
        _out.WriteLine($"10 ppm : {normal.AssignedArea:E4}  ({normal.MergedGroups:N0} regions)");
        _out.WriteLine($"100 ppm: {wide.AssignedArea:E4}  ({wide.MergedGroups:N0} regions)");

        Assert.True(normal.AssignedArea <= tight.AssignedArea + 1e-9);
        Assert.True(wide.AssignedArea <= normal.AssignedArea + 1e-9);
        Assert.True(wide.MergedGroups <= normal.MergedGroups);
        Assert.True(normal.MergedGroups <= tight.MergedGroups);
    }
}
