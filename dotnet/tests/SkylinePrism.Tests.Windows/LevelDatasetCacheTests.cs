using System;
using System.Threading;
using System.Threading.Tasks;
using SkylinePrism.App;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The dataset cache the Markers pane and the Dynamic Range detail window share: each level read
/// once, neither level evicting the other, and a read already running joined rather than repeated.
/// </summary>
public class LevelDatasetCacheTests
{
    private sealed record Matrix(string Dir, FeatureLevel Level);

    private int _reads;

    private Func<Matrix> Read(string dir, FeatureLevel level) => () =>
    {
        Interlocked.Increment(ref _reads);
        return new Matrix(dir, level);
    };

    [Fact]
    public async Task BothLevelsStayCached_NeitherEvictsTheOther()
    {
        var cache = new LevelDatasetCache<Matrix>();

        var protein = await cache.GetAsync("run", FeatureLevel.Protein, Read("run", FeatureLevel.Protein));
        var peptide = await cache.GetAsync("run", FeatureLevel.Peptide, Read("run", FeatureLevel.Peptide));
        var proteinAgain = await cache.GetAsync("run", FeatureLevel.Protein, Read("run", FeatureLevel.Protein));
        var peptideAgain = await cache.GetAsync("RUN", FeatureLevel.Peptide, Read("run", FeatureLevel.Peptide));

        Assert.Same(protein, proteinAgain);
        Assert.Same(peptide, peptideAgain); // a directory is matched as Windows matches it
        Assert.Equal(2, _reads);
    }

    /// <summary>A click and a follow tick arriving before the first read finishes share that read.</summary>
    [Fact]
    public async Task ASecondCallerWhileTheReadRuns_WaitsForItInsteadOfStartingAnother()
    {
        var cache = new LevelDatasetCache<Matrix>();
        var release = new ManualResetEventSlim();
        Func<Matrix> slow = () =>
        {
            Interlocked.Increment(ref _reads);
            release.Wait();
            return new Matrix("run", FeatureLevel.Protein);
        };

        var first = cache.GetAsync("run", FeatureLevel.Protein, slow);
        var second = cache.GetAsync("run", FeatureLevel.Protein, slow);
        release.Set();

        Assert.Same(await first, await second);
        Assert.Equal(1, _reads);
    }

    [Fact]
    public async Task AFailedRead_IsNotKept()
    {
        var cache = new LevelDatasetCache<Matrix>();

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            cache.GetAsync("run", FeatureLevel.Protein, () => throw new InvalidOperationException("locked")));
        var retried = await cache.GetAsync("run", FeatureLevel.Protein, Read("run", FeatureLevel.Protein));

        Assert.Equal(new Matrix("run", FeatureLevel.Protein), retried);
    }

    [Fact]
    public async Task ANewDirectory_OrAClear_ReadsAgain()
    {
        var cache = new LevelDatasetCache<Matrix>();

        await cache.GetAsync("run1", FeatureLevel.Protein, Read("run1", FeatureLevel.Protein));
        var other = await cache.GetAsync("run2", FeatureLevel.Protein, Read("run2", FeatureLevel.Protein));
        Assert.Equal("run2", other.Dir);

        cache.Clear();
        await cache.GetAsync("run2", FeatureLevel.Protein, Read("run2", FeatureLevel.Protein));

        Assert.Equal(3, _reads);
    }
}
