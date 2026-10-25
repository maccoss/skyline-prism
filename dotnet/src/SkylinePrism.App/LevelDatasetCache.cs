using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using SkylinePrism.Core.DifferentialAnalysis;

namespace SkylinePrism.App;

/// <summary>
/// The corrected matrices of ONE output directory, read once per level and shared by every pane that
/// reads them - the Markers pane, and the Dynamic Range pane's per-feature window.
/// </summary>
/// <remarks>
/// <para>It replaced a single-slot cache the Markers pane owned. Once the Dynamic Range click read
/// through it too, a pane on the other level evicted the Markers pane's matrix - so switching back
/// re-read the parquet - and a click plus a Skyline-follow tick arriving before the first read
/// finished both started a read of the same file.</para>
/// <para>Holding the LOAD (a task) rather than the result is what makes a second caller wait for
/// the read already running instead of starting its own. A failed read is not kept, so the next
/// call tries again.</para>
/// <para>At most one entry per level: a new directory drops everything, since a matrix at
/// peptide level is the size of the cohort and the old directory's will not be asked for again.
/// Used from the UI thread only; nothing here is locked.</para>
/// </remarks>
internal sealed class LevelDatasetCache<T> where T : class
{
    private readonly Dictionary<FeatureLevel, Task<T>> _loads = new();
    private string? _dir;

    /// <summary>
    /// The level's dataset, starting <paramref name="load"/> on a worker thread only if no read of
    /// that level has succeeded or is running.
    /// </summary>
    /// <param name="load">
    /// Reads the level. Passed per call, so it can carry what was current when it was asked for -
    /// the additional metadata file to join - rather than what was current when the cache was made.
    /// </param>
    public Task<T> GetAsync(string dir, FeatureLevel level, Func<T> load)
    {
        if (!string.Equals(dir, _dir, StringComparison.OrdinalIgnoreCase))
        {
            _loads.Clear();
            _dir = dir;
        }

        if (_loads.TryGetValue(level, out var running) && !running.IsFaulted && !running.IsCanceled)
            return running;

        var started = Task.Run(load);
        _loads[level] = started;
        return started;
    }

    /// <summary>
    /// Forget everything: a re-run into the same directory, or a metadata table joined since the
    /// matrices were read. A read still running completes for whoever awaits it, but is not reused.
    /// </summary>
    public void Clear()
    {
        _loads.Clear();
        _dir = null;
    }
}
