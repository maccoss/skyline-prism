using System;
using System.Collections.Generic;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The Differential pane's progress bar: shown only once work outlasts the delay, and hidden only
/// when the last of several overlapping pieces of work finishes.
/// </summary>
public class DelayedBusyIndicatorTests
{
    /// <summary>A delay that fires only when the test says so.</summary>
    private sealed class FakeDelay
    {
        private readonly List<(Action Action, Cancel Handle)> _pending = new();

        public IDisposable Schedule(Action action)
        {
            var handle = new Cancel();
            _pending.Add((action, handle));
            return handle;
        }

        public int Scheduled => _pending.Count;

        public void Elapse()
        {
            var due = _pending.ToArray();
            _pending.Clear();
            foreach (var (action, handle) in due)
                if (!handle.Cancelled)
                    action();
        }

        public sealed class Cancel : IDisposable
        {
            public bool Cancelled { get; private set; }

            public void Dispose() => Cancelled = true;
        }
    }

    private static (DelayedBusyIndicator Indicator, FakeDelay Delay, List<bool> Shown) Make()
    {
        var delay = new FakeDelay();
        var shown = new List<bool>();
        return (new DelayedBusyIndicator(shown.Add, delay.Schedule), delay, shown);
    }

    [Fact]
    public void WorkThatFinishesInsideTheDelay_NeverShowsTheBar()
    {
        var (busy, delay, shown) = Make();

        busy.Begin().Dispose();
        delay.Elapse(); // the timer would have fired after the work ended

        Assert.DoesNotContain(true, shown);
        Assert.False(busy.IsBusy);
    }

    [Fact]
    public void WorkThatOutlastsTheDelay_ShowsTheBarAndHidesItAtTheEnd()
    {
        var (busy, delay, shown) = Make();

        var work = busy.Begin();
        delay.Elapse();
        Assert.Equal(new[] { true }, shown);

        work.Dispose();
        Assert.Equal(new[] { true, false }, shown);
    }

    /// <summary>
    /// A level switch loads, then runs the contrast; a selector can start one contrast while another
    /// is in flight. The bar stays up until the last piece finishes, and the delay starts once.
    /// </summary>
    [Fact]
    public void OverlappingWork_HidesOnlyWhenTheLastPieceEnds()
    {
        var (busy, delay, shown) = Make();

        var first = busy.Begin();
        var second = busy.Begin();
        Assert.Equal(1, delay.Scheduled);
        delay.Elapse();

        first.Dispose();
        Assert.Equal(new[] { true }, shown);
        Assert.True(busy.IsBusy);

        second.Dispose();
        Assert.Equal(new[] { true, false }, shown);
        Assert.False(busy.IsBusy);
    }

    [Fact]
    public void DisposingAScopeTwice_DoesNotEndOtherWork()
    {
        var (busy, delay, _) = Make();

        var first = busy.Begin();
        var second = busy.Begin();
        first.Dispose();
        first.Dispose();

        Assert.True(busy.IsBusy);
        second.Dispose();
        Assert.False(busy.IsBusy);
    }

    [Fact]
    public void NewWorkAfterIdle_StartsAFreshDelay()
    {
        var (busy, delay, shown) = Make();

        busy.Begin().Dispose();
        var later = busy.Begin();
        Assert.Equal(2, delay.Scheduled);
        delay.Elapse(); // the first, cancelled timer must not show the bar; the second does

        Assert.Equal(new[] { false, true }, shown);
        later.Dispose();
    }
}
