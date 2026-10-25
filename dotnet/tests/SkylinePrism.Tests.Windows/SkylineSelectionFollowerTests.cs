using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The rules every pane that follows Skyline's selection relies on, pinned without Skyline or a
/// dispatcher: the RPC is passed in, and a pane is a <see cref="SkylineSelectionFollower.Poll"/>.
/// </summary>
public class SkylineSelectionFollowerTests
{
    private sealed class Harness
    {
        public readonly List<string> Asked = new();
        public readonly List<(string Locator, object? Item)> Applied = new();
        public readonly List<string> Logged = new();
        public readonly List<string> Traced = new();
        public Func<string, string?> Selection = _ => "Molecule:/sp|P1|A/PEPC[+57.021464]K";
        public Func<string, object?> Resolve = locator => "feature:" + locator;
        public readonly SkylineSelectionFollower Follower;

        public Harness() => Follower = new SkylineSelectionFollower(
            type =>
            {
                Asked.Add(type);
                return Selection(type);
            },
            Traced.Add,
            Logged.Add);

        public SkylineSelectionFollower.Poll Poll(string type = "Molecule") =>
            new(type, l => Resolve(l), (l, item) => Applied.Add((l, item)));
    }

    [Fact]
    public async Task APoll_AsksForThePanesLevel_AndAppliesWhatItResolvedTo()
    {
        var h = new Harness();
        h.Follower.Follow(() => h.Poll("MoleculeGroup"));

        await h.Follower.PollOnceAsync();

        Assert.Equal(new[] { "MoleculeGroup" }, h.Asked);
        var (locator, item) = Assert.Single(h.Applied);
        Assert.Equal("Molecule:/sp|P1|A/PEPC[+57.021464]K", locator);
        Assert.Equal("feature:" + locator, item);
    }

    [Fact]
    public async Task NoPane_OrAPaneWithNothingToFollow_AsksSkylineNothing()
    {
        var h = new Harness();
        await h.Follower.PollOnceAsync();               // no pane
        h.Follower.Follow(() => null);                  // a pane with nothing plotted
        await h.Follower.PollOnceAsync();

        Assert.Empty(h.Asked);
        Assert.Empty(h.Applied);
    }

    [Fact]
    public async Task NothingSelectedInSkyline_AppliesNothing()
    {
        var h = new Harness { Selection = _ => null };
        h.Follower.Follow(() => h.Poll());

        await h.Follower.PollOnceAsync();

        Assert.Single(h.Asked);
        Assert.Empty(h.Applied);
    }

    /// <summary>A busy or closed Skyline is retried every tick, never counted toward giving up.</summary>
    [Fact]
    public async Task AFailedRpc_IsRetried_AndNeverStopsFollowing()
    {
        var h = new Harness { Selection = _ => throw new TimeoutException("Skyline is busy") };
        h.Follower.Follow(() => h.Poll());

        for (var i = 0; i < SkylineSelectionFollower.MaxFailures + 2; i++)
            await h.Follower.PollOnceAsync();

        Assert.False(h.Follower.Stopped);
        Assert.Equal(SkylineSelectionFollower.MaxFailures + 2, h.Asked.Count);
        Assert.Empty(h.Logged);
    }

    /// <summary>
    /// Anything else that throws is counted; after MaxFailures in a row following stops and says so,
    /// rather than failing on every tick forever.
    /// </summary>
    [Fact]
    public async Task RepeatedFailures_StopFollowing_AndSaySo()
    {
        var h = new Harness { Resolve = _ => throw new InvalidOperationException("index broke") };
        h.Follower.Follow(() => h.Poll());

        for (var i = 0; i < SkylineSelectionFollower.MaxFailures; i++)
        {
            Assert.False(h.Follower.Stopped);
            await h.Follower.PollOnceAsync();
        }

        Assert.True(h.Follower.Stopped);
        Assert.Equal(SkylineSelectionFollower.MaxFailures, h.Traced.Count);
        Assert.Contains(h.Logged, m => m.StartsWith("Stopped following Skyline's selection", StringComparison.Ordinal));

        // Stopped means stopped: nothing more is asked until the pane is shown again.
        var asked = h.Asked.Count;
        await h.Follower.PollOnceAsync();
        Assert.Equal(asked, h.Asked.Count);

        // Showing a pane again is the retry.
        h.Resolve = l => l;
        h.Follower.Follow(() => h.Poll());
        Assert.False(h.Follower.Stopped);
        await h.Follower.PollOnceAsync();
        Assert.Single(h.Applied);
    }

    [Fact]
    public async Task ASuccess_ResetsTheFailureCount()
    {
        var h = new Harness();
        var fail = true;
        h.Resolve = l => fail ? throw new InvalidOperationException("x") : l;
        h.Follower.Follow(() => h.Poll());

        for (var round = 0; round < 3; round++)
        {
            fail = true;
            for (var i = 0; i < SkylineSelectionFollower.MaxFailures - 1; i++)
                await h.Follower.PollOnceAsync();
            fail = false;
            await h.Follower.PollOnceAsync();
        }

        Assert.False(h.Follower.Stopped);
        Assert.Equal(3, h.Applied.Count);
    }

    /// <summary>An answer that arrives after another pane took over belongs to neither.</summary>
    [Fact]
    public async Task AnAnswerForAPaneNoLongerShown_IsDropped()
    {
        var h = new Harness();
        var release = new TaskCompletionSource();
        h.Selection = _ =>
        {
            release.Task.Wait();
            return "MoleculeGroup:/sp|P1|A";
        };
        h.Follower.Follow(() => h.Poll());

        var inFlight = h.Follower.PollOnceAsync();
        h.Follower.Follow(() => null); // the user switched panes
        release.SetResult();
        await inFlight;

        Assert.Empty(h.Applied);
    }

    /// <summary>A slow Skyline makes polling quieter; it never piles up requests.</summary>
    [Fact]
    public async Task APollWhileOneIsInFlight_DoesNothing()
    {
        var h = new Harness();
        var release = new TaskCompletionSource();
        h.Selection = _ =>
        {
            release.Task.Wait();
            return "MoleculeGroup:/sp|P1|A";
        };
        h.Follower.Follow(() => h.Poll());

        var first = h.Follower.PollOnceAsync();
        await h.Follower.PollOnceAsync(); // returns at once
        release.SetResult();
        await first;

        Assert.Single(h.Asked);
        Assert.Single(h.Applied);
    }
}
