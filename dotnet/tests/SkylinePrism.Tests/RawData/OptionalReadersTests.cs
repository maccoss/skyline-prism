using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.RawData;
using Xunit;

namespace SkylinePrism.Tests.RawData;

/// <summary>
/// Loading the optional instrument-file reader.
/// </summary>
/// <remarks>
/// <para>These assert what is true in BOTH builds, because both ship: the reader assembly exists
/// only when a pwiz-sharp checkout was present at compile time, so this same code takes the "found
/// it" path on a developer machine and the "not in this build" path in CI. A test that assumed
/// either would pass in one place and fail in the other.</para>
///
/// <para>What is invariant is the contract: attempt once, never throw, and say which way it went.
/// That last part matters more than it looks - a build whose reader failed to load behaves exactly
/// like a build that never had one, and the only difference a user can see is the line this writes.</para>
/// </remarks>
[Collection("IonAccountingRun")]
public class OptionalReadersTests : IDisposable
{
    public OptionalReadersTests()
    {
        OptionalReaders.Reset();
        Ms2SignalReaders.Clear();
    }

    public void Dispose()
    {
        OptionalReaders.Reset();
        Ms2SignalReaders.Clear();
    }

    /// <summary>
    /// Registration is attempted ONCE per process however often it is called - both entry points
    /// call it, and the WPF tool calls it at startup while a command may call it again.
    /// </summary>
    [Fact]
    public void RegistrationIsAttemptedOnlyOnce()
    {
        var log = new List<string>();

        OptionalReaders.Register(log.Add);
        var afterFirst = log.Count;
        Assert.True(afterFirst > 0, "the first attempt should say which way it went");

        OptionalReaders.Register(log.Add);
        OptionalReaders.Register(log.Add);

        Assert.Equal(afterFirst, log.Count);
    }

    /// <summary>
    /// It says which way it went, and the two outcomes are the only two - a reader was registered,
    /// or the build has none. Anything else means the load failed silently.
    /// </summary>
    [Fact]
    public void ItReportsWhetherAReaderWasFound()
    {
        var log = new List<string>();

        OptionalReaders.Register(log.Add);

        var line = Assert.Single(log);
        var registered = Ms2SignalReaders.All.Count > 0;
        if (registered)
        {
            Assert.Contains("registered", line, StringComparison.OrdinalIgnoreCase);
            // And the line names what was registered, so a mixed cohort is visible.
            Assert.All(Ms2SignalReaders.All, r => Assert.Contains(r.Describe(), line, StringComparison.Ordinal));
        }
        else
        {
            Assert.Contains("not in this build", line, StringComparison.OrdinalIgnoreCase);
        }
    }

    /// <summary>
    /// A reader that cannot load is a missing denominator, not a reason to fail startup - so this is
    /// called from the WPF tool's OnStartup and must not throw for any reason, including no log.
    /// </summary>
    [Fact]
    public void RegisteringWithoutALogDoesNotThrow()
    {
        OptionalReaders.Register();
        OptionalReaders.Register(null);
    }

    /// <summary>
    /// Reset is what lets a test observe the first attempt. Without it the second test in a run
    /// would see an already-attempted registration and log nothing, which would look like a pass.
    /// </summary>
    [Fact]
    public void ResetAllowsTheAttemptToBeObservedAgain()
    {
        var first = new List<string>();
        OptionalReaders.Register(first.Add);
        Assert.Single(first);

        var suppressed = new List<string>();
        OptionalReaders.Register(suppressed.Add);
        Assert.Empty(suppressed);

        OptionalReaders.Reset();
        var second = new List<string>();
        OptionalReaders.Register(second.Add);
        Assert.Single(second);
        Assert.Equal(first[0], second[0]);
    }

    /// <summary>
    /// Whatever this build has, the accounting-capable readers are a subset of the registered ones -
    /// so a build with a reader that cannot do ion accounting reports no accounting rather than
    /// throwing when one is asked for.
    /// </summary>
    [Fact]
    public void AccountingReadersAreASubsetOfRegisteredReaders()
    {
        OptionalReaders.Register();

        Assert.True(IonAccountingReaders.All.Count <= Ms2SignalReaders.All.Count);
        Assert.Equal(IonAccountingReaders.Available, IonAccountingReaders.All.Count > 0);
        Assert.All(IonAccountingReaders.All, r => Assert.Contains(r, Ms2SignalReaders.All));
    }
}
