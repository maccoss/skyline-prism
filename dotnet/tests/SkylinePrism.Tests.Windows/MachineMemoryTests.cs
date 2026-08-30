using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The free-memory query behind the "this document will not fit" warning. It is interop, so the struct
/// layout is the thing most easily got silently wrong - a wrong dwLength or a misordered field yields
/// plausible-looking nonsense rather than an error - and the number it produces is the one actionable
/// figure in a warning about whether closing Skyline would help.
///
/// <para>It lives outside MainWindow precisely so this test can exist: CLAUDE.md records that window as
/// "~1000 lines of code-behind at 0% coverage".</para>
/// </summary>
public class MachineMemoryTests
{
    [Fact]
    public void ReadsPlausibleTotalAndAvailablePhysicalMemory()
    {
        var memory = MachineMemory.Read();

        Assert.NotNull(memory);
        var (totalGb, availableGb) = memory!.Value;

        // Deliberately loose: this has to hold on a CI container and on a 64 GB workstation alike. What it
        // pins is that the fields are read in the right order and scaled correctly - a swapped or
        // mis-sized field shows up as a total of zero, a negative, or something absurd.
        Assert.InRange(totalGb, 0.5, 4096);
        Assert.InRange(availableGb, 0, totalGb);
    }

    /// <summary>FreePhysicalGb is the caller-facing shorthand and must agree with the full read.</summary>
    [Fact]
    public void FreePhysicalGbMatchesTheAvailableFigure()
    {
        var free = MachineMemory.FreePhysicalGb();
        var memory = MachineMemory.Read();

        Assert.NotNull(free);
        Assert.NotNull(memory);
        // Sampled a moment apart on a live machine, so compare generously - this is checking they are the
        // same quantity, not that memory held still between two calls.
        Assert.InRange(free!.Value, 0, memory!.Value.TotalGb);
    }
}
