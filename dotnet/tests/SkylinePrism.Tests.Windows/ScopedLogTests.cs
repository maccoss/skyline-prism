using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Inputs are exported concurrently, and the deepest log lines come from Skyline's own console
/// ("Opening file...", "2%") which say nothing about which document produced them - two documents at
/// once emit a stream of identical-looking pairs. <see cref="PrismInput.Scoped"/> tags every line for an
/// input with its batch label, applied once at the boundary so it covers all three input kinds, the
/// runner output, and the Skyline-selection messages alike.
/// </summary>
public class ScopedLogTests
{
    [Fact]
    public void EveryMessageIsTaggedWithTheLabel()
    {
        var lines = new List<string>();
        var log = PrismInput.Scoped(lines.Add, "PlateA");

        log("Opening file...");
        log("2%");

        Assert.Equal(new[] { "[PlateA] Opening file...", "[PlateA] 2%" }, lines);
    }

    [Fact]
    public void IdenticalLinesFromTwoDocumentsBecomeDistinguishable()
    {
        // The exact symptom: Skyline emits the same generic text from both processes.
        var lines = new List<string>();
        var a = PrismInput.Scoped(lines.Add, "Plt1");
        var b = PrismInput.Scoped(lines.Add, "Plt2");

        a("Opening file...");
        b("Opening file...");

        Assert.Equal(2, lines.Distinct().Count());
        Assert.Contains("[Plt1] Opening file...", lines);
        Assert.Contains("[Plt2] Opening file...", lines);
    }

    [Fact]
    public void IndentationFromSkylineOutputIsPreserved()
    {
        // Runner output is indented to set it apart from PRISM's own messages; the tag goes in front of
        // the indent so the structure still reads.
        var lines = new List<string>();
        PrismInput.Scoped(lines.Add, "PlateA")("    Success! Imported Reports");

        Assert.Equal("[PlateA]     Success! Imported Reports", lines[0]);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void BlankLinesArePassedThroughUntagged(string blank)
    {
        // Blank lines separate sections; tagging them would turn every gap into visual noise.
        var lines = new List<string>();
        PrismInput.Scoped(lines.Add, "PlateA")(blank);

        Assert.Equal(blank, lines[0]);
    }

    [Fact]
    public void ScopingIsThreadSafeForConcurrentExports()
    {
        // Two scoped loggers writing at once must not interleave within a line.
        var sink = new System.Collections.Concurrent.ConcurrentBag<string>();
        void Sink(string m) => sink.Add(m);

        System.Threading.Tasks.Parallel.Invoke(
            () => { for (var i = 0; i < 200; i++) PrismInput.Scoped(Sink, "Plt1")($"{i}%"); },
            () => { for (var i = 0; i < 200; i++) PrismInput.Scoped(Sink, "Plt2")($"{i}%"); });

        Assert.Equal(400, sink.Count);
        Assert.Equal(200, sink.Count(l => l.StartsWith("[Plt1] ", StringComparison.Ordinal)));
        Assert.Equal(200, sink.Count(l => l.StartsWith("[Plt2] ", StringComparison.Ordinal)));
    }

    [Fact]
    public void APreExportedReportInputIsAlsoTagged()
    {
        // All three input kinds go through the same boundary, so none is left unattributed.
        var dir = Path.Combine(Path.GetTempPath(), "prism_scoped_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var report = Path.Combine(dir, "Plate_07.parquet");
        File.WriteAllText(report, "x");

        var lines = new List<string>();
        PrismInput.FromReportFile(report)
            .Prepare(dir, null, null, null, lines.Add, CancellationToken.None);

        Assert.NotEmpty(lines);
        Assert.All(lines, l => Assert.StartsWith("[Plate_07] ", l));
    }
}
