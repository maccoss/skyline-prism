using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Qc;
using SkylinePrism.Core.RawData;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The across-the-gradient section of the QC report: one quantity, three panels, nothing else.
/// </summary>
/// <remarks>
/// <para>Left untested when it was written, and said so in the PR: it was verified once by hand, by
/// rendering a report over a real 48-replicate cache and reading it. That is not a test, and the
/// section had already been wrong once - it drew three FRACTION panels and then a fourth of absolute
/// MS2 IONS, so two quantities in two units sat under one heading and read as a series.</para>
///
/// <para>Driven through <see cref="QcReport.Generate"/> rather than the private section builder,
/// because the heading and the panel count are what a reader sees.</para>
/// </remarks>
public class IonGradientSectionTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "prism-gradient-" + Guid.NewGuid().ToString("N"));

    public IonGradientSectionTests()
    {
        Directory.CreateDirectory(_dir);
        foreach (var file in Directory.GetFiles(
                     Path.Combine(AppContext.BaseDirectory, "fixtures", "mini", "e2e-sum", "output")))
        {
            File.Copy(file, Path.Combine(_dir, Path.GetFileName(file)));
        }
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    [Fact]
    public void TheGradientSectionIsThreePanelsOfOneQuantity()
    {
        IonAccountingStore.Write(_dir, Measured(withSignal: true));

        var html = Render();
        var headings = Headings(html);

        var gradient = headings.Where(h => h.Contains("Across the Gradient", StringComparison.Ordinal))
            .ToArray();
        Assert.Single(gradient);

        // Signal, named in the heading - not "Assigned Share", and not ions.
        Assert.Contains("Fraction of Acquired MS2 Signal", gradient[0], StringComparison.Ordinal);

        // Best, median and worst. A fourth panel is how the two-quantity bug looked.
        var captions = Captions(html)
            .Where(c => c.Contains("replicate by assigned fraction", StringComparison.Ordinal))
            .ToArray();
        Assert.Equal(3, captions.Length);
        Assert.Contains(captions, c => c.Contains("Best replicate", StringComparison.Ordinal));
        Assert.Contains(captions, c => c.Contains("Median replicate", StringComparison.Ordinal));
        Assert.Contains(captions, c => c.Contains("Worst replicate", StringComparison.Ordinal));

        // Nothing absolute under that heading: "ions per cycle" was the fourth panel's caption.
        Assert.DoesNotContain("absolute ions per cycle", html, StringComparison.Ordinal);
    }

    /// <summary>
    /// A cache measured before the summed TIC existed still gets the section, in ions - and the
    /// heading says ions, so the panels and their caption never disagree.
    /// </summary>
    [Fact]
    public void ACacheWithNoSignalGetsTheSectionInIons()
    {
        IonAccountingStore.Write(_dir, Measured(withSignal: false));

        var gradient = Headings(Render())
            .Single(h => h.Contains("Across the Gradient", StringComparison.Ordinal));

        Assert.Contains("Fraction of Acquired MS2 Ions", gradient, StringComparison.Ordinal);
    }

    private string Render()
    {
        QcReport.Generate(_dir, new PrismConfig(), savePlots: false);
        return File.ReadAllText(Path.Combine(_dir, "qc_report.html"));
    }

    private static string[] Headings(string html) =>
        Regex.Matches(html, @"<h[23][^>]*>(.*?)</h[23]>", RegexOptions.Singleline)
            .Select(m => m.Groups[1].Value).ToArray();

    private static string[] Captions(string html) =>
        Regex.Matches(html, @"<div class=""cap"">(.*?)</div>", RegexOptions.Singleline)
            .Select(m => m.Groups[1].Value).ToArray();

    /// <summary>Five replicates with distinguishable fractions, so best/median/worst are unambiguous.</summary>
    private static IonAccountingResult Measured(bool withSignal)
    {
        var rows = new IonAccountingRow[5];
        var cycles = new System.Collections.Generic.List<IonCycleRow>();
        for (var i = 0; i < rows.Length; i++)
        {
            var sample = $"r{i + 1}";
            // Ion and signal fractions deliberately rank the cohort the SAME way here; the ranking
            // quantity has its own test. This one is about the section's shape.
            double assigned = (i + 1) * 40;
            rows[i] = new IonAccountingRow(
                sample, "experimental", sample + ".raw", Ms2ReadStatus.Ok, "test", 10, 100,
                Ms1Acquired: 1000, Ms2Acquired: 1000,
                Ms1Assigned: assigned, Ms2Assigned: assigned,
                Ms2Explained: assigned * 2, HasExplained: true,
                0, 30, 0, 0, 0, 20,
                Array.Empty<double>(), Array.Empty<double>(),
                AcquiredUtc: null,
                Ms1Signal: withSignal ? 1000 : 0, Ms2Signal: withSignal ? 1000 : 0,
                Ms1SignalAssigned: withSignal ? assigned : 0,
                Ms2SignalAssigned: withSignal ? assigned : 0,
                Ms2SignalExplained: withSignal ? assigned * 2 : 0,
                HasSignal: withSignal);

            for (var c = 0; c < 20; c++)
            {
                cycles.Add(new IonCycleRow(
                    sample, c, c * 0.5, (c + 1) * 0.5, 1, 10,
                    50, 100, assigned / 20, assigned / 10, assigned / 5,
                    withSignal ? 50 : 0, withSignal ? 100 : 0,
                    withSignal ? assigned / 20 : 0, withSignal ? assigned / 10 : 0,
                    withSignal ? assigned / 5 : 0));
            }
        }

        return new IonAccountingResult(
            "key", "+/-10 ppm (centroided)", "+/-10 ppm (centroided)", "167 windows",
            Array.Empty<string>(), 4321, true, rows, cycles);
    }
}
