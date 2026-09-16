using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Visualization;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// The ion-accounting sections of the QC report.
/// </summary>
public static partial class QcReport
{
    /// <summary>
    /// Ions acquired and assigned, per replicate and across the gradient - or nothing at all when
    /// this directory has not been measured.
    /// </summary>
    /// <remarks>
    /// <para><b>A pure file read.</b> Everything comes from <c>ion_accounting.parquet</c> and
    /// <c>ion_cycles.parquet</c>, so <c>prism qc -d</c> renders these on a directory whose
    /// <c>merged_data/</c> and instrument files are long gone, and omits the section rather than
    /// recomputing on one that was never measured. Measuring is <c>prism ion-accounting</c>, which
    /// reads every instrument file in the cohort.</para>
    ///
    /// <para><b>The captions carry the settings.</b> The cache is keyed on both extraction
    /// tolerances, the isolation scheme and the instrument files, and a reader of the HTML has no
    /// other way to know which produced these numbers - so each caption names them rather than
    /// leaving the figures to be read as settings-independent.</para>
    /// </remarks>
    /// <param name="cohortReplicates">
    /// How many replicates the ANALYSIS has - the denominator for "how much of it has been measured".
    /// Taken from the run's own sample list rather than from the cache, because the cache is not one
    /// while a measurement is running: progress is saved after every replicate and the rows for
    /// replicates with no file of their own are added only once the scan loop ends, so a cache read
    /// mid-run holds exactly the replicates measured so far and could only ever say "N of N".
    /// </param>
    private static List<PlotSection> RenderIonAccountingSection(
        string outputDir, int cohortReplicates, bool savePlots, string plotsDir, Action<string>? log)
    {
        var sections = new List<PlotSection>();

        var result = IonAccountingStore.Read(outputDir);
        if (result is null || result.Rows.Count == 0)
            return sections;

        var usable = result.Usable;
        if (usable.Count == 0)
        {
            log?.Invoke(
                "  Ion accounting: the cache holds no replicate with usable numbers, so the report "
                + "omits the section.");
            return sections;
        }

        log?.Invoke(
            $"  Ion accounting: plotting {usable.Count:N0} measured replicate(s) from "
            + $"{IonAccountingStore.FileName}.");

        // How many of the cohort these numbers cover. A partly measured cache is ordinary - a --max
        // spot check, a measurement still running, or instrument files that could not be found - and
        // the plots cannot say it themselves once they are drawn over the replicates that HAVE
        // numbers. Against the analysis's replicate count, not the cache's: see cohortReplicates.
        var coverage = usable.Count >= cohortReplicates
            ? ""
            : $"{usable.Count:N0} of {cohortReplicates:N0} replicates measured so far. ";
        var settings = coverage
            + $"Product {result.ProductTolerance}, precursor "
            + $"{result.PrecursorTolerance}; isolation scheme {result.IsolationScheme}; "
            + $"{result.AssignedPeptides:N0} peptides claimed signal.";

        // A cached result carries no run log, and a units error leaves the FRACTION correct - so the
        // plots look entirely normal and only the absolute totals are wrong. Said in the caption
        // because that is the only place a reader of the HTML would ever find it.
        var offScale = usable.Count(r => r.IonScaleImplausible);
        if (offScale > 0)
        {
            var worst = usable.First(r => r.IonScaleImplausible);
            settings = $"WARNING: {offScale:N0} replicate(s) report an impossible number of "
                + $"ions per scan ({worst.MeanMs1IonsPerScan:E2} at MS1, "
                + $"{worst.MeanMs2IonsPerScan:E2} at MS2, against an AGC target of perhaps 1e6). "
                + "The totals are in the wrong unit; the fractions are unaffected.\n" + settings;
        }

        // ---- Per replicate, one panel per MS level. Never one axis for both: measured over a whole
        // run MS2 acquires about three times the ions of MS1 while the assigned fraction goes the
        // other way, so a shared axis would flatten one of them.
        var bars = new List<PlotImage>();
        foreach (var level in new[] { PlotRenderer.IonLevel.Ms2, PlotRenderer.IonLevel.Ms1 })
        {
            var name = level.ToString().ToUpperInvariant();
            // Only what is NOT on the image. The title carries the medians and the legend names every
            // series, so what is left is the spread, the cases where a figure is withheld as
            // impossible, and the settings - each a fact with no other home. Describing the bars in
            // prose as well made this the one section of the report with a paragraph under every
            // figure.
            var caption = string.Join("\n", new[]
            {
                "- " + FractionCaption(usable, level),
                ExplainedCaption(usable, level),
                // Once per section, under the first panel - it is the same settings for both MS
                // levels, and printing them twice is half of what made this a wall of text.
                bars.Count == 0 ? settings : "",
            }.Where(line => line.Length > 2));

            Render(
                bars, caption, AccountingTitle(name, usable),
                $"ion_accounting_{name.ToLowerInvariant()}.png", savePlots, plotsDir,
                // The replicates that HAVE numbers. Drawn over every row, a cache measured for one
                // replicate of thirty-nine was one bar and thirty-eight empty slots, which reads as a
                // cohort that acquired nothing rather than as a measurement still to finish.
                () => PlotRenderer.IonAccountingPng(
                    result with { Rows = usable }, level, AccountingTitle(name, usable)));
        }
        sections.Add(new PlotSection(
            usable.Any(r => r.HasExplained)
                ? "Ions Acquired, Quantified and Explained"
                : "Ions Acquired and Assigned to a Peptide",
            bars));

        AddIonProfileSection(sections, outputDir, result, settings, savePlots, plotsDir, log);
        return sections;
    }

    /// <summary>
    /// The assigned fraction across the gradient, for the best, median and worst replicate rather
    /// than all of them.
    /// </summary>
    /// <remarks>
    /// <para>Three panels because 192 is a section nobody scrolls, and because the question these
    /// answer - is the analysis explaining a steady fraction of what was acquired, or losing it
    /// somewhere in particular - is answered by the extremes and the middle.</para>
    /// <para>The FRACTION rather than the absolute traces: both absolute curves rise and fall with
    /// the elution envelope, so a stretch the analysis cannot explain is invisible in them. The
    /// absolute pair is still drawn for the median replicate, because the fraction alone does not
    /// say whether a low stretch carried much signal at all.</para>
    /// </remarks>
    /// <summary>
    /// The gradient bin, in minutes. 0.01 is 0.6 s - shorter than one acquisition cycle on the
    /// instruments this was built for - so it bins essentially nothing and the trace is drawn at
    /// the rate the run was acquired at. A minute-wide bin averaged sixty cycles together, which is
    /// the one thing these panels exist to show. Matches the pane's own default.
    /// </summary>
    private const double GradientBinMinutes = 0.01;

    private static void AddIonProfileSection(
        List<PlotSection> sections, string outputDir, IonAccountingResult result, string settings,
        bool savePlots, string plotsDir, Action<string>? log)
    {
        // ONE quantity for the whole section. It used to draw three fraction panels and then an
        // absolute one, so "fraction of acquired MS2 signal" and "MS2 ions" sat under the same
        // heading meaning different things - two quantities and two units, read as a series.
        var signal = result.Usable.Any(r => r.HasSignal);
        var representatives = result.Representatives(signal);
        if (representatives.Count == 0)
            return;

        var withCycles = IonAccountingStore.SamplesWithCycles(outputDir);
        if (withCycles.Count == 0)
        {
            log?.Invoke(
                $"  Ion accounting: no {IonAccountingStore.CyclesFile}, so the across-the-gradient "
                + "panels are omitted. The per-replicate bars above are unaffected.");
            return;
        }

        var quantity = signal ? PlotRenderer.IonQuantity.Signal : PlotRenderer.IonQuantity.Ions;
        var noun = signal ? "signal" : "ions";
        var labels = Labels(representatives.Count);
        var images = new List<PlotImage>();

        for (var i = 0; i < representatives.Count; i++)
        {
            var row = representatives[i];
            if (!withCycles.Contains(row.Sample, StringComparer.Ordinal))
                continue;

            var cycles = IonAccountingStore.ReadCycles(outputDir, row.Sample);
            if (cycles.Count == 0)
                continue;

            // The panel is titled with the replicate and its legend names both traces and the
            // whole-run figure, so the caption carries only what is not already on the image: the
            // settings, once per section, and the one case where a fraction is withheld.
            var lines = new List<string>();
            if (row.ExceededIn(signal))
            {
                lines.Add("- This replicate assigned more than it acquired, which is impossible, so "
                    + "no whole-run fraction is stated.");
            }
            if (images.Count == 0)
                lines.Add(settings);

            Render(
                images, string.Join("\n", lines),
                $"{labels[i]} replicate by assigned fraction: {row.Sample}",
                $"ion_fraction_{labels[i].ToLowerInvariant()}.png", savePlots, plotsDir,
                () => PlotRenderer.IonFractionProfilePng(
                    cycles, PlotRenderer.IonLevel.Ms2, GradientBinMinutes,
                    title: $"{labels[i]}: {row.Sample}", quantity: quantity));
        }

        if (images.Count > 0)
        {
            sections.Add(new PlotSection(
                $"Fraction of Acquired MS2 {char.ToUpperInvariant(noun[0])}{noun[1..]} Across the "
                + "Gradient",
                images));
        }
    }

    /// <summary>
    /// What to say about the cohort's fraction, and what to refuse to say. A fraction over 1 is
    /// impossible, so where one occurs the figure is withheld and named as a defect rather than
    /// quietly excluded from a median.
    /// </summary>
    /// <summary>
    /// The second number, when there is one: the fraction the peptides can ACCOUNT FOR against the
    /// fraction they are QUANTIFIED on.
    ///
    /// <para>Returns empty at MS1 and on a cache that measured no explained total, so a report over
    /// an older directory reads exactly as it did before - silence rather than a zero, which would
    /// say the peptides explain nothing.</para>
    /// </summary>
    internal static string ExplainedCaption(
        IReadOnlyList<IonAccountingRow> usable, PlotRenderer.IonLevel level)
    {
        if (level != PlotRenderer.IonLevel.Ms2)
            return "";

        var impossible = usable.Count(r => r.ExplainedImpossible);
        if (impossible > 0)
        {
            return $"- NOTE: {impossible:N0} replicate(s) explained more than was acquired, or "
                + "less than they quantified - both impossible, so no explained figure is reported. "
                + "The quantified one is unaffected.";
        }

        var fractions = usable
            .Where(r => r.HasExplained && !r.Exceeded && !r.ExplainedImpossible)
            .Select(r => r.Ms2ExplainedFraction)
            .Where(double.IsFinite)
            .OrderBy(f => f)
            .ToArray();
        if (fractions.Length == 0)
            return "";

        var median = fractions.Length % 2 == 1
            ? fractions[fractions.Length / 2]
            : (fractions[fractions.Length / 2 - 1] + fractions[fractions.Length / 2]) / 2;
        var span = fractions.Length > 1
            ? $", ranging {IonAccountingStore.Percent(fractions[0])} to "
              + $"{IonAccountingStore.Percent(fractions[^1])}"
            : "";

        // The number and which bar it is, and no more. What "explained" counts - every theoretical b
        // and y ion at 1+ and 2+ plus the surviving precursor and its first two isotopes - is in the
        // plot's own title in short form and in docs/skyline-tool.md in full; spelled out here it was
        // most of what made this the one section of the report with a paragraph under every figure.
        return $"- Median {IonAccountingStore.Percent(median)} explained{span} (the lighter bar).";
    }

    internal static string FractionCaption(
        IReadOnlyList<IonAccountingRow> usable, PlotRenderer.IonLevel level)
    {
        var exceeded = usable.Count(r => r.Exceeded);
        if (exceeded > 0)
        {
            return $"NOTE: {exceeded:N0} of {usable.Count:N0} replicates assigned more signal than "
                + "was acquired, which is impossible - so no fraction is reported here. Check that "
                + "the isolation scheme matches the acquisition and that the extraction tolerances "
                + "are the document's own.";
        }

        var fractions = usable
            .Select(r => level == PlotRenderer.IonLevel.Ms1 ? r.Ms1Fraction : r.Ms2Fraction)
            .Where(double.IsFinite)
            .OrderBy(f => f)
            .ToArray();
        if (fractions.Length == 0)
            return "No replicate carried a usable acquired total, so no fraction is reported.";

        var median = fractions.Length % 2 == 1
            ? fractions[fractions.Length / 2]
            : (fractions[fractions.Length / 2 - 1] + fractions[fractions.Length / 2]) / 2;

        var span = fractions.Length > 1
            ? $", ranging {IonAccountingStore.Percent(fractions[0])} to "
              + $"{IonAccountingStore.Percent(fractions[^1])}"
            : "";
        var coverage = usable.Count(r => !string.IsNullOrEmpty(r.DataFile));
        var over = coverage == usable.Count
            ? ""
            : $" (over the {coverage:N0} replicates whose data file could be read)";

        return $"Median {IonAccountingStore.Percent(median)} assigned{span}{over}.";
    }

    private static string[] Labels(int count) => count switch
    {
        1 => new[] { "Representative" },
        2 => new[] { "Best", "Worst" },
        _ => new[] { "Best", "Median", "Worst" },
    };

    /// <summary>
    /// Render one panel, saving it beside the report when asked, and turning a render failure into a
    /// caption rather than into a missing report. Every other failure in this file logs and carries
    /// on; a plot is not worth losing a run's whole QC report over.
    /// </summary>
    /// <param name="alt">
    /// What the panel shows, for a reader who cannot see it. Its own sentence because the caption is
    /// now the settings and the exceptions only - most panels here carry none at all, and an image
    /// with no alt text describes nothing.
    /// </param>
    private static void Render(
        List<PlotImage> images, string caption, string alt, string fileName, bool savePlots,
        string plotsDir, Func<byte[]> render)
    {
        try
        {
            var png = render();
            if (savePlots && png.Length > 0)
            {
                Directory.CreateDirectory(plotsDir);
                File.WriteAllBytes(Path.Combine(plotsDir, fileName), png);
            }
            images.Add(new PlotImage(caption, png, alt));
        }
        catch (Exception ex)
        {
            // The caption is kept, not replaced: this is the panel that carries the settings and any
            // impossible-measurement warning for its whole section, and losing a render is no reason
            // to lose them as well.
            var failure = (alt.Length > 0 ? alt + " " : "")
                + "(render failed: " + ex.GetType().Name + ")";
            images.Add(new PlotImage(
                caption.Length > 0 ? failure + "\n" + caption : failure, Array.Empty<byte>(), alt));
        }
    }

    /// <summary>
    /// The plot heading. "Assigned to a Peptide" describes one numerator, which is the whole answer
    /// only until there are two.
    /// </summary>
    private static string AccountingTitle(string level, IReadOnlyList<IonAccountingRow> usable) =>
        level == "MS2" && usable.Any(r => r.HasExplained)
            ? "MS2 Ions Acquired, Quantified and Explained"
            : $"{level} Ions Acquired and Assigned to a Peptide";

}
