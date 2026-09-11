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
    private static List<PlotSection> RenderIonAccountingSection(
        string outputDir, bool savePlots, string plotsDir, Action<string>? log)
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

        var settings = $"Extraction windows: product {result.ProductTolerance}, precursor "
            + $"{result.PrecursorTolerance}. Isolation scheme: {result.IsolationScheme}. "
            + $"{result.AssignedPeptides:N0} peptides claimed signal.";

        // A cached result carries no run log, and a units error leaves the FRACTION correct - so the
        // plots look entirely normal and only the absolute totals are wrong. Said in the caption
        // because that is the only place a reader of the HTML would ever find it.
        var offScale = usable.Count(r => r.IonScaleImplausible);
        if (offScale > 0)
        {
            var worst = usable.First(r => r.IonScaleImplausible);
            settings = $"WARNING: {offScale:N0} replicate(s) report an impossible number of ions per "
                + $"scan (e.g. {worst.MeanMs1IonsPerScan:E2} at MS1, {worst.MeanMs2IonsPerScan:E2} at "
                + "MS2, against an AGC target of perhaps 1e6). The TOTALS are in the wrong unit; the "
                + "fractions are unaffected, which is why the plots still look right. " + settings;
        }

        // ---- Per replicate, one panel per MS level. Never one axis for both: measured over a whole
        // run MS2 acquires about three times the ions of MS1 while the assigned share goes the other
        // way, so a shared axis would flatten one of them.
        var bars = new List<PlotImage>();
        foreach (var level in new[] { PlotRenderer.IonLevel.Ms2, PlotRenderer.IonLevel.Ms1 })
        {
            var name = level.ToString().ToUpperInvariant();
            var caption = $"{name}: ions the instrument acquired, and the part of them falling "
                + "inside a region some peptide of this analysis claims. Shared signal is counted "
                + "once - two peptides whose fragments fall within the extraction tolerance of each "
                + "other read the same detector counts. " + FractionCaption(usable, level)
                + " " + settings;

            Render(
                bars, caption, $"ion_accounting_{name.ToLowerInvariant()}.png", savePlots, plotsDir,
                () => PlotRenderer.IonAccountingPng(
                    result, level, $"{name} Ions Acquired and Assigned to a Peptide"));
        }
        sections.Add(new PlotSection("Ions Acquired and Assigned to a Peptide", bars));

        AddIonProfileSection(sections, outputDir, result, settings, savePlots, plotsDir, log);
        return sections;
    }

    /// <summary>
    /// The assigned share across the gradient, for the best, median and worst replicate rather than
    /// all of them.
    /// </summary>
    /// <remarks>
    /// <para>Three panels because 192 is a section nobody scrolls, and because the question these
    /// answer - is the analysis explaining a steady share of what was acquired, or losing it
    /// somewhere in particular - is answered by the extremes and the middle.</para>
    /// <para>The SHARE rather than the absolute traces: both absolute curves rise and fall with the
    /// elution envelope, so a stretch the analysis cannot explain is invisible in them. The absolute
    /// pair is still drawn for the median replicate, because the share alone does not say whether a
    /// low stretch carried much signal at all.</para>
    /// </remarks>
    private static void AddIonProfileSection(
        List<PlotSection> sections, string outputDir, IonAccountingResult result, string settings,
        bool savePlots, string plotsDir, Action<string>? log)
    {
        var representatives = result.Representatives();
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

        var labels = Labels(representatives.Count);
        var images = new List<PlotImage>();
        var median = representatives.Count >= 3 ? representatives[1] : representatives[0];

        for (var i = 0; i < representatives.Count; i++)
        {
            var row = representatives[i];
            if (!withCycles.Contains(row.Sample, StringComparer.Ordinal))
                continue;

            var cycles = IonAccountingStore.ReadCycles(outputDir, row.Sample);
            if (cycles.Count == 0)
                continue;

            var share = row.Exceeded
                ? "This replicate assigned more than it acquired, which is impossible, so no share "
                  + "is stated."
                : $"Whole run: {IonAccountingStore.Percent(row.Ms2Fraction)} of acquired MS2 ions "
                  + $"and {IonAccountingStore.Percent(row.Ms1Fraction)} of acquired MS1 ions.";

            Render(
                images,
                $"{labels[i]} by assigned share: {row.Sample}. The line is the share of each "
                + $"cycle's acquired MS2 ions that a peptide explains. The axis starts at zero "
                + $"and fits the data above it. "
                + share + " " + settings,
                $"ion_share_{labels[i].ToLowerInvariant()}.png", savePlots, plotsDir,
                () => PlotRenderer.IonFractionProfilePng(
                    cycles, PlotRenderer.IonLevel.Ms2, binMinutes: 1.0,
                    title: $"{labels[i]}: {row.Sample}"));

            // The absolute pair, for the median only. A low share where almost nothing was acquired
            // is a different finding from a low share at the peak of the elution.
            if (ReferenceEquals(row, median))
            {
                Render(
                    images,
                    $"Median replicate {row.Sample}, absolute ions per cycle: acquired MS2 filled, "
                    + "with the part assigned to a peptide drawn over it. Read together with the "
                    + "share above - a dip in the share matters more where the acquired trace is "
                    + "high. " + settings,
                    "ion_profile_median.png", savePlots, plotsDir,
                    () => PlotRenderer.IonProfilePng(
                        cycles, PlotRenderer.IonLevel.Ms2, binMinutes: 1.0,
                        title: $"{row.Sample}: MS2 ions across the gradient"));
            }
        }

        if (images.Count > 0)
            sections.Add(new PlotSection("Assigned Share Across the Gradient", images));
    }

    /// <summary>
    /// What to say about the cohort's fraction, and what to refuse to say. A fraction over 1 is
    /// impossible, so where one occurs the figure is withheld and named as a defect rather than
    /// quietly excluded from a median.
    /// </summary>
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
    private static void Render(
        List<PlotImage> images, string caption, string fileName, bool savePlots, string plotsDir,
        Func<byte[]> render)
    {
        try
        {
            var png = render();
            if (savePlots && png.Length > 0)
            {
                Directory.CreateDirectory(plotsDir);
                File.WriteAllBytes(Path.Combine(plotsDir, fileName), png);
            }
            images.Add(new PlotImage(caption, png));
        }
        catch (Exception ex)
        {
            images.Add(new PlotImage(
                caption + " (render failed: " + ex.GetType().Name + ")", Array.Empty<byte>()));
        }
    }
}
