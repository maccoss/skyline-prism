using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Every number the Ion Accounting pane states must be the quantity the plot beside it is drawing.
/// </summary>
/// <remarks>
/// <para>Ions and signal are different numbers, not two units for one: the ion count weights each
/// scan by its ion injection time and the summed TIC does not. So <c>Ms2Fraction</c> and
/// <c>Ms2SignalFraction</c> are both correct and they disagree - by whatever the injection times
/// were doing across the run. That shipped: with Quantity set to Signal the plot drew TIC
/// percentages while the status line beneath it reported the ion-weighted ones, both plausible, with
/// nothing on screen to say they were different quantities.</para>
///
/// <para>A source check rather than a behavioral one, like <see cref="UiThreadSafetyTests"/> and for
/// the same reason: these are private methods on <c>MainWindow</c>, which needs a real window and
/// dispatcher. And a source check fails for the NEXT instance of the mistake too - the failure mode
/// is a plausible number, so nothing about it is visible at run time.</para>
/// </remarks>
public class IonQuantityConsistencyTests
{
    private static string AppDir =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));

    /// <summary>
    /// The quantity-blind fraction properties, which read correctly and are wrong half the time.
    /// Their quantity-aware forms - <c>Ms1FractionIn(signal)</c> and friends - take the choice out
    /// of the caller's hands, which is the only reliable way to make it.
    /// </summary>
    private static readonly string[] QuantityBlind =
    {
        "Ms1Fraction", "Ms2Fraction", "Ms2ExplainedFraction",
    };

    [Fact]
    public void TheIonPaneNeverReadsAQuantityBlindFraction()
    {
        var source = CodeOnly();

        foreach (var property in QuantityBlind)
        {
            // "Ms2Fraction" but not "Ms2FractionIn", which is the one that is allowed.
            var uses = Regex.Matches(source, $@"\.{Regex.Escape(property)}\b(?!In)")
                .Select(m => LineOf(source, m.Index))
                .ToArray();

            Assert.True(
                uses.Length == 0,
                $"MainWindow.IonAccounting.cs reads {property} directly at line(s) "
                + $"{string.Join(", ", uses)}. Ions and signal have DIFFERENT fractions, so a number "
                + $"stated beside the plot must come from {property}In(signal) - otherwise the "
                + "status line quotes ion percentages under a signal plot and nothing says so.");
        }
    }

    /// <summary>
    /// The impossibility check has the same trap: <c>Exceeded</c> is the ion one, and it does not
    /// imply <c>SignalExceeded</c>. Withholding on the wrong one shows an impossible fraction with
    /// no warning at all.
    /// </summary>
    [Fact]
    public void TheIonPaneChecksImpossibilityInTheQuantityItIsShowing()
    {
        var source = CodeOnly();

        var uses = Regex.Matches(source, @"\.Exceeded\b(?!In)")
            .Select(m => LineOf(source, m.Index))
            .ToArray();

        Assert.True(
            uses.Length == 0,
            "MainWindow.IonAccounting.cs reads Exceeded directly at line(s) "
            + $"{string.Join(", ", uses)}. Use ExceededIn(signal): a fault confined to short "
            + "injection scans leaves the ion fraction under 1 while the signal fraction is over "
            + "it, so the check has to be made in the quantity being drawn.");
    }

    /// <summary>
    /// The file with its line comments blanked out, line numbering preserved.
    /// </summary>
    /// <remarks>
    /// A comment naming <c>IonAccountingRow.Exceeded</c> - which is the right thing for a comment to
    /// do, since that is where the rule is written down - is not a use of it. Blanking rather than
    /// removing keeps the reported line numbers pointing at the real file.
    /// </remarks>
    private static string CodeOnly()
    {
        var lines = File.ReadAllLines(Path.Combine(AppDir, "MainWindow.IonAccounting.cs"));
        for (var i = 0; i < lines.Length; i++)
        {
            var comment = lines[i].IndexOf("//", StringComparison.Ordinal);
            if (comment >= 0)
                lines[i] = lines[i][..comment];
        }
        return string.Join("\n", lines);
    }

    private static int LineOf(string source, int index) =>
        source.Take(index).Count(c => c == '\n') + 1;
}
