using System;
using System.IO;
using SkylinePrism.App;
using SkylinePrism.Core.DifferentialAnalysis;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The Differential pane's contrast survives the pickers being rebuilt: a level switch, a failed load
/// and its retry, a re-run, and a round trip through another Group-by column. Reported as "set up
/// Purpose KI vs KI_control at protein level, switch to peptide, and nothing can be grouped any more;
/// every change resets the other settings".
/// </summary>
public class DiffContrastMemoryTests
{
    private static readonly string[] PurposeValues = { "KI", "KI_control", "WT" };

    [Fact]
    public void Arms_ComeBackForTheColumnTheyWerePickedUnder()
    {
        var memory = new DiffContrastMemory();
        memory.RememberArms("Purpose", new[] { "KI_control" }, new[] { "KI" });
        memory.RememberArms("Genotype", new[] { "WT" }, new[] { "Q111" });

        Assert.True(memory.TryGetArms("Purpose", PurposeValues, out var a, out var b));
        Assert.Equal(new[] { "KI_control" }, a);
        Assert.Equal(new[] { "KI" }, b);
        Assert.False(memory.TryGetArms("Sex", new[] { "Male", "Female" }, out _, out _));
    }

    /// <summary>A value the newly loaded matrix does not have is dropped, never restored as a phantom tick.</summary>
    [Fact]
    public void Arms_AreKeptToTheValuesTheColumnStillHas()
    {
        var memory = new DiffContrastMemory();
        memory.RememberArms("Purpose", new[] { "KI_control", "Retired" }, new[] { "KI" });

        Assert.True(memory.TryGetArms("Purpose", PurposeValues, out var a, out _));
        Assert.Equal(new[] { "KI_control" }, a);
    }

    /// <summary>
    /// Restoring one arm beside a defaulted other could tick the same value in both, so an arm that
    /// cannot come back sends both to their defaults.
    /// </summary>
    [Fact]
    public void Arms_AreNotRestored_WhenEitherArmHasNothingLeft()
    {
        var memory = new DiffContrastMemory();
        memory.RememberArms("Purpose", new[] { "KI_control" }, new[] { "Retired" });

        Assert.False(memory.TryGetArms("Purpose", PurposeValues, out _, out _));
    }

    /// <summary>An emptied arm is a state passed through while re-ticking; it must not overwrite a real choice.</summary>
    [Fact]
    public void Arms_AnEmptyArmDoesNotReplaceTheLastRealChoice()
    {
        var memory = new DiffContrastMemory();
        memory.RememberArms("Purpose", new[] { "KI_control" }, new[] { "KI" });
        memory.RememberArms("Purpose", Array.Empty<string>(), new[] { "KI" });

        Assert.True(memory.TryGetArms("Purpose", PurposeValues, out var a, out _));
        Assert.Equal(new[] { "KI_control" }, a);
    }

    /// <summary>A failed load empties the picker; that is not the user choosing "no Group by".</summary>
    [Fact]
    public void GroupByAndSubject_IgnoreAnEmptiedPicker()
    {
        var memory = new DiffContrastMemory();
        memory.RememberGroupBy("Purpose");
        memory.RememberSubject("Mouse ID");
        memory.RememberGroupBy(null);
        memory.RememberSubject(null);

        Assert.Equal("Purpose", memory.GroupBy);
        Assert.Equal("Mouse ID", memory.Subject);
    }

    [Fact]
    public void Covariates_KeepTheirTickAndSwitchedType()
    {
        var memory = new DiffContrastMemory();
        memory.RememberCovariates(new (string, bool, CovariateKind?)[]
        {
            ("Sex", true, null),
            ("Age (mo)", true, CovariateKind.Categorical),
            ("Genotype", false, null),
        });

        Assert.True(memory.TryGetCovariate("Sex", out var sex));
        Assert.Null(sex);
        Assert.True(memory.TryGetCovariate("Age (mo)", out var age));
        Assert.Equal(CovariateKind.Categorical, age);
        Assert.False(memory.TryGetCovariate("Genotype", out _));
    }

    /// <summary>
    /// The picker never offers the Group-by column. Making a ticked covariate the Group by and then
    /// moving off it again brings its tick back; unticking an offered column forgets it.
    /// </summary>
    [Fact]
    public void Covariates_OnlyTheColumnsOfferedAreUpdated()
    {
        var memory = new DiffContrastMemory();
        memory.RememberCovariates(new (string, bool, CovariateKind?)[] { ("Sex", true, null), ("Genotype", true, null) });

        // Group by is now Genotype, so the picker lists Sex but not Genotype - and Sex was unticked.
        memory.RememberCovariates(new (string, bool, CovariateKind?)[] { ("Sex", false, null) });

        Assert.False(memory.TryGetCovariate("Sex", out _));
        Assert.True(memory.TryGetCovariate("Genotype", out _));
    }

    /// <summary>
    /// The pane takes its restored Group by from the memory, not from a capture that only a successful
    /// load at the other level ever made. Source check, as in <see cref="DefaultContrastColumnTests"/>:
    /// the path needs a real window and a run to exercise.
    /// </summary>
    [Fact]
    public void Pane_RestoresGroupByFromMemoryAndRemembersBeforeClearing()
    {
        var source = File.ReadAllText(Path.Combine(AppDir(), "MainWindow.Differential.cs"));

        Assert.Contains("_diffMemory.GroupBy is { } kept && ds.MetadataColumns.Contains(kept)", source, StringComparison.Ordinal);
        Assert.DoesNotContain("CaptureContrastChoices", source, StringComparison.Ordinal);

        // ClearDiffPickers must remember before it empties, or a failed load loses the contrast.
        var clear = source.IndexOf("private void ClearDiffPickers()", StringComparison.Ordinal);
        var remember = source.IndexOf("RememberDiffChoices();", clear, StringComparison.Ordinal);
        var empty = source.IndexOf("DiffGroupByCombo.ItemsSource = null;", clear, StringComparison.Ordinal);
        Assert.True(clear >= 0 && remember > clear && remember < empty);
    }

    private static string AppDir() => Path.GetFullPath(Path.Combine(
        AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));
}
