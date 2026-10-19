using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text.Json;
using SkylinePrism.Core.Config;
using Xunit;

namespace SkylinePrism.Tests.Config;

/// <summary>
/// <see cref="PrismConfig.Overlay"/>: a YAML config applied on top of another config, as
/// <c>prism run --from-provenance ... -c overrides.yaml</c> applies it on top of an earlier run's.
/// </summary>
public class ConfigOverlayTests
{
    private static string Json(PrismConfig config) => JsonSerializer.Serialize(config);

    private static string Yaml(PrismConfig config) =>
        new YamlDotNet.Serialization.SerializerBuilder()
            .WithNamingConvention(YamlDotNet.Serialization.NamingConventions.UnderscoredNamingConvention.Instance)
            .Build()
            .Serialize(config);

    /// <summary>
    /// Every settable value in every section moved off its default, so a key the overlay failed to
    /// bind cannot hide behind a default that happens to agree.
    /// </summary>
    private static void Perturb(object section)
    {
        foreach (var p in section.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (!p.CanWrite)
                continue;
            var t = p.PropertyType;
            if (t.DeclaringType == typeof(PrismConfig))
            {
                var child = p.GetValue(section) ?? Activator.CreateInstance(t)!;
                Perturb(child);
                p.SetValue(section, child);
            }
            else if (t == typeof(string))
                p.SetValue(section, "x_" + p.Name);
            else if (t == typeof(int) || t == typeof(int?))
                p.SetValue(section, ((int?)p.GetValue(section) ?? 0) + 7);
            else if (t == typeof(double) || t == typeof(double?))
                p.SetValue(section, ((double?)p.GetValue(section) ?? 0) + 0.5);
            else if (t == typeof(bool) || t == typeof(bool?))
                p.SetValue(section, !((bool?)p.GetValue(section) ?? false));
            else if (t == typeof(List<string>))
                p.SetValue(section, new List<string> { "x_" + p.Name });
            else
                throw new InvalidOperationException($"Perturb does not know {t} ({p.Name}) - add it here.");
        }
    }

    private static PrismConfig Perturbed(bool withLibraryAssistBlock)
    {
        var config = new PrismConfig();
        Perturb(config);
        if (!withLibraryAssistBlock)
            config.TransitionRollup.LibraryAssist = null;
        return config;
    }

    /// <summary>
    /// An override means what the same YAML means on its own: over the defaults, every key PRISM
    /// reads binds to the value <see cref="PrismConfig.Parse"/> gives it.
    /// </summary>
    /// <remarks>
    /// The YAML is a full dump of a config with every value moved, so this covers every key in every
    /// section - including ones a hand-written test would never think to set - and a key added to the
    /// config later is covered without touching this test. The nested library_assist block is left
    /// out here because Parse folds it over the flat library_* keys, which would mask a flat key the
    /// overlay missed; it has its own tests below.
    /// </remarks>
    [Fact]
    public void Overlay_OntoTheDefaults_BindsEveryKeyAsParseDoes()
    {
        var yaml = Yaml(Perturbed(withLibraryAssistBlock: false));
        var expected = PrismConfig.Parse(yaml);
        Assert.NotEqual(Json(new PrismConfig()), Json(expected));

        var overlaid = new PrismConfig();
        var applied = overlaid.Overlay(yaml);

        Assert.Equal(Json(expected), Json(overlaid));
        Assert.Contains("global_normalization.rt_lowess.n_grid_points", applied);
        Assert.Contains("protein_rollup.ibaq.max_peptide_length", applied);
        Assert.Contains("sample_annotations.qc_pattern", applied);
    }

    /// <summary>
    /// The other direction: a YAML that names every key replaces every value of a base that differs
    /// from it everywhere, the nested library_assist block included.
    /// </summary>
    [Fact]
    public void Overlay_OfEveryKey_ReplacesEveryValueOfTheBase()
    {
        var yaml = Yaml(new PrismConfig());
        var config = Perturbed(withLibraryAssistBlock: true);

        config.Overlay(yaml);

        Assert.Equal(Json(PrismConfig.Parse(yaml)), Json(config));
    }

    [Fact]
    public void Overlay_ChangesTheKeysItNames_AndKeepsEveryOther()
    {
        var config = PrismConfig.Parse("""
            transition_rollup:
              method: topn
              topn_count: 5
            global_normalization:
              method: rt_lowess
              rt_lowess:
                frac: 0.5
                n_grid_points: 50
            sample_annotations:
              reference_pattern: [Ref]
              qc_pattern: [QC]
            """);

        var applied = config.Overlay("""
            transition_rollup:
              topn_count: 7
            global_normalization:
              rt_lowess:
                frac: 0.2
            sample_annotations:
              reference_pattern: [Pool, Anchor]
            """);

        Assert.Equal(
            new[] { "transition_rollup.topn_count", "global_normalization.rt_lowess.frac",
                    "sample_annotations.reference_pattern" },
            applied);
        Assert.Equal("topn", config.TransitionRollup.Method);
        Assert.Equal(7, config.TransitionRollup.TopnCount);
        Assert.Equal("rt_lowess", config.GlobalNormalization.Method);
        // A subsection is merged key by key ...
        Assert.Equal(0.2, config.GlobalNormalization.RtLowess.Frac);
        Assert.Equal(50, config.GlobalNormalization.RtLowess.NGridPoints);
        // ... and a list is replaced whole, never appended to.
        Assert.Equal(new[] { "Pool", "Anchor" }, config.SampleAnnotations.ReferencePattern);
        Assert.Equal(new[] { "QC" }, config.SampleAnnotations.QcPattern);
    }

    /// <summary>
    /// A flat library_* key overrides a base whose YAML used the nested block, rather than being
    /// undone by that block when the overlay folds library_assist again.
    /// </summary>
    [Fact]
    public void Overlay_AFlatLibraryKey_IsNotUndoneByTheBasesNestedBlock()
    {
        var config = PrismConfig.Parse("""
            transition_rollup:
              method: library_assist
              library_assist:
                library_path: a.blib
                mz_tolerance: 0.05
            """);
        Assert.Equal("a.blib", config.TransitionRollup.LibraryPath);

        config.Overlay("transition_rollup:\n  library_path: b.blib\n");

        Assert.Equal("b.blib", config.TransitionRollup.LibraryPath);
        Assert.Equal(0.05, config.TransitionRollup.LibraryMzTolerance);
    }

    /// <summary>
    /// Within one YAML the nested block still wins over a flat key, as it does in Parse.
    /// </summary>
    [Fact]
    public void Overlay_ANestedLibraryKey_WinsOverAFlatOne_AsInParse()
    {
        var config = PrismConfig.Parse("transition_rollup:\n  library_path: a.blib\n  library_mz_tolerance: 0.05\n");
        var yaml = """
            transition_rollup:
              library_path: d.blib
              library_assist:
                library_path: c.blib
            """;

        var applied = config.Overlay(yaml);

        Assert.Equal(PrismConfig.Parse(yaml).TransitionRollup.LibraryPath, config.TransitionRollup.LibraryPath);
        Assert.Equal("c.blib", config.TransitionRollup.LibraryPath);
        Assert.Equal(0.05, config.TransitionRollup.LibraryMzTolerance); // not named, so kept
        Assert.Contains("transition_rollup.library_assist.library_path", applied);
    }

    [Fact]
    public void Overlay_ASectionHeaderWithNothingUnderIt_SetsNothing()
    {
        var config = new PrismConfig();
        config.QcReport.Enabled = false;

        var applied = config.Overlay("qc_report:\n");

        Assert.Empty(applied);
        Assert.NotNull(config.QcReport);
        Assert.False(config.QcReport.Enabled);
    }

    [Fact]
    public void Overlay_SkipsAKeyPrismDoesNotRead()
    {
        var config = new PrismConfig();
        var before = Json(config);

        var applied = config.Overlay("transition_rollup:\n  min_transition: 5\nnot_a_section: 1\n");

        Assert.Empty(applied);
        Assert.Equal(before, Json(config));
    }

    /// <summary>
    /// The file form warns about an unknown key exactly as a plain <c>-c</c> does, and refuses a
    /// method PRISM does not implement whether it came from the base or the override.
    /// </summary>
    [Fact]
    public void ApplyOverrides_WarnsAboutUnknownKeys_AndRefusesAnUnimplementedMethod()
    {
        var path = Path.Combine(Path.GetTempPath(), "prism_overlay_" + Guid.NewGuid().ToString("N") + ".yaml");
        try
        {
            File.WriteAllText(path, "transition_rollup:\n  min_transitions: 2\n  min_transition: 5\n");
            var warnings = new List<string>();
            var config = new PrismConfig();

            var applied = config.ApplyOverrides(path, warnings.Add);

            Assert.Equal(new[] { "transition_rollup.min_transitions" }, applied);
            Assert.Equal(2, config.TransitionRollup.MinTransitions);
            var warning = Assert.Single(warnings);
            Assert.Contains("'transition_rollup.min_transition' is not recognized", warning, StringComparison.Ordinal);

            File.WriteAllText(path, "transition_rollup:\n  method: adaptive\n");
            Assert.Throws<NotSupportedException>(() => new PrismConfig().ApplyOverrides(path));

            File.WriteAllText(path, "qc_report:\n  enabled: false\n");
            var adaptive = new PrismConfig();
            adaptive.TransitionRollup.Method = "adaptive";
            Assert.Throws<NotSupportedException>(() => adaptive.ApplyOverrides(path));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
