using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using SkylinePrism.Core.Config;
using SkylinePrism.Core.Pipeline;
using Xunit;

namespace SkylinePrism.Tests.Pipeline;

/// <summary>
/// The stage-cache's safety property, tested rather than trusted.
///
/// <para>Reuse is only correct if <see cref="StageDependencies.ByStage"/> names every config key that
/// can change a stage's output. A key missing from the table is not a compile error and not a test
/// failure anywhere else - it is a run that silently reuses a stale file and reports numbers computed
/// with the OLD setting. These tests make adding a key without classifying it fail the build.</para>
/// </summary>
public class StageDependencyCoverageTests
{
    [Fact]
    public void EveryConfigKeyIsEitherUsedByAStageOrDeclaredIrrelevant()
    {
        var claimed = StageDependencies.ByStage.Values
            .SelectMany(keys => keys.SelectMany(StageDependencies.ExpandToLeaves))
            .ToHashSet(StringComparer.Ordinal);

        var irrelevant = StageDependencies.OutputIrrelevant.Keys
            .SelectMany(StageDependencies.ExpandToLeaves)
            .ToHashSet(StringComparer.Ordinal);

        var unclassified = StageDependencies.AllLeafKeys()
            .Where(k => !claimed.Contains(k) && !irrelevant.Contains(k))
            .OrderBy(k => k, StringComparer.Ordinal)
            .ToList();

        Assert.True(unclassified.Count == 0,
            "These config keys are not classified in StageDependencies, so the stage cache cannot know "
            + "whether changing them invalidates anything. Add each to the stage(s) that read it, or to "
            + "OutputIrrelevant with the reason:\n  " + string.Join("\n  ", unclassified));
    }

    [Fact]
    public void NoKeyIsBothUsedAndDeclaredIrrelevant()
    {
        var claimed = StageDependencies.ByStage.Values
            .SelectMany(keys => keys.SelectMany(StageDependencies.ExpandToLeaves))
            .ToHashSet(StringComparer.Ordinal);
        var both = StageDependencies.OutputIrrelevant.Keys
            .SelectMany(StageDependencies.ExpandToLeaves)
            .Where(claimed.Contains)
            .ToList();

        Assert.True(both.Count == 0,
            "Declared irrelevant while also feeding a stage's fingerprint: " + string.Join(", ", both));
    }

    [Fact]
    public void EveryDeclaredKeyExists()
    {
        // A renamed config key must not linger here, silently contributing nothing to a fingerprint.
        var all = StageDependencies.AllLeafKeys().ToHashSet(StringComparer.Ordinal);
        var declared = StageDependencies.ByStage.Values.SelectMany(v => v)
            .Concat(StageDependencies.OutputIrrelevant.Keys)
            .SelectMany(StageDependencies.ExpandToLeaves)
            .Distinct(StringComparer.Ordinal);

        var missing = declared.Where(k => !all.Contains(k)).OrderBy(k => k).ToList();
        Assert.True(missing.Count == 0,
            "Declared in StageDependencies but not a key of PrismConfig: " + string.Join(", ", missing));
    }

    /// <summary>
    /// The property that makes the table meaningful: mutating a key a stage declares MUST change that
    /// stage's fingerprint, and mutating one it does not declare must NOT. Driven by reflection over
    /// every leaf key, so it covers keys added later too.
    /// </summary>
    [Fact]
    public void MutatingAKeyChangesExactlyTheStagesThatDeclareIt()
    {
        var failures = new List<string>();

        foreach (var key in StageDependencies.AllLeafKeys())
        {
            var baseline = new PrismConfig();
            var mutated = new PrismConfig();
            if (!TryMutate(mutated, key))
                continue; // no representative alternative value (a list, or an unsupported type)

            foreach (var stage in StageDependencies.ByStage.Keys)
            {
                var declares = StageDependencies.ByStage[stage]
                    .SelectMany(StageDependencies.ExpandToLeaves)
                    .Contains(key, StringComparer.Ordinal);

                var changed = StageCache.Fingerprint(stage, baseline)
                    != StageCache.Fingerprint(stage, mutated);

                if (declares && !changed)
                    failures.Add($"{stage} declares '{key}' but its fingerprint ignored a change to it");
                if (!declares && changed)
                    failures.Add($"{stage} does not declare '{key}' yet its fingerprint changed");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    /// <summary>Set a leaf key to a value different from its default; false when we cannot.</summary>
    private static bool TryMutate(PrismConfig config, string key)
    {
        var segments = key.Split('.');
        object? owner = config;
        for (var i = 0; i < segments.Length - 1; i++)
        {
            owner = StageDependencies.ReadPath(owner, segments[i]);
            if (owner is null)
                return false;
        }
        var prop = owner!.GetType().GetProperty(
            StageDependencies.ToPascal(segments[^1]), BindingFlags.Public | BindingFlags.Instance);
        if (prop is null || !prop.CanWrite)
            return false;

        var current = prop.GetValue(owner);
        var type = Nullable.GetUnderlyingType(prop.PropertyType) ?? prop.PropertyType;
        object? next;
        if (type == typeof(string))
            next = (current as string) == "prism-test" ? "prism-test-2" : "prism-test";
        else if (type == typeof(bool))
            next = !(bool)(current ?? false);
        else if (type == typeof(int))
            next = ((int?)current ?? 0) + 7;
        else if (type == typeof(double))
            next = ((double?)current ?? 0) + 0.5;
        else
            return false;

        prop.SetValue(owner, next);
        return true;
    }
}
