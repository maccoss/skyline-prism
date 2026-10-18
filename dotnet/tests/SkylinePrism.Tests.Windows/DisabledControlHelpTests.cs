using System;
using System.IO;
using System.Text.RegularExpressions;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// A control this window GRAYS rather than hides has to be able to say why.
/// </summary>
/// <remarks>
/// WPF suppresses tooltips on disabled elements unless <c>ToolTipService.ShowOnDisabled</c> is set,
/// so a grayed control's explanation is invisible at exactly the moment it is the only thing the
/// reader needs. That is a one-attribute mistake to make and an invisible one to find - a reviewer
/// sees a tooltip in the markup and assumes it shows - so it is pinned here rather than trusted.
/// </remarks>
public class DisabledControlHelpTests
{
    private static string Xaml => File.ReadAllText(Path.GetFullPath(Path.Combine(
        AppContext.BaseDirectory, "..", "..", "..", "..", "..",
        "src", "SkylinePrism.App", "MainWindow.xaml")));

    /// <summary>
    /// The prior-source checkbox is grayed when the run has no control replicates - deliberately,
    /// rather than hidden, so its tooltip can say why it is off. That only works if the tooltip
    /// survives being disabled.
    /// </summary>
    [Fact]
    public void ThePriorSourceCheckbox_ShowsItsTooltipWhileDisabled()
    {
        var element = ElementMarkup("DiffPriorFromControlsCheck");

        Assert.Contains("ToolTipService.ShowOnDisabled=\"True\"", element, StringComparison.Ordinal);
    }

    /// <summary>
    /// Its help text is set in code, not markup, because the disabled case prefixes it with the
    /// reason. A static ToolTip in the XAML would silently win over that.
    /// </summary>
    [Fact]
    public void ThePriorSourceCheckbox_HasNoStaticTooltipToOverrideTheDynamicOne()
    {
        var element = ElementMarkup("DiffPriorFromControlsCheck");

        Assert.DoesNotContain("ToolTip=\"", element, StringComparison.Ordinal);
    }

    /// <summary>The markup of one named element, from its opening tag to the end of that tag.</summary>
    private static string ElementMarkup(string name)
    {
        var xaml = Xaml;
        var start = xaml.IndexOf($"x:Name=\"{name}\"", StringComparison.Ordinal);
        Assert.True(start >= 0, $"{name} was not found in MainWindow.xaml - was it renamed?");

        // Back up to the opening '<' of this element, then forward to the tag's close.
        var open = xaml.LastIndexOf('<', start);
        var end = xaml.IndexOf('>', start);
        Assert.True(open >= 0 && end > open, $"could not delimit the {name} element");
        return xaml[open..end];
    }
}

/// <summary>
/// The Differential pane's handler-suppression flag must only be written through its scope.
/// </summary>
/// <remarks>
/// The flag nests: assigning a combo's SelectedIndex raises SelectionChanged synchronously, and
/// that handler suppresses around its own assignments. A bare `_diffSuppress = false` on the way
/// out of the inner one ends the OUTER window too, so the rest of a load runs its handlers
/// unsuppressed against a half-initialized pane. SuppressScope saves and restores instead - but
/// only if every site goes through it, which is what this pins.
/// </remarks>
public class DiffSuppressionScopeTests
{
    [Fact]
    public void TheSuppressionFlag_IsOnlyWrittenInsideItsScope()
    {
        var path = Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..",
            "src", "SkylinePrism.App", "MainWindow.Differential.cs"));
        var lines = File.ReadAllLines(path);

        var writes = lines
            .Select((text, i) => (text, line: i + 1))
            .Where(t => t.text.Contains("_diffSuppress = ", StringComparison.Ordinal)
                        || t.text.Contains("_diffSuppress =", StringComparison.Ordinal))
            .Where(t => !t.text.Contains("owner._diffSuppress = true", StringComparison.Ordinal)
                        && !t.text.Contains("_owner._diffSuppress = _previous", StringComparison.Ordinal))
            .ToList();

        Assert.True(writes.Count == 0,
            "write the flag through SuppressDiff() instead, so it restores rather than clears: "
            + string.Join("; ", writes.Select(w => $"line {w.line}: {w.text.Trim()}")));
    }
}
