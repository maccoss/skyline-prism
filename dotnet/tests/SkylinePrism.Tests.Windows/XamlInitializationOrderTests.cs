using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// A rule with a specific history: setting <c>SelectedIndex</c> on a ComboBox <b>in XAML</b> makes WPF
/// raise <c>SelectionChanged</c> from that ComboBox's <c>EndInit</c> - part way through
/// <c>InitializeComponent</c>. Every control declared AFTER it in the XAML is still unbuilt and its
/// generated field is null.
///
/// <para>A handler that touches one there throws out of the window's constructor. That is not a handler
/// error that shows up as a dialog and leaves a working tool behind: it is a startup crash, and the tool
/// never opens at all. It shipped once, when the density View picker's handler set the visibility of the
/// colormap label declared a few lines below it.</para>
///
/// <para>The two handlers already on preselected combos did not crash, but only because the state they
/// happened to test (<c>_densityMap</c>, <c>_rangeTabShown</c>) is falsy during construction - an
/// accident of ordering rather than a guard. This test requires the invariant to be stated:
/// <c>IsInitialized</c> is false for the whole of <c>InitializeComponent</c> and true forever after, so
/// checking it is the one condition that means what it says here.</para>
///
/// <para>Note it is specifically XAML that is dangerous. The many <c>Combo.SelectedIndex = 0</c> lines in
/// the MainWindow constructor run AFTER <c>InitializeComponent</c>, when every control exists, and are
/// not covered by this rule.</para>
/// </summary>
public class XamlInitializationOrderTests
{
    private static string AppDir =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));

    [Fact]
    public void EveryPreselectedComboHandler_GuardsAgainstFiringDuringInitialization()
    {
        var handlers = PreselectedComboHandlers();
        Assert.NotEmpty(handlers); // the XAML scan must actually have found something

        var members = Members();
        var unguarded = new List<string>();
        foreach (var (control, handler) in handlers)
        {
            Assert.True(members.ContainsKey(handler),
                $"{control}'s SelectionChanged handler {handler} was not found - was it renamed?");
            // Comments stripped first. Without that, the explanatory comment above a guard satisfies the
            // check on its own - which it did, and this test passed on a handler whose guard had been
            // deleted. A rule that a comment can satisfy is not a rule.
            if (!StripComments(members[handler]).Contains("IsInitialized", StringComparison.Ordinal))
                unguarded.Add($"{control} -> {handler}");
        }

        Assert.True(unguarded.Count == 0,
            "These ComboBoxes set SelectedIndex in XAML, so their SelectionChanged fires during "
            + "InitializeComponent - before the controls declared after them exist. Their handlers must "
            + "return early on !IsInitialized, or the first one to touch a later control crashes the "
            + "tool on startup:" + Environment.NewLine
            + string.Join(Environment.NewLine, unguarded));
    }

    /// <summary>
    /// Every <c>&lt;ComboBox&gt;</c> in the window's XAML that sets both <c>SelectedIndex</c> and
    /// <c>SelectionChanged</c>, as (control name, handler name).
    /// </summary>
    private static List<(string Control, string Handler)> PreselectedComboHandlers()
    {
        var xaml = File.ReadAllText(Path.Combine(AppDir, "MainWindow.xaml"));
        var found = new List<(string, string)>();
        // Attributes may be spread over several lines, so match the whole opening tag.
        foreach (Match tag in Regex.Matches(xaml, @"<ComboBox\b[^>]*>", RegexOptions.Singleline))
        {
            var text = tag.Value;
            if (!Regex.IsMatch(text, @"\bSelectedIndex\s*="))
                continue;
            var handler = Regex.Match(text, @"\bSelectionChanged\s*=\s*""(\w+)""");
            if (!handler.Success)
                continue;
            var name = Regex.Match(text, @"\bx:Name\s*=\s*""(\w+)""");
            found.Add((name.Success ? name.Groups[1].Value : "(unnamed ComboBox)", handler.Groups[1].Value));
        }
        return found;
    }

    /// <summary>
    /// Drop <c>//</c> and <c>/* */</c> comments so only executable code is inspected. Crude - it would
    /// also cut a <c>//</c> inside a string literal - which is fine for asking whether a body tests a
    /// property, and much better than counting prose as a guard.
    /// </summary>
    private static string StripComments(string body) =>
        Regex.Replace(Regex.Replace(body, @"/\*.*?\*/", "", RegexOptions.Singleline), @"//[^\r\n]*", "");

    /// <summary>Method name -> body, across every partial of MainWindow.</summary>
    private static Dictionary<string, string> Members()
    {
        var members = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var file in Directory.GetFiles(AppDir, "MainWindow*.cs"))
        {
            var src = File.ReadAllText(file);
            foreach (Match m in Regex.Matches(
                         src, @"^\s*(?:private|internal|public|protected)[^\r\n=;]*?\b(\w+)\s*\(",
                         RegexOptions.Multiline))
            {
                var name = m.Groups[1].Value;
                if (name is "if" or "for" or "foreach" or "while" or "switch" or "return" or "using")
                    continue;
                if (!members.ContainsKey(name))
                    members[name] = BlockAt(src, m.Index);
            }
        }
        return members;
    }

    /// <summary>The brace-balanced block starting at or after <paramref name="from"/>.</summary>
    private static string BlockAt(string src, int from)
    {
        var open = src.IndexOf('{', from);
        if (open < 0)
            return "";
        var depth = 0;
        for (var i = open; i < src.Length; i++)
        {
            if (src[i] == '{') depth++;
            else if (src[i] == '}' && --depth == 0)
                return src[open..(i + 1)];
        }
        return src[open..];
    }
}
