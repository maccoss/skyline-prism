using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// Two rules the Dynamic Range selection poll broke, both of which produced the same symptom: error
/// dialogs, hundreds of them, until the tool was killed.
///
/// <list type="number">
/// <item>Code reached from a worker thread must not touch a WPF control. The poll built its locator
/// index off the UI thread (150,000 entries would freeze the window), and the method it called read
/// <c>RangeLevel</c> - a property that reads a ComboBox - so every tick threw
/// <c>The calling thread cannot access this object because a different thread owns it</c>.</item>
/// <item>An <c>async void</c> handler must not be able to throw. An exception escaping one becomes an
/// unhandled application exception; on a 750 ms timer that is a dialog every 750 ms, forever.</item>
/// </list>
///
/// <para>These are source checks rather than behavioral ones because the code under test lives in
/// <c>MainWindow</c> and needs a real window and dispatcher to exercise. A source check still fails on
/// the exact defect that shipped, and - unlike a test of one method - it fails for the next instance of
/// the same class of mistake too.</para>
/// </summary>
public class UiThreadSafetyTests
{
    private static string AppDir =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "SkylinePrism.App"));

    /// <summary>Methods the selection poll runs on a worker thread, and must therefore keep UI-free.</summary>
    private static readonly string[] WorkerThreadMethods =
    {
        "FindEntryByLocator",
        "ResolveLocator",
    };

    [Fact]
    public void WorkerThreadMethods_TouchNoWpfControls()
    {
        var controls = ControlNames();
        Assert.NotEmpty(controls); // the XAML parse must actually have found something

        var members = Members();
        var uiBound = UiBoundMembers(members, controls);

        // The regression itself: ResolveLocator did not name a control, it called RangeLevel, which
        // reads RangeLevelCombo. Only a transitive check catches that.
        Assert.Contains("RangeLevel", uiBound);

        foreach (var method in WorkerThreadMethods)
        {
            Assert.True(members.ContainsKey(method), $"{method} not found - was it renamed?");
            Assert.False(uiBound.Contains(method),
                $"{method} runs on the selection poll's worker thread but reaches a WPF control "
                + $"(via {string.Join(" / ", Reaches(method, members, uiBound))}). "
                + "Pass the value in as a parameter instead, or marshal to the dispatcher.");
        }
    }

    [Fact]
    public void EveryAsyncVoidHandler_CatchesItsOwnFailures()
    {
        var bare = new List<string>();
        foreach (var file in Directory.GetFiles(AppDir, "*.cs"))
        {
            var src = File.ReadAllText(file);
            foreach (Match m in Regex.Matches(src, @"\basync void\s+(\w+)\s*\("))
            {
                var body = BlockAt(src, m.Index);
                if (!CatchesEverything(body))
                    bare.Add($"{Path.GetFileName(file)}: {m.Groups[1].Value}");
            }
        }

        Assert.True(bare.Count == 0,
            "async void cannot propagate an exception to a caller, so an escaping one becomes an "
            + "unhandled application exception - a modal dialog, and on a timer one per tick until the "
            + "tool is killed. These must catch and report:" + Environment.NewLine
            + string.Join(Environment.NewLine, bare));
    }

    /// <summary>
    /// Whether a body has a catch that really does catch everything: a bare <c>catch</c>, or
    /// <c>catch (Exception ...)</c> with no <c>when</c> filter.
    /// <para>
    /// The distinction matters. <c>catch (IOException)</c>, or <c>catch (Exception e) when (...)</c>,
    /// look like protection and are not - whatever falls outside still escapes, and for an
    /// <c>async void</c> handler that is the unhandled-exception dialog again.
    /// </para>
    /// </summary>
    private static bool CatchesEverything(string body)
    {
        foreach (Match c in Regex.Matches(body, @"\bcatch\s*(?:\(([^)]*)\))?\s*(when\s*\()?"))
        {
            // Only the HANDLER'S OWN catch counts - depth 1, i.e. a statement of the method body
            // itself. A catch nested inside a lambda is not protection: PollSkylineSelection wraps
            // its RPC call in `Task.Run(() => { try { ... } catch { ... } })`, and counting that let
            // this check pass on the exact code that shipped without a handler-level catch at all.
            if (DepthAt(body, c.Index) != 1)
                continue;
            if (c.Groups[2].Success)
                continue; // filtered: only catches some of it
            var caught = c.Groups[1].Value.Trim();
            if (caught.Length == 0 || Regex.IsMatch(caught, @"^(System\.)?Exception\b"))
                return true;
        }
        return false;
    }

    /// <summary>Brace nesting at <paramref name="index"/>; the body's own statements sit at 1.</summary>
    private static int DepthAt(string body, int index)
    {
        var depth = 0;
        for (var i = 0; i < index; i++)
        {
            if (body[i] == '{') depth++;
            else if (body[i] == '}') depth--;
        }
        return depth;
    }

    // ------------------------------------------------------------------ source model

    /// <summary>Every x:Name in the window's XAML: the members that belong to the UI thread.</summary>
    private static HashSet<string> ControlNames()
    {
        var xaml = File.ReadAllText(Path.Combine(AppDir, "MainWindow.xaml"));
        return Regex.Matches(xaml, @"x:Name=""(\w+)""")
            .Select(m => m.Groups[1].Value)
            .ToHashSet(StringComparer.Ordinal);
    }

    /// <summary>Member name -> body, across every partial of MainWindow.</summary>
    private static Dictionary<string, string> Members()
    {
        var members = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var file in Directory.GetFiles(AppDir, "MainWindow*.cs"))
        {
            var src = File.ReadAllText(file);
            // Methods, and expression/block-bodied properties - RangeLevel is a property.
            foreach (Match m in Regex.Matches(
                         src, @"^\s*(?:private|internal|public|protected)[^\r\n=;]*?\b(\w+)\s*(\(|=>|\r?\n\s*\{)",
                         RegexOptions.Multiline))
            {
                var name = m.Groups[1].Value;
                if (name is "if" or "for" or "foreach" or "while" or "switch" or "return" or "using")
                    continue;
                var body = m.Groups[2].Value == "=>" ? StatementAt(src, m.Index) : BlockAt(src, m.Index);
                if (!members.ContainsKey(name))
                    members[name] = body;
            }
        }
        return members;
    }

    /// <summary>Members that touch a control, directly or through another member. Fixpoint.</summary>
    private static HashSet<string> UiBoundMembers(
        Dictionary<string, string> members, HashSet<string> controls)
    {
        var bound = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (name, body) in members)
            if (!MarshalsToDispatcher(body)
                && controls.Any(c => Regex.IsMatch(body, $@"\b{Regex.Escape(c)}\b")))
                bound.Add(name);

        bool grew;
        do
        {
            grew = false;
            foreach (var (name, body) in members)
            {
                if (bound.Contains(name) || MarshalsToDispatcher(body))
                    continue;
                if (bound.Any(b => Regex.IsMatch(body, $@"\b{Regex.Escape(b)}\b")))
                    grew = bound.Add(name);
            }
        } while (grew);

        return bound;
    }

    /// <summary>
    /// Whether a member hands its UI work to the dispatcher - the other legitimate way to touch a
    /// control from a worker thread, so it is neither UI-bound nor a route to something that is.
    /// <para>
    /// <c>Log</c> is why this matters: it writes to the log TextBox but checks
    /// <c>Dispatcher.CheckAccess()</c> first, so calling it from a worker is correct. Without this,
    /// every method that logs would be flagged, and the check would be useless.
    /// </para>
    /// </summary>
    private static bool MarshalsToDispatcher(string body) =>
        Regex.IsMatch(body, @"Dispatcher\s*\.\s*(CheckAccess|Invoke|BeginInvoke)");

    /// <summary>Which UI-bound members a method reaches, for a failure message that names the path.</summary>
    private static IEnumerable<string> Reaches(
        string method, Dictionary<string, string> members, HashSet<string> uiBound)
    {
        var body = members[method];
        return uiBound.Where(b => b != method && Regex.IsMatch(body, $@"\b{Regex.Escape(b)}\b"));
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

    /// <summary>An expression body: everything up to the terminating semicolon.</summary>
    private static string StatementAt(string src, int from)
    {
        var end = src.IndexOf(';', from);
        return end < 0 ? src[from..] : src[from..end];
    }
}
