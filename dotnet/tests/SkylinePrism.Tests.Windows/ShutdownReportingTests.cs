using System;
using System.IO;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// A fault raised while the tool is exiting must be logged and not shown.
///
/// <para>The regression: a completed run - 224 samples, 66,379 peptides, every output written, "Done."
/// in the log - ended with a modal "Skyline-PRISM error" reporting
/// <c>System.DllNotFoundException</c> at
/// <c>&lt;CrtImplementationDetails&gt;.ModuleUninitializer.SingletonDomainUnload</c>. That is WPF's
/// own <c>DirectWriteForwarder.dll</c> failing to resolve <c>vcruntime140</c> as the CLR unloads the
/// domain at process exit - no PRISM frame on the stack, nothing the user can do, and
/// <c>AppDomain.UnhandledException</c> cannot stop the process anyway. The dialog was the only part
/// of it that reached the user.</para>
///
/// <para>Source checks rather than behavioral ones: <c>App.Report</c> needs a real
/// <c>Application</c>, dispatcher and message loop to exercise, and the fault itself originates in
/// the runtime's teardown, which a test cannot stage. The same rationale as
/// <see cref="UiThreadSafetyTests"/>. What these do catch is the guard being dropped, which is the
/// way this defect would come back.</para>
/// </summary>
public class ShutdownReportingTests
{
    /// <summary>
    /// The App source, read as text. Asserted to exist rather than left to throw: the path is five
    /// levels up from the test binary, so a change to the output layout - as the move to
    /// net10.0-windows10.0.19041.0 was - otherwise fails all three tests with a bare
    /// FileNotFoundException that gives no hint the fixture is a source file.
    /// </summary>
    private static string ReadAppSource()
    {
        var path = Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..",
            "src", "SkylinePrism.App", "App.xaml.cs"));
        Assert.True(File.Exists(path), $"App.xaml.cs not found at {path} - has the output layout moved?");
        return File.ReadAllText(path);
    }

    [Fact]
    public void TheShutdownFlagIsSetFromAShutdownHook()
    {
        var text = ReadAppSource();

        // OnExit is the hook that fires on the app's only shutdown path; ShutdownStarted is a guard
        // for a path that does not exist yet. Both are asserted by name, but the COUNT deliberately
        // is not: pinning it at exactly two would fail on adding a correct third hook (ProcessExit,
        // say) and equally on dropping the redundant one, neither of which is a regression.
        Assert.Contains("protected override void OnExit", text);
        Assert.Contains("Dispatcher.ShutdownStarted", text);
        Assert.True(
            CountOf(text, "_shutdownStarted = true") >= 1,
            "At least one shutdown hook must set the flag before teardown.");
    }

    [Fact]
    public void ReportChecksTheShutdownFlagBeforeShowingADialog()
    {
        var text = ReadAppSource();

        // One dialog call site, so its position alone says whether the guard precedes it.
        Assert.Equal(1, CountOf(text, "MessageBox.Show"));

        var guard = text.IndexOf("if (_shutdownStarted)", StringComparison.Ordinal);
        var dialog = text.IndexOf("MessageBox.Show", StringComparison.Ordinal);

        Assert.True(guard >= 0, "App.Report must return early once shutdown has begun.");
        Assert.True(
            guard < dialog,
            "The shutdown guard must come BEFORE the dialog, or a teardown fault is shown anyway.");

        // The guard has to return, not merely log: falling through reaches the dialog.
        Assert.Contains("return;", text[guard..dialog]);
    }

    [Fact]
    public void TheFaultIsStillWrittenToTheLog()
    {
        var text = ReadAppSource();

        // Suppressing the dialog must not suppress the diagnosis - that log line is the only record
        // of a shutdown fault, and it is how the reported case was identified.
        //
        // Matched loosely on purpose. Pinning the exact statement text, interpolation and all, meant
        // rewrapping the line or renaming Report's parameter broke this test while the behavior it
        // checks was untouched. What matters is only that the exception reaches the log BEFORE the
        // guard can return.
        var guard = text.IndexOf("if (_shutdownStarted)", StringComparison.Ordinal);
        Assert.True(guard >= 0, "App.Report must return early once shutdown has begun.");

        var logged = text.IndexOf("UNHANDLED", StringComparison.Ordinal);
        Assert.True(logged >= 0, "Report must log the exception under an UNHANDLED marker.");
        Assert.True(
            logged < guard,
            "The exception must be logged before the shutdown guard returns.");
        Assert.Contains("WriteLog", text[..guard]);
    }

    private static int CountOf(string haystack, string needle)
    {
        var n = 0;
        for (var i = haystack.IndexOf(needle, StringComparison.Ordinal);
             i >= 0;
             i = haystack.IndexOf(needle, i + needle.Length, StringComparison.Ordinal))
            n++;
        return n;
    }
}
