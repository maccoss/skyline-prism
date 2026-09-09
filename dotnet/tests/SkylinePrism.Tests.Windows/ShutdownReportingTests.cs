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
    private static string AppSource =>
        Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..",
            "src", "SkylinePrism.App", "App.xaml.cs"));

    [Fact]
    public void TheShutdownFlagIsSetFromBothShutdownHooks()
    {
        var source = File.ReadAllText(AppSource);

        // ShutdownStarted fires as the dispatcher tears down, OnExit after the last window closes,
        // and which arrives first depends on how the tool was closed - so both must set it.
        Assert.Contains("Dispatcher.ShutdownStarted", source);
        Assert.Contains("protected override void OnExit", source);
        Assert.Equal(2, CountOf(source, "_shutdownStarted = true"));
    }

    [Fact]
    public void ReportChecksTheShutdownFlagBeforeShowingADialog()
    {
        var source = File.ReadAllText(AppSource);

        // One dialog call site, so its position alone says whether the guard precedes it.
        Assert.Equal(1, CountOf(source, "MessageBox.Show"));

        var guard = source.IndexOf("if (_shutdownStarted)", StringComparison.Ordinal);
        var dialog = source.IndexOf("MessageBox.Show", StringComparison.Ordinal);

        Assert.True(guard >= 0, "App.Report must return early once shutdown has begun.");
        Assert.True(
            guard < dialog,
            "The shutdown guard must come BEFORE the dialog, or a teardown fault is shown anyway.");

        // The guard has to return, not merely log: falling through reaches the dialog.
        var betweenGuardAndDialog = source[guard..dialog];
        Assert.Contains("return;", betweenGuardAndDialog);
    }

    [Fact]
    public void TheFaultIsStillWrittenToTheLog()
    {
        var source = File.ReadAllText(AppSource);

        // Suppressing the dialog must not suppress the diagnosis - this log line is the only record
        // of a shutdown fault, and it is how the reported case was identified.
        var guard = source.IndexOf("if (_shutdownStarted)", StringComparison.Ordinal);
        Assert.True(guard >= 0);
        Assert.Contains("WriteLog($\"UNHANDLED ({source}): {ex}\")", source);
        Assert.True(
            source.IndexOf("WriteLog($\"UNHANDLED ({source}): {ex}\")", StringComparison.Ordinal) < guard,
            "The exception must be logged before the shutdown guard returns.");
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
