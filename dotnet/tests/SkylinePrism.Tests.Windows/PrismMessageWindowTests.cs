using System;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

/// <summary>
/// The error window that replaced the run-failure message box: the whole error can be copied, and
/// the locked-file case offers Retry. Built for real on an STA thread, as in
/// <see cref="PanoramaPublishWindowTests"/>.
/// </summary>
public class PrismMessageWindowTests
{
    private static T OnSta<T>(Func<T> body)
    {
        T result = default!;
        Exception? failure = null;
        var thread = new Thread(() =>
        {
            try { result = body(); }
            catch (Exception ex) { failure = ex; }
        });
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        thread.Join();
        if (failure is not null)
            throw new Xunit.Sdk.XunitException("The window threw while being built: " + failure);
        return result;
    }

    [Fact]
    public void CopyText_CarriesTheMessageTheDetailsAndTheLog()
    {
        var text = OnSta(() =>
        {
            var w = new PrismMessageWindow("PRISM run failed", "The process cannot access the file",
                "System.IO.IOException: ...\n   at PrismPipeline.WriteSampleMetadata", isError: true,
                new[] { "OK" }, @"C:\logs\prism-tool.log");
            var copy = w.CopyText;
            w.Close();
            return copy;
        });

        Assert.Contains("PRISM run failed", text, StringComparison.Ordinal);
        Assert.Contains("The process cannot access the file", text, StringComparison.Ordinal);
        Assert.Contains("at PrismPipeline.WriteSampleMetadata", text, StringComparison.Ordinal);
        Assert.Contains(@"C:\logs\prism-tool.log", text, StringComparison.Ordinal);
    }

    /// <summary>
    /// Retry is the default (Enter), Cancel answers Escape, and closing the window with its X is a
    /// Cancel - never a silent Retry.
    /// </summary>
    [Fact]
    public void RetryCancel_RetryIsDefault_AndClosingCancels()
    {
        var state = OnSta(() =>
        {
            var w = new PrismMessageWindow("Close files to run PRISM", "sample_metadata.csv is open",
                @"W:\out\sample_metadata.csv", isError: false, new[] { "Retry", "Cancel" }, logPath: null);
            var buttons = Buttons(w);
            var s = (Result: w.Result,
                Labels: buttons.Select(b => b.Content as string).ToArray(),
                Default: buttons.Single(b => b.IsDefault).Content as string,
                Cancel: buttons.Single(b => b.IsCancel).Content as string);
            w.Close();
            return s;
        });

        Assert.Equal("Cancel", state.Result);
        Assert.Contains("Copy to clipboard", state.Labels);
        Assert.DoesNotContain("Open log", state.Labels); // no log path given
        Assert.Equal("Retry", state.Default);
        Assert.Equal("Cancel", state.Cancel);
    }

    private static Button[] Buttons(DependencyObject root) =>
        LogicalTreeHelper.GetChildren(root).OfType<DependencyObject>()
            .SelectMany(c => (c is Button b ? new[] { b } : Array.Empty<Button>()).Concat(Buttons(c)))
            .ToArray();
}
