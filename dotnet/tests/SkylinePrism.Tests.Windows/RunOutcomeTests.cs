using System;
using SkylinePrism.App;
using Xunit;

namespace SkylinePrism.Tests.Windows;

public class RunOutcomeTests
{
    // The shape Stop produced during a headless export: PrismInput's per-input wrapper around the
    // runner's cancellation. It was reported as "PRISM run failed" with a stack trace.
    private static Exception WrappedExportCancel() =>
        new InvalidOperationException(
            "Input 'plate1.sky' (batch 'plate1') failed: The operation was canceled.",
            new OperationCanceledException());

    [Fact]
    public void Stop_DuringAnExport_IsAStop_NotAFailure() =>
        Assert.True(RunOutcome.IsCancellation(WrappedExportCancel(), stopRequested: true));

    [Fact]
    public void AWrappedCancellation_WithoutStop_IsStillAFailure() =>
        // Could be a timeout: that must still be reported.
        Assert.False(RunOutcome.IsCancellation(WrappedExportCancel(), stopRequested: false));

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ABareCancellation_IsAStop(bool stopRequested) =>
        Assert.True(RunOutcome.IsCancellation(new OperationCanceledException(), stopRequested));

    [Fact]
    public void AnAggregate_IsAStop_OnlyWhenEveryPartIs()
    {
        Assert.True(RunOutcome.IsCancellation(
            new AggregateException(new OperationCanceledException(), WrappedExportCancel()), stopRequested: true));
        Assert.False(RunOutcome.IsCancellation(
            new AggregateException(new OperationCanceledException(), new System.IO.IOException("disk full")),
            stopRequested: true));
    }

    [Fact]
    public void ARealFailure_DuringStop_IsStillAFailure() =>
        Assert.False(RunOutcome.IsCancellation(
            new InvalidOperationException("Input 'x' failed", new System.IO.IOException("disk full")),
            stopRequested: true));
}
