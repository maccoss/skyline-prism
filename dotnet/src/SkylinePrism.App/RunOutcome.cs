using System;
using System.Linq;

namespace SkylinePrism.App;

/// <summary>
/// How a run that threw is reported: stopped by the user, or failed.
/// </summary>
/// <remarks>
/// Its own class, like <see cref="VizNavigation"/>, so the rule can be tested without a window.
/// </remarks>
internal static class RunOutcome
{
    /// <summary>
    /// Whether an exception that ended a run is the run being stopped rather than failing.
    /// </summary>
    /// <remarks>
    /// Cancellation arrives in several shapes depending on which stage was interrupted: bare, in an
    /// <see cref="AggregateException"/>, or wrapped by the per-input "Input '...' failed" exception
    /// that names the document being prepared. The wrapped shape went unrecognized, so pressing Stop
    /// during an export showed the "PRISM run failed" dialog with a stack trace.
    /// <para>A wrapped cancellation counts only when Stop was pressed. An
    /// <see cref="OperationCanceledException"/> inside another exception can also be a timeout, and
    /// a timeout is a failure to report.</para>
    /// </remarks>
    /// <param name="ex">The exception that ended the run.</param>
    /// <param name="stopRequested">Whether the user pressed Stop (or closed the window) during the run.</param>
    public static bool IsCancellation(Exception ex, bool stopRequested) => ex switch
    {
        OperationCanceledException => true,
        AggregateException agg => agg.InnerExceptions.Count > 0
                                  && agg.InnerExceptions.All(e => IsCancellation(e, stopRequested)),
        { InnerException: { } inner } when stopRequested => IsCancellation(inner, stopRequested),
        _ => false,
    };
}
