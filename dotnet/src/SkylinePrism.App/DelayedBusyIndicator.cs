using System;

namespace SkylinePrism.App;

/// <summary>
/// A "working" indicator that appears only once work has run past a delay, and hides when the LAST
/// piece of overlapping work finishes.
/// </summary>
/// <remarks>
/// <para>A count rather than a flag: in the Differential pane a level switch loads and then runs, and
/// every selector can start a contrast while another is still in flight, so "done" is when the last
/// one finishes, not the first.</para>
/// <para>The delay is the point. Most contrasts finish well inside a second, and a bar that flashes on
/// every tick of a checkbox reads as something going wrong.</para>
/// <para>Its own class for the same reason as <see cref="VizNavigation"/>: pure state with no window
/// in it, so the rule can be tested without a dispatcher. The window supplies the two effects - show
/// or hide the bar, and run a callback after the delay.</para>
/// </remarks>
internal sealed class DelayedBusyIndicator
{
    private readonly Action<bool> _setVisible;
    private readonly Func<Action, IDisposable> _afterDelay;
    private int _count;
    private IDisposable? _pending;

    /// <param name="setVisible">Shows (true) or hides (false) the indicator.</param>
    /// <param name="afterDelay">
    /// Runs the callback once, after the delay, on the same thread as everything else here; disposing
    /// the result cancels it.
    /// </param>
    public DelayedBusyIndicator(Action<bool> setVisible, Func<Action, IDisposable> afterDelay)
    {
        _setVisible = setVisible;
        _afterDelay = afterDelay;
    }

    /// <summary>Whether any work is running.</summary>
    public bool IsBusy => _count > 0;

    /// <summary>Mark work as running until the returned scope is disposed.</summary>
    public IDisposable Begin()
    {
        if (_count++ == 0)
            _pending = _afterDelay(() =>
            {
                _pending = null;
                if (_count > 0)
                    _setVisible(true);
            });
        return new Scope(this);
    }

    private void End()
    {
        if (--_count > 0)
            return;
        _count = 0;
        _pending?.Dispose();
        _pending = null;
        _setVisible(false);
    }

    private sealed class Scope : IDisposable
    {
        private DelayedBusyIndicator? _owner;

        public Scope(DelayedBusyIndicator owner) => _owner = owner;

        // Idempotent: a scope disposed twice must not end someone else's work.
        public void Dispose()
        {
            _owner?.End();
            _owner = null;
        }
    }
}
