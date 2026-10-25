using System;
using System.Threading.Tasks;

namespace SkylinePrism.App;

/// <summary>
/// Follows Skyline's selection for whichever pane is on screen: asks Skyline what is selected, has
/// the pane resolve it to one of its own points, and hands the answer back to the pane.
/// </summary>
/// <remarks>
/// <para>Skyline's RPC is request/response and cannot push events, so following it means polling -
/// one lightweight call, and only while a pane that follows it is visible. The window owns the timer
/// and calls <see cref="PollOnceAsync"/> on each tick; everything else is here, once, for every pane
/// that follows: the Dynamic Range plot and the Volcano. It used to be written out in the Dynamic
/// Range pane and borrowed by the Volcano through that pane's fields.</para>
/// <para>What this guarantees, each of it learned the hard way in the pane code it replaces:</para>
/// <list type="bullet">
/// <item>The RPC and the pane's resolve run OFF the UI thread - the call waits on a busy
/// application, and the first resolve builds an index over every feature (150,000 at peptide level).
/// The pane's <see cref="Poll.Prepare"/> half runs ON it, so that is where a pane reads its
/// controls.</item>
/// <item>One poll at a time: a Skyline that has gone slow makes this quieter rather than piling up
/// requests.</item>
/// <item>A failed RPC (Skyline busy, closed, mid-document-change) is just retried on the next tick.
/// Anything else that throws is counted, and after <see cref="MaxFailures"/> in a row following
/// stops and says so - on a 750 ms timer, an unhandled exception was an error dialog every tick until
/// the tool was killed.</item>
/// <item>An answer that arrives after the pane has changed is dropped.</item>
/// </list>
/// <para>Its own class, with the RPC passed in, so these rules are tested without Skyline or a
/// dispatcher.</para>
/// </remarks>
internal sealed class SkylineSelectionFollower
{
    /// <summary>Consecutive failures before following gives up.</summary>
    public const int MaxFailures = 3;

    /// <summary>
    /// One pane's side of a poll, built on the UI thread by <see cref="Prepare"/> - which captures
    /// whatever the resolve needs, since it must not read a control from the worker.
    /// </summary>
    /// <param name="ElementType">What to ask Skyline for: "MoleculeGroup" (proteins) or "Molecule"
    /// (peptides). Skyline resolves the ancestor at that level, so a precursor or transition selection,
    /// or a row in a document grid, still lands on the right element.</param>
    /// <param name="Resolve">Locator to the pane's point, or null. Runs on a worker thread.</param>
    /// <param name="Apply">The locator and what it resolved to, on the UI thread.</param>
    public sealed record Poll(string ElementType, Func<string, object?> Resolve, Action<string, object?> Apply);

    private readonly Func<string, string?> _selectedLocator;
    private readonly Action<string> _trace;
    private readonly Action<string> _log;
    private bool _busy;
    private int _failures;

    /// <param name="selectedLocator">
    /// Skyline's selection at an element type, or null when nothing is selected there. Runs on a
    /// worker thread; an exception means the call failed and is retried.
    /// </param>
    /// <param name="trace">Where each failure goes in full - the tool's log file.</param>
    /// <param name="log">Where giving up is said - the Log tab the user reads.</param>
    public SkylineSelectionFollower(Func<string, string?> selectedLocator, Action<string> trace, Action<string> log)
    {
        _selectedLocator = selectedLocator;
        _trace = trace;
        _log = log;
    }

    /// <summary>
    /// The visible pane's poll builder, or null when no visible pane follows Skyline. It returns null
    /// itself when the pane has nothing to follow yet (no run loaded, nothing plotted).
    /// </summary>
    public Func<Poll?>? Pane { get; private set; }

    /// <summary>True once following has given up; <see cref="Follow"/> starts it again.</summary>
    public bool Stopped { get; private set; }

    /// <summary>
    /// Follow for <paramref name="pane"/> from now on, or for nothing when null. A new pane - or the
    /// same one coming back into view - is a fresh start, which is how a user retries after it gave up.
    /// </summary>
    public void Follow(Func<Poll?>? pane)
    {
        Pane = pane;
        _failures = 0;
        Stopped = false;
    }

    /// <summary>One poll. Call it from the UI thread; it never throws.</summary>
    public async Task PollOnceAsync()
    {
        var pane = Pane;
        if (_busy || Stopped || pane is null)
            return;
        _busy = true;
        try
        {
            var poll = pane();
            if (poll is null)
                return;

            var found = await Task.Run<(string? Locator, object? Item)>(() =>
            {
                string? locator;
                try
                {
                    locator = _selectedLocator(poll.ElementType);
                }
                catch (Exception)
                {
                    return (null, null); // busy, closed, or mid-document-change; retry next tick
                }
                return locator is null ? (null, null) : (locator, poll.Resolve(locator));
            });

            // The pane may have gone, or another taken its place, while the call was in flight.
            if (found.Locator is null || !ReferenceEquals(pane, Pane))
                return;

            poll.Apply(found.Locator, found.Item);
            _failures = 0;
        }
        catch (Exception ex)
        {
            _trace("Following Skyline's selection failed: " + ex);
            if (++_failures >= MaxFailures)
            {
                Stopped = true;
                _log($"Stopped following Skyline's selection after {MaxFailures} consecutive failures "
                    + $"({ex.Message}). Clicking a point still works; switch away from this pane and "
                    + "back to retry.");
            }
        }
        finally
        {
            _busy = false;
        }
    }
}
