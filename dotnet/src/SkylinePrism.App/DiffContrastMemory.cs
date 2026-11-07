using System;
using System.Collections.Generic;
using System.Linq;
using SkylinePrism.Core.DifferentialAnalysis;

namespace SkylinePrism.App;

/// <summary>
/// The contrast the user set up in the Differential pane - Group by, each column's two arms, the
/// Subject column and the Adjust-for ticks - kept by NAME, apart from the pickers that show it.
/// </summary>
/// <remarks>
/// <para>The pickers are rebuilt from the loaded matrix whenever it changes: a level switch, a new
/// output directory, a re-run, a new Group by. Each rebuild used to start from defaults, carrying
/// over only what one code path remembered to capture. So a contrast survived a level switch only
/// if the load succeeded - a failed one (sample_metadata.csv open in Excel) emptied every picker,
/// and the retry came back on the default column - and going from Group by "Purpose" to another
/// column and back reseeded the arms and dropped every Adjust-for tick.</para>
/// <para>Here the choices outlive the pickers. A rebuild takes what it can by name and falls back to
/// the default for anything the new matrix does not have, so nothing is ever restored that the data
/// cannot support.</para>
/// <para>Its own file, not nested in MainWindow, so it can be tested without a window.</para>
/// </remarks>
internal sealed class DiffContrastMemory
{
    private readonly Dictionary<string, (string[] A, string[] B)> _arms = new(StringComparer.Ordinal);

    /// <summary>Ticked covariates, with the type the user switched to, or null where it was left as inferred.</summary>
    private readonly Dictionary<string, CovariateKind?> _covariates = new(StringComparer.Ordinal);

    /// <summary>The Group-by column last chosen, or null before any was.</summary>
    public string? GroupBy { get; private set; }

    /// <summary>The Subject column last chosen, or null before any was.</summary>
    public string? Subject { get; private set; }

    /// <summary>Record a Group-by choice. Null is ignored: an emptied picker is not a choice.</summary>
    public void RememberGroupBy(string? column)
    {
        if (!string.IsNullOrEmpty(column))
            GroupBy = column;
    }

    /// <summary>Record a Subject choice. Null is ignored, as for <see cref="RememberGroupBy"/>.</summary>
    public void RememberSubject(string? column)
    {
        if (!string.IsNullOrEmpty(column))
            Subject = column;
    }

    /// <summary>
    /// Record the arms ticked under <paramref name="column"/>. An arm left empty is not recorded: it
    /// is a state the user passes through while re-ticking, and restoring it would restore an error.
    /// </summary>
    public void RememberArms(string column, IReadOnlyCollection<string> a, IReadOnlyCollection<string> b)
    {
        if (a.Count == 0 || b.Count == 0)
            return;
        _arms[column] = (a.ToArray(), b.ToArray());
    }

    /// <summary>
    /// The arms last ticked under <paramref name="column"/>, kept to the values it still has. False -
    /// keep the defaults - unless BOTH arms keep at least one value, since restoring one arm beside a
    /// defaulted other could put the same value in both.
    /// </summary>
    public bool TryGetArms(string column, IReadOnlyCollection<string> available,
        out IReadOnlyList<string> a, out IReadOnlyList<string> b)
    {
        a = b = Array.Empty<string>();
        if (!_arms.TryGetValue(column, out var arms))
            return false;
        var have = new HashSet<string>(available, StringComparer.Ordinal);
        var keptA = arms.A.Where(have.Contains).ToList();
        var keptB = arms.B.Where(have.Contains).ToList();
        if (keptA.Count == 0 || keptB.Count == 0)
            return false;
        a = keptA;
        b = keptB;
        return true;
    }

    /// <summary>
    /// Record the Adjust-for picker as it stands. Only the columns it OFFERS are updated - a ticked one
    /// is kept, an unticked one forgotten - so a column it does not list right now (the Group-by
    /// column, which is never a covariate of itself) keeps its tick for when it is offered again.
    /// </summary>
    public void RememberCovariates(IEnumerable<(string Name, bool Selected, CovariateKind? Switched)> offered)
    {
        foreach (var (name, selected, switched) in offered)
        {
            if (selected)
                _covariates[name] = switched;
            else
                _covariates.Remove(name);
        }
    }

    /// <summary>
    /// Whether <paramref name="column"/> was ticked in Adjust for, and the type it was switched to
    /// (null where it was left as inferred).
    /// </summary>
    public bool TryGetCovariate(string column, out CovariateKind? switched) =>
        _covariates.TryGetValue(column, out switched);
}
