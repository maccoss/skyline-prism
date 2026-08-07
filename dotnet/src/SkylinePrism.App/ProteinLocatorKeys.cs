using System.Collections.Generic;
using SkylinePrism.Core.Qc;

namespace SkylinePrism.App;

/// <summary>
/// The names a PRISM protein group might be found under in the Skyline document tree, best first.
///
/// <para>PRISM runs its own parsimony, so its groups need not line up with the document's. A group of
/// several proteins also reports its accessions and gene names as one <c>" / "</c>-joined string
/// (<c>"A0A075B5J9 / A0A075B5K3"</c>), which matches no node at all - so each member is offered
/// separately, which is what lets a group whose leading protein is absent from the document still
/// find one of its other members.</para>
///
/// <para>Measured on an 11,320-group mouse cohort: the joined forms alone resolved 99.1% of groups,
/// and splitting them resolves 100%.</para>
/// </summary>
internal static class ProteinLocatorKeys
{
    public static IEnumerable<string> For(AbundanceEntry entry)
    {
        foreach (var field in new[] { entry.Key, entry.ProteinName, entry.Accession, entry.Gene })
        {
            if (string.IsNullOrWhiteSpace(field))
                continue;

            yield return field!;
            if (!field!.Contains('/'))
                continue;

            foreach (var part in field.Split('/'))
            {
                var trimmed = part.Trim();
                if (trimmed.Length > 0)
                    yield return trimmed;
            }
        }
    }
}
