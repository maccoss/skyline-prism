using System.Linq;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// <see cref="ProteinListMatcher.MatchPeptide"/> - deciding whether a PEPTIDE belongs to a protein
/// list, from the <c>;</c>-separated group columns <c>corrected_peptides.parquet</c> carries.
///
/// <para><b>Why these exist.</b> The rule was private to <c>MarkerNormalizeStage.MarkerPeptideRows</c>
/// and is now shared with signal accounting. The marker tests looked like adequate cover for the
/// extraction - and are not: deleting the gene and protein-name branches entirely left all 23 of them
/// passing, because their fixtures match on accession. So the two untested branches are tested here,
/// on the rule itself.</para>
/// </summary>
public class ProteinListMatchPeptideTests
{
    private static ProteinListMatcher MatcherOf(params string[] members)
    {
        var list = new ProteinList { Name = "Panel", Visible = false };  // visibility must not matter
        foreach (var m in members)
            list.Members.Add(m);
        return ProteinListSet.MatcherFor(list);
    }

    /// <summary>The branch the marker tests already covered.</summary>
    [Fact]
    public void MatchesOnAccession()
    {
        var m = MatcherOf("P02768");
        Assert.NotNull(m.MatchPeptide("P02768", null, null));
        Assert.Null(m.MatchPeptide("P99999", null, null));
    }

    /// <summary>The branch a mutation proved was NOT covered by the marker tests.</summary>
    [Fact]
    public void MatchesOnGene()
    {
        var m = MatcherOf("ALB");
        Assert.NotNull(m.MatchPeptide(null, "ALB", null));
        Assert.Null(m.MatchPeptide(null, "APOA1", null));
    }

    /// <summary>The other branch that mutation showed uncovered.</summary>
    [Fact]
    public void MatchesOnProteinName()
    {
        var m = MatcherOf("ALBU_HUMAN");
        Assert.NotNull(m.MatchPeptide(null, null, "ALBU_HUMAN"));
        Assert.Null(m.MatchPeptide(null, null, "APOA1_HUMAN"));
    }

    /// <summary>
    /// The point of splitting rather than taking the first group. A shared peptide names every group
    /// it belongs to, and the signal genuinely came from a member protein of the list even when that
    /// protein is not the first one listed. <c>DynamicRange.FirstGroup</c> takes only the first and
    /// would miss this - the two rules disagree, and this is the one that follows the signal.
    /// </summary>
    [Fact]
    public void MatchesViaASecondGroupOfASharedPeptide()
    {
        var byGene = MatcherOf("APOA1");
        Assert.NotNull(byGene.MatchPeptide(null, "ALB;APOA1", null));
        Assert.NotNull(byGene.MatchPeptide(null, null, "ALBU_HUMAN;APOA1_HUMAN"));

        var byAccession = MatcherOf("P02647");
        Assert.NotNull(byAccession.MatchPeptide("P02768;P02647", null, null));

        // ...and in every case the FIRST group alone would not have matched.
        Assert.Null(byGene.MatchPeptide(null, "ALB", null));
        Assert.Null(byAccession.MatchPeptide("P02768", null, null));
    }

    /// <summary>Group columns arrive with whatever spacing the writer used.</summary>
    [Fact]
    public void TrimsAndIgnoresEmptyGroups()
    {
        var m = MatcherOf("APOA1");
        Assert.NotNull(m.MatchPeptide(null, " ALB ; APOA1 ", null));
        Assert.NotNull(m.MatchPeptide(null, ";;APOA1;;", null));
        Assert.Null(m.MatchPeptide(null, ";;;", null));
    }

    /// <summary>
    /// A peptide with no group columns at all - an older peptide file, or one written before parsimony
    /// stamped them on - is simply unmatched, not an error.
    /// </summary>
    [Fact]
    public void NullAndEmptyColumnsMatchNothing()
    {
        var m = MatcherOf("APOA1");
        Assert.Null(m.MatchPeptide(null, null, null));
        Assert.Null(m.MatchPeptide("", "", ""));
    }

    /// <summary>
    /// Visibility must not be consulted. All 65 shipped panels ship <c>Visible = false</c>, so a
    /// consumer that reached for <c>BuildMatcher()</c> would silently match nothing; the accounting
    /// asks per selected list via <c>MatcherFor</c>, where the caller's selection is the only gate.
    /// </summary>
    [Fact]
    public void IgnoresListVisibility()
    {
        var hidden = ProteinList.BuiltIns.First().Clone();
        Assert.False(hidden.Visible);
        var member = ProteinList.MatchToken(hidden.Members[0]);

        Assert.NotNull(ProteinListSet.MatcherFor(hidden).MatchPeptide(member, member, member));
    }

    /// <summary>
    /// <see cref="ProteinListMatcher.SplitGroups"/> is the shared reading of these columns; pin it
    /// directly so a caller that needs the members rather than the verdict has the same rule.
    /// </summary>
    [Fact]
    public void SplitGroupsIsTheSharedReadingOfAGroupColumn()
    {
        Assert.Equal(new[] { "A", "B" }, ProteinListMatcher.SplitGroups("A;B"));
        Assert.Equal(new[] { "A", "B" }, ProteinListMatcher.SplitGroups(" A ; B "));
        Assert.Empty(ProteinListMatcher.SplitGroups(null));
        Assert.Empty(ProteinListMatcher.SplitGroups(""));
        Assert.Empty(ProteinListMatcher.SplitGroups(" ; ; "));
    }
}
