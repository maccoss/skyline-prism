using System;
using System.IO;
using System.Linq;
using SkylinePrism.Core.Qc;
using Xunit;

namespace SkylinePrism.Tests.Qc;

/// <summary>
/// The protein panels PRISM ships. They are a user-facing default rather than an implementation detail:
/// they name a normalization in a config file, they fill the GUI's pickers, and one of them (Tubular
/// contamination) is a readout that is meant to be looked at rather than normalized to. So the things
/// worth pinning are the ones a rename or an accidental edit would silently break.
/// </summary>
public class BuiltInProteinListTests
{
    /// <summary>
    /// The name constants ARE the config surface - `marker_normalization.protein_list: Glomerulus` is
    /// resolved by string. Renaming a list without the constant, or the constant without the list, would
    /// break every config that names it, with no compile error to catch it.
    /// </summary>
    [Fact]
    public void EveryShippedListIsReachableByItsPublishedName()
    {
        var names = new[]
        {
            ProteinList.EvMarkersName,
            ProteinList.GlomerulusName,
            ProteinList.TubularContaminationName,
        };

        Assert.Equal(names.Length, ProteinList.BuiltIns.Count);
        foreach (var name in names)
        {
            var found = new ProteinListSet().Find(name);
            Assert.NotNull(found);
            Assert.NotEmpty(found!.Members);
        }
        // Case-insensitive, because the name is typed by hand into YAML.
        Assert.NotNull(new ProteinListSet().Find("glomerulus"));
    }

    /// <summary>
    /// Shipped lists arrive unticked. Opening the Dynamic Range tab on a machine that has never saved a
    /// list must not color points for a panel the user did not ask for - three panels' worth of
    /// highlighting on an unrelated cohort would be noise presented as a finding.
    /// </summary>
    [Fact]
    public void ShippedListsAreOffUntilAskedFor()
    {
        Assert.All(ProteinList.BuiltIns, l => Assert.False(l.Visible));
        Assert.Empty(new ProteinListSet().BuildMatcher().Lists);
    }

    /// <summary>
    /// ...but they are reachable by the plot once ticked, WITHOUT first being copied into the user's
    /// file. This is the whole point of unioning the built-ins into the matcher: before this, a shipped
    /// list could name a normalization but could never highlight anything.
    /// </summary>
    [Fact]
    public void AShippedListHighlightsOnceVisible()
    {
        var set = new ProteinListSet();
        var glom = set.WithBuiltIns().Single(l => l.Name == ProteinList.GlomerulusName).Clone();
        glom.Visible = true;
        set.Lists.Add(glom);

        var matcher = set.BuildMatcher();
        Assert.NotNull(matcher.Match("P53420", "COL4A3", "CO4A3_HUMAN"));
        Assert.Equal(ProteinList.GlomerulusName, matcher.Match(null, "PODXL", null)?.Name);
        // A protein in no panel stays unhighlighted.
        Assert.Null(matcher.Match("P02768", "ALB", "ALBU_HUMAN"));
    }

    /// <summary>
    /// A user list of the same name replaces the shipped one rather than sitting beside it - otherwise
    /// the picker shows "Glomerulus" twice and the config's name is ambiguous.
    /// </summary>
    [Fact]
    public void AUserListOfTheSameNameWins()
    {
        var set = new ProteinListSet
        {
            Lists = { new ProteinList { Name = "glomerulus", Members = { "NPHS1" } } },
        };

        var resolved = set.WithBuiltIns()
            .Where(l => string.Equals(l.Name, "glomerulus", StringComparison.OrdinalIgnoreCase))
            .ToList();
        Assert.Single(resolved);
        Assert.Equal(new[] { "NPHS1" }, resolved[0].Members);
    }

    /// <summary>
    /// The kidney panels are two halves of one workflow and must not overlap: the glomerular one
    /// estimates how much glomerulus a dissection captured, the tubular one flags what came along with
    /// it. A protein in both would be regressed out as "capture" and then reported as "contamination".
    /// </summary>
    [Fact]
    public void TheKidneyPanelsDoNotShareAProtein()
    {
        var glom = ProteinList.BuiltIns.Single(l => l.Name == ProteinList.GlomerulusName).Members;
        var tubular = ProteinList.BuiltIns
            .Single(l => l.Name == ProteinList.TubularContaminationName).Members;

        Assert.Empty(glom.Intersect(tubular, StringComparer.OrdinalIgnoreCase));
        Assert.True(glom.Count >= 3, "a marker score needs at least three markers to exist at all");
    }

    /// <summary>
    /// Deliberate exclusions, recorded as tests because the reasoning is invisible in the member list.
    /// COL4A1/COL4A2 are ubiquitous basement membrane, so including them would make the score track any
    /// BM rather than the GBM; NPHS1/NPHS2 are the podocyte-loss phenotype itself, so a score built on
    /// them would regress out the finding along with the capture.
    /// </summary>
    [Fact]
    public void TheGlomerularPanelExcludesTheProteinsThatWouldRegressOutTheFinding()
    {
        var glom = ProteinList.BuiltIns.Single(l => l.Name == ProteinList.GlomerulusName).Members;
        foreach (var excluded in new[] { "COL4A1", "COL4A2", "NPHS1", "NPHS2" })
            Assert.DoesNotContain(excluded, glom, StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Marker normalization resolves a shipped list on a machine with no saved lists at all - the CLI
    /// case, where there is no GUI to have curated anything.
    /// </summary>
    [Fact]
    public void MarkerNormalizationResolvesAShippedListWithNoSavedFile()
    {
        var absent = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "none.json");

        var list = ProteinListSet.Resolve(ProteinList.TubularContaminationName, null, absent);

        Assert.NotNull(list);
        Assert.Contains("UMOD", list!.Members, StringComparer.OrdinalIgnoreCase);
    }
}
