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
        var empty = new ProteinListSet();

        Assert.All(ProteinList.BuiltIns, panel =>
        {
            var found = empty.Find(panel.Name);
            Assert.NotNull(found);
            Assert.NotEmpty(found!.Members);
        });

        // The constants are the config surface - a rename that misses one breaks every config naming it.
        foreach (var name in new[]
                 {
                     ProteinList.EvMarkersName, ProteinList.EvExtendedName,
                     ProteinList.GlomerulusName, ProteinList.TubularContaminationName,
                 })
        {
            Assert.NotNull(empty.Find(name));
        }

        // Case-insensitive, because the name is typed by hand into YAML.
        Assert.NotNull(empty.Find("glomerulus"));
    }

    /// <summary>Two panels sharing a name would make a config ambiguous and the picker unreadable.</summary>
    [Fact]
    public void ShippedPanelNamesAreUnique()
    {
        var names = ProteinList.BuiltIns.Select(l => l.Name).ToList();

        Assert.Equal(names.Count, names.Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.All(names, n => Assert.False(string.IsNullOrWhiteSpace(n)));
    }

    /// <summary>
    /// A name that identified a shipped panel in a released version still resolves. "EV markers" shipped
    /// plain in v26.19.0/v26.20.0; after the core/extended split a config naming it would otherwise abort
    /// the run with "not found" rather than doing anything the user could act on.
    /// </summary>
    [Fact]
    public void ARetiredPanelNameStillResolves()
    {
        var found = new ProteinListSet().Find("EV markers");

        Assert.NotNull(found);
        Assert.Equal(ProteinList.EvMarkersName, found!.Name);
        Assert.Equal(18, found.Members.Count);
    }

    /// <summary>...but a user list may take a retired name back - the exact match is tried first.</summary>
    [Fact]
    public void AUserListOutranksARetiredName()
    {
        var set = new ProteinListSet
        {
            Lists = { new ProteinList { Name = "EV markers", Members = { "CD9" } } },
        };

        Assert.Equal(new[] { "CD9" }, set.Find("EV markers")!.Members);
    }

    /// <summary>
    /// The contaminant panel is accessions, not gene symbols, and that is load-bearing: these are
    /// non-human proteins, so "ALB" for bovine serum albumin would match HUMAN albumin - the most
    /// abundant protein in a plasma sample - and "TRYP" would match human trypsin-1.
    /// </summary>
    [Fact]
    public void TheContaminantPanelUsesAccessionsNotGeneSymbols()
    {
        var crap = ProteinList.BuiltIns.Single(l => l.Name.Contains("cRAP", StringComparison.Ordinal));

        Assert.Contains("P00761", crap.Members);           // porcine trypsin
        Assert.Contains("P02769", crap.Members);           // bovine serum albumin
        foreach (var dangerous in new[] { "ALB", "TRYP", "PRSS1", "CASB", "LYZ" })
            Assert.DoesNotContain(dangerous, crap.Members, StringComparer.OrdinalIgnoreCase);
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
    /// A user list of the same name WINS that name - curating over a shipped default is the point, and a
    /// config naming it must keep meaning what it meant.
    /// </summary>
    [Fact]
    public void AUserListOfTheSameNameWinsThatName()
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
    /// ...but the shipped panel is still REACHABLE, under a suffixed name. Dropping it was a real trap:
    /// a saved "EV Markers" curated as 34 proteins for highlighting on the dynamic-range plot silently
    /// replaced the 18-protein panel the method was validated on - same name, different purpose,
    /// different score, no way to select the shipped one and no sign it existed.
    /// </summary>
    [Fact]
    public void AShadowedShippedListIsStillSelectable()
    {
        var set = new ProteinListSet
        {
            Lists = { new ProteinList { Name = "glomerulus", Members = { "NPHS1", "NPHS2" } } },
        };

        var shipped = set.Find(ProteinList.GlomerulusName + ProteinListSet.ShippedSuffix);

        Assert.NotNull(shipped);
        Assert.Equal(18, shipped!.Members.Count);
        Assert.Contains("COL4A3", shipped.Members);
        // The user's list still owns the bare name.
        Assert.Equal(2, set.Find("Glomerulus")!.Members.Count);
    }

    /// <summary>
    /// The suffixed copy is a clone, so editing the user's list cannot mutate the shipped panel that
    /// every other run on the machine resolves.
    /// </summary>
    [Fact]
    public void TheShippedCopyIsIndependentOfTheBuiltInDefinition()
    {
        var set = new ProteinListSet
        {
            Lists = { new ProteinList { Name = ProteinList.EvMarkersName, Members = { "CD9" } } },
        };

        var shipped = set.Find(ProteinList.EvMarkersName + ProteinListSet.ShippedSuffix)!;
        shipped.Members.Add("SOMETHING_ELSE");

        Assert.Equal(
            18, ProteinList.BuiltIns.Single(l => l.Name == ProteinList.EvMarkersName).Members.Count);
    }

    /// <summary>
    /// A user who has taken the suffixed name too keeps it; the shipped one is dropped rather than
    /// appearing twice under names that both belong to the user.
    /// </summary>
    [Fact]
    public void AUserListMayTakeTheSuffixedNameToo()
    {
        var set = new ProteinListSet
        {
            Lists =
            {
                new ProteinList { Name = ProteinList.EvMarkersName, Members = { "CD9" } },
                new ProteinList
                {
                    Name = ProteinList.EvMarkersName + ProteinListSet.ShippedSuffix,
                    Members = { "CD63" },
                },
            },
        };

        var names = set.WithBuiltIns().Select(l => l.Name).ToList();
        Assert.Equal(names.Count, names.Distinct(StringComparer.OrdinalIgnoreCase).Count());
        Assert.Equal(new[] { "CD63" },
            set.Find(ProteinList.EvMarkersName + ProteinListSet.ShippedSuffix)!.Members);
    }

    /// <summary>
    /// An indistinguishable protein group names all its members slash-joined, and a panel naming any one
    /// of them has to match the group. Found on real data: a 158-member histone panel matched four
    /// proteins in a cohort whose histones were nearly all in slash-joined groups.
    /// </summary>
    [Fact]
    public void ASlashJoinedProteinGroupMatchesOnAnyMember()
    {
        var set = new ProteinListSet
        {
            Lists = { new ProteinList { Name = "histones", Visible = true, Members = { "H2AC18" } } },
        };
        var matcher = set.BuildMatcher();

        Assert.NotNull(matcher.Match(null, "H2AC11 / H2AC18 / H2AJ / H2AC14", null));
        Assert.NotNull(matcher.Match(null, null, "H2A1_HUMAN / H2AC18"));
        // A group that does not contain the member still does not match.
        Assert.Null(matcher.Match(null, "H2BC14 / H2BC12 / H2BC13", null));
    }

    /// <summary>
    /// ENO1 must not be in the housekeeping panel: yeast enolase is a routine spike-in standard and
    /// shares the symbol, so the member would fire on the spike-in rather than on a housekeeper. The
    /// spike-in is carried by accession in the contaminants panel instead.
    /// </summary>
    [Fact]
    public void HousekeepingExcludesEno1_AndTheSpikeInIsTrackedByAccession()
    {
        var house = ProteinList.BuiltIns.Single(l => l.Name == "Housekeeping proteins");
        var crap = ProteinList.BuiltIns.Single(l => l.Name.Contains("cRAP", StringComparison.Ordinal));

        Assert.DoesNotContain("ENO1", house.Members, StringComparer.OrdinalIgnoreCase);
        Assert.Contains("P00924", crap.Members);
    }

    /// <summary>
    /// The histone panel is comprehensive by design - the proteomic ruler sums ALL histone signal
    /// (doi:10.1074/mcp.M113.037309) - and carries both nomenclatures, because PRISM matches tokens
    /// exactly and most documents still use the legacy HIST1H* names.
    /// </summary>
    [Fact]
    public void TheHistonePanelCarriesBothNomenclatures()
    {
        var histones = ProteinList.BuiltIns.Single(l => l.Name.StartsWith("Histones", StringComparison.Ordinal));

        Assert.Contains("H4C1", histones.Members);        // current HGNC
        Assert.Contains("HIST1H4A", histones.Members);    // legacy
        Assert.Contains("H4_HUMAN", histones.Members);    // UniProt entry name
        Assert.True(histones.Members.Count > 100, "a curated subset would not be the ruler it is named after");
    }

    /// <summary>
    /// Panels carry mouse symbols where the ortholog is named differently. Matching is case-insensitive,
    /// so a conserved symbol needs no help - but hemoglobin is not conserved, and Hemolysis is a READOUT,
    /// so a mouse sample that had lysed would have looked clean.
    /// </summary>
    [Fact]
    public void PanelsCarryMouseOrthologsWhereTheSymbolDiffers()
    {
        var hemolysis = ProteinList.BuiltIns.Single(l => l.Name == "Hemolysis");

        Assert.Contains("HBB", hemolysis.Members);
        Assert.Contains("Hbb-bs", hemolysis.Members);
        Assert.Contains("Hba-a1", hemolysis.Members);

        // Case-insensitivity does the rest: a conserved symbol matches either species as written.
        var set = new ProteinListSet { Lists = { new ProteinList { Name = "x", Visible = true, Members = { "ALB" } } } };
        Assert.NotNull(set.BuildMatcher().Match(null, "Alb", null));
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
