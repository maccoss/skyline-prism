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
        var tokens = crap.Members.Select(ProteinList.MatchToken).ToList();

        Assert.Contains("P00761", tokens);                 // porcine trypsin
        Assert.Contains("P02769", tokens);                 // bovine serum albumin
        foreach (var dangerous in new[] { "ALB", "TRYP", "PRSS1", "CASB", "LYZ" })
            Assert.DoesNotContain(dangerous, tokens, StringComparer.OrdinalIgnoreCase);

        // Every member is labeled, and the label is what a reader sees. The names are the reason the
        // panel is readable at all, so a member that lost its label is a regression, not a detail.
        Assert.All(crap.Members, m => Assert.NotEqual(ProteinList.MatchToken(m), ProteinList.DisplayName(m)));
        Assert.Equal("Trypsin (porcine)", ProteinList.DisplayName(crap.Members.First(m => m.StartsWith("P00761", StringComparison.Ordinal))));
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
        Assert.Contains("P00924", crap.Members.Select(ProteinList.MatchToken));
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

        var list = ProteinListSet.Resolve(ProteinList.EvMarkersName, null, absent);

        Assert.NotNull(list);
        Assert.Contains("CD9", list!.Members, StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// A display-only panel cannot define a normalization, and says why. Readouts and pathways share one
    /// reason: their abundance IS the signal, so dividing by it removes the thing being looked for - a
    /// readout hides the problem, a pathway hides the finding. This used to be documentation, which meant
    /// the failure was silent: the run succeeded and the numbers looked plausible.
    /// </summary>
    [Theory]
    [InlineData("Tubular contamination")]
    [InlineData("Hemolysis")]
    [InlineData("Common contaminants (cRAP)")]
    [InlineData("Glycolysis")]
    [InlineData("Oxidative phosphorylation")]
    public void ADisplayOnlyPanelCannotDefineANormalization(string name)
    {
        var absent = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "none.json");

        var ex = Assert.Throws<InvalidOperationException>(
            () => ProteinListSet.Resolve(name, null, absent));

        Assert.Contains("display-only", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(name, ex.Message);
    }

    /// <summary>
    /// ...but the same panel is perfectly legitimate to break MS2 signal down BY. "How much of the
    /// labelled signal is contamination" is the question, not a mistake - so ResolveForDisplay accepts
    /// exactly the panels Resolve refuses. Same names as the test above, asserting the opposite verdict.
    /// </summary>
    [Theory]
    [InlineData("Tubular contamination")]
    [InlineData("Hemolysis")]
    [InlineData("Common contaminants (cRAP)")]
    [InlineData("Glycolysis")]
    [InlineData("Oxidative phosphorylation")]
    public void ADisplayOnlyPanelIsFineToDisplay(string name)
    {
        var absent = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "none.json");

        var lists = ProteinListSet.ResolveForDisplay(new[] { name }, null, absent);

        Assert.Single(lists);
        Assert.Equal(name, lists[0].Name);
        Assert.True(lists[0].DisplayOnly, $"{name} is expected to be a display-only panel");
    }

    /// <summary>
    /// Selection order is the caller's and is meaningful - matching stops at the first list claiming a
    /// peptide - so it must survive resolution rather than being sorted or de-duplicated.
    /// </summary>
    [Fact]
    public void ResolveForDisplayKeepsTheCallersOrder()
    {
        var absent = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "none.json");
        var names = new[] { "Hemolysis", "Common contaminants (cRAP)" };

        var lists = ProteinListSet.ResolveForDisplay(names, null, absent);

        Assert.Equal(names, lists.Select(l => l.Name));
        Assert.Equal(names.Reverse(),
            ProteinListSet.ResolveForDisplay(names.Reverse(), null, absent).Select(l => l.Name));
    }

    /// <summary>An unknown name is the only failure, and it says what is available.</summary>
    [Fact]
    public void ResolveForDisplayRejectsOnlyAnUnknownName()
    {
        var absent = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "none.json");

        Assert.Empty(ProteinListSet.ResolveForDisplay(null, null, absent));
        Assert.Empty(ProteinListSet.ResolveForDisplay(Array.Empty<string>(), null, absent));

        var ex = Assert.Throws<InvalidOperationException>(
            () => ProteinListSet.ResolveForDisplay(new[] { "No Such Panel" }, null, absent));
        Assert.Contains("No Such Panel", ex.Message);
        Assert.Contains("Available:", ex.Message);
    }

    /// <summary>
    /// Every pathway is display-only, and the panels meant to normalize are not - if a normalizer were
    /// ever flagged by accident the feature would fail closed with a confusing message.
    /// </summary>
    [Fact]
    public void NormalizersAreNotFlaggedDisplayOnly()
    {
        foreach (var name in new[]
                 {
                     ProteinList.EvMarkersName, ProteinList.GlomerulusName,
                     "Histones (proteomic ruler)", "Ribosomal proteins", "Mitochondrial content",
                 })
        {
            var panel = ProteinList.BuiltIns.Single(l => l.Name == name);
            Assert.False(panel.DisplayOnly, $"{name} is meant to be usable as a normalizer");
        }
    }

    /// <summary>
    /// A member is "&lt;token&gt; = &lt;label&gt;", and only the token is matched. This is what lets the
    /// contaminants panel read as protein names while still matching on accessions - the two halves are
    /// independent, and the label may be anything, including text that would be a terrible match token.
    /// </summary>
    [Theory]
    [InlineData("P00761 = Trypsin (porcine)", "P00761", "Trypsin (porcine)")]
    [InlineData("P00761=Trypsin", "P00761", "Trypsin")]           // spaces around the separator optional
    [InlineData("  ALB  ", "ALB", "ALB")]                          // unlabeled: the member is both
    [InlineData("CD9", "CD9", "CD9")]
    [InlineData("P02769 = ", "P02769", "P02769")]                  // empty label falls back to the token
    [InlineData("A = B = C", "A", "B = C")]                        // first separator wins; label keeps the rest
    public void AMemberSplitsIntoAMatchTokenAndALabel(string member, string token, string label)
    {
        Assert.Equal(token, ProteinList.MatchToken(member));
        Assert.Equal(label, ProteinList.DisplayName(member));
    }

    /// <summary>
    /// The label must not reach the matcher. A contaminant labeled "Serum albumin (bovine, BSA)" that
    /// matched on its label would fire on human albumin - exactly the collision the accession-only rule
    /// exists to prevent, reintroduced through the readability change.
    /// </summary>
    [Fact]
    public void ALabelIsDisplayedButNeverMatched()
    {
        var crap = ProteinList.BuiltIns.Single(l => l.Name.Contains("cRAP", StringComparison.Ordinal)).Clone();
        crap.Visible = true;
        var matcher = ProteinListSet.MatcherFor(crap);

        Assert.NotNull(matcher.Match("P02769", null, null));            // bovine BSA, by accession
        Assert.Null(matcher.Match("P02768", "ALB", "ALBU_HUMAN"));      // human albumin stays clean
        Assert.Null(matcher.Match(null, "Albumin", null));
        Assert.Null(matcher.Match(null, "Trypsin", null));
    }

    /// <summary>
    /// Every shipped panel sits under a heading in the Predefined tab. A panel with no category would
    /// land in a stray "Other" group - not broken, but the kind of thing nobody notices until a user
    /// asks where a panel went.
    /// </summary>
    [Fact]
    public void EveryShippedPanelHasACategory()
    {
        Assert.All(ProteinList.BuiltIns, panel =>
            Assert.False(string.IsNullOrWhiteSpace(panel.Category), $"{panel.Name} has no category"));

        // Endothelial and epithelial are deliberately SEPARATE headings: they are different tissues
        // that happen to alliterate, and one combined group reads as a filing error.
        var categories = ProteinList.BuiltIns.Select(l => l.Category).Distinct().ToList();
        Assert.Contains("Endothelial", categories);
        Assert.Contains("Epithelial", categories);
    }

    /// <summary>A category survives the clone the GUI takes when a panel is copied or ticked.</summary>
    [Fact]
    public void CloningKeepsTheCategory()
    {
        var panel = ProteinList.BuiltIns.First(l => l.Category == "Endothelial");

        Assert.Equal(panel.Category, panel.Clone().Category);
    }

    /// <summary>
    /// Insulin signaling is display-only for a reason worth pinning: the pathway is regulated by
    /// PHOSPHORYLATION, so normalizing abundance to it would divide by something that does not move
    /// with the biology being asked about.
    /// </summary>
    [Theory]
    [InlineData("Cell cycle and proliferation")]
    [InlineData("Epithelial-mesenchymal transition")]
    [InlineData("Hypoxia response")]
    [InlineData("Glucose and lipid metabolism")]
    [InlineData("Insulin signaling")]
    public void TheMetabolicAndProliferativePanelsShipDisplayOnly(string name)
    {
        var panel = ProteinList.BuiltIns.Single(l => l.Name == name);

        Assert.True(panel.DisplayOnly);
        Assert.False(panel.Visible);
        Assert.Equal("Pathways and processes", panel.Category);
        Assert.True(panel.Members.Count >= 20, $"{name} is too thin to read a programme from");
    }

    /// <summary>
    /// The contaminants panel must not claim a HUMAN protein. This is the test the panel needed and did
    /// not have: it carried the UniProt entry names ENO1_YEAST and TRYP_PIG, and the matcher strips
    /// species suffixes so panels work across human and mouse - so they reduced to the tokens ENO1 and
    /// TRYP, which are exactly the two collisions the panel's own comment says accessions exist to
    /// prevent. Every human run colored abundant alpha-enolase as a contaminant and, because this panel
    /// is declared before Glycolysis, took ENO1 from the panel it belongs to.
    /// <para>
    /// The existing accession/gene-symbol tests could not see it: they check the RAW member strings,
    /// and "ENO1_YEAST" is neither the symbol "ENO1" nor a collision until the matcher tokenizes it.
    /// This one asks the matcher.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData("P06733", "ENO1", "ENOA_HUMAN")]      // alpha-enolase: abundant in every human sample
    [InlineData("P07477", "PRSS1", "TRY1_HUMAN")]     // human trypsin-1
    [InlineData("P02768", "ALB", "ALBU_HUMAN")]       // human serum albumin
    [InlineData("P04406", "GAPDH", "G3P_HUMAN")]
    public void TheContaminantPanelNeverClaimsAHumanProtein(string accession, string gene, string name)
    {
        var crap = ProteinList.BuiltIns.Single(l => l.Name.Contains("cRAP", StringComparison.Ordinal)).Clone();
        crap.Visible = true;

        Assert.Null(ProteinListSet.MatcherFor(crap).Match(accession, gene, name));
    }

    /// <summary>
    /// ...and no shipped panel may contain a member that tokenizes to a bare human gene symbol it did
    /// not mean. Stated as a rule rather than a list, so the next entry name added anywhere trips it.
    /// </summary>
    [Fact]
    public void TheContaminantPanelIsAccessionsOnly()
    {
        var crap = ProteinList.BuiltIns.Single(l => l.Name.Contains("cRAP", StringComparison.Ordinal));

        Assert.All(crap.Members, m =>
        {
            var token = ProteinList.MatchToken(m);
            Assert.DoesNotContain('_', token);   // an entry name; its species suffix is stripped away
            // The official UniProt accession pattern. A gene symbol ("ENO1", "ALB") and an entry name
            // both fail it, which is the whole point.
            Assert.Matches("^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$", token);
        });
    }

    /// <summary>
    /// Every shipped panel belongs to one of the declared headings. The Predefined tab groups on
    /// Category and substitutes "Other" for a blank one, so a panel added without a category does not
    /// fail anything - it just appears under an eighth heading that exists only because of the fallback,
    /// inside a collapsed group where nobody looks. Stated as a closed set so a typo trips it too.
    /// </summary>
    [Fact]
    public void EveryShippedPanelHasADeclaredCategory()
    {
        var declared = new[]
        {
            "Normalizers", "Plasma and blood", "Endothelial", "Epithelial",
            "Readouts and contamination", "Pathways and processes", "Brain and neurodegeneration",
        };

        Assert.All(ProteinList.BuiltIns, panel =>
        {
            Assert.False(string.IsNullOrWhiteSpace(panel.Category), $"'{panel.Name}' has no category");
            Assert.Contains(panel.Category, declared);
        });
    }

    /// <summary>
    /// An '=' only separates a display label when it comes before any field separator. A pasted
    /// spreadsheet row can carry one that means nothing here, and swallowing the whole row into a single
    /// member gave it a match token of the entire line - matching nothing, and reported as permanently
    /// "not quantified" with no visible cause.
    /// </summary>
    [Theory]
    [InlineData("P02769 = Serum albumin (bovine, BSA)", 1, "P02769")]  // label owns the commas
    [InlineData("P02768,ALB,Albumin", 3, "P02768")]                    // plain spreadsheet row
    [InlineData("P02768,ALB,ratio H/L=1.2", 3, "P02768")]              // '=' after a comma is not a label
    [InlineData("P00761;P02769", 2, "P00761")]
    public void SplitMemberLine_TreatsEqualsAsALabelOnlyBeforeAnyFieldSeparator(
        string line, int expectedCount, string expectedFirstToken)
    {
        var members = ProteinList.SplitMemberLine(line).ToList();

        Assert.Equal(expectedCount, members.Count);
        Assert.Equal(expectedFirstToken, ProteinList.MatchToken(members[0]));
    }

    /// <summary>
    /// Whoever wins ENO1 across the whole shipped set, it is not the contaminants panel. List order is
    /// priority, so a contaminant member that over-matches does not merely add a wrong color - it takes
    /// the protein away from the panel that should have had it.
    /// </summary>
    [Fact]
    public void GlycolysisKeepsEno1AcrossTheWholeShippedSet()
    {
        var set = new ProteinListSet();
        foreach (var panel in ProteinList.BuiltIns)
        {
            var copy = panel.Clone();
            copy.Visible = true;
            set.Lists.Add(copy);
        }

        Assert.Equal("Glycolysis", set.BuildMatcher().Match("P06733", "ENO1", "ENOA_HUMAN")?.Name);
    }

    /// <summary>
    /// A comma inside a display label does not split the member. The editor and the file importer both
    /// treat commas as member separators - a pasted spreadsheet row is the common case - and three
    /// shipped contaminants carry a comma inside their label, so the naive split turned
    /// "P02769 = Serum albumin (bovine, BSA)" into a member that still matched and a "BSA)" that never
    /// could.
    /// </summary>
    [Theory]
    [InlineData("P02769 = Serum albumin (bovine, BSA)", 1)]
    [InlineData("P00924 = Enolase 1 (yeast, spike-in)", 1)]
    [InlineData("CD9, CD63, CD81", 3)]                  // no label: still a spreadsheet row
    [InlineData("CD9; CD63", 2)]
    [InlineData("", 0)]
    public void ALabelKeepsItsCommas(string line, int expected)
    {
        var members = ProteinList.SplitMemberLine(line).ToList();

        Assert.Equal(expected, members.Count);
        if (expected == 1)
            Assert.Equal(line.Trim(), members[0]);
    }
}
