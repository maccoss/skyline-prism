using System.Collections.Generic;

namespace SkylinePrism.Core.Qc;

/// <summary>
/// The protein panels PRISM ships.
///
/// <para>They exist so a normalization or a plot highlight can be defined without curating anything
/// first, and so a panel means the same thing on every machine - which is what makes one citable in a
/// methods section. A user list of the same name still wins that name; the shipped one stays reachable
/// under its <see cref="ProteinListSet.ShippedSuffix"/> form.</para>
///
/// <para>Every panel ships <b>invisible</b>: opening a plot must not color points for a panel nobody
/// asked for. Three kinds are mixed here deliberately, and the difference matters more than the
/// membership does:</para>
///
/// <list type="bullet">
/// <item><b>Normalizers</b> - proportional to how much of the material of interest was captured, and NOT
/// to the phenotype. EV markers (core) and Glomerulus are built for this.</item>
/// <item><b>Identity panels</b> - what a cell or tissue type is. Useful for highlighting; usable as a
/// normalizer only if the biology under study does not change the cell type's abundance.</item>
/// <item><b>Readouts</b> - Hemolysis, Fibrinogen, Keratin contamination, Tubular contamination, Common
/// contaminants. Their abundance IS the problem being looked for, so normalizing to one removes the
/// evidence. Tick them on a plot; never name them in <c>marker_normalization</c>.</item>
/// </list>
///
/// <para>Gene symbols throughout, except the contaminant panel - see its own note for why that one has
/// to be accessions.</para>
/// </summary>
internal static class BuiltInProteinPanels
{
    public static IReadOnlyList<ProteinList> All { get; } = new[]
    {
        new ProteinList
        {
            Name = "EV markers (core)",
            Category = "Normalizers",
            ColorHex = "#2ca02c",
            Visible = false,
            // The panel the marker normalization was validated on (PC1 = 70.4% of marker variance on the FLARE
            // cohort). ACTB and GAPDH are deliberately absent: including the two most abundant co-isolates would
            // make the score partly a total-protein score, which is what a normalizer must be independent of.
            // Use this one to normalize; use the extended panel to look.
            Members =
            {
                "CD9", "CD63", "CD81", "TSG101", "PDCD6IP", "SDCBP", "VPS4B", "FLOT1", "FLOT2",
                "ANXA2", "ANXA5", "ANXA6", "RAB7A", "RAB5C", "RAB14", "HSPA8", "ITGB1", "EHD1",
            },
        },

        new ProteinList
        {
            Name = "EV markers (extended)",
            Category = "Plasma and blood",
            ColorHex = "#1f77b4",
            Visible = false,
            // A broad EV-association panel for HIGHLIGHTING on the dynamic-range plot, not for normalizing - it
            // carries ACTB, GAPDH and several cell-type-specific members. Kept as its own name so it cannot be
            // confused with the core panel; the two differ in both directions, not by inclusion.
            Members =
            {
                "CD9", "CD63", "PDCD6IP", "FLOT1", "FLOT2", "TSG101", "SDCBP", "NCAM1", "CD40",
                "SEPTIN2", "ATP5F1A", "HSP90B1", "ANXA5", "VPS4B", "ATP1A1", "SLC3A2", "ITGB1",
                "BSG", "LAMP1", "LAMP2", "TFRC", "STOM", "SDC4", "GPC1", "RAB7A", "RAB5C", "RAB1A",
                "RAP1B", "GNAI2", "EHD4", "ANXA11", "HSPA8", "ACTB", "GAPDH",
            },
        },

        new ProteinList
        {
            Name = "Classic plasma proteins",
            Category = "Plasma and blood",
            ColorHex = "#d62728",
            Visible = false,
            // HBA1 and HBB are deliberately NOT here. Hemoglobin is intracellular; finding it in plasma means the
            // sample lysed, which is a QC finding rather than a plasma measurement - see the Hemolysis panel.
            Members =
            {
                "ALB", "TF", "HPX", "A2M", "ORM1", "ORM2", "HP", "CP", "TTR", "RBP4", "GC", "AFM",
                "ATRN", "BCHE", "CNDP1", "GPX3", "SELENOP", "A1BG", "GPLD1", "BTD", "ADAMTS13",
                "KLKB1", "KNG1", "PLG", "F2", "FGL1",
            },
        },

        new ProteinList
        {
            Name = "Free soluble acidic plasma proteins",
            Category = "Plasma and blood",
            ColorHex = "#2ca02c",
            Visible = false,
            // The broadest plasma panel here, and it contains Platelet microparticles entirely, 14/15 of the
            // lipoproteins and 26/28 of Classic plasma proteins. On the plot the earliest visible list wins a
            // shared protein's color, so turning several of these on together is worth doing deliberately.
            Members =
            {
                "CP", "FGL1", "GPX3", "RBP4", "SERPIND1", "TTR", "IGFALS", "ADIPOQ", "BCHE",
                "SERPINF1", "GC", "CNDP1", "IGFBP3", "HPR", "HGFAC", "FN1", "HP", "SERPING1",
                "ORM2", "SERPINA7", "F7", "F13B", "ADAMTS13", "ORM1", "PI16", "ECM1", "STC2",
                "BTD", "SERPINA3", "Serpina3n", "SERPINF2", "KLKB1", "GPLD1", "MST1", "A1BG", "SERPINC1",
                "ITIH4", "SERPINA6", "ALB", "AFM", "SELENOP", "SERPINA11", "CFHR2", "SERPINA5",
                "SERPINA1", "HPX", "SERPINA10", "TF", "A2M", "ATRN", "SERPINA4", "AMBP", "ITIH3",
                "ITIH2", "ITIH1", "F9", "F2", "F10", "ARG1", "APCS", "ELANE", "MPO", "HRG", "PLG",
                "ANGPTL3", "KNG1", "PCSK9", "CRP", "LBP", "SHBG", "LCAT", "PON1", "APOL1", "APOD",
                "APOF", "PON3", "APOA1", "APOM", "SAA2", "APOA2", "SAA1", "CLU", "APOE", "APOA5",
                "APOC2", "APOC1", "APOB", "APOC3", "APOA4", "CD5L", "VWF", "MFGE8", "ITGA2B",
                "SRGN", "THBS1", "PPBP", "FETUB", "APOH", "LGALS3BP", "PF4", "AHSG",
            },
        },

        new ProteinList
        {
            Name = "Immunoglobulin and complement",
            Category = "Plasma and blood",
            ColorHex = "#9467bd",
            Visible = false,
            // Complement is completed with C1QA/B/C, C2, C4A/C4B, C7, C8B, CFB, CFD, CFI and CFP, and the lambda
            // constant regions IGLC1-3 join IGKC and replace IGLL1, which is the pre-B surrogate light chain
            // rather than a lambda constant region and is rarely meaningful in plasma.
            Members =
            {
            // Mouse aliases, so the panel works on either species. Matching is case-insensitive, so a
            // conserved symbol needs no help; these are the orthologs that carry a DIFFERENT symbol and
            // would otherwise be missed in silence. They cost nothing on human data - they simply never
            // match.
                // Mouse splits IgG differently (Ighg2a/2b/2c) and has a single C4 gene, C4b.
                "Ighg2a", "Ighg2b", "Ighg2c", "Igha", "Ighm", "C4b",
                "IGHM", "IGHD", "JCHAIN", "IGKC", "IGHG1", "IGHG2", "IGHG3", "IGHG4",
                "IGHA1", "IGHA2", "C1R", "C1S", "C3", "C5", "C6", "C8A", "C8G", "C9", "CFH",
                "CFHR2", "CFHR3", "CFHR4", "CFHR5", "CPN1", "CPN2", "MASP2", "MBL2", "FCN1",
                "FCN3", "CD5L", "APCS", "VSIG4", "C1QA", "C1QB", "C1QC", "C2", "C4A", "C4B", "C7",
                "C8B", "CFB", "CFD", "CFI", "CFP", "IGLC1", "IGLC2", "IGLC3",
            },
        },

        new ProteinList
        {
            Name = "Lipoproteins (LDL/VLDL/HDL)",
            Category = "Plasma and blood",
            ColorHex = "#1f77b4",
            Visible = false,
            ShowLabels = true,
            Members =
            {
                "APOB", "APOA4", "APOC4", "APOA1", "APOA2", "APOD", "APOF", "APOM", "APOL1",
                "APOC1", "APOC3", "LCAT", "PON1", "PON3", "CLU", "APOE", "APOC2", "APOA5", "LPA",
            },
        },

        new ProteinList
        {
            Name = "Platelet microparticles",
            Category = "Plasma and blood",
            ColorHex = "#ff7f0e",
            Visible = false,
            // Split from a combined 'Platelet MP / Calciprotein' panel: two unrelated particle types with
            // different causes and different remedies, so a hit on the combined panel said nothing on its
            // own. The calciprotein members (AHSG, FETUB) are not shipped as a panel of their own - they
            // remain in the soluble plasma panel.
            Members =
            {
                "PF4", "PPBP", "THBS1", "ITGA2B", "VWF", "SRGN", "TUBB1", "SELP", "GP1BA", "F13A1",
            },
        },

        new ProteinList
        {
            Name = "Arterial endothelial markers",
            Category = "Endothelial",
            ColorHex = "#2ca02c",
            Visible = false,
            ShowLabels = true,
            Members =
            {
                "EFNB2", "DLL4", "NOTCH1", "NOTCH4", "HEY1", "HEY2", "GJA4", "GJA5", "SOX17",
            },
        },

        new ProteinList
        {
            Name = "Venous endothelial markers",
            Category = "Endothelial",
            ColorHex = "#ff7f0e",
            Visible = false,
            ShowLabels = true,
            Members =
            {
                "NR2F2", "EPHB4", "NT5E", "ACKR1",
            },
        },

        new ProteinList
        {
            Name = "Capillary endothelial markers",
            Category = "Endothelial",
            ColorHex = "#9467bd",
            Visible = false,
            ShowLabels = true,
            Members =
            {
                "CA4", "RGCC", "CD36", "FABP4", "AQP1",
            },
        },

        new ProteinList
        {
            Name = "Pan-endothelial markers",
            Category = "Endothelial",
            ColorHex = "#8c564b",
            Visible = false,
            // KDR, not VEGFR2: the latter is a protein nickname and matches nothing. PRISM matches accession, gene
            // name and protein-name tokens, and the UniProt entry gives VGFR2_HUMAN -> VGFR2.
            // CAV1 is ubiquitous rather than endothelial-specific and will fire on fibroblast content too.
            Members =
            {
                "PECAM1", "CDH5", "VWF", "KDR", "TEK", "TIE1", "ENG", "THBD", "CLDN5", "ESAM",
                "ICAM2", "EGFL7", "MCAM", "CD34", "PLVAP", "EMCN", "PTPRB", "ROBO4", "NOS3",
                "CAV1", "CALCRL", "RAMP2", "S1PR1", "AQP1",
            },
        },

        new ProteinList
        {
            Name = "Brain/BBB endothelial markers",
            Category = "Endothelial",
            ColorHex = "#e377c2",
            Visible = false,
            ShowLabels = true,
            Members =
            {
                "SLC2A1", "MFSD2A", "SLC7A5", "ABCB1", "ABCG2", "CLDN5", "OCLN", "TJP1", "LSR",
            },
        },

        new ProteinList
        {
            Name = "Liver sinusoidal endothelial markers",
            Category = "Endothelial",
            ColorHex = "#bcbd22",
            Visible = false,
            ShowLabels = true,
            Members =
            {
                "STAB1", "STAB2", "CLEC4G", "MRC1", "FCGR2B", "OIT3", "LYVE1",
            },
        },

        new ProteinList
        {
            Name = "Kidney glomerular endothelial markers",
            Category = "Endothelial",
            ColorHex = "#17becf",
            Visible = false,
            ShowLabels = true,
            // EMCN, ESM1 and KDR added: the pair EHD3/PLVAP was below the three-marker minimum, so the panel could
            // highlight but could never define a marker score - the one thing single-glomerulus work needs.
            Members =
            {
                "EHD3", "PLVAP", "EMCN", "ESM1", "KDR",
            },
        },

        new ProteinList
        {
            Name = "Pan-epithelial markers",
            Category = "Epithelial",
            ColorHex = "#1f77b4",
            Visible = false,
            Members =
            {
                "EPCAM", "CDH1", "KRT8", "KRT18", "KRT19", "KRT7", "CLDN3", "CLDN4", "CLDN7",
                "OCLN", "TJP1", "TJP2", "TJP3", "F11R", "DSP", "PKP2", "PKP3", "JUP", "DSG2",
                // LGALS3BP is deliberately absent: broadly secreted rather than epithelial-specific, so it
                // fires on plasma content. It remains in the soluble plasma panel, where it belongs.
                "DSC2", "ITGA6", "ITGB4", "MUC1", "CD24", "SDC1",
            },
        },

        new ProteinList
        {
            Name = "Kidney tubule epithelial markers",
            Category = "Epithelial",
            ColorHex = "#d62728",
            Visible = false,
            ShowLabels = true,
            // Overlaps Tubular contamination by design. This panel asks 'is this tubule?'; that one asks 'how much
            // tubule came along with my glomerulus?'. Same proteins, opposite questions.
            Members =
            {
                "PAX2", "PAX8", "LRP2", "CUBN", "SLC34A1", "UMOD", "AQP1", "AQP2", "SLC12A1",
                "SLC12A3", "GATA3",
            },
        },

        new ProteinList
        {
            Name = "Intestine epithelial markers",
            Category = "Epithelial",
            ColorHex = "#17becf",
            Visible = false,
            ShowLabels = true,
            Members =
            {
            // Mouse aliases, so the panel works on either species. Matching is case-insensitive, so a
            // conserved symbol needs no help; these are the orthologs that carry a DIFFERENT symbol and
            // would otherwise be missed in silence. They cost nothing on human data - they simply never
            // match.
                // Mouse lysozyme is two genes; the alpha-defensins are a large, differently numbered family,
                // so DEFA5 has no single ortholog to name here.
                "Lyz1", "Lyz2",
                "CDX1", "CDX2", "VIL1", "LGR5", "OLFM4", "MUC2", "TFF3", "CHGA", "LYZ", "DEFA5",
                "FABP2", "ALPI",
            },
        },

        new ProteinList
        {
            Name = "Lung epithelial markers",
            Category = "Epithelial",
            ColorHex = "#d62728",
            Visible = false,
            ShowLabels = true,
            Members =
            {
                "NKX2-1", "SFTPC", "SFTPB", "SFTPA1", "NAPSA", "SCGB1A1", "SCGB3A2", "FOXJ1",
                "MUC5AC", "MUC5B", "KRT5", "AGER", "LAMP3",
            },
        },

        new ProteinList
        {
            Name = "Hemolysis",
            Category = "Readouts and contamination",
            DisplayOnly = true,
            ColorHex = "#a8341f",
            Visible = false,
            ShowLabels = true,
            // A READOUT, never a normalizer. Erythrocyte cytosol in plasma or serum means the sample lysed before
            // or during preparation, which shifts the whole abundance profile.
            Members =
            {
                "HBA1", "HBB", "HBD", "CA1", "PRDX2", "BLVRB", "CAT",
            // Mouse aliases, so the panel works on either species. Matching is case-insensitive, so a
            // conserved symbol needs no help; these are the orthologs that carry a DIFFERENT symbol and
            // would otherwise be missed in silence. They cost nothing on human data - they simply never
            // match.
                "Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt", "Hbb-b1", "Hbb-b2",
            },
        },

        new ProteinList
        {
            Name = "Fibrinogen",
            Category = "Readouts and contamination",
            DisplayOnly = true,
            ColorHex = "#ff7f0e",
            Visible = false,
            ShowLabels = true,
            // The plasma-versus-serum discriminator, and a readout for incomplete clotting. Absent from every
            // curated panel this set was built from, which made it the largest single gap.
            Members =
            {
                "FGA", "FGB", "FGG",
            },
        },

        new ProteinList
        {
            Name = "Keratin contamination",
            Category = "Readouts and contamination",
            DisplayOnly = true,
            ColorHex = "#7f7f7f",
            Visible = false,
            ShowLabels = true,
            // Skin and dust keratins from sample handling. Deliberately NOT KRT8/KRT18/KRT19, which are genuine
            // epithelial markers and live in Pan-epithelial markers - mixing the two makes both useless.
            Members =
            {
                "KRT1", "KRT2", "KRT9", "KRT10",
            },
        },

        new ProteinList
        {
            Name = "Common contaminants (cRAP)",
            Category = "Readouts and contamination",
            DisplayOnly = true,
            ColorHex = "#7f7f7f",
            Visible = false,
            ShowLabels = true,
            // Written "<accession> = <name>": everything left of the '=' is matched, everything right of
            // it is only displayed. Accessions are unreadable and names are unsafe, so the panel carries
            // both rather than choosing.
            //
            // Listed by ACCESSION, not gene symbol, and that is not a style choice. These are non-human proteins:
            // 'ALB' for bovine serum albumin would match human albumin - the most abundant protein in a plasma
            // sample - and 'TRYP' would match human trypsin-1. Accessions cannot collide that way.
            //
            // The UniProt ENTRY name is not a way around that, which is subtler and cost this panel two
            // members: the matcher strips species suffixes so panels work across human and mouse, so
            // 'ENO1_YEAST' reduces to the token 'ENO1' and 'TRYP_PIG' to 'TRYP' - the very collisions
            // named above. Carrying them meant every human run colored abundant alpha-enolase as a
            // contaminant, labeled it "yeast, spike-in", and - because this panel is declared before
            // Glycolysis, and the first list to claim a protein wins - took ENO1 away from the panel it
            // belongs to. The accessions cover both spike-ins. Do not add an entry name back here, in any
            // species.
            //
            // A starting set, not an exhaustive one. Import the cRAP FASTA your search actually uses to extend it.
            Members =
            {
                "P00761 = Trypsin (porcine)",
                "P00760 = Trypsin (bovine)",
                "P02769 = Serum albumin (bovine, BSA)",
                "P00698 = Lysozyme C (chicken)",
                "P02701 = Avidin (chicken)",
                "P22629 = Streptavidin (S. avidinii)",
                "P02662 = alpha-S1-casein (bovine)",
                "P00924 = Enolase 1 (yeast, spike-in)",
                "P02663 = alpha-S2-casein (bovine)",
                "P02666 = beta-casein (bovine)",
                "P02668 = kappa-casein (bovine)",
                "P00921 = Carbonic anhydrase 2 (bovine)"
            },
        },

        new ProteinList
        {
            Name = "Glomerulus",
            Category = "Normalizers",
            ColorHex = "#1f77b4",
            Visible = false,
            // Structural markers of glomerular tissue, for normalizing single-glomerulus work by how
            // much glomerulus a dissection actually captured:
            //   GBM collagen IV      COL4A3/4/5 - the alpha3-4-5 network, GBM-specific. COL4A1/COL4A2
            //                        are deliberately absent: they are ubiquitous basement membrane and
            //                        would make the score track any BM, not this one.
            //   GBM laminin          LAMA5, LAMB2, LAMC1 (laminin-521)
            //   BM proteoglycan      AGRN, HSPG2, NID1
            //   glomerular endothel. EHD3 (glomerular-endothelium enriched), EMCN, PECAM1
            //   mesangium            ITGA8, PDGFRB
            //   podocyte             PODXL, SYNPO, CD2AP, PTPRO
            //
            // Weighted toward structure ON PURPOSE. The obvious podocyte markers - NPHS1, NPHS2 - are
            // left out because podocyte loss IS the phenotype in most glomerular disease, and a score
            // dominated by them would regress out the finding rather than the capture. The four
            // podocyte proteins here are a minority of the panel so it still tracks podocyte-bearing
            // tissue without PC1 becoming a podocyte-injury axis.
            //
            // Check before trusting it on a new cohort: how many are quantified, and whether PC1
            // separates large sections from small ones (capture, what you want) or diseased from
            // control (pathology, which you do not want to remove).
            Members =
            {
                "COL4A3", "COL4A4", "COL4A5",
                "LAMA5", "LAMB2", "LAMC1",
                "AGRN", "HSPG2", "NID1",
                "EHD3", "EMCN", "PECAM1",
                "ITGA8", "PDGFRB",
                "PODXL", "SYNPO", "CD2AP", "PTPRO",
            },
        },

        new ProteinList
        {
            Name = "Tubular contamination",
            Category = "Readouts and contamination",
            DisplayOnly = true,
            ColorHex = "#d62728",
            Visible = false,
            ShowLabels = true,
            // NOT a normalizer - a readout. Hand-dissected or microdissected glomeruli carry tubular
            // fragments, and these proteins are abundant enough that a little carry-over is obvious on
            // the dynamic-range plot. Spread across nephron segments so the plot says WHICH segment
            // came along:
            //   proximal tubule    LRP2 (megalin), CUBN (cubilin), SLC34A1, SLC5A2, MIOX, ACSM2A,
            //                      ACSM2B, ANPEP, GGT1, PDZK1
            //   thick ascending    UMOD (uromodulin), SLC12A1
            //   distal / collecting SLC12A3, AQP2
            //   proximal + thin    AQP1
            //
            // Normalizing to these would be backwards: their abundance is the contamination, not the
            // thing being measured.
            Members =
            {
                "LRP2", "CUBN", "SLC34A1", "SLC5A2", "MIOX", "ACSM2A", "ACSM2B", "ANPEP", "GGT1",
                "PDZK1",
                "UMOD", "SLC12A1",
                "SLC12A3", "AQP2",
                "AQP1",
            },
        },

        new ProteinList
        {
            Name = "Histones (proteomic ruler)",
            Category = "Normalizers",
            ColorHex = "#8c564b",
            Visible = false,
            // Wisniewski, Hein, Cox & Mann, Mol Cell Proteomics 2014 (doi:10.1074/mcp.M113.037309): the summed MS
            // signal of histones is proportional to the DNA in the sample, and therefore to the number of cells.
            // It is the closest thing the proteome has to a cell counter, which is what makes it the right
            // denominator when what varies between samples is how much TISSUE was captured rather than how much
            // protein was loaded - hand-dissected glomeruli, for instance.
            //
            // COMPREHENSIVE ON PURPOSE. The paper sums 'all histone-derived peptides, irrespective of which histone
            // they mapped to or how they were assembled in protein groups', so a curated subset would measure
            // something narrower than the ruler it is named after. Both nomenclatures are carried - current HGNC
            // (H2BC*, H4C*, H1-*) and the legacy HIST1H* names most FASTA files and Skyline documents still use -
            // plus a few UniProt entry names, because PRISM matches tokens exactly and a panel in one nomenclature
            // silently misses whatever share of the signal uses the other.
            //
            // Two caveats the paper settles or raises. PTMs: searching acetyl/phospho/methyl moved the CUMULATIVE
            // histone fraction by only 5-10% (H3 individually is the exception), so a standard search is fine.
            // Depth: the histone fraction only stabilizes from roughly 12,000 peptides down, so a shallow run gives
            // an unreliable ruler.
            //
            // NOT the paper's absolute scaling. PRISM removes the axis these move along, giving a RELATIVE per-cell
            // adjustment; the ruler's copy-numbers-per-cell need the histone fraction of total signal, which is a
            // different calculation and not what marker_normalization does.
            Members =
            {
                "H1-0", "H1-1", "H1-2", "H1-3", "H1-4", "H1-5", "H1-6", "H1-10", "H1F0",
                "HIST1H1A", "HIST1H1C", "HIST1H1D", "HIST1H1E", "HIST1H1B", "HIST1H1T", "H1FX",
                "H2AC4", "H2AC6", "H2AC8", "H2AC11", "H2AC12", "H2AC13", "H2AC14", "H2AC15",
                "H2AC16", "H2AC17", "H2AC18", "H2AC19", "H2AC20", "H2AC21", "H2AZ1", "H2AZ2",
                "H2AX", "H2AJ", "MACROH2A1", "MACROH2A2", "HIST1H2AB", "HIST1H2AC", "HIST1H2AD",
                "HIST1H2AE", "HIST1H2AG", "HIST1H2AH", "HIST1H2AJ", "HIST1H2AK", "HIST1H2AL",
                "HIST1H2AM", "HIST2H2AA3", "HIST2H2AC", "H2AFZ", "H2AFV", "H2AFX", "H2AFJ",
                "H2AFY", "H2AFY2", "H2BC1", "H2BC3", "H2BC4", "H2BC5", "H2BC6", "H2BC7", "H2BC8",
                "H2BC9", "H2BC10", "H2BC11", "H2BC12", "H2BC13", "H2BC14", "H2BC15", "H2BC17",
                "H2BC18", "H2BC21", "H2BU1", "HIST1H2BA", "HIST1H2BB", "HIST1H2BC", "HIST1H2BD",
                "HIST1H2BE", "HIST1H2BF", "HIST1H2BG", "HIST1H2BH", "HIST1H2BI", "HIST1H2BJ",
                "HIST1H2BK", "HIST1H2BL", "HIST1H2BM", "HIST1H2BN", "HIST1H2BO", "HIST2H2BE",
                "HIST2H2BF", "HIST3H2BB", "H3C1", "H3C2", "H3C3", "H3C4", "H3C6", "H3C7", "H3C8",
                "H3C10", "H3C11", "H3C12", "H3-3A", "H3-3B", "H3-4", "H3-5", "CENPA", "HIST1H3A",
                "HIST1H3B", "HIST1H3C", "HIST1H3D", "HIST1H3E", "HIST1H3F", "HIST1H3G", "HIST1H3H",
                "HIST1H3I", "HIST1H3J", "HIST2H3A", "HIST2H3C", "HIST3H3", "H3F3A", "H3F3B",
                "H4C1", "H4C2", "H4C3", "H4C4", "H4C5", "H4C6", "H4C8", "H4C9", "H4C11", "H4C12",
                "H4C13", "H4C14", "H4C15", "H4-16", "HIST1H4A", "HIST1H4B", "HIST1H4C", "HIST1H4D",
                "HIST1H4E", "HIST1H4F", "HIST1H4H", "HIST1H4I", "HIST1H4J", "HIST1H4K", "HIST1H4L",
                "HIST2H4A", "HIST2H4B", "HIST4H4", "H4_HUMAN", "H2A1_HUMAN", "H2B1K_HUMAN",
                "H31_HUMAN", "H32_HUMAN", "H33_HUMAN", "H12_HUMAN", "H14_HUMAN", "H2AZ_HUMAN",
                "H2AX_HUMAN",
            },
        },

        new ProteinList
        {
            Name = "Ribosomal proteins",
            Category = "Normalizers",
            ColorHex = "#9467bd",
            Visible = false,
            // Validated as a proteomic ruler for cellular RNA in the same paper: summed ribosomal protein signal was
            // 3.6-5.3% of total, and tracked biochemically measured RNA within a factor of 1.01 +/- 0.13. As a
            // normalizer this asks 'per unit of biosynthetic capacity' rather than 'per cell'.
            // Be careful in any study where translation itself is the phenotype - then this is the finding, not the
            // denominator.
            Members =
            {
                "RPLP0", "RPLP1", "RPLP2", "RPSA", "RPL3", "RPL4", "RPL5", "RPL6", "RPL7", "RPL7A",
                "RPL8", "RPL9", "RPL10", "RPL10A", "RPL11", "RPL12", "RPL13", "RPL13A", "RPL14",
                "RPL15", "RPL17", "RPL18", "RPL18A", "RPL19", "RPL21", "RPL22", "RPL23", "RPL23A",
                "RPL24", "RPL26", "RPL27", "RPL27A", "RPL28", "RPL29", "RPL30", "RPL31", "RPL32",
                "RPL34", "RPL35", "RPL35A", "RPL36", "RPL37", "RPL37A", "RPL38", "RPL39", "RPL41",
                "RPS2", "RPS3", "RPS3A", "RPS4X", "RPS5", "RPS6", "RPS7", "RPS8", "RPS9", "RPS10",
                "RPS11", "RPS12", "RPS13", "RPS14", "RPS15", "RPS15A", "RPS16", "RPS17", "RPS18",
                "RPS19", "RPS20", "RPS21", "RPS23", "RPS24", "RPS25", "RPS26", "RPS27", "RPS27A",
                "RPS28", "RPS29",
            },
        },

        new ProteinList
        {
            Name = "Mitochondrial content",
            Category = "Normalizers",
            ColorHex = "#e377c2",
            Visible = false,
            // For work where mitochondrial content is the thing that varies between samples - muscle, metabolic
            // tissue. Answers 'per unit of mitochondrion'. Citrate synthase (CS) is the classical biochemical
            // equivalent, and is kept here so the two can be compared.
            Members =
            {
                "VDAC1", "VDAC2", "VDAC3", "TOMM20", "TOMM22", "TIMM23", "CS", "ATP5F1A",
                "ATP5F1B", "SLC25A3", "SLC25A5", "HSPD1", "HSPA9", "MDH2", "SDHA", "UQCRC1",
                "UQCRC2", "COX4I1", "COX5A", "NDUFS1", "PHB1",
            },
        },

        new ProteinList
        {
            Name = "Housekeeping proteins",
            Category = "Readouts and contamination",
            DisplayOnly = true,
            ColorHex = "#7f7f7f",
            Visible = false,
            ShowLabels = true,
            // A READOUT, not a normalizer, and the distinction matters more here than anywhere else in this file.
            // These are Western blot loading controls: one band standing in for total protein. PRISM already does
            // that job at Stage 2b across every peptide in the run, so normalizing to twenty proteins would redo
            // the loading step on a far noisier estimate - and these proteins are, famously, not stable anyway
            // (ACTB and GAPDH move with proliferation, hypoxia and cell cycle).
            // Useful for LOOKING: if these are flat after normalization, loading normalization did its job.
            Members =
            {
                // ENO1 is deliberately absent: yeast enolase is a routine spike-in standard and shares the
                // symbol with the human gene, so this member would fire on the spike-in rather than on a
                // housekeeper. It is listed by accession in the contaminants panel, where seeing it is the point.
                "ACTB", "GAPDH", "TUBB", "TUBA1B", "PPIA", "YWHAZ", "YWHAE", "VCL", "PGK1",
                "LDHA", "HSP90AA1", "HSP90AB1", "HSPA8", "B2M", "UBC", "EEF1A1", "EEF2", "CFL1",
                "PKM",
            },
        },

        // ---- Pathways -------------------------------------------------------------------------------
        // Display sets, not denominators: every one is DisplayOnly, because normalizing to a pathway
        // removes the biology being studied. Hand-written rather than imported, which keeps them free of
        // any third party's redistribution terms - KEGG in particular cannot be shipped - and lets each
        // one be sized for a plot (15-35 members) instead of the several hundred an ontology would give.
        // For a specific pathway not here, import a gene list into a list of your own.

        new ProteinList
        {
            Name = "Oxidative phosphorylation",
            Category = "Pathways and processes",
            ColorHex = "#8c564b",
            Visible = false,
            DisplayOnly = true,
            // Complexes I-V of the electron transport chain. The most abundant mitochondrial signal in most
            // tissues, and the one that moves with mitochondrial content - compare against the Mitochondrial
            // content panel, which asks how much mitochondrion there is rather than how much OXPHOS machinery.
            Members =
            {
                "NDUFA4", "NDUFA9", "NDUFB10", "NDUFS1", "NDUFS2", "NDUFS3", "NDUFV1", "NDUFV2",
                "SDHA", "SDHB", "UQCRC1", "UQCRC2", "UQCRFS1", "CYC1", "UQCRB", "CYCS", "COX4I1",
                "COX5A", "COX5B", "COX6B1", "COX6C", "COX7A2", "ATP5F1A", "ATP5F1B", "ATP5F1C",
                "ATP5PB", "ATP5PO", "ATP5MC3",
            },
        },

        new ProteinList
        {
            Name = "Glycolysis",
            Category = "Pathways and processes",
            ColorHex = "#ff7f0e",
            Visible = false,
            DisplayOnly = true,
            // The canonical ten steps plus the common isoenzymes. Nearly always detected and usually abundant, so
            // it reads clearly on an abundance plot.
            // ENO1 is present here as the human gene: if the lab spikes yeast enolase, that is a DIFFERENT protein
            // tracked by accession in Common contaminants (cRAP).
            Members =
            {
                "HK1", "HK2", "GPI", "PFKL", "PFKM", "PFKP", "ALDOA", "ALDOB", "ALDOC", "TPI1",
                "GAPDH", "PGK1", "PGAM1", "ENO1", "ENO2", "ENO3", "PKM", "LDHA", "LDHB",
            },
        },

        new ProteinList
        {
            Name = "TCA cycle",
            Category = "Pathways and processes",
            ColorHex = "#2ca02c",
            Visible = false,
            DisplayOnly = true,
            // Citrate synthase through malate dehydrogenase. CS is shared with Mitochondrial content, where it is
            // the classical biochemical measure of mitochondrial mass rather than a pathway member.
            Members =
            {
                "CS", "ACO2", "IDH2", "IDH3A", "IDH3B", "OGDH", "DLST", "DLD", "SUCLA2", "SUCLG1",
                "SDHA", "SDHB", "FH", "MDH2", "PDHA1", "PDHB", "PC",
            },
        },

        new ProteinList
        {
            Name = "Proteasome",
            Category = "Pathways and processes",
            ColorHex = "#9467bd",
            Visible = false,
            DisplayOnly = true,
            // The 20S core (PSMA/PSMB) and the 19S regulatory particle (PSMC/PSMD). A tight, well-detected complex,
            // which makes it a good sanity check: its members should move together.
            Members =
            {
                "PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7", "PSMB1", "PSMB2",
                "PSMB3", "PSMB4", "PSMB5", "PSMB6", "PSMB7", "PSMC1", "PSMC2", "PSMC3", "PSMC4",
                "PSMC5", "PSMC6", "PSMD1", "PSMD2", "PSMD3", "PSMD6", "PSMD7", "PSMD11", "PSMD14",
            },
        },

        new ProteinList
        {
            Name = "Lysosome",
            Category = "Pathways and processes",
            ColorHex = "#d62728",
            Visible = false,
            DisplayOnly = true,
            // Cathepsins, glycosidases and membrane proteins. Prominent in EV and secretome preparations, where a
            // high lysosomal signal usually says something about the isolation rather than the biology.
            Members =
            {
                "LAMP1", "LAMP2", "CTSA", "CTSB", "CTSD", "CTSL", "CTSS", "CTSZ", "CTSK", "GBA",
                "HEXA", "HEXB", "GUSB", "GLA", "GAA", "NPC2", "PSAP", "TPP1", "GRN", "ASAH1",
                "NAGLU", "MAN2B1",
            },
        },

        new ProteinList
        {
            Name = "Spliceosome and hnRNP",
            Category = "Pathways and processes",
            ColorHex = "#17becf",
            Visible = false,
            DisplayOnly = true,
            // Core snRNP and heterogeneous nuclear ribonucleoproteins - a nuclear signal, so its presence in a
            // secretome or EV prep is a contamination readout rather than a finding.
            Members =
            {
                "SNRPA", "SNRPB", "SNRPD1", "SNRPD2", "SNRPD3", "SNRPE", "SNRPF", "SF3A1", "SF3B1",
                "SF3B2", "SF3B3", "SRSF1", "SRSF2", "SRSF3", "SRSF7", "HNRNPA1", "HNRNPA2B1",
                "HNRNPC", "HNRNPK", "HNRNPM", "HNRNPU", "PRPF8", "EFTUD2", "DDX5",
            },
        },

        new ProteinList
        {
            Name = "Extracellular matrix",
            Category = "Pathways and processes",
            ColorHex = "#bcbd22",
            Visible = false,
            DisplayOnly = true,
            // Collagens, laminins and the proteoglycans that hold them together. Overlaps the Glomerulus panel by
            // design - the same proteins are structure there and matrix here, which is the difference between
            // asking how much glomerulus was captured and asking how much matrix a tissue contains.
            Members =
            {
                "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "COL5A1", "COL6A1", "COL6A2",
                "COL6A3", "COL12A1", "FN1", "LAMA4", "LAMA5", "LAMB1", "LAMB2", "LAMC1", "NID1",
                "NID2", "HSPG2", "AGRN", "FBN1", "EMILIN1", "DCN", "LUM", "BGN", "FMOD", "VCAN",
                "POSTN", "TNC", "SPARC", "THBS1", "LTBP1",
            },
        },

        new ProteinList
        {
            Name = "Actin cytoskeleton",
            Category = "Pathways and processes",
            ColorHex = "#e377c2",
            Visible = false,
            DisplayOnly = true,
            // Actin, its motors and its regulators. Ubiquitous and abundant, so this is most useful as a reference
            // band on an abundance plot rather than as a finding in itself.
            Members =
            {
                "ACTB", "ACTG1", "ACTN1", "ACTN4", "MYH9", "MYH10", "MYL6", "MYL12A", "TLN1",
                "VCL", "ZYX", "PXN", "FLNA", "FLNB", "CFL1", "PFN1", "TPM1", "TPM3", "TPM4",
                "CAPZA1", "CAPZB", "ARPC1B", "ARPC2", "ARPC3", "ACTR2", "ACTR3", "GSN", "EZR",
                "MSN", "RDX",
            },
        },

        new ProteinList
        {
            Name = "Antigen presentation (MHC)",
            Category = "Pathways and processes",
            ColorHex = "#1f77b4",
            Visible = false,
            DisplayOnly = true,
            // Class I and class II with their loading machinery. B2M is also a routine plasma analyte, so on a
            // plasma plot expect it to sit far above the rest of the panel.
            Members =
            {
                "HLA-A", "HLA-B", "HLA-C", "HLA-E", "B2M", "TAP1", "TAP2", "TAPBP", "CALR", "CANX",
                "PDIA3", "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1",
                "CD74", "HLA-DMA", "HLA-DMB",
            },
        },

        new ProteinList
        {
            Name = "Acute phase response",
            Category = "Pathways and processes",
            ColorHex = "#a8341f",
            Visible = false,
            DisplayOnly = true,
            // Positive acute-phase reactants, plus the two classic NEGATIVE ones - ALB and TF fall when the others
            // rise, so a panel that shows them moving in opposite directions is showing the response, not noise.
            // Overlaps the plasma panels heavily; that overlap is the point.
            Members =
            {
                "CRP", "SAA1", "SAA2", "HP", "HPX", "LBP", "ORM1", "ORM2", "SERPINA1", "SERPINA3",
                "FGA", "FGB", "FGG", "C3", "C9", "CP", "ITIH3", "ITIH4", "LCN2", "S100A8",
                "S100A9", "ALB", "TF", "TTR", "RBP4",
            },
        },

        new ProteinList
        {
            Name = "Chaperones and proteostasis",
            Category = "Pathways and processes",
            ColorHex = "#7f7f7f",
            Visible = false,
            DisplayOnly = true,
            // HSP70/90 families, the CCT/TRiC folding complex and their co-chaperones. HSPA8 and HSP90B1 are also
            // EV-associated, so the EV panels and this one will light up together in a vesicle prep.
            Members =
            {
                "HSPA1A", "HSPA4", "HSPA5", "HSPA8", "HSPA9", "HSP90AA1", "HSP90AB1", "HSP90B1",
                "DNAJA1", "DNAJB1", "DNAJC7", "STIP1", "BAG3", "CDC37", "PTGES3", "TCP1", "CCT2",
                "CCT3", "CCT4", "CCT5", "CCT6A", "CCT7", "CCT8", "PPIA", "PPIB", "PDIA3", "PDIA6",
                "P4HB",
            },
        },

        new ProteinList
        {
            Name = "Redox and antioxidant",
            Category = "Pathways and processes",
            ColorHex = "#2c6b3f",
            Visible = false,
            DisplayOnly = true,
            // Superoxide dismutases, peroxiredoxins, the glutathione and thioredoxin systems. PRDX2 and CAT are
            // shared with Hemolysis, where they mean erythrocyte lysis rather than oxidative biology - which is
            // why one is a readout and this is not.
            Members =
            {
                "SOD1", "SOD2", "SOD3", "CAT", "GPX1", "GPX3", "GPX4", "PRDX1", "PRDX2", "PRDX3",
                "PRDX4", "PRDX5", "PRDX6", "TXN", "TXN2", "TXNRD1", "GSR", "GSTP1", "GSTO1",
                "GLRX", "NQO1", "HMOX1", "GCLM",
            },
        },

        // ---- Neurobiology and cellular processes ----------------------------------------------------
        // The brain cell-type panels are IDENTITY sets and may normalize; the disease and process panels
        // are DisplayOnly, because in a study of them their abundance is the result. Synaptic proteins
        // are the closest call and are display-only deliberately - see that panel's note.

        new ProteinList
        {
            Name = "Neuronal markers",
            Category = "Brain and neurodegeneration",
            ColorHex = "#1f77b4",
            Visible = false,
            ShowLabels = true,
            // Neuron identity: neurofilaments, neuron-specific tubulin and enolase, and the cytoskeletal and
            // vesicle proteins that go with them. An IDENTITY panel, so it can serve as a denominator - 'per unit
            // of neuron' - but only where neuronal loss is not itself the phenotype. In most neurodegeneration it
            // is; use Histones (per cell) or a structural panel there instead.
            // NEFL doubles as a CSF/plasma axonal-damage biomarker, so in a fluid matrix expect it to behave as a
            // readout rather than as tissue identity.
            Members =
            {
                "NEFL", "NEFM", "NEFH", "INA", "TUBB3", "ENO2", "MAP2", "MAPT", "SYN1", "SYT1",
                "STMN2", "UCHL1", "RBFOX3", "SNAP25", "NRGN", "GAP43", "CAMK2A", "THY1", "BASP1",
            },
        },

        new ProteinList
        {
            Name = "Astrocyte markers",
            Category = "Brain and neurodegeneration",
            ColorHex = "#2ca02c",
            Visible = false,
            ShowLabels = true,
            // Astrocyte identity. GFAP and AQP4 are the canonical pair; ALDH1L1 and SLC1A2/SLC1A3 (GLT-1/GLAST)
            // add metabolic and transporter coverage. GFAP rises with reactive astrogliosis, so a high signal can
            // mean more astrocytes OR the same astrocytes activated - the panel cannot tell those apart.
            Members =
            {
                "GFAP", "AQP4", "ALDH1L1", "SLC1A2", "SLC1A3", "S100B", "GJA1", "GLUL", "CLU",
                "APOE", "VIM", "SPARCL1", "ALDOC", "MT3", "CD44",
            },
        },

        new ProteinList
        {
            Name = "Microglia markers",
            Category = "Brain and neurodegeneration",
            ColorHex = "#d62728",
            Visible = false,
            ShowLabels = true,
            // Microglial identity, split between homeostatic (P2RY12, TMEM119, CX3CR1) and activation-associated
            // (CD68, TREM2, TYROBP, C1Q) members. That split is the useful part: the same panel reads differently
            // depending on which half is carrying it, so look at the members rather than the total.
            Members =
            {
                "AIF1", "P2RY12", "TMEM119", "CX3CR1", "CSF1R", "ITGAM", "CD68", "TREM2", "TYROBP",
                "C1QA", "C1QB", "C1QC", "SPI1", "LAPTM5", "CTSS", "HEXB", "FCER1G",
            },
        },

        new ProteinList
        {
            Name = "Oligodendrocyte and myelin",
            Category = "Brain and neurodegeneration",
            ColorHex = "#9467bd",
            Visible = false,
            ShowLabels = true,
            // Myelin and the cells that make it. Among the most abundant proteins in white matter, so on a brain
            // abundance plot this panel sits at the top and is a good orientation marker.
            // White-matter content varies with dissection, which makes this a plausible capture denominator for
            // brain tissue - and a bad one in demyelinating disease, where myelin loss is the finding.
            Members =
            {
                "MBP", "PLP1", "MOG", "MAG", "CNP", "MOBP", "MAL", "CLDN11", "SIRT2", "UGT8",
                "ERMN", "ASPA", "OLIG1", "OLIG2", "SOX10", "TF", "QDPR",
            },
        },

        new ProteinList
        {
            Name = "Alzheimer's disease",
            Category = "Brain and neurodegeneration",
            ColorHex = "#a8341f",
            Visible = false,
            ShowLabels = true,
            DisplayOnly = true,
            // The proteins defining the AD lesions and the genetics around them: APP and its secretase machinery,
            // MAPT, APOE, and the risk-gene products detectable by MS. VGF, SMOC1, CHI3L1 and NPTX2 are included as
            // CSF/plasma markers that have held up across cohorts.
            // DISPLAY ONLY, emphatically: in an AD study these ARE the finding, and normalizing to them would
            // regress out the disease and leave a tidy null result.
            // MAPT and APOE also sit in the neuronal and astrocyte panels - the same protein is identity there and
            // pathology here, which is the distinction these panels exist to keep straight.
            Members =
            {
                "APP", "MAPT", "APOE", "PSEN1", "PSEN2", "BACE1", "NCSTN", "APH1A", "PSENEN",
                "ITM2B", "CLU", "CR1", "PICALM", "TREM2", "SORL1", "BIN1", "CD2AP", "MS4A4A",
                "INPP5D", "VGF", "SMOC1", "CHI3L1", "NPTX2", "SCG2", "GAP43",
            },
        },

        new ProteinList
        {
            Name = "Parkinson's disease",
            Category = "Brain and neurodegeneration",
            ColorHex = "#7a5c12",
            Visible = false,
            ShowLabels = true,
            DisplayOnly = true,
            // PD_MARKERS from the SEA-AD pilot notebooks, kept as the lab uses it. Two companion panels
            // below carve out the parts that ask different questions: the LRRK2-RAB substrates, and the
            // lysosomal arm.
            //
            // Two halves, and the split is the useful part. The familial-PD gene products (SNCA, LRRK2, PRKN,
            // PINK1, PARK7, VPS35, GBA1) say what is going wrong; the dopaminergic markers (TH, SLC6A3, SLC18A2,
            // DDC, ALDH1A1) say how much substantia nigra is left, since those neurons are what PD destroys.
            // DISPLAY ONLY for both halves. The gene products are the pathology, and the dopaminergic markers are
            // the loss being measured - normalizing to either erases the result.
            // LRRK2 is large and low-abundance; do not read its absence as biology.
            Members =
            {
                "SNCA", "SNCB", "SNCG", "LRRK2", "PRKN", "PINK1", "PARK7", "VPS35", "DNAJC6",
                "GBA", "GLB1", "RAB10", "RAB8A", "RAB12", "RAB29", "RAB35", "GAK", "ATP13A2",
                "SYNJ1", "TMEM175", "SCARB2", "VPS13C", "GALC", "SMPD1", "UCHL1", "GRN", "CTSB",
                "CTSD", "CTSL", "CTSS", "CTSZ", "CTSA", "CTSF", "TH", "DDC", "CALB1",
            },
        },

        new ProteinList
        {
            Name = "ALS and FTD",
            Category = "Brain and neurodegeneration",
            ColorHex = "#0f6e78",
            Visible = false,
            ShowLabels = true,
            DisplayOnly = true,
            // The two share genetics and pathology, so one panel covers both: the RNA-binding proteins that
            // mislocalize (TARDBP, FUS, MATR3, HNRNPA1/A2B1), the autophagy-adjacent genes (SQSTM1, OPTN, VCP,
            // UBQLN2, TBK1), SOD1, and the neurofilaments that report axonal damage in CSF and plasma.
            // DISPLAY ONLY. NEFL in particular is a damage READOUT - it rises because axons are dying, so
            // normalizing to it would divide the signal by the disease.
            // C9orf72 is the commonest genetic cause and among the hardest to detect by bottom-up MS; treat its
            // absence as a coverage statement, not a biological one.
            Members =
            {
                "TARDBP", "FUS", "SOD1", "C9orf72", "OPTN", "SQSTM1", "VCP", "UBQLN2", "TBK1",
                "MATR3", "HNRNPA1", "HNRNPA2B1", "PFN1", "CHMP2B", "ANXA11", "TIA1", "SETX",
                "ATXN2", "NEFL", "NEFH", "CHAT", "SLC18A3",
            },
        },

        new ProteinList
        {
            Name = "Huntington's disease",
            Category = "Brain and neurodegeneration",
            ColorHex = "#9467bd",
            Visible = false,
            ShowLabels = true,
            DisplayOnly = true,
            // HTT itself, its interactors, and the striatal medium spiny neuron markers that report the cell type
            // HD destroys (PPP1R1B/DARPP-32, PDE10A, DRD1, DRD2, ADORA2A, GPR88, RGS9, ARPP21, PENK).
            // DISPLAY ONLY: the MSN markers ARE the degeneration being measured.
            // A limit worth stating plainly - bottom-up MS cannot distinguish mutant from wild-type HTT. The polyQ
            // expansion sits in one N-terminal tryptic region and standard workflows do not resolve it, so this
            // panel measures HTT abundance and striatal content, never allele status.
            Members =
            {
                "HTT", "HAP1", "HIP1", "HIP1R", "BDNF", "PPP1R1B", "PDE10A", "DRD1", "DRD2",
                "ADORA2A", "GPR88", "RGS9", "ARPP21", "PENK", "TAC1", "CALB1", "ATXN1", "ATXN2",
                "ATXN3", "TBP",
            },
        },

        new ProteinList
        {
            Name = "Synaptic proteins",
            Category = "Brain and neurodegeneration",
            ColorHex = "#e377c2",
            Visible = false,
            ShowLabels = true,
            DisplayOnly = true,
            // Pre- and postsynaptic machinery: SNAREs, synaptic vesicle proteins and the postsynaptic density.
            // DISPLAY ONLY by default because synapse loss is the phenotype in most neurodegeneration, so
            // normalizing to it removes the very thing being measured. A synaptosome preparation that genuinely
            // wants 'per unit of synapse' is a legitimate exception - duplicate this panel into a list of your own
            // and use that, which makes the choice explicit rather than accidental.
            Members =
            {
                "SNAP25", "SYP", "SYN1", "SYN2", "SYT1", "VAMP2", "STX1A", "STX1B", "STXBP1",
                "CPLX1", "CPLX2", "SV2A", "DLG4", "SHANK3", "HOMER1", "GRIN1", "GRIA2", "NRGN",
                "SYNGR1", "SYNPO", "RAB3A", "NSF", "AP2B1",
            },
        },

        new ProteinList
        {
            Name = "DNA damage repair",
            Category = "Pathways and processes",
            ColorHex = "#8c564b",
            Visible = false,
            DisplayOnly = true,
            // Sensing, signaling and the major repair pathways together: MRN and the ATM/ATR kinases, homologous
            // recombination, non-homologous end joining, mismatch repair, nucleotide and base excision.
            // Most of these are low-abundance nuclear proteins, so expect sparse coverage outside a deep
            // whole-cell measurement - and treat their presence in a secretome or EV prep as contamination.
            Members =
            {
                "ATM", "ATR", "CHEK1", "CHEK2", "TP53BP1", "MDC1", "MRE11", "RAD50", "NBN",
                "RAD51", "BRCA1", "BRCA2", "PARP1", "XRCC1", "XRCC5", "XRCC6", "PRKDC", "LIG1",
                "LIG3", "LIG4", "ERCC1", "XPA", "XPC", "MSH2", "MSH6", "MLH1", "PMS2", "PCNA",
                "RPA1", "RPA2", "FEN1", "APEX1", "OGG1", "UNG", "POLB", "SMC1A", "RIF1",
            },
        },

        new ProteinList
        {
            Name = "Autophagy",
            Category = "Pathways and processes",
            ColorHex = "#17becf",
            Visible = false,
            DisplayOnly = true,
            // Cargo receptors, the ATG conjugation machinery and the LC3/GABARAP family. SQSTM1 and the LC3s are
            // the ones usually read as a flux marker, with the standing caveat that a single timepoint cannot
            // distinguish more autophagy from blocked degradation.
            Members =
            {
                "SQSTM1", "MAP1LC3B", "GABARAP", "GABARAPL1", "GABARAPL2", "NBR1", "OPTN",
                "CALCOCO2", "TAX1BP1", "ATG3", "ATG5", "ATG7", "ATG12", "ATG16L1", "BECN1",
                "WIPI2", "ULK1", "RB1CC1", "VPS35", "LAMP1", "CTSD",
            },
        },

        new ProteinList
        {
            Name = "Unfolded protein response",
            Category = "Pathways and processes",
            ColorHex = "#7a5c12",
            Visible = false,
            DisplayOnly = true,
            // ER stress sensing and the ER folding machinery it drives. HSPA5 (BiP) is the hub and is also in the
            // chaperone panel; PDIA3, CALR and CANX are shared with antigen presentation, where they are loading
            // machinery rather than a stress response.
            Members =
            {
                "HSPA5", "DDIT3", "ATF4", "ATF6", "EIF2AK3", "ERN1", "XBP1", "EDEM1", "SEL1L",
                "HERPUD1", "DNAJB9", "DNAJC3", "PDIA3", "PDIA4", "PDIA6", "P4HB", "CALR", "CANX",
                "HYOU1", "SDF2L1", "MANF", "CRELD2",
            },
        },

        new ProteinList
        {
            Name = "Innate immune signaling",
            Category = "Pathways and processes",
            ColorHex = "#ff7f0e",
            Visible = false,
            DisplayOnly = true,
            // Pattern recognition through to the inflammasome and its cytokines. In plasma the abundant members
            // (S100A8/A9, LBP, CD14) dominate and the signaling components are usually absent - a panel that
            // reads very differently in tissue than in fluid.
            Members =
            {
                "TLR2", "TLR4", "CD14", "LBP", "MYD88", "IRAK4", "TRAF6", "NFKB1", "NFKB2", "RELA",
                "NLRP3", "PYCARD", "CASP1", "IL1B", "IL18", "S100A8", "S100A9", "S100A12", "HMGB1",
                "TNFAIP3", "STAT1", "STAT3", "ISG15", "MX1",
            },
        },

        new ProteinList
        {
            Name = "Apoptosis",
            Category = "Pathways and processes",
            ColorHex = "#5b6b70",
            Visible = false,
            DisplayOnly = true,
            // Initiator and executioner caspases, the BCL2 family on both sides, and the apoptosome. CYCS is
            // shared with oxidative phosphorylation - it is an ETC component until it is in the cytosol, which no
            // abundance measurement can distinguish.
            Members =
            {
                "CASP3", "CASP6", "CASP7", "CASP8", "CASP9", "BAX", "BAK1", "BID", "BCL2",
                "BCL2L1", "MCL1", "BAD", "CYCS", "APAF1", "DIABLO", "XIAP", "BIRC2", "BIRC5",
                "PARP1", "FAS", "FADD", "TP53", "AIFM1", "ENDOG",
            },
        },

        new ProteinList
        {
            Name = "Fatty acid oxidation",
            Category = "Pathways and processes",
            ColorHex = "#2c6b3f",
            Visible = false,
            DisplayOnly = true,
            // Carnitine shuttle and the mitochondrial beta-oxidation spiral. Sits inside the mitochondrion, so
            // read it alongside Mitochondrial content: a rise in both is more mitochondrion, a rise in this one
            // alone is a shift in what those mitochondria are doing.
            Members =
            {
                "CPT1A", "CPT2", "SLC25A20", "CRAT", "ACADVL", "ACADM", "ACADS", "ACADL", "ACAD9",
                "HADHA", "HADHB", "ECHS1", "ECI1", "ACAA2", "ACAT1", "HADH", "ETFA", "ETFB",
                "ETFDH", "ACOX1", "CROT",
            },
        },

        // ---- Tissue composition ---------------------------------------------------------------------
        // What a dissection actually contained, as opposed to which cell types are present. The pair is
        // meant to be read as a ratio - see either panel's note.

        new ProteinList
        {
            Name = "White matter",
            Category = "Normalizers",
            ColorHex = "#5b6b70",
            Visible = false,
            ShowLabels = true,
            // The five-marker myelin set used across the SEA-AD pilot notebooks (WM_GENES), kept exactly
            // as the lab uses it so a panel here reproduces those figures rather than approximating them.
            // Deliberately minimal - PLP1 and MBP alone carry most of the signal. If a cohort needs
            // sharper discrimination, CLDN11, ASPA, OPALIN, MOBP and UGT8 are the usual additions;
            // duplicate this panel rather than editing it, so the SEA-AD definition stays intact.
            //
            // How much white matter a dissection captured. Myelin dominates it, so this overlaps
            // 'Oligodendrocyte and myelin' heavily - and the two exist separately for the same reason 'Kidney
            // tubule epithelial' and 'Tubular contamination' do. That panel asks which cell type this is; this one
            // asks what got dissected, which is the question when cortical samples vary in how much subcortical
            // white matter came along.
            // NOT display-only: white-matter content is a capture variable, so 'per unit of grey matter' is a
            // legitimate denominator for cortical work - pair it with Grey matter below and use the ratio.
            // In a demyelinating disease it stops being capture and becomes the finding; do not normalize then.
            Members =
            {
                "PLP1", "MBP", "MOG", "MAG", "CNP",
            },
        },

        new ProteinList
        {
            Name = "Grey matter",
            Category = "Normalizers",
            ColorHex = "#0f6e78",
            Visible = false,
            ShowLabels = true,
            // The four-marker synaptic set used across the SEA-AD pilot notebooks (GM_GENES), kept exactly
            // as the lab uses it.
            //
            // MEASURED CAUTION, from the SEA-AD MTG pilot (73 donors with both GM area and pathology):
            // this panel correlates with measured GM fraction at r = +0.37, but it also declines with
            // pathology - r = -0.14 with Braak and -0.22 with CERAD. All four members are synaptic, and
            // synapse loss is the AD phenotype, so in a neurodegeneration cohort part of what it measures
            // IS the disease. Fine for reporting composition; treat it with suspicion as a denominator
            // there, because residualizing on it removes some of the finding.
            // (Of the four, only SYN1 was independent of both pathology scales.) For sharper cell-type discrimination SLC17A7, GAD1/GAD2 and ATP1A3 are
            // the usual additions - duplicate rather than edit, so the SEA-AD definition stays intact.
            //
            // The complement of White matter: neuronal and synaptic density, which is what grey matter is made of.
            // Discriminating members rather than merely abundant ones - SLC17A7 and GAD1/GAD2 for excitatory and
            // inhibitory neurons, ATP1A3 for the neuronal sodium pump, CAMK2A and NRGN for cortical neuropil.
            // Used as a pair with White matter, the RATIO is the useful readout: it says how consistent the
            // dissection was across a cohort before any of the biology is interpreted.
            // The same caution as Neuronal markers - where neuronal loss IS the phenotype, this measures the
            // disease rather than the capture, and Histones (per cell) is the safer denominator.
            Members =
            {
                "SYP", "SNAP25", "SYN1", "DLG4",
            },
        },

        new ProteinList
        {
            Name = "Parkinson's disease (RAB substrates)",
            Category = "Brain and neurodegeneration",
            ColorHex = "#d9b654",
            Visible = false,
            ShowLabels = true,
            DisplayOnly = true,
            // PD_RAB from the SEA-AD notebooks: the RAB GTPases LRRK2 phosphorylates. Separate from the main PD
            // panel because they ask a mechanistic question rather than a pathology one - LRRK2 activity, read
            // through its substrates.
            // Bottom-up abundance cannot see that activity; the informative measurement is the phosphosite (RAB10
            // T73 and its equivalents), which needs a phospho-enriched run. Read this panel as coverage - are the
            // substrates present at all - not as activation.
            Members =
            {
                "RAB10", "RAB8A", "RAB12", "RAB29", "RAB35", "RAB3A",
            },
        },

        new ProteinList
        {
            Name = "Parkinson's disease (lysosomal)",
            Category = "Brain and neurodegeneration",
            ColorHex = "#8c564b",
            Visible = false,
            ShowLabels = true,
            DisplayOnly = true,
            // PD_LYSOSOMAL from the SEA-AD notebooks: the lysosomal arm of PD genetics, with GBA1 the commonest
            // genetic risk factor and the cathepsins and glycosidases around it.
            // Overlaps the generic Lysosome panel heavily and deliberately - that one asks how much lysosome a
            // sample has, this one asks about the PD-implicated subset. Same proteins, different question.
            Members =
            {
                "GBA", "GBA2", "GLB1", "GALC", "SMPD1", "GRN", "SCARB2", "ATP13A2", "TMEM175",
                "PSAP", "HEXA", "HEXB", "LAMP1", "LAMP2", "NEU1", "ASAH1", "NPC1", "NPC2", "ARSA",
                "MAN2B1", "GUSB", "NAGLU", "TPP1", "GALNS", "CTSB", "CTSD", "CTSL", "CTSZ", "CTSA",
                "CTSF", "CTSS", "CTSH",
            },
        },

        new ProteinList
        {
            Name = "Brain fluid-like proteins",
            Category = "Brain and neurodegeneration",
            ColorHex = "#5b6b70",
            Visible = false,
            ShowLabels = true,
            DisplayOnly = true,
            // FLUID_LIKE from the SEA-AD notebooks: brain proteins that behave the way CSF/plasma markers do, so a
            // tissue measurement of them can be read alongside fluid cohorts.
            // A bridging set rather than a pathway - useful for asking whether a tissue finding has any prospect of
            // being seen in an accessible matrix.
            Members =
            {
                "VGF", "CHI3L1", "GPNMB", "SPP1", "CLU", "A2M", "APOD", "SERPINA3", "CST3", "B2M",
                "YWHAZ", "ALDOA", "ENO2", "NPTX2", "PKM", "YWHAG", "CALB1", "PEBP1", "CNTN1",
                "NCAM1",
            },
        },

        // ---- Metabolic and proliferative programmes --------------------------------------------------
        // All display-only. Insulin signaling in particular is a phospho story - see its own note.

        new ProteinList
        {
            Name = "Cell cycle and proliferation",
            Category = "Pathways and processes",
            ColorHex = "#d62728",
            Visible = false,
            DisplayOnly = true,
            // Replication licensing (MCM2-7), the replication machinery, and the mitotic kinases. Abundant in
            // proliferating tissue and close to absent in post-mitotic tissue, which makes it one of the more
            // interpretable panels here: a signal in brain or muscle usually means infiltrating or dividing cells,
            // not a change in the resident population.
            // MKI67 is the histology standard and among the least MS-friendly members - large, low-abundance and
            // poorly digested. Read the MCM complex instead.
            Members =
            {
                "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "PCNA", "RFC1", "RFC4", "TOP2A",
                "CCNB1", "CDK1", "CDK2", "CCNA2", "AURKA", "AURKB", "PLK1", "BUB3", "MAD2L1",
                "RRM1", "RRM2", "TK1", "TYMS", "MKI67", "KIF11", "KIF23",
            },
        },

        new ProteinList
        {
            Name = "Epithelial-mesenchymal transition",
            Category = "Pathways and processes",
            ColorHex = "#ff7f0e",
            Visible = false,
            DisplayOnly = true,
            // The clearest signature in this file: the epithelial half (CDH1, the claudins and keratins) and the
            // mesenchymal half (VIM, CDH2, FN1, TAGLN, ACTA2, S100A4) move in opposite directions, so the panel is
            // read as a BALANCE rather than a total.
            // Overlaps Pan-epithelial and Extracellular matrix on purpose - the same proteins are identity there
            // and a transition here.
            // The driving transcription factors (SNAI1, TWIST1, ZEB1) are low-abundance nuclear proteins and are
            // usually not detected; their absence says nothing.
            Members =
            {
                "CDH1", "CDH2", "VIM", "FN1", "SNAI2", "ZEB1", "TWIST1", "TAGLN", "ACTA2",
                "S100A4", "SPARC", "MMP2", "MMP14", "TIMP1", "ITGB1", "ITGA5", "THBS1", "COL1A1",
                "LOXL2", "CAV1", "KRT8", "KRT18", "EPCAM", "CTNNB1", "JUP",
            },
        },

        new ProteinList
        {
            Name = "Hypoxia response",
            Category = "Pathways and processes",
            ColorHex = "#7a5c12",
            Visible = false,
            DisplayOnly = true,
            // The downstream, detectable half of the hypoxia programme: the glycolytic shift, the pH and matrix
            // enzymes, and the mitophagy receptors.
            // HIF1A itself is deliberately included but rarely seen - it is degraded within minutes in normoxia,
            // so its absence is the expected result rather than evidence against hypoxia. CA9 and NDRG1 are the
            // more reliable reporters.
            // Overlaps Glycolysis heavily; the distinction is that a coordinated rise HERE with CA9 and NDRG1 is a
            // hypoxia reading, whereas glycolysis alone is not.
            Members =
            {
                "HIF1A", "CA9", "NDRG1", "BNIP3", "BNIP3L", "VEGFA", "SLC2A1", "LDHA", "PGK1",
                "ALDOA", "ENO1", "PDK1", "P4HA1", "P4HA2", "PLOD2", "LOX", "ADM", "EGLN1", "VHL",
                "HK2", "ANGPTL4", "SERPINE1",
            },
        },

        new ProteinList
        {
            Name = "Glucose and lipid metabolism",
            Category = "Pathways and processes",
            ColorHex = "#2ca02c",
            Visible = false,
            DisplayOnly = true,
            // The metabolic layer of diabetes and metabolic disease - the part bottom-up MS actually measures.
            // Glycogen handling, gluconeogenesis, the pentose phosphate shunt, de novo lipogenesis and lipid
            // droplets, plus the plasma readouts (ADIPOQ, LEP, RBP4, the IGFBPs) that travel with them.
            // Companion to Insulin signaling below: this panel is what the signaling CHANGES, and unlike the
            // signaling it is abundant enough to see.
            Members =
            {
                "SLC2A1", "SLC2A4", "GYS1", "PYGL", "PYGM", "PPP1R3A", "GBE1", "AGL", "PCK1",
                "PCK2", "FBP1", "ALDOB", "G6PD", "PGD", "TKT", "TALDO1", "FASN", "ACACA", "ACLY",
                "SCD", "ELOVL6", "DGAT1", "PLIN2", "PLIN3", "CPT1A", "ADIPOQ", "LEP", "RBP4",
                "IGFBP1", "IGFBP2", "IGFBP3", "SHBG",
            },
        },

        new ProteinList
        {
            Name = "Insulin signaling",
            Category = "Pathways and processes",
            ColorHex = "#9467bd",
            Visible = false,
            DisplayOnly = true,
            // The canonical INSR -> IRS -> PI3K -> AKT -> mTOR cascade and its brakes (PTEN, TSC1/2, FOXO1).
            // 
            // READ THIS ONE DIFFERENTLY. Insulin signaling is regulated by PHOSPHORYLATION, not by abundance:
            // these proteins are present whether or not the pathway is active, so a flat panel is the expected
            // result and says nothing about signaling. Several members (INSR, IRS1, IRS2, MTOR) are also large,
            // low-abundance and poorly covered in a standard bottom-up run, so gaps here are coverage rather than
            // biology.
            // It earns its place on a PHOSPHO-enriched run, where the sites are the measurement - AKT1 S473,
            // IRS1 S307, RPS6 S235/S236, EIF4EBP1 T37/T46. For abundance work, read Glucose and lipid metabolism
            // above instead: it measures what this pathway does rather than what it is made of.
            Members =
            {
                "INSR", "IGF1R", "IRS1", "IRS2", "PIK3R1", "PIK3CA", "PDPK1", "AKT1", "AKT2",
                "GSK3A", "GSK3B", "FOXO1", "FOXO3", "MTOR", "RPTOR", "RICTOR", "TSC1", "TSC2",
                "RHEB", "RPS6KB1", "RPS6", "EIF4EBP1", "EIF4E", "PTEN", "SHC1", "GRB2", "SOS1",
                "PPP2CA", "PRKAA1", "PRKAB1",
            },
        },
    };
}
