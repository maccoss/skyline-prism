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
            ColorHex = "#7f7f7f",
            Visible = false,
            ShowLabels = true,
            // Listed by ACCESSION, not gene symbol, and that is not a style choice. These are non-human proteins:
            // 'ALB' for bovine serum albumin would match human albumin - the most abundant protein in a plasma
            // sample - and 'TRYP' would match human trypsin-1. Accessions cannot collide that way.
            // A starting set, not an exhaustive one. Import the cRAP FASTA your search actually uses to extend it.
            Members =
            {
                "P00761", "TRYP_PIG", "P00760", "P02769", "P00698", "P02701", "P22629", "P02662",
                "P00924", "ENO1_YEAST",
                "P02663", "P02666", "P02668", "P00921",
            },
        },

        new ProteinList
        {
            Name = "Glomerulus",
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
            Name = "Mitochondrial mass",
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
    };
}
