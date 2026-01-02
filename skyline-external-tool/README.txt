Skyline-PRISM External Tool
===========================

PRISM (Proteomics Reference-Integrated Signal Modeling) provides:
- RT-aware normalization (RT-LOWESS, median)
- ComBat batch correction
- Tukey median polish for robust quantification
- Protein parsimony with flexible shared peptide handling

Requirements
------------
- Python 3.10 or later
- skyline-prism package (auto-installed)

Usage
-----
1. Install this tool via Skyline: Tools > External Tools > Add from File
2. Open a Skyline document with quantitative data
3. Run: Tools > PRISM > Run PRISM Analysis
4. Configure parameters in the dialog
5. View results in the PRISM Viewer

The tool will:
- Export your data using the Skyline-PRISM report
- Export sample metadata using the Replicates report
- Run the PRISM normalization and batch correction pipeline
- Launch an interactive viewer to explore results

Output Files
------------
Results are saved to a 'prism-output' subdirectory next to your Skyline document:
- corrected_peptides.parquet - Normalized peptide abundances
- corrected_proteins.parquet - Normalized protein abundances
- qc_report.html - QC plots and diagnostics
- metadata.json - Processing provenance

Documentation
-------------
https://github.com/maccoss/skyline-prism

MacCoss Lab, University of Washington
