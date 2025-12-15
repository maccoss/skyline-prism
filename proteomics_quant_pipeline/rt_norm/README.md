# RT-Norm: RT-Aware Normalization for Proteomics

RT-Norm is a Python package for retention time-aware normalization of LC-MS proteomics data, with robust protein quantification using Tukey median polish.

## Key Features

- **Reference-anchored RT correction**: Uses inter-experiment reference samples to learn and correct RT-dependent technical variation
- **Dual-control validation**: Uses intra-experiment pool samples to validate that corrections work without overfitting
- **Robust protein quantification**: Tukey median polish handles outlier peptides without pre-identification
- **Flexible protein inference**: Multiple strategies for handling shared peptides (all_groups, unique_only, razor)
- **Multiple rollup methods**: median_polish, Top-N, iBAQ, maxLFQ, directLFQ

## Installation

```bash
pip install rt-norm
```

Or for development:

```bash
git clone https://github.com/maccosslab/rt-norm
cd rt-norm
pip install -e ".[dev,viz]"
```

## Quick Start

### 1. Merge Skyline reports

```bash
rt-norm merge report1.csv report2.csv -o unified_data.parquet -m sample_metadata.tsv
```

### 2. Run normalization

```bash
rt-norm normalize -i unified_data.parquet -o normalized_peptides.parquet -c config.yaml
```

### 3. Roll up to proteins

```bash
rt-norm rollup -i normalized_peptides.parquet -o proteins.parquet -g protein_groups.tsv
```

### 4. Validate results

```bash
rt-norm validate --before unified_data.parquet --after normalized_peptides.parquet --report qc_report.html
```

## Experimental Design Requirements

For optimal results, include these controls in each batch:

| Control Type | Description | Replicates/Batch |
|--------------|-------------|------------------|
| Inter-experiment reference | Commercial pool (e.g., Golden West CSF) | 2-4 |
| Intra-experiment pool | Pooled experimental samples | 2-4 |

Plus internal QCs in all samples:
- **Protein QC**: Yeast enolase (16 ng/μg sample)
- **Peptide QC**: PRTC (30-150 fmol/injection)

## Configuration

See `config_template.yaml` for all options. Key settings:

```yaml
rt_correction:
  enabled: true
  method: spline
  spline_df: 5
  per_batch: true

global_normalization:
  method: median  # or vsn

parsimony:
  shared_peptide_handling: all_groups  # recommended

protein_rollup:
  method: median_polish
  min_peptides: 3
```

## Shared Peptide Handling

Three strategies available:

- **`all_groups`** (default, recommended): Apply shared peptides to ALL protein groups. Acknowledges proteoform complexity; avoids assumptions based on FASTA annotations.

- **`unique_only`**: Only use peptides unique to a single protein group. Most conservative.

- **`razor`**: Assign shared peptides to group with most peptides (MaxQuant-style).

## Rollup Methods

| Method | Description |
|--------|-------------|
| `median_polish` | Tukey median polish - robust to outliers (default) |
| `topn` | Average of top N most intense peptides |
| `ibaq` | Sum intensity / theoretical peptide count |
| `maxlfq` | Maximum peptide ratio extraction |
| `directlfq` | Direct LFQ estimation |

## Citation

If you use RT-Norm, please cite:

Tsantilas KA et al. "A framework for quality control in quantitative proteomics." 
J Proteome Res. 2024. DOI: 10.1021/acs.jproteome.4c00363

## License

MIT License - see LICENSE file.
