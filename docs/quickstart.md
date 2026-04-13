# PANICLE Quick Start Guide

This guide will get you up and running with PANICLE for genome-wide association studies (GWAS).

## Installation

```bash
# Clone the repository
git clone https://github.com/jschnable/PANICLE.git
cd PANICLE

# Install in development mode
pip install -e .
```

## Basic GWAS Analysis

Here's a minimal example to run a GWAS analysis:

```python
from panicle.pipelines.gwas import GWASPipeline

# 1. Initialize the pipeline
pipeline = GWASPipeline(output_dir='./my_gwas_results')

# 2. Load your data
pipeline.load_data(
    phenotype_file='my_phenotypes.csv',
    genotype_file='my_genotypes.vcf.gz',
    # map_file='my_genotypes.map'  # Optional unless format lacks positions
)

# 3. Align samples between phenotype and genotype data
pipeline.align_samples()

# 4. Compute population structure (PCs and optional kinship)
pipeline.compute_population_structure(
    n_pcs=3,              # Number of principal components
    calculate_kinship=True # Needed for MLM (auto-computed if omitted, but slower)
)

# 5. Run GWAS analysis
pipeline.run_analysis(
    traits=['Height', 'FloweringTime'],
    methods=['GLM', 'MLM']
)

# Results are automatically saved to ./my_gwas_results/
```

## File Formats

### Phenotype File Format
CSV file with an individual ID column and numeric trait columns:

```csv
ID,Height,FloweringTime,YieldTonPerHa
Ind001,1.85,72,8.5
Ind002,1.92,68,9.2
Ind003,1.78,75,7.8
```

**ID Column Auto-Detection**: PANICLE automatically detects common ID column names including: `ID`, `IID`, `Sample`, `Taxa`, `Genotype`, `Accession`. If none are found, the first column is used. The detected column is printed during data loading.

### Genotype File Formats
Supported formats:
- **VCF/VCF.GZ**: Standard variant call format (recommended)
- **CSV/TSV**: Numeric matrix (rows=samples, cols=markers)
- **HapMap**: TASSEL HapMap format
- **Plink**: Binary plink format (.bed/.bim/.fam)

### Genetic Map File Format (Optional but recommended)
CSV/TSV with `MARKER`, `CHROM`, and `POS` columns (legacy `SNP` and aliases like `Chr`, `Pos` are accepted).
Recommended for numeric genotype matrices and for LOCO-based methods like `MLM`.

## Understanding the Output

After running the analysis, your output directory will contain:

```
my_gwas_results/
├── GWAS_Height_all_results.csv        # Full results for Height
├── GWAS_Height_significant.csv           # Only significant markers
├── GWAS_Height_GLM_manhattan.png         # Manhattan plot
├── GWAS_Height_GLM_qq.png                # QQ plot
├── GWAS_FloweringTime_all_results.csv # Full results for FloweringTime
└── GWAS_summary_by_traits_methods.csv    # Summary statistics
```

**Note:** Full results files are written as plain CSV by default. You can gzip them yourself if disk space is a concern.

### Reading Your Results

```python
import pandas as pd

# Load full results
results = pd.read_csv('my_gwas_results/GWAS_Height_all_results.csv')

# Results contain:
# - MARKER: Marker ID (legacy SNP alias also present)
# - CHROM: Chromosome
# - POS: Position
# - MAF: Minor allele frequency
# - GLM_P: P-values from GLM
# - GLM_Effect: Effect sizes from GLM
# - MLM_P: P-values from MLM (if you ran it)
# - MLM_Effect: Effect sizes from MLM

# Get significant markers (p < 0.05/n_markers Bonferroni)
sig_markers = results[results['GLM_P'] < 0.05 / len(results)]
print(f"Found {len(sig_markers)} significant markers")

# Top 10 most significant
top_markers = results.nsmallest(10, 'GLM_P')
print(top_markers[['MARKER', 'CHROM', 'POS', 'GLM_P', 'GLM_Effect']])
```

## Common Analysis Scenarios

### Scenario 1: Quick GLM Analysis (No Population Structure)

```python
pipeline = GWASPipeline(output_dir='./quick_analysis')
pipeline.load_data(phenotype_file='phenos.csv', genotype_file='genos.vcf.gz')
pipeline.align_samples()

# Run GLM only (faster, no kinship needed)
pipeline.run_analysis(
    traits=['MyTrait'],
    methods=['GLM']
)
```

### Scenario 2: MLM with Population Structure Correction

```python
pipeline = GWASPipeline(output_dir='./mlm_analysis')
pipeline.load_data(phenotype_file='phenos.csv', genotype_file='genos.vcf.gz')
pipeline.align_samples()

# Compute population structure
pipeline.compute_population_structure(
    n_pcs=5,                    # Use 5 PCs as covariates
    calculate_kinship=True      # Needed for MLM (auto-computed if omitted)
)

# Run MLM (accounts for population structure)
pipeline.run_analysis(
    traits=['MyTrait'],
    methods=['MLM']
)
```

### Scenario 3: Using External Covariates

```python
pipeline = GWASPipeline(output_dir='./covariate_analysis')

# Load data with external covariates
# Example: controlling for flowering time when analyzing plant height
pipeline.load_data(
    phenotype_file='phenos.csv',
    genotype_file='genos.vcf.gz',
    covariate_file='covariates.csv',      # CSV with ID, Cov1, Cov2, ...
    covariate_columns=['DaysToFlower']    # Which columns to use
)

pipeline.align_samples()
pipeline.compute_population_structure(n_pcs=3, calculate_kinship=True)  # kinship for MLM

# PCs and external covariates are automatically combined
pipeline.run_analysis(
    traits=['PlantHeight'],
    methods=['MLM']
)
```

### Scenario 4: Multiple Methods Comparison

```python
pipeline = GWASPipeline(output_dir='./method_comparison')
pipeline.load_data(phenotype_file='phenos.csv', genotype_file='genos.vcf.gz')
pipeline.align_samples()
pipeline.compute_population_structure(n_pcs=3, calculate_kinship=True)  # kinship for MLM

# Run multiple methods at once
pipeline.run_analysis(
    traits=['MyTrait'],
    methods=['GLM', 'MLM', 'FarmCPU', 'BLINK']
    # Add 'FarmCPUResampling' if needed (resampling is slow)
)

# All results are in the same all_results.csv file
results = pd.read_csv('method_comparison/GWAS_MyTrait_all_results.csv')
# Contains GLM_P, MLM_P, FarmCPU_P, BLINK_P columns
```

## Advanced Options

### Custom Significance Threshold

```python
# Use a specific p-value threshold instead of Bonferroni
pipeline.run_analysis(
    traits=['MyTrait'],
    methods=['MLM'],
    significance=1e-5  # Fixed threshold
)
```

### Control Output Files

```python
# Choose which outputs to generate
pipeline.run_analysis(
    traits=['MyTrait'],
    methods=['MLM'],
    outputs=['all_marker_pvalues', 'manhattan', 'qq']
    # Options: 'all_marker_pvalues', 'significant_marker_pvalues', 'manhattan', 'qq'
)
```

### Using Effective Tests for Multiple Testing Correction

```python
# Automatically calculate effective number of independent tests
pipeline.load_data(
    phenotype_file='phenos.csv',
    genotype_file='genos.vcf.gz',
    loader_kwargs={
        'compute_effective_tests': True  # Calculates M_eff (Li et al. 2012)
    }
)

# The pipeline will use M_eff instead of total markers for Bonferroni
pipeline.run_analysis(
    traits=['MyTrait'],
    methods=['MLM'],
)
```

## Command-Line Interface

PANICLE includes a CLI script for running GWAS analyses without writing Python code.

### Basic Usage

```bash
python scripts/run_GWAS.py \
  --phenotype phenos.csv \
  --genotype genos.vcf.gz \
  --traits PlantHeight,DaysToFlower \
  --methods GLM,MLM \
  --outputdir ./results
```

### Common CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--phenotype`, `-p` | Phenotype CSV/TSV file (required) | — |
| `--genotype`, `-g` | Genotype file (required) | — |
| `--traits` | Comma-separated trait names (case-sensitive) | All numeric columns |
| `--methods` | Comma-separated methods: GLM, MLM, BAYESLOCO, FarmCPU, BLINK, FarmCPUResampling | GLM,MLM,FarmCPU |
| `--n-pcs` | Number of principal components | 3 |
| `--outputdir`, `-o` | Output directory | ./GWAS_results |
| `--format`, `-f` | Genotype format (auto-detected if omitted) | Auto |
| `--compute-effective-tests` | Use effective test count for Bonferroni | Off |
| `--include-standard-errors` | Include `{METHOD}_SE` columns in merged CSV output files | Off |
| `--min-maf` | Minimum minor allele frequency filter | 0.0 |
| `--max-missing` | Maximum missing data proportion | 1.0 |
| `--significance` | Fixed p-value threshold (overrides Bonferroni) | — |

### Examples

```bash
# Quick GLM scan on a single trait
python scripts/run_GWAS.py -p phenos.csv -g genos.vcf.gz \
  --traits PlantHeight --methods GLM

# Full analysis with effective tests correction
python scripts/run_GWAS.py -p phenos.csv -g genos.vcf.gz \
  --methods GLM,MLM,FarmCPU,BLINK \
  --compute-effective-tests --n-pcs 5

# Only generate plots (no CSV output)
python scripts/run_GWAS.py -p phenos.csv -g genos.vcf.gz \
  --methods MLM --outputs manhattan qq
```

Run `python scripts/run_GWAS.py --help` for the full list of options.

### Pre-caching Genotypes

For repeated analyses on the same genotype file, you can pre-convert it to PANICLE's
binary cache format for faster loading (~26x speedup on subsequent runs):

```bash
panicle-cache-genotype -i genotypes.vcf.gz -o genotypes_cached
```

**Note:** The `panicle-cache-genotype` command is only available after installing the
package with `pip install -e .` (or `pip install panicle`). If you skip installation,
genotype caching still happens automatically on first load — this tool just lets you
do it ahead of time.

## Troubleshooting

### Problem: "Sample mismatch" or "No common individuals"
**Solution:** Check that individual IDs match exactly between phenotype and genotype files (case-sensitive).

### Problem: "Kinship matrix missing"
**Solution:** Run `pipeline.compute_population_structure(calculate_kinship=True)` for MLM. If omitted, MLM will auto-compute kinship internally, but precomputing avoids redundant work when analyzing multiple traits. FarmCPU and BLINK do not use kinship. For LOCO methods, ensure a map is available (VCF/PLINK/HapMap or `map_file`).

### Problem: Analysis is very slow
**Solution:**
- Use GLM for initial screening (much faster)
- Avoid FarmCPUResampling unless you need RMIP stability
- Consider filtering low MAF variants before analysis

### Problem: Many warnings about VCF parsing
**Solution:** These are usually harmless warnings from htslib about VCF metadata. Your analysis results are still valid.

## Next Steps

- **Try the [Sorghum GWAS Tutorial](gwas_sorghum_tutorial.ipynb)**: Interactive Jupyter notebook with real data
- **See [examples/](../examples/)**: More detailed example scripts with test data
- **See [api_reference.md](api_reference.md)**: Complete API documentation
- **See [output_files.md](output_files.md)**: Detailed output format specifications
- **See [README.md](../README.md)**: Algorithm descriptions and benchmarks

## Getting Help

- Check the documentation in `docs/`
- Run example scripts in `examples/`
- Open an issue on GitHub
