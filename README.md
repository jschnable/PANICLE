# PANICLE: Python Algorithms for Nucleotide-phenotype Inference and Chromosome-wide Locus Evaluation

[![PyPI version](https://badge.fury.io/py/panicle.svg)](https://pypi.org/project/panicle/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/jschnable/PANICLE/actions/workflows/publish.yml/badge.svg)](https://github.com/jschnable/PANICLE/actions/workflows/publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PANICLE is a **Python package for Genome Wide Association Studies (GWAS)**. It implements GLM, MLM, FarmCPU, and BLINK. PANICLE seeks to achieve speeds comparable or better to other implementations while supporting multiple input data formats, providing multiple quality of life features (native effect marker number testing, leave one chromosome out MLM, calculation of resampling model inclusion probabilities, etc), and allowing modern GWAS algorithms to be natively integrated into python-based data analysis pipelines and ecosystems.

## Key Features

*   **Multiple Algorithms**: GLM, MLM, FarmCPU, BLINK
*   **Supported Genotype Formats**: VCF/BCF, PLINK, HapMap, CSV/TSV with optional caching of genotype data in binary during initial run (speeds future data loading dramatically).
*   **Robustness**: Graceful handling of missing data.

## Installation

Requires Python 3.9+.

```bash
pip install panicle
```

*With optional dependencies for PLINK format support:*
```bash
pip install panicle[plink]
```

*Or install all optional dependencies:*
```bash
pip install panicle[all]
```

### Development Installation

To install from source for development:
```bash
git clone https://github.com/jschnable/PANICLE.git
cd PANICLE
pip install -e .[all]
```

### Dependencies

**Core dependencies** (installed automatically):
- `numpy` ≥1.19.0
- `scipy` ≥1.6.0
- `pandas` ≥1.2.0
- `h5py` ≥3.0.0 (HDF5 support)
- `matplotlib` ≥3.3.0 (plotting)
- `numba` ≥0.50.0 (JIT compilation for performance)
- `cyvcf2` ≥0.30.0 (fast VCF/BCF parsing)

**Optional dependencies**:
- `bed-reader` ≥1.0.0 — PLINK .bed/.bim/.fam format support (`pip install panicle[plink]`)
- `joblib` ≥1.0.0 — Parallel processing for LOCO methods (`pip install panicle[parallel]`)

## Quick Start (Python API)

```python
from panicle import PANICLE

# Run GWAS with a single function call
results = PANICLE(
    phe="data/phenotype.csv",
    geno="data/genotypes.vcf.gz",
    map_data="data/map.csv",       # optional for VCF (map is extracted automatically)
    method=["GLM", "MLM", "FarmCPU"]
)

# Results are also saved to CSV files automatically
```

For more control over data loading, use the loader functions directly:

```python
from panicle import load_genotype_vcf, load_phenotype_file, match_individuals
from panicle import PANICLE_MLM, PANICLE_K_VanRaden, PANICLE_PCA

# Load data
genotype, sample_ids, snp_map = load_genotype_vcf("data/genotypes.vcf.gz")
phenotypes = load_phenotype_file("data/phenotype.csv")

# Align samples and loop over traits
for trait in phenotypes.columns[1:]:
    phe_trait = phenotypes[["ID", trait]].dropna()
    phe_aligned, _, geno_idx, _ = match_individuals(phe_trait, sample_ids)

    geno_subset = genotype.subset_individuals(geno_idx)
    phe_array = phe_aligned.values  # (n, 2) array: [ID, value]

    K = PANICLE_K_VanRaden(geno_subset)
    results = PANICLE_MLM(phe=phe_array, geno=geno_subset, K=K)
    df = results.to_dataframe()
    print(f"{trait}: {(df['P'] < 5e-8).sum()} significant markers")
```

## CLI Usage (Quick Start)

The `run_GWAS.py` script provides a command-line interface for batch processing.

```bash
python scripts/run_GWAS.py \
  --phenotype data/phenotype.csv \
  --genotype data/genotypes.vcf.gz \
  --traits Trait1,Trait2 \
  --methods GLM,MLM,FarmCPU,BLINK \
  --n-pcs 5 \
  --compute-effective-tests \
  --outputs manhattan qq significant_marker_pvalues \
  --outputdir ./results
```

For a small demo dataset included in the repo, see `examples/EXAMPLE_DATA.md` and try:
```bash
python scripts/run_GWAS.py \
  --phenotype examples/example_phenotypes.csv \
  --genotype examples/example_genotypes.vcf.gz \
  --traits PlantHeight \
  --methods GLM \
  --outputdir ./results
```

### Parameters

| Argument | Description | Default |
| :--- | :--- | :--- |
| **`--phenotype`** | Path to phenotype CSV/TSV (must contain ID column). | **Required** |
| **`--phenotype-id-column`** | ID column name in phenotype file. | ID |
| **`--genotype`** | Path to genotype VCF/BCF/CSV. | **Required** |
| **`--map`** | Optional map file (SNP, CHROM, POS). Recommended for numeric CSV/TSV and LOCO methods. | None |
| **`--format`** | Genotype format override: `vcf`, `plink`, `hapmap`, `csv`, `tsv`, `numeric`. | Auto |
| **`--traits`** | Comma-separated list of columns to analyze. | All numeric |
| **`--methods`** | GWAS methods: `GLM`, `MLM`, `FarmCPU`, `BLINK`, `FarmCPUResampling`. | GLM,MLM,FarmCPU,BLINK |
| **`--n-pcs`** | Number of Principal Components for population structure. | 3 |
| **`--compute-effective-tests`** | Calculate Effective SNP Number (Me) and use it for Bonferroni correction. | False |
| **`--alpha`** | Significance level (e.g., 0.05). Threshold = `alpha / Me` (or `M`). | 0.05 |
| **`--significance`** | Fixed p-value threshold (overrides Bonferroni). | None |
| **`--n-eff`** | Effective number of markers (overrides Me). | None |
| **`--covariates`** | External covariate file. | None |
| **`--covariate-columns`** | Comma-separated covariate column names. | All except ID |
| **`--covariate-id-column`** | ID column name in covariate file. | ID |
| **`--max-iterations`** | Max iterations for FarmCPU/BLINK. | 10 |
| **`--max-genotype-dosage`** | Max dosage (e.g., 2 for diploid). | 2.0 |
| **`--outputdir`** | Output directory. | ./GWAS_results |
| **`--outputs`** | Outputs to generate: `all_marker_pvalues`, `significant_marker_pvalues`, `manhattan`, `qq` (see [docs/output_files.md](docs/output_files.md)). | All |

Other useful filters:
- `--max-missing` (default 1.0), `--min-maf` (default 0.0)
- `--drop-monomorphic` / `--keep-monomorphic`
- `--snps-only`, `--no-split-multiallelic`

## Python API Usage

Integrate PANICLE into scripts or Jupyter Notebooks via the `GWASPipeline` class.

```python
from panicle.pipelines.gwas import GWASPipeline

# 1. Initialize
pipeline = GWASPipeline(output_dir="./results")

# 2. Load Data (Auto-caches for speed)
pipeline.load_data(
    phenotype_file="data/phenotype.csv",
    genotype_file="data/genotype.vcf.gz",
    map_file="data/genotype.map",  # Optional unless format lacks positions
    trait_columns=["Height", "Yield"],
    loader_kwargs={'compute_effective_tests': True}  # Enable Me calculation
)

# 3. Pre-process
pipeline.align_samples()
pipeline.compute_population_structure(n_pcs=5)

# 4. Run Analysis (runs in parallel by default)
pipeline.run_analysis(
    methods=['GLM', 'MLM', 'FARMCPU', 'BLINK'],
    alpha=0.05
)
```

## Input Formats

### Phenotype & Covariates
CSV or TSV files with an **ID column** and numeric columns for traits/covariates. PANICLE auto-detects ID columns named `ID`, `id`, `IID`, `sample`, `Sample`, `Taxa`, `taxa`, `Genotype`, `genotype`, `Accession`, `accession` (if multiple, it uses the leftmost). If none match, it uses the first column. Use `--phenotype-id-column` (or `--covariate-id-column`) to specify a custom ID column name.

### Genotype
*   **VCF/BCF**: `.vcf`, `.vcf.gz`, `.bcf` (Preferred for performance).
*   **CSV/TSV**: Numeric matrix (rows=samples, cols=markers) + genetic map file with `SNP`, `CHROM`, and `POS` columns (case-insensitive aliases like `Chr`, `Pos` are accepted).
*   **PLINK**: `.bed` + `.bim` + `.fam`.
*   **HapMap**: `.hmp.txt`.

**Performance notes:** VCF is typically the slowest format on the first run, but PANICLE caches parsed marker data so subsequent loads are competitive with other formats. BCF is roughly ~2x faster than VCF on the first run, and PLINK/bed is roughly ~4x faster than VCF on the first run (exact speedups depend on marker count, sample size, and hardware).

## Tips

1.  **Effective Tests**: Use `--compute-effective-tests` to calculate a less stringent, more accurate Bonferroni threshold based on marker linkage (`Me`).
2.  **Genotype Subsetting**: If you align or filter samples manually, use `GenotypeMatrix.subset_individuals(...)` to preserve pre-imputed fast paths.

## Documentation & Examples

### Documentation

Detailed documentation is available in the [`docs/`](docs/) directory:

- **[Quick Start Guide](docs/quickstart.md)** - Get up and running in 5 minutes
- **[API Reference](docs/api_reference.md)** - Complete API documentation for all functions and classes
- **[Output Files](docs/output_files.md)** - Understanding result file formats and columns

### Interactive Tutorial

- **[Sorghum GWAS Tutorial](docs/sorghum_gwas_tutorial.ipynb)** - Jupyter notebook with complete GWAS workflow

### Example Scripts

The [`examples/`](examples/) directory contains runnable example scripts with included test data:

| Example | Description |
|---------|-------------|
| [01_basic_gwas.py](examples/01_basic_gwas.py) | Simplest GWAS with GLM |
| [02_mlm_with_structure.py](examples/02_mlm_with_structure.py) | MLM with population structure correction |
| [03_hybrid_mlm.py](examples/03_hybrid_mlm.py) | Hybrid MLM (Wald + LRT) for accurate p-values |
| [04_with_covariates.py](examples/04_with_covariates.py) | Including external covariates |
| [05_reading_results.py](examples/05_reading_results.py) | Analyzing and visualizing results |

Run any example:
```bash
cd examples
python 01_basic_gwas.py
```

## Algorithms

### GLM

General Linear Model for fast single-marker association testing. Uses the Frisch-Waugh-Lovell (FWL) theorem combined with QR decomposition for computational efficiency. The algorithm residualizes the phenotype and genotypes against the covariate matrix (PCs + intercept), then computes per-marker regression statistics in vectorized batches. GLM is the fastest GWAS method but may generate overly optimistic significance values.

### MLM

Mixed Linear Model accounting for population structure and cryptic relatedness via a kinship matrix.

**Key design decisions:**
- **LOCO by default**: Leave-One-Chromosome-Out kinship avoids proximal contamination (testing a marker against a kinship matrix that includes that marker), increasing power to detect true associations.
- **Eigenspace transformation**: Data is transformed via eigendecomposition of the kinship matrix, converting the correlated mixed model into an equivalent weighted least squares problem.
- **REML variance components**: Heritability (h²) is estimated using Brent's method optimization of the REML likelihood.

**MLM_Hybrid** extends MLM with a two-phase approach: (1) fast Wald screening of all markers, (2) Likelihood Ratio Test (LRT) refinement for markers passing a screening threshold (default p < 1e-4). LRT re-estimates variance components per marker, providing more accurate p-values but would be slow to run for many millions of markers.

### FarmCPU

Fixed and random model Circulating Probability Unification. FarmCPU iteratively alternates between a fixed-effect model (GLM) and random-effect model to identify associated markers while controlling for polygenic background. FarmCPU can often detect more independent loci linked to variation in the same trait since it controls for the impact of each significant signal when determining the significance of other signals.

This means FarmCPU will NOT give the "towers" most of us expect from classical manhattan plots which are the result of many different markers in LD with the same causal variant. Instead it will identify only one marker since once the effect of this marker is controlled for the significance of any markers in LD with that marker decline to baseline levels. 

FarmCPU Citation: Liu, X., Huang, M., Fan, B., Buckler, E. S., & Zhang, Z. (2016). Iterative usage of fixed and random effect models for powerful and efficient genome-wide association studies. _PLoS genetics_, _12_(2), e1005767.

### BLINK

Bayesian-information and Linkage-disequilibrium Iteratively Nested Keyway. BLINK builds on FarmCPU's iterative framework but uses BIC-based model selection to optimize the pseudo-QTN set. Like FarmCPU, BLINK can often identify larger numbers of independent causal variants from the same phenotype/genotype set than GLM or MLM. Like FarmCPU, it will typically identify only one significant marker per causal variant and lacks the expected "towers" in manhattan plots caused by groups of markers that are all in LD. 

Blink Citation: Huang, M., Liu, X., Zhou, Y., Summers, R. M., & Zhang, Z. (2019). BLINK: a package for the next level of genome-wide association studies with both individuals and markers in the millions. _Gigascience_, _8_(2), giy154.

### Effective Marker Number Estimates

PANICLE includes a python-based based implementation of the effective marker number estimation method implemented in GEC. Accounts for linkage disequilibrium between markers to provide a less conservative multiple testing correction than standard Bonferroni.

GEC citation: Li MX, Yeung JM, Cherny SS, Sham PC. Evaluating the effective numbers of independent tests and significant p-value thresholds in commercial genotyping arrays and public imputation reference datasets. Hum Genet. 2012 May;131(5):747-56.

## Benchmarks

Benchmarks based on traits measured from 862 samples, each scored for 5,751,024 markers and run on an Apple M4 CPU (cached VCF).

### Data Loading

| Step                | Time    |
|---------------------|---------|
| Genotype loading    | 1.34s   |
| Phenotype loading   | 0.005s  |
| Sample alignment    | 11.12s  |
| PCA (3 components)  | 2.08s   |
| **Total**           | **14.55s** |

*Note: First run with a given genetic marker file requires substantial time for parsing (≈9 minutes for 5M markers scored for 1000 individuals); subsequent runs use binary cache and load in seconds.*

### Analysis Times (5.75M markers, 862 samples; excludes data loading/result writing)

| Method      | Time    | Notes                                     |
|-------------|---------|-------------------------------------------|
| GLM         | 8.94s   | ~643K markers/second                      |
| MLM         | 28.18s  | LOCO kinship precompute +15.95s = 44.13s total |
| MLM_Hybrid  | 30.67s  | Includes LOCO kinship reuse               |
| FarmCPU     | 41.90s  | 10 max iterations                         |
| BLINK       | 60.81s  | 10 max iterations                         |

### Scaling by Marker Count (862 samples; includes cached load, alignment, PCA, kinship where relevant)

| Markers    | GLM     | MLM     | MLM_Hybrid | FarmCPU  | BLINK   |
|------------|---------|---------|------------|----------|---------|
| 50,000     | 12.09s  | 12.86s  | 12.37s     | 12.29s   | 12.42s  |
| 500,000    | 12.78s  | 15.72s  | 15.03s     | 14.66s   | 15.74s  |
| 5,000,000  | 19.49s  | 47.12s  | 45.63s     | 46.37s   | 58.60s  |

## License

Distributed under the MIT license. See [LICENSE](LICENSE).

---
**Disclaimer:** This is an independent Python implementation of algorithms developed by others. Any errors are mine alone. -James
