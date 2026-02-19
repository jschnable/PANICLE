# PANICLE API Reference

Complete reference for the PANICLE package.

## Table of Contents
- [GWASPipeline](#gwaspipeline)
- [Association Methods](#association-methods)
- [Data Loaders](#data-loaders)
- [Utility Functions](#utility-functions)

---

## GWASPipeline

The main high-level interface for running GWAS analyses.

### Class: `panicle.pipelines.gwas.GWASPipeline`

```python
GWASPipeline(output_dir='./GWAS_results')
```

**Parameters:**
- `output_dir` (str): Directory where results and plots will be saved. Default: `'./GWAS_results'`

**Attributes:**
- `genotype_matrix` (GenotypeMatrix): Aligned genotype data (n_individuals × n_markers)
- `geno_map` (GenotypeMap): Genetic map with SNP information (ID, chromosome, position)
- `phenotype_df` (DataFrame): Aligned phenotype data with 'ID' column + trait columns
- `covariate_df` (DataFrame): External covariates (if loaded)
- `pcs` (ndarray): Principal components (n_individuals × n_pcs)
- `kinship` (ndarray): Kinship matrix (n_individuals × n_individuals)
- `output_dir` (Path): Output directory path
- `effective_tests_info` (dict): Effective number of independent tests (if computed)

**Example:**
```python
from panicle.pipelines.gwas import GWASPipeline

pipeline = GWASPipeline(output_dir='./my_results')
```

---

### Method: `load_data()`

Load and validate phenotype, genotype, and optional covariate data.

```python
pipeline.load_data(
    phenotype_file,
    genotype_file,
    map_file=None,
    genotype_format=None,
    trait_columns=None,
    covariate_file=None,
    covariate_columns=None,
    covariate_id_column='ID',
    loader_kwargs=None
)
```

**Parameters:**
- `phenotype_file` (str): Path to phenotype CSV file (first column must be 'ID' or individual IDs)
- `genotype_file` (str): Path to genotype file (VCF, HapMap, or Plink format)
- `map_file` (str, optional): Override genetic map file. If None, map is extracted from genotype file
- `genotype_format` (str, optional): Format of genotype file ('vcf', 'hapmap', 'plink'). Auto-detected if None
- `trait_columns` (list, optional): Which phenotype columns to load. If None, loads all numeric columns
- `covariate_file` (str, optional): Path to external covariate CSV file
- `covariate_columns` (list, optional): Which covariate columns to use
- `covariate_id_column` (str): Column name for individual IDs in covariate file. Default: `'ID'`
- `loader_kwargs` (dict, optional): Additional arguments for genotype loader:
  - `compute_effective_tests` (bool): Calculate effective number of independent tests
  - `effective_test_kwargs` (dict): Parameters for effective test calculation

**Side Effects:**
- Sets `self.phenotype_df`, `self.genotype_matrix`, `self.geno_map`
- Sets `self.covariate_df` if covariate file provided
- Sets `self.effective_tests_info` if `compute_effective_tests=True`

**Example:**
```python
pipeline.load_data(
    phenotype_file='phenotypes.csv',
    genotype_file='genotypes.vcf.gz',
    covariate_file='field_locations.csv',
    covariate_columns=['Field', 'Year']
)
```

---

### Method: `align_samples()`

Match and align individuals between phenotype, genotype, and covariate datasets.

```python
pipeline.align_samples()
```

**Parameters:** None

**Side Effects:**
- Updates `self.phenotype_df` with only matched individuals
- Updates `self.genotype_matrix` with only matched individuals
- Updates `self.covariate_df` with only matched individuals (if applicable)
- Prints summary of sample matching

**Raises:**
- `ValueError`: If no common individuals found between datasets

**Example:**
```python
pipeline.align_samples()
# Output:
#    Original phenotypes: 1000
#    Original genotypes: 800
#    Matched Intersection: 750
```

---

### Method: `compute_population_structure()`

Calculate principal components and/or kinship matrix for population structure correction.

```python
pipeline.compute_population_structure(
    n_pcs=3,
    calculate_kinship=True
)
```

**Parameters:**
- `n_pcs` (int): Number of principal components to compute. Default: 3. Set to 0 to skip PCA
- `calculate_kinship` (bool): Whether to calculate kinship matrix. Default: True

**Side Effects:**
- Sets `self.pcs` (n_individuals × n_pcs array)
- Sets `self.pc_names` (list of PC names: ['PC1', 'PC2', ...])
- Sets `self.kinship` (n_individuals × n_individuals matrix) if `calculate_kinship=True`

**Notes:**
- PCs are automatically used as covariates in subsequent analyses
- Kinship matrix is required for MLM, FarmCPU, and BLINK methods
- Uses VanRaden (2008) method for kinship calculation

**Example:**
```python
# Compute 5 PCs and kinship matrix
pipeline.compute_population_structure(n_pcs=5, calculate_kinship=True)

# Skip PCA, only compute kinship
pipeline.compute_population_structure(n_pcs=0, calculate_kinship=True)
```

---

### Method: `run_analysis()`

Run GWAS analysis using specified methods and traits.

```python
pipeline.run_analysis(
    traits=None,
    methods=['GLM', 'MLM'],
    max_iterations=10,
    significance=None,
    alpha=0.05,
    n_eff=None,
    max_genotype_dosage=2.0,
    farmcpu_params=None,
    blink_params=None,
    outputs=['all_marker_pvalues', 'significant_marker_pvalues', 'manhattan', 'qq']
)
```

**Parameters:**
- `traits` (list, optional): Which traits to analyze. If None, analyzes all numeric columns in phenotype data
- `methods` (list): GWAS methods to run. Options:
  - `'GLM'`: General Linear Model (fast, no population structure correction)
  - `'MLM'`: Mixed Linear Model with LOCO kinship (includes automatic LRT refinement for top hits)
  - `'FarmCPU'`: Fixed and random model Circulating Probability Unification
  - `'BLINK'`: Bayesian-information and Linkage-disequilibrium Iteratively Nested Keyway
- `max_iterations` (int): Maximum iterations for iterative methods (FarmCPU, BLINK). Default: 10
- `significance` (float, optional): Fixed p-value threshold. If None, uses Bonferroni correction
- `alpha` (float): Significance level for Bonferroni correction. Default: 0.05
- `n_eff` (int, optional): Effective number of tests for Bonferroni. If None, uses total markers or M_eff
- `max_genotype_dosage` (float): Maximum genotype dosage for MAF calculation. Default: 2.0
- `farmcpu_params` (dict, optional): Parameters for FarmCPU
- `blink_params` (dict, optional): Parameters for BLINK
- `outputs` (list): Which outputs to generate. Options:
  - `'all_marker_pvalues'`: Full results CSV
  - `'significant_marker_pvalues'`: Significant SNPs only CSV
  - `'manhattan'`: Manhattan plot PNG
  - `'qq'`: QQ plot PNG

**Side Effects:**
- Writes result files to `self.output_dir/`
- Prints analysis progress and summary statistics

**Output Files** (per trait):
- `GWAS_{trait}_all_results.csv`: Full association results for all markers
- `GWAS_{trait}_significant.csv`: Only significant markers (p < threshold)
- `GWAS_{trait}_{method}_manhattan.png`: Manhattan plot
- `GWAS_{trait}_{method}_qq.png`: QQ plot
- `GWAS_summary_by_traits_methods.csv`: Summary table (shared across traits)

**Example:**
```python
# Run MLM for multiple traits
pipeline.run_analysis(
    traits=['Height', 'FloweringTime'],
    methods=['MLM']
)

# Run all methods with custom threshold
pipeline.run_analysis(
    traits=['Yield'],
    methods=['GLM', 'MLM', 'FarmCPU', 'BLINK'],
    significance=1e-6,
    outputs=['all_marker_pvalues', 'manhattan']
)
```

---

## Association Methods

Low-level association testing functions. These are called internally by `GWASPipeline` but can also be used directly.

### `PANICLE_GLM()`

General Linear Model association test.

```python
from panicle.association.glm import PANICLE_GLM

results = PANICLE_GLM(
    phe,           # Phenotype array (n × 2): [ID, value]
    geno,          # Genotype matrix (n × m)
    CV=None,       # Covariates (n × p)
    verbose=True
)
```

**Returns:** `AssociationResults` object with attributes:
- `effects`: Effect sizes (m,)
- `se`: Standard errors (m,)
- `pvalues`: P-values (m,)

---

### `PANICLE_MLM()`

Mixed Linear Model association test with global kinship matrix.

```python
from panicle.association.mlm import PANICLE_MLM

results = PANICLE_MLM(
    phe,              # Phenotype array (n × 2)
    geno,             # Genotype matrix (n × m)
    K,                # Kinship matrix (n × n)
    eigenK=None,      # Pre-computed eigendecomposition (optional)
    CV=None,          # Covariates (n × p)
    vc_method='BRENT', # Variance component estimation: 'BRENT' only
    maxLine=1000,     # Batch size
    cpu=1,            # Number of CPU cores
    verbose=True
)
```

**Returns:** `AssociationResults`

---

### `PANICLE_MLM_LOCO()`

MLM with LOCO (Leave-One-Chromosome-Out) kinship and automatic LRT refinement.

```python
from panicle.association.mlm_loco import PANICLE_MLM_LOCO

results = PANICLE_MLM_LOCO(
    phe,                    # Phenotype array (n × 2)
    geno,                   # Genotype matrix (n × m)
    map_data,               # Genetic map (required for LOCO)
    loco_kinship=None,      # Pre-computed LOCO kinship (optional)
    CV=None,                # Covariates (n × p)
    vc_method='BRENT',      # Variance component estimation
    maxLine=1000,           # Batch size
    cpu=1,                  # Number of CPU cores
    lrt_refinement=True,    # Apply LRT refinement to top hits
    screen_threshold=5e-4,  # P-value cutoff for LRT refinement
    lrt_solver='GEMMA',     # Exact LRT solver: 'GEMMA' (fast) or 'BRENT' (legacy)
    lrt_batch_size=2048,    # Candidate chunk size for batched eigenspace transforms
    verbose=True
)
```

**Returns:** `AssociationResults` with LRT-refined p-values for markers passing screen

**Notes:**
- Requires `map_data` with chromosome information for LOCO kinship
- By default, markers with Wald p-value < 5e-4 are re-tested using exact LRT
- `lrt_solver='GEMMA'` uses a derivative-based exact optimizer with automatic fallback
  to bounded Brent on numerical edge cases
- If `loco_kinship` is not provided, it will be computed automatically
- Set `lrt_refinement=False` to skip LRT refinement (faster, Wald p-values only)

---

### `PANICLE_FarmCPU()`

Fixed and random model Circulating Probability Unification.

```python
from panicle.association.farmcpu import PANICLE_FarmCPU

results = PANICLE_FarmCPU(
    phe,                    # Phenotype array (n × 2)
    geno,                   # Genotype matrix (n × m)
    map_data,               # Genetic map (SNP, Chr, Pos)
    CV=None,                # Covariates (n × p)
    maxLoop=10,             # Maximum iterations
    p_threshold=0.05,       # P-value threshold for QTN selection
    QTN_threshold=0.01,     # Threshold for QTN optimization
    bin_size=None,          # Bin sizes for multi-scale binning (default: [5e5, 5e6, 5e7])
    method_bin='static',    # Binning method
    maxLine=5000,           # Batch size
    cpu=1,                  # Number of CPU cores
    reward_method='min',    # Substitution method for pseudo-QTNs
    verbose=True
)
```

**Returns:** `AssociationResults`

---

### `PANICLE_BLINK()`

Bayesian-information and Linkage-disequilibrium Iteratively Nested Keyway.

```python
from panicle.association.blink import PANICLE_BLINK

results = PANICLE_BLINK(
    phe,                    # Phenotype array (n × 2)
    geno,                   # Genotype matrix (n × m)
    map_data,               # Genetic map (SNP, Chr, Pos)
    CV=None,                # Covariates (n × p)
    maxLoop=10,             # Maximum iterations
    converge=1.0,           # Jaccard similarity threshold for convergence
    ld_threshold=0.7,       # LD threshold for pruning
    bic_method='naive',     # BIC evaluation strategy
    method_sub='reward',    # Substitution method for pseudo-QTNs
    p_threshold=None,       # P-value threshold (auto-calculated if None)
    verbose=True
)
```

**Returns:** `AssociationResults`

---

## Data Loaders

Functions for loading different data file formats.

### `load_phenotype_file()`

```python
from panicle.data.loaders import load_phenotype_file

pheno_df = load_phenotype_file(
    filename,
    trait_columns=None  # Load all if None
)
```

**Returns:** DataFrame with 'ID' column + trait columns

---

### `load_genotype_file()`

```python
from panicle.data.loaders import load_genotype_file

geno_matrix, individual_ids, geno_map = load_genotype_file(
    filename,
    file_format='vcf',  # or 'hapmap', 'plink'
    compute_effective_tests=False,
    effective_test_kwargs=None
)
```

**Returns:** Tuple of (GenotypeMatrix, list of IDs, GenotypeMap)

Best practice when aligning samples manually:

```python
from panicle.data.loaders import match_individuals

aligned_pheno, _, sample_indices, _ = match_individuals(
    phenotype_df=phenotype_df,
    individual_ids=individual_ids
)
geno_matrix = geno_matrix.subset_individuals(sample_indices)
```

---

### `load_covariate_file()`

```python
from panicle.data.loaders import load_covariate_file

cov_df = load_covariate_file(
    filename,
    covariate_columns=None,
    id_column='ID'
)
```

**Returns:** DataFrame with covariates

---

## Utility Functions

### Population Structure

```python
from panicle.matrix.pca import PANICLE_PCA
from panicle.matrix.kinship import PANICLE_K_VanRaden

# Compute PCs
pcs = PANICLE_PCA(M, pcs_keep=5, verbose=True)

# Compute kinship matrix
K = PANICLE_K_VanRaden(M, verbose=True)
```

---

### Statistical Functions

```python
from panicle.utils.stats import genomic_inflation_factor, calculate_maf_from_genotypes

# Calculate genomic inflation factor (lambda GC)
lambda_gc = genomic_inflation_factor(pvalues)

# Calculate minor allele frequencies
maf = calculate_maf_from_genotypes(geno_matrix, max_dosage=2.0)
```

---

### Effective Tests

```python
from panicle.utils.effective_tests import estimate_effective_tests_from_genotype

eff_info = estimate_effective_tests_from_genotype(
    geno_matrix,
    geno_map,
    composite_ld_threshold=0.2,
    window_size_bp=1000000
)

# Returns dict with:
# - 'Me': Effective number of independent tests
# - 'total_snps': Total number of SNPs
# - 'ld_blocks': Number of LD blocks detected
```

---

### Visualization

```python
from panicle.visualization.manhattan import PANICLE_Report

# Generate Manhattan and QQ plots
PANICLE_Report(
    results=assoc_results,
    map_data=geno_map,
    output_prefix='my_gwas',
    threshold=0.05/n_markers,
    file_type='png',
    dpi=300
)
```

---

## Data Types

### AssociationResults

Named tuple containing GWAS results:

```python
from panicle.utils.data_types import AssociationResults

results = AssociationResults(
    effects=effect_array,   # Effect sizes
    se=se_array,           # Standard errors
    pvalues=pvalue_array   # P-values
)

# Access attributes
print(results.pvalues)
print(results.effects)
```

### GenotypeMatrix

Wrapper around genotype data (usually memory-mapped):

```python
import numpy as np
from panicle.utils.data_types import GenotypeMatrix

geno = GenotypeMatrix(data_array)

# Attributes
geno.n_individuals  # Number of samples
geno.n_markers      # Number of markers

# Indexing (returns a numpy array)
subset = geno[0:10, :]  # First 10 individuals, all markers

# Best practice: keep GenotypeMatrix when subsetting individuals
subset_geno = geno.subset_individuals(range(10))

# Fetch non-contiguous markers with imputation handled
marker_block = geno.get_columns_imputed([0, 10, 20], dtype=np.float32)
```

---

## Common Workflows

### Workflow 1: Standard GWAS

```python
from panicle.pipelines.gwas import GWASPipeline

pipeline = GWASPipeline(output_dir='./results')
pipeline.load_data('phenos.csv', 'genos.vcf.gz')
pipeline.align_samples()
pipeline.compute_population_structure(n_pcs=3, calculate_kinship=True)
pipeline.run_analysis(traits=['Trait1'], methods=['GLM', 'MLM'])
```

### Workflow 2: Direct Association Testing

```python
from panicle.association.mlm import PANICLE_MLM
from panicle.matrix.kinship import PANICLE_K_VanRaden
import numpy as np

# Prepare data
phe = np.column_stack([np.arange(n), phenotype_values])
K = PANICLE_K_VanRaden(genotype_matrix)

# Run MLM
results = PANICLE_MLM(phe, genotype_matrix, K)

# Get significant SNPs
sig_indices = results.pvalues < 0.05/len(results.pvalues)
```

---

## See Also

- [Quick Start Guide](quickstart.md)
- [Output File Formats](output_files.md)
- [Examples](../examples/)
- [Main README](../README.md) - Algorithm descriptions and benchmarks
