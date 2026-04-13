# Changelog

All notable changes to PANICLE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-04-13

### Added
- Optional CLI and pipeline support to export GWAS standard errors.
- Grouped multi-trait GLM execution paths with pipeline auto-dispatch.
- eQTL multi-trait acceleration tutorial documentation.
- Test coverage for effective tests, stats utilities, visualization, CLI utilities, and expanded GWAS pipeline paths.

### Changed
- Optimized LOCO MLM multi-trait execution and genotype alignment/subsetting paths.
- Optimized effective marker number calculations and added CPU control wiring.
- Improved PCA and kinship-related data flow for large analyses.

### Fixed
- Corrected phenotype parsing and BLINK option forwarding behavior.
- Aligned reported genomic inflation lambda with the QQ plot lambda computation.

## [0.1.0] - 2026-01-25

### Changed
- **Package rebranded from pyMVP to PANICLE** (Python Algorithms for Nucleotide-phenotype Inference and Chromosome-wide Locus Evaluation)
- All `MVP_*` functions renamed to `PANICLE_*` (e.g., `MVP_GLM` → `PANICLE_GLM`)
- Package name changed from `pymvp` to `panicle` in imports
- Cache file extensions changed from `.pymvp.*` to `.panicle.*`
- CLI command renamed from `pymvp-cache-genotype` to `panicle-cache-genotype`

### Added
- Initial public release of PANICLE
- Core GWAS methods: GLM, MLM, FarmCPU, BLINK
- **Hybrid MLM method** combining Wald test screening with LRT refinement
  - 2-3% runtime overhead vs standard MLM
  - Orders of magnitude p-value improvement for significant associations
- High-level `GWASPipeline` API for streamlined workflows
- Multiple genotype format support:
  - VCF/BCF with automatic binary caching (~26x faster loading)
  - PLINK binary format (.bed/.bim/.fam)
  - HapMap format
  - CSV/TSV matrices
- Automatic population structure correction:
  - Step-wise PCA calculation
  - VanRaden kinship matrix computation
- Effective tests calculation for accurate Bonferroni correction
- Parallel execution of multiple GWAS methods
- Comprehensive visualization:
  - Manhattan plots with decimated rendering
  - QQ plots with genomic inflation factor
  - Results comparison plots
- Command-line interface via `scripts/run_GWAS.py`
- Binary genotype caching tool: `panicle-cache-genotype`

### Documentation
- Complete API reference for all classes and functions
- Quick start guide with 6 common scenarios
- Output file format specifications
- 5 runnable example scripts demonstrating different workflows
- Interactive Jupyter notebook for Hybrid MLM demonstration
- PDF report generator for publication-ready results
- Detailed algorithm documentation for Hybrid MLM method

### Performance
- Vectorized VCF loading with cyvcf2 optimization
- Binary caching for instant subsequent loads (~1.5s)
- Numba JIT acceleration for computationally intensive operations
- 2-4x faster than R-based rMVP implementation

### Dependencies
- Core: numpy, scipy, pandas, h5py, tables, statsmodels, scikit-learn, matplotlib, seaborn, tqdm, numba
- Optional: cyvcf2 (VCF support), bed-reader (PLINK support)

[0.1.0]: https://github.com/jschnable/PANICLE/releases/tag/v0.1.0
[0.3.1]: https://github.com/jschnable/PANICLE/releases/tag/v0.3.1
