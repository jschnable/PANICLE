# PANICLE Documentation

Welcome to the PANICLE documentation! This guide will help you perform genome-wide association studies (GWAS) with PANICLE.

## Documentation Index

### Getting Started
- **[Quick Start Guide](quickstart.md)** - Get up and running in 5 minutes
  - Basic GWAS workflow
  - Common analysis scenarios
  - Troubleshooting tips

### Reference Documentation
- **[API Reference](api_reference.md)** - Complete API documentation
  - GWASPipeline class methods
  - Association testing functions
  - Data loaders and utilities

- **[Output Files](output_files.md)** - Understanding your results
  - File formats and column descriptions
  - How to read and analyze results
  - Working with output in Python/R

### Tutorials & Examples
- **[Sorghum GWAS Tutorial](gwas_sorghum_tutorial.ipynb)** - Interactive Jupyter notebook
  - Complete GWAS workflow with real sorghum data (725 samples, 170K markers)
  - Demonstrates GLM, MLM, FarmCPU, and BLINK methods
  - Visualizations and result interpretation

- **[Multi-Trait eQTL Acceleration Tutorial](eqtl_multitrait_acceleration_tutorial.ipynb)** - Interactive Jupyter notebook
  - Demonstrates eQTL-style many-traits LOCO MLM workflow
  - Shows chromosome-major acceleration (`PANICLE_MLM_LOCO_MULTI`)
  - Includes writing top QTL hits to CSV output

- **[Example Scripts](../examples/README.md)** - Runnable Python scripts
  - 01: Basic GWAS with GLM
  - 02: MLM with population structure
  - 03: Hybrid MLM for increased power
  - 04: Including covariates (flowering time example)
  - 05: Reading and analyzing results

## Quick Links

### Installation
```bash
git clone https://github.com/jschnable/PANICLE.git
cd PANICLE
pip install -e .
```

### Minimal Example
```python
from panicle.pipelines.gwas import GWASPipeline

pipeline = GWASPipeline(output_dir='./results')
pipeline.load_data('phenos.csv', 'genos.vcf.gz')
pipeline.align_samples()
pipeline.compute_population_structure(n_pcs=3, calculate_kinship=True)
pipeline.run_analysis(traits=['Height'], methods=['MLM'])
```

### CLI Quick Start
```bash
python scripts/run_GWAS.py \
  --phenotype examples/example_phenotypes.csv \
  --genotype examples/example_genotypes.vcf.gz \
  --traits PlantHeight \
  --methods GLM \
  --outputdir ./results
```

## Workflow Overview

```
1. Initialize Pipeline
   ↓
2. Load Data (phenotypes, genotypes, optional covariates)
   ↓
3. Align Samples (match individuals across datasets)
   ↓
4. Compute Population Structure (PCs and optional kinship)
   ↓
5. Run Analysis (choose methods: GLM, MLM, MLM_Hybrid, FarmCPU, BLINK, FarmCPUResampling)
   ↓
6. Results saved to output directory
```

## Available Methods

| Method | Speed | Population Control | Best For |
|--------|-------|-------------------|----------|
| **GLM** | Very Fast | PCs (optional) | Quick screening, homogeneous populations |
| **MLM** | Moderate | LOCO (if map) or Kinship + PCs | Diverse populations, related individuals |
| **MLM_Hybrid** | Moderate | LOCO (map required) + PCs | LRT-quality p-values for top hits |
| **FarmCPU** | Slow | Kinship + PCs | Controlling false positives |
| **BLINK** | Slow | Kinship + PCs | Sparse signals with large marker sets |
| **FarmCPUResampling** | Very Slow | Kinship + PCs | Stability via RMIP resampling |

## File Formats

### Input Files

**Phenotype (CSV)**
```csv
ID,Height,FloweringTime
Sample1,1.85,72
Sample2,1.92,68
```

**Genotype (VCF/HapMap/Plink/CSV/TSV)**
- VCF: Standard variant call format (`.vcf` or `.vcf.gz`)
- CSV/TSV: Numeric matrix (rows=samples, cols=markers)
- HapMap: TASSEL HapMap format
- Plink: Binary format (`.bed/.bim/.fam`)

**Covariates (CSV, optional)**
```csv
ID,Field,Year
Sample1,A,2023
Sample2,B,2023
```

**Genetic Map (CSV/TSV, optional but recommended)**
CSV/TSV with `MARKER`, `CHROM`, and `POS` columns (legacy `SNP` and aliases like `Chr`, `Pos` are accepted).
Recommended for numeric genotype matrices and for LOCO-based methods like `MLM_Hybrid`.

### Output Files

- `GWAS_{trait}_all_results.csv` - Full association results
- `GWAS_{trait}_significant.csv` - Significant markers only
- `GWAS_{trait}_{method}_manhattan.png` - Manhattan plot
- `GWAS_{trait}_{method}_qq.png` - QQ plot

See [Output Files](output_files.md) for detailed format specifications.

## Common Questions

### Q: Which method should I use?
- **Start with GLM** for quick screening
- **Use MLM** if you have population structure or relatedness
- **Use Hybrid MLM** when you need accurate p-values for top hits with minimal runtime cost

### Q: How many PCs should I include?
- Typically 3-10 PCs depending on population structure
- Check QQ plots: λ_GC should be close to 1.0
- More PCs if you have strong stratification

### Q: Why are my results files large?
- Full results contain all markers × all methods
- Compress with `gzip`: `gzip GWAS_*_all_results.csv`
- Or only output significant markers: `outputs=['significant_marker_pvalues']`

### Q: Sample IDs don't match?
- IDs must match **exactly** (case-sensitive)
- Check first column of phenotype file is 'ID'
- Run `pipeline.align_samples()` and check the output

## Performance Tips

1. **Use binary genotype formats** - Genotype files are cached automatically (VCF/BCF/PLINK/HapMap/CSV/TSV)
2. **Filter low MAF variants** before analysis
3. **Start with GLM** for initial screening
4. **Use Hybrid MLM** instead of full LRT for production analyses
5. **Run methods in parallel** - pipeline handles this automatically

## Getting Help

- **Check examples**: See `examples/` directory
- **Read docs**: Start with [quickstart.md](quickstart.md)
- **Run tests**: `pytest tests/` (if available)
- **Open an issue**: GitHub issues for bugs/questions

## Citation

If you use PANICLE in your research, please cite:

```
PANICLE: A Python package for efficient multivariate GWAS analysis
[Full citation to be added]
```

## See Also

- [Example Scripts](../examples/README.md) - Runnable example code with test data
- [Main README](../README.md) - Project overview and algorithm descriptions
- [GitHub Repository](https://github.com/jschnable/PANICLE) - Source code and issues

---
