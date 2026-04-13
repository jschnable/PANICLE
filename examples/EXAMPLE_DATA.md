# Example Data for PANICLE

This directory contains real sorghum data curated for testing and demonstration purposes.

## Files

### `example_phenotypes.csv`
- **Description**: Plant height measurements from field trials
- **Source**: Subset of SbDiv diversity panel grown in Nebraska 2021
- **Samples**: 738 individuals
- **Traits**: 2 (PlantHeight in meters, DaysToFlower in days)
- **Format**: CSV with columns `ID,PlantHeight,DaysToFlower`

### `example_genotypes.vcf.gz`
- **Description**: SNP genotype data
- **Source**: RNA-seq derived markers from SbDiv diversity panel
- **Samples**: 738 individuals (matching phenotypes)
- **Markers**: 6,533 SNPs across all 10 chromosomes
- **Format**: Compressed VCF (standard variant call format)

### `example_covariates.csv`
- **Description**: Flowering time data for use as a covariate
- **Source**: Same SbDiv diversity panel, Nebraska 2021
- **Samples**: 738 individuals (matching phenotypes and genotypes)
- **Covariates**: 1 (DaysToFlower, days from planting to flowering)
- **Format**: CSV with columns `ID,DaysToFlower`
- **Use case**: Demonstrates controlling for flowering time when analyzing plant height

## Data Curation

The example data was created by:
1. Extracting all 738 samples that have both genotype and phenotype data
2. Including all Bonferroni-significant markers from an MLM run on PlantHeight
3. Adding 5,000 random markers sampled across all 10 chromosomes

This keeps true signals in the dataset while providing a realistic background of
random markers for benchmarking and demos.

## Usage

All example scripts in this directory are designed to work with these files:

```python
from panicle.pipelines.gwas import GWASPipeline

# Basic GWAS
pipeline = GWASPipeline(output_dir='./results')
pipeline.load_data(
    phenotype_file='example_phenotypes.csv',
    genotype_file='example_genotypes.vcf.gz'
)
pipeline.align_samples()
pipeline.compute_population_structure(n_pcs=3, calculate_kinship=True)
pipeline.run_analysis(traits=['PlantHeight'], methods=['MLM'])

# GWAS with covariates
pipeline = GWASPipeline(output_dir='./results_with_covariates')
pipeline.load_data(
    phenotype_file='example_phenotypes.csv',
    genotype_file='example_genotypes.vcf.gz',
    covariate_file='example_covariates.csv',
    covariate_columns=['DaysToFlower']
)
pipeline.align_samples()
pipeline.compute_population_structure(n_pcs=3, calculate_kinship=True)
pipeline.run_analysis(traits=['PlantHeight'], methods=['MLM'])
```

## Expected Results

When analyzing this data with appropriate population structure correction (MLM):
- You should detect many of the included significant PlantHeight markers
- Lambda GC inflation factors should be reasonable (~1.0-1.3) with proper correction

## Data Size

- Phenotype file: ~31 KB (text)
- Genotype file: ~1.0 MB (compressed VCF)
- Covariate file: ~18 KB (text)
- Total: ~1.1 MB

This small dataset allows for:
- Fast testing (~10-60 seconds per analysis)
- Demonstration of all PANICLE features
- Verification of installation and setup
- Learning the PANICLE workflow

## Original Data Source

Data derives from:
- **SbDiv diversity panel**: 378 sorghum lines representing global diversity
- **Field trials**: Nebraska 2021 growing season
- **Genotyping**: RNA-seq based SNP calling (Mangal et al. 2025)

For the full dataset, see the `sorghum_data/` directory in the parent repository.

## Citation

If you use this example data in publications, please cite:
- The PANICLE package version you used
- Original SbDiv panel and genotyping work (Mangal et al. 2025, in prep)
