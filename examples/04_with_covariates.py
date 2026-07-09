#!/usr/bin/env python3
"""
Example 04: GWAS with External Covariates

This example shows how to include external covariates in your GWAS analysis.

In this case, we're analyzing PlantHeight while controlling for DaysToFlower
(flowering time) as a covariate. This is a realistic scenario since:
- Flowering time can be correlated with plant height
- Controlling for flowering time helps isolate height-specific genetic effects
- This demonstrates how to use any measured trait as a covariate

External covariates can help control for:
- Other traits (as shown here)
- Environmental effects (location, year)
- Experimental design factors (block, treatment)
- Other measured confounders
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panicle.pipelines.gwas import GWASPipeline

HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Example 04: GWAS with Covariates")
    parser.add_argument("--phenotype", default=str(HERE / "example_phenotypes.csv"))
    parser.add_argument("--genotype", default=str(HERE / "example_genotypes.vcf.gz"))
    parser.add_argument("--map", default=None)
    parser.add_argument("--trait", default="PlantHeight")
    parser.add_argument("--covariate-file", default=str(HERE / "example_covariates.csv"))
    parser.add_argument("--covariate-columns", default="DaysToFlower")
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--outputdir", default="./example04_results")
    return parser.parse_args()

def main():
    print("=" * 70)
    print("EXAMPLE 04: GWAS with External Covariates")
    print("=" * 70)

    args = parse_args()

    # Initialize pipeline
    pipeline = GWASPipeline(output_dir=args.outputdir)

    # Load data including external covariates
    print("\n1. Loading data with covariates...")
    covariate_list = [c.strip() for c in args.covariate_columns.split(',') if c.strip()]
    pipeline.load_data(
        phenotype_file=args.phenotype,
        genotype_file=args.genotype,
        map_file=args.map,
        covariate_file=args.covariate_file,
        covariate_columns=covariate_list
    )
    print(f"   Loaded covariate(s): {', '.join(covariate_list) if covariate_list else 'None'}")

    # Align samples
    print("\n2. Aligning samples...")
    pipeline.align_samples()

    # Compute PCs; they are combined with external covariates for the fixed model.
    # Default LOCO MLM does not need a precomputed global kinship matrix.
    print("\n3. Computing population structure...")
    pipeline.compute_population_structure(
        n_pcs=args.n_pcs,
        calculate_kinship=False,
    )
    if covariate_list:
        cov_label = ", ".join(covariate_list + [f"PC{i + 1}" for i in range(args.n_pcs)])
    else:
        cov_label = ", ".join([f"PC{i + 1}" for i in range(args.n_pcs)])
    print(f"   Final covariates: {cov_label}")

    # Run MLM with combined covariates
    print("\n4. Running MLM with all covariates...")
    pipeline.run_analysis(
        traits=[args.trait],
        methods=['MLM']
    )

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {args.outputdir}/")
    print("\nCovariates used in analysis:")
    if covariate_list:
        print(f"- External: {', '.join(covariate_list)}")
    else:
        print("- External: None")
    print(f"- PCs: {', '.join([f'PC{i + 1}' for i in range(args.n_pcs)])}")
    print("\nThis helps control for:")
    print("- Trait correlations or measured confounders")
    print("- Population stratification")
    print("- Increases power to detect height-specific genetic associations")
    print("\nNext steps:")
    print("- Compare results with example02 (MLM without flowering time covariate)")
    print("- Check if controlling for flowering time reveals new height-specific QTLs")


if __name__ == '__main__':
    main()
