#!/usr/bin/env python3
"""
Example 01: Basic GWAS Analysis

This example demonstrates the simplest possible GWAS workflow using PANICLE.
We'll run a GLM analysis (no population structure correction) on a single trait.

Prerequisites:
- phenotype.csv: CSV file with ID column and trait columns
- genotypes.vcf.gz: VCF file with genetic variants
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
    parser = argparse.ArgumentParser(description="Example 01: Basic GWAS Analysis")
    parser.add_argument("--phenotype", default=str(HERE / "example_phenotypes.csv"))
    parser.add_argument("--genotype", default=str(HERE / "example_genotypes.vcf.gz"))
    parser.add_argument("--map", default=None)
    parser.add_argument("--trait", default="PlantHeight")
    parser.add_argument("--outputdir", default="./example01_results")
    return parser.parse_args()

def main():
    print("=" * 70)
    print("EXAMPLE 01: Basic GWAS Analysis")
    print("=" * 70)

    args = parse_args()

    # Initialize the pipeline with output directory
    pipeline = GWASPipeline(output_dir=args.outputdir)

    # Load phenotype and genotype data
    print("\n1. Loading data...")
    pipeline.load_data(
        phenotype_file=args.phenotype,
        genotype_file=args.genotype,
        map_file=args.map
    )

    # Align samples (match individuals between phenotype and genotype)
    print("\n2. Aligning samples...")
    pipeline.align_samples()

    # Run GLM analysis
    # GLM is fast and doesn't require population structure correction
    print("\n3. Running GLM analysis...")
    pipeline.run_analysis(
        traits=[args.trait],  # Analyze the requested trait
        methods=['GLM']          # Use General Linear Model
    )

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {args.outputdir}/")
    print(f"- GWAS_{args.trait}_all_results.csv       (all markers)")
    print(f"- GWAS_{args.trait}_significant.csv       (significant markers only)")
    print(f"- GWAS_{args.trait}_GLM_manhattan.png (Manhattan plot)")
    print(f"- GWAS_{args.trait}_GLM_qq.png       (QQ plot)")
    print("\nNext steps:")
    print("- Open the Manhattan plot to visualize results")
    print("- Check the QQ plot for genomic inflation")
    print(f"- Load GWAS_{args.trait}_all_results.csv in Python/R for further analysis")


if __name__ == '__main__':
    main()
