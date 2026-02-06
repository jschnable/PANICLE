#!/usr/bin/env python3
"""
Example 05: Reading and Analyzing GWAS Results

This example demonstrates how to load and analyze PANICLE output files,
including filtering, sorting, and extracting information.

Prerequisites: Run example 02 first to generate MLM results.
This script reads from './example02_results/' directory.

Alternatively, you can modify the paths to read from example01, example03, or example04 results.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Example 05: Reading GWAS Results")
    parser.add_argument("--results", default=None, help="Path to GWAS *_all_results.csv")
    parser.add_argument("--trait", default="PlantHeight")
    parser.add_argument("--method", default="MLM")
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def load_and_explore(results_path: Path):
    """Load results and show basic exploration."""
    print("=" * 70)
    print("1. Loading and Exploring Results")
    print("=" * 70)

    # Load full results (from example02)
    results = pd.read_csv(results_path)

    print(f"\nTotal markers: {len(results):,}")
    print(f"Columns: {list(results.columns)}")
    print(f"\nFirst few rows:")
    print(results.head())

    # Basic statistics
    print(f"\nP-value statistics:")
    print(f"  Min: {results['MLM_P'].min():.2e}")
    print(f"  Median: {results['MLM_P'].median():.2e}")
    print(f"  Max: {results['MLM_P'].max():.2e}")

    print(f"\nMAF statistics:")
    print(f"  Mean: {results['MAF'].mean():.3f}")
    print(f"  Min: {results['MAF'].min():.3f}")
    print(f"  Max: {results['MAF'].max():.3f}")

    return results


def filter_and_sort(results, method):
    """Demonstrate filtering and sorting operations."""
    print("\n" + "=" * 70)
    print("2. Filtering and Sorting")
    print("=" * 70)

    # Calculate Bonferroni threshold
    bonferroni = 0.05 / len(results)
    print(f"\nBonferroni threshold (0.05/{len(results):,}): {bonferroni:.2e}")

    # Get significant SNPs
    p_col = f"{method}_P"
    sig_snps = results[results[p_col] < bonferroni].copy()
    print(f"Significant SNPs: {len(sig_snps)}")

    if len(sig_snps) > 0:
        print("\nTop 5 significant SNPs:")
        top5 = sig_snps.nsmallest(5, p_col)
        effect_col = f"{method}_Effect"
        print(top5[['SNP', 'CHROM', 'POS', p_col, effect_col, 'MAF']])

    # Filter by chromosome
    chr1 = results[results['CHROM'] == 'Chr01']
    print(f"\nMarkers on Chr01: {len(chr1):,}")

    # Filter by MAF
    common = results[results['MAF'] >= 0.05]
    print(f"Common variants (MAF >= 0.05): {len(common):,}")

    return sig_snps


def compare_methods(results, method):
    """Compare results from multiple methods."""
    print("\n" + "=" * 70)
    print("3. Comparing Methods")
    print("=" * 70)

    if 'GLM_P' in results.columns and f"{method}_P" in results.columns:
        # Calculate correlation
        log_glm = -np.log10(results['GLM_P'].clip(lower=1e-300))
        log_mlm = -np.log10(results[f"{method}_P"].clip(lower=1e-300))

        corr, pval = stats.spearmanr(log_glm, log_mlm)
        print(f"\nSpearman correlation (GLM vs MLM): {corr:.3f}")

        # Find method-specific hits
        bonf = 0.05 / len(results)
        glm_only = results[(results['GLM_P'] < bonf) & (results[f"{method}_P"] >= bonf)]
        mlm_only = results[(results[f"{method}_P"] < bonf) & (results['GLM_P'] >= bonf)]
        both = results[(results[f"{method}_P"] < bonf) & (results['GLM_P'] < bonf)]

        print(f"\nSignificant in:")
        print(f"  GLM only: {len(glm_only)}")
        print(f"  MLM only: {len(mlm_only)}")
        print(f"  Both:     {len(both)}")
    else:
        print("\nOnly one method found in results")


def extract_candidate_regions(results, sig_snps):
    """Extract genomic regions around significant SNPs."""
    print("\n" + "=" * 70)
    print("4. Extracting Candidate Regions")
    print("=" * 70)

    if len(sig_snps) == 0:
        print("\nNo significant SNPs to extract regions for")
        return

    # Define window size (e.g., ±100kb)
    window = 100000  # 100 kb

    print(f"\nExtracting regions ±{window/1000:.0f}kb around significant SNPs:")

    for idx, snp in sig_snps.head(3).iterrows():
        chrom = snp['CHROM']
        pos = snp['POS']

        # Get markers in window
        window_snps = results[
            (results['CHROM'] == chrom) &
            (results['POS'] >= pos - window) &
            (results['POS'] <= pos + window)
        ]

        print(f"\n  {snp['SNP']} ({chrom}:{pos:,})")
        print(f"    Markers in region: {len(window_snps)}")
        print(f"    Position range: {window_snps['POS'].min():,} - {window_snps['POS'].max():,}")


def create_custom_plots(results, method, output_dir: Path):
    """Create custom visualizations."""
    print("\n" + "=" * 70)
    print("5. Creating Custom Plots")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: P-value distribution
    p_col = f"{method}_P"
    effect_col = f"{method}_Effect"
    axes[0, 0].hist(-np.log10(results[p_col].clip(lower=1e-300)),
                    bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('-log10(P-value)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('P-value Distribution')

    # Plot 2: MAF distribution
    axes[0, 1].hist(results['MAF'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Minor Allele Frequency')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('MAF Distribution')

    # Plot 3: Effect size vs p-value
    axes[1, 0].scatter(results[effect_col],
                       -np.log10(results[p_col].clip(lower=1e-300)),
                       alpha=0.5, s=1)
    axes[1, 0].set_xlabel('Effect Size')
    axes[1, 0].set_ylabel('-log10(P-value)')
    axes[1, 0].set_title('Effect Size vs Significance')
    axes[1, 0].axhline(-np.log10(0.05/len(results)), color='red',
                       linestyle='--', alpha=0.5)

    # Plot 4: MAF vs effect size
    axes[1, 1].scatter(results['MAF'],
                       np.abs(results[effect_col]),
                       alpha=0.5, s=1)
    axes[1, 1].set_xlabel('Minor Allele Frequency')
    axes[1, 1].set_ylabel('|Effect Size|')
    axes[1, 1].set_title('MAF vs Effect Size')

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "custom_analysis.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved custom plots to: {out_path}")

    plt.close()


def export_for_external_tools(sig_snps, output_dir: Path, method):
    """Export results in formats useful for other tools."""
    print("\n" + "=" * 70)
    print("6. Exporting for External Tools")
    print("=" * 70)

    if len(sig_snps) == 0:
        print("\nNo significant SNPs to export")
        return

    # BED format (for genome browsers)
    bed = sig_snps[['CHROM', 'POS']].copy()
    bed['END'] = bed['POS'] + 1
    bed['NAME'] = sig_snps['SNP']
    p_col = f"{method}_P"
    bed['SCORE'] = (-np.log10(sig_snps[p_col]) * 10).astype(int)
    bed = bed[['CHROM', 'POS', 'END', 'NAME', 'SCORE']]
    output_dir.mkdir(parents=True, exist_ok=True)
    bed_path = output_dir / "significant_snps.bed"
    bed.to_csv(bed_path, sep='\t', header=False, index=False)
    print(f"\nExported to BED format: {bed_path}")

    # Simple list of SNP IDs
    snp_path = output_dir / "significant_snp_ids.txt"
    sig_snps['SNP'].to_csv(snp_path, index=False, header=False)
    print(f"Exported SNP IDs: {snp_path}")


def main():
    print("\n" + "=" * 70)
    print("EXAMPLE 05: Reading and Analyzing GWAS Results")
    print("=" * 70)

    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.results is None:
        results_path = HERE / "example02_results" / f"GWAS_{args.trait}_all_results.csv"
    else:
        results_path = Path(args.results)

    # Load and explore
    results = load_and_explore(results_path)

    # Filter and sort
    sig_snps = filter_and_sort(results, args.method)

    # Compare methods (if multiple available)
    compare_methods(results, args.method)

    # Extract candidate regions
    extract_candidate_regions(results, sig_snps)

    # Create custom plots
    create_custom_plots(results, args.method, output_dir)

    # Export for external tools
    export_for_external_tools(sig_snps, output_dir, args.method)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
