import argparse
from typing import List, Optional

OUTPUT_CHOICES = (
    'all_marker_pvalues',
    'significant_marker_pvalues',
    'manhattan',
    'qq',
)

VALID_FORMATS = ('csv', 'tsv', 'numeric', 'vcf', 'plink', 'hapmap')
FORMAT_ALIASES = {'bcf': 'vcf'}

def normalize_format(fmt: Optional[str]) -> Optional[str]:
    """Normalize a genotype format string with case-insensitive matching.

    Tries an exact match first, then case-insensitive, then stripped +
    case-insensitive.  ``bcf`` is accepted as an alias for ``vcf``.
    Returns *None* when *fmt* is None (auto-detect).
    """
    if fmt is None:
        return None

    # 1. Exact match
    if fmt in VALID_FORMATS:
        return fmt
    if fmt in FORMAT_ALIASES:
        return FORMAT_ALIASES[fmt]

    # 2. Case-insensitive match
    key = fmt.lower()
    if key in VALID_FORMATS:
        return key
    if key in FORMAT_ALIASES:
        return FORMAT_ALIASES[key]

    # 3. Stripped + case-insensitive
    key = fmt.strip().lower()
    if key in VALID_FORMATS:
        return key
    if key in FORMAT_ALIASES:
        return FORMAT_ALIASES[key]

    valid = ', '.join(sorted(set(VALID_FORMATS) | set(FORMAT_ALIASES)))
    raise argparse.ArgumentTypeError(
        f"Unrecognized genotype format: '{fmt}'. "
        f"Valid formats: {valid}"
    )


def normalize_outputs(outputs: List[str]) -> List[str]:
    """Helper to normalize output choices"""
    if not outputs:
        return list(OUTPUT_CHOICES)
    # Simplified normalization
    valid = []
    for o in outputs:
        o = o.strip().lower()
        if o in OUTPUT_CHOICES:
            valid.append(o)
    return valid if valid else list(OUTPUT_CHOICES)

def parse_args():
    """Parse command line arguments for GWAS pipeline"""
    parser = argparse.ArgumentParser(
        description="Comprehensive GWAS Analysis using PANICLE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--phenotype", "-p", required=True,
                       help="Phenotype file (CSV/TSV with ID column and trait columns)")
    parser.add_argument("--phenotype-id-column", default='ID',
                       help="Column name for sample IDs in phenotype file. "
                            "Auto-detected if column is named: ID, IID, Sample, "
                            "Taxa, Genotype, or Accession. If none match, the "
                            "first column is used.")
    parser.add_argument("--genotype", "-g", required=True,
                       help="Genotype file")
    
    # Optional arguments
    parser.add_argument("--map", "-m", default=None,
                       help="Genetic map file (CSV/TSV)")
    parser.add_argument("--outputdir", "-o", default="./GWAS_results",
                       help="Output directory")
    parser.add_argument("--covariates", default=None,
                       help="Optional covariate file")
    parser.add_argument("--covariate-columns", default=None,
                       help="Comma-separated list of covariate column names")
    parser.add_argument("--covariate-id-column", default='ID',
                       help="Column name for sample IDs in covariate file")
    parser.add_argument("--format", "-f", default=None,
                       type=normalize_format,
                       help="Genotype file format (csv, tsv, numeric, vcf, bcf, plink, hapmap)")
    parser.add_argument("--methods", default="GLM,MLM,FarmCPU",
                       help="Methods to run (comma-separated: GLM, MLM, BAYESLOCO, FarmCPU, BLINK)")
    parser.add_argument("--n-pcs", type=int, default=3,
                       help="Number of PCs")
    parser.add_argument(
        "--ncpus",
        type=int,
        default=1,
        help="CPUs used by each method engine (0 = all available cores)",
    )
    parser.add_argument(
        "--parallel-mode",
        choices=["auto", "off", "on"],
        default="auto",
        help="Method execution mode: off forces single CPU, auto/on use --ncpus",
    )
    
    # Thresholds
    parser.add_argument("--significance", type=float, default=None,
                       help="Fixed p-value threshold")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Bonferroni alpha")
    parser.add_argument("--n-eff", type=int, default=None,
                       help="Effective number of markers")
    parser.add_argument("--compute-effective-tests", action='store_true',
                       help="Compute effective tests")

    # Options
    parser.add_argument("--max-genotype-dosage", type=float, default=2.0,
                       help="Max dosage (e.g. 2 for diploid)")
    parser.add_argument("--max-iterations", type=int, default=10,
                       help="Max iterations for FarmCPU/BLINK")
    parser.add_argument("--traits", default=None,
                       help="Comma-separated traits")
    
    # Output
    parser.add_argument("--outputs", nargs='+',
                       choices=list(OUTPUT_CHOICES),
                       default=list(OUTPUT_CHOICES),
                       help="Outputs to generate")
    parser.add_argument(
        "--include-standard-errors",
        action='store_true',
        help="Include {METHOD}_SE columns in GWAS_*_all_results.csv and GWAS_*_significant.csv",
    )

    # FarmCPU resampling options
    parser.add_argument("--resampling", action='store_true',
                       help="Enable FarmCPU resampling")
    parser.add_argument("--farmcpu-resampling-runs", type=int, default=100,
                       help="Number of FarmCPU resampling runs")
    parser.add_argument("--farmcpu-resampling-mask-proportion", type=float, default=0.1,
                       help="Proportion of phenotype values masked per resampling run")
    parser.add_argument("--farmcpu-resampling-significance", type=float, default=None,
                       help="Fixed p-value threshold for resampling (overrides alpha)")
    parser.add_argument("--farmcpu-resampling-alpha", type=float, default=None,
                       help="Alpha for resampling Bonferroni threshold (alpha/n)")
    parser.add_argument("--farmcpu-resampling-cluster", action='store_true',
                       help="Cluster resampling hits by LD")
    parser.add_argument("--farmcpu-resampling-ld-threshold", type=float, default=0.7,
                       help="LD threshold for resampling clustering")
    parser.add_argument("--farmcpu-resampling-seed", type=int, default=None,
                       help="Random seed for resampling")

    # FarmCPU thresholds
    parser.add_argument("--farmcpu-qtn-threshold", type=float, default=None,
                       help="Fixed p-value threshold for FarmCPU QTN selection")
    parser.add_argument("--farmcpu-qtn-alpha", type=float, default=None,
                       help="Alpha for FarmCPU QTN Bonferroni threshold (alpha/n)")
    parser.add_argument("--farmcpu-p-threshold", type=float, default=None,
                       help="Fixed p-value threshold for FarmCPU early stopping")

    # BAYESLOCO options
    parser.add_argument("--bayesloco-max-iter", type=int, default=None,
                       help="BAYESLOCO max fitting epochs")
    parser.add_argument("--bayesloco-patience", type=int, default=None,
                       help="BAYESLOCO early-stopping patience")
    parser.add_argument("--bayesloco-loco-mode", choices=["subtract_only", "refine"], default=None,
                       help="BAYESLOCO LOCO mode")
    parser.add_argument("--bayesloco-test-method", choices=["score", "wald"], default=None,
                       help="BAYESLOCO association test method")
    parser.add_argument("--bayesloco-calibration", choices=["none", "gc", "unrelated_subset"], default=None,
                       help="BAYESLOCO test-stat calibration mode")
    parser.add_argument("--bayesloco-unrelated-indices", default=None,
                       help="Path to newline-delimited unrelated sample row indices for BAYESLOCO calibration")
    parser.add_argument("--bayesloco-prior-pi-grid", default=None,
                       help="Comma-separated BAYESLOCO prior pi grid (e.g. 0.005,0.02,0.05)")
    parser.add_argument("--bayesloco-prior-slab-scale-grid", default=None,
                       help="Comma-separated BAYESLOCO slab-scale grid (e.g. 0.75,1.25)")
    parser.add_argument("--bayesloco-batch-fit", type=int, default=None,
                       help="BAYESLOCO marker batch size for fitting")
    parser.add_argument("--bayesloco-batch-test", type=int, default=None,
                       help="BAYESLOCO marker batch size for testing")
    parser.add_argument("--bayesloco-refine-iter", type=int, default=None,
                       help="BAYESLOCO LOCO refinement epochs per chromosome")

    # Loader options (simplified subset)
    parser.add_argument("--drop-monomorphic", action='store_true', dest='drop_monomorphic',
                       help="Drop monomorphic markers (now default behavior, kept for backward compatibility)")
    parser.add_argument("--keep-monomorphic", action='store_false', dest='drop_monomorphic',
                       help="Keep monomorphic markers (override default)")
    parser.add_argument("--max-missing", type=float, default=1.0)
    parser.add_argument("--min-maf", type=float, default=0.0)
    parser.add_argument("--snps-only", action='store_true')
    parser.add_argument("--no-split-multiallelic", action='store_true')

    # Set defaults
    parser.set_defaults(drop_monomorphic=True)

    return parser.parse_args()
