"""
PANICLE: Python Algorithms for Nucleotide-phenotype Inference and
Chromosome-wide Locus Evaluation

A comprehensive, memory-efficient, and parallel-accelerated
genome-wide association study (GWAS) tool.
Based on the original rMVP package design.
"""

import logging
import os
import warnings

# Configure library-level logging so that INFO+ messages are visible by default
# (preserves the existing print()-based behavior). Users can override via:
#   logging.getLogger('panicle').setLevel(logging.WARNING)
_pkg_logger = logging.getLogger('panicle')
if not _pkg_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    _pkg_logger.addHandler(_handler)
    _pkg_logger.setLevel(logging.INFO)

# Suppress OpenMP deprecation warnings that occur with Numba parallel processing
# This is a known issue with newer OpenMP versions and Numba's parallel features
# The warning is cosmetic and does not affect functionality
os.environ.setdefault('KMP_WARNINGS', 'off')

# Filter out the specific OpenMP deprecation warning
warnings.filterwarnings('ignore', message='.*omp_set_nested.*deprecated.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*omp_set_nested.*')

__version__ = "0.1.0"
__author__ = "James C. Schnable"

from .core.mvp import PANICLE
from .matrix.kinship import PANICLE_K_VanRaden, PANICLE_K_IBS
from .matrix.pca import PANICLE_PCA
from .association.glm import PANICLE_GLM
from .association.mlm import PANICLE_MLM
from .association.farmcpu import PANICLE_FarmCPU
from .association.blink import PANICLE_BLINK
from .association.farmcpu_resampling import PANICLE_FarmCPUResampling
from .visualization.manhattan import PANICLE_Report
from .data.loaders import (
    load_genotype_file,
    load_genotype_vcf,
    load_genotype_plink,
    load_genotype_hapmap,
    load_phenotype_file,
    load_map_file,
    match_individuals,
)

__all__ = [
    'PANICLE',
    'PANICLE_K_VanRaden',
    'PANICLE_K_IBS',
    'PANICLE_PCA',
    'PANICLE_GLM',
    'PANICLE_MLM',
    'PANICLE_FarmCPU',
    'PANICLE_BLINK',
    'PANICLE_FarmCPUResampling',
    'PANICLE_Report',
    'load_genotype_file',
    'load_genotype_vcf',
    'load_genotype_plink',
    'load_genotype_hapmap',
    'load_phenotype_file',
    'load_map_file',
    'match_individuals',
]
