"""
GWAS Pipeline Module

This module enables a modular, object-oriented approach to running GWAS analyses.
It encapsulates data loading, sample alignment, population structure correction,
association testing, and result reporting into a reusable pipeline class.
"""

from ..core.workflow import (
    PreparedTrait, TraitCacheKey, TraitPreparation, MethodRunResult, MethodOptions,
    retained_samples, group_sample_indices, select_markers, association_genotype,
    run_method, run_trait_group,
)

from ..core.methods import ordered_pipeline_methods
from ..core.thresholds import base_threshold as resolve_base_threshold, trait_thresholds
from ..reporting.models import MethodReport, TraitReport, ReportOptions
from ..reporting.pipeline import TraitOutputContext, write_trait_results, write_trait_report

import dataclasses
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple

from ..data.loaders import (
    load_phenotype_file, load_genotype_file, load_map_file,
    load_covariate_file, match_individuals, detect_file_format
)
from ..utils.stats import (
    compute_mac_keep_indices,
    pad_association_results,
    qq_compatible_genomic_inflation_factor,
)
from ..utils.data_types import (
    GenotypeMap,
    GenotypeMatrix,
    AssociationResults,
)
from ..utils.effective_tests import estimate_effective_tests_from_genotype
from ..utils.perf import available_cpu_count, format_blas_runtime
from ..association.farmcpu_resampling import (
    PANICLE_FarmCPUResampling,
)
from ..association.glm import PANICLE_GLM, PANICLE_GLM_MULTI
from ..association.mlm import PANICLE_MLM
from ..association.mlm_loco import PANICLE_MLM_LOCO, PANICLE_MLM_LOCO_MULTI
from ..association.bayes_loco import PANICLE_BayesLOCO
from ..association.bayes_loco.config import BayesLocoConfig
from ..association.farmcpu import PANICLE_FarmCPU
from ..association.blink import PANICLE_BLINK
from ..matrix.pca import PANICLE_PCA
from ..matrix.ld import LDDecay, ld_decay, ld_prune
from ..matrix.admixture import fit_admixture
from ..postgwas.heritability import REMLHeritability, gower_scale, reml_heritability
from ..postgwas.trait import PostGWASOptions, run_post_gwas, write_heritability
from ..matrix.kinship import PANICLE_K_VanRaden
from ..matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from ..visualization.manhattan import PANICLE_Report

# Internal helper for resampling progress logging.
class _FarmCPUResamplingProgressReporter:
    def __init__(self, log_fn, trait_name: str):
        self._log = log_fn
        self._trait_name = trait_name
        self._started = False
        self._total_elapsed = 0.0

    def __call__(self, run_idx: int, total_runs: int, run_seconds: float) -> None:
        if not self._started:
            self._log(f"[{self._trait_name}] started resampling ({total_runs} runs)")
            self._started = True
        self._total_elapsed += float(run_seconds)
        self._log(
            f"[{self._trait_name}] run {run_idx}/{total_runs} "
            f"finished in {run_seconds:.2f}s (total {self._total_elapsed:.2f}s)"
        )
        if run_idx >= total_runs:
            self._log(f"[{self._trait_name}] finished resampling in {self._total_elapsed:.0f}s")


def _resolve_method_cpu(ncpus: int, parallel_mode: str) -> int:
    """Resolve effective CPU count for per-method internal parallelism."""
    mode = str(parallel_mode).strip().lower()
    if mode not in {"auto", "off", "on"}:
        raise ValueError("parallel_mode must be one of: 'auto', 'off', 'on'")
    try:
        requested = int(ncpus)
    except (TypeError, ValueError):
        raise ValueError("ncpus must be an integer")
    if requested < 0:
        raise ValueError("ncpus must be >= 0")
    if mode == "off":
        return 1
    if requested == 0:
        return available_cpu_count()
    return max(1, requested)


def _map_has_non_numeric_chrom_labels(geno_map) -> bool:
    """Return True if any chromosome label is non-numeric (e.g. ``chr1``).

    Uses the cached chromosome order on ``GenotypeMap`` so this does not
    materialize a full marker-map DataFrame. On a cache-backed VCF that
    DataFrame is 12M+ rows and was previously built only to decide whether
    to print an htslib contig warning.
    """
    if geno_map is None:
        return False
    try:
        if hasattr(geno_map, "get_chromosome_order"):
            labels = geno_map.get_chromosome_order()
        else:
            return False
        return any(not str(label).isdigit() for label in labels)
    except (KeyError, AttributeError, TypeError, ValueError):
        return False


def normalize_mlm_mode(mlm_mode: Optional[str]) -> str:
    """Normalize MLM mode to ``'loco'`` or ``'global'``."""
    if mlm_mode is None:
        return "loco"
    mode = str(mlm_mode).strip().lower().replace("-", "_")
    if mode in {"loco", "leave_one_chromosome_out"}:
        return "loco"
    if mode in {"global", "full", "classic"}:
        return "global"
    raise ValueError(
        f"Invalid mlm_mode={mlm_mode!r}; expected 'loco' or 'global'"
    )


# Helper function for method dispatch
def _execute_prepared_method(method, prepared: PreparedTrait, options: MethodOptions):
    """Execute a prepared trait; retain pipeline failure and option policies."""
    try:
        if method == 'GLM':
            completed = run_method(
                "GLM", prepared,
                runner=PANICLE_GLM,
                options=dict(
                    cpu=options.ncpus,
                    verbose=False,
                ),
            )

        elif method == 'MLM':
            mlm_kwargs = options.mlm or {}
            mode = normalize_mlm_mode(options.mlm_mode)
            use_loco = mode == "loco" and prepared.geno_map is not None
            if use_loco:
                completed = run_method(
                    "MLM_LOCO", prepared,
                    runner=PANICLE_MLM_LOCO,
                    options=dict(
                        loco_kinship=options.loco_kinship,
                        verbose=False,
                        **mlm_kwargs,
                    ),
                )
            else:
                if prepared.kinship is None:
                    return MethodRunResult("MLM", error="Kinship matrix missing")
                completed = run_method(
                    "MLM", prepared,
                    runner=PANICLE_MLM,
                    options=dict(
                        K=prepared.kinship,
                        cpu=options.ncpus,
                        verbose=False,
                    ),
                )

        elif method == 'FARMCPU':
            # Leave p_threshold as None unless the caller set it explicitly so
            # PANICLE_FarmCPU can use its rMVP-style default early-stop
            # (0.01 / n_tests) and keep QTN_threshold at the uncorrected 0.01
            # default.  Defaulting to `alpha` (0.05) previously forced
            # QTN_threshold = max(0.05, 0.01) and disabled the 0.01/n stop.
            fc_p = options.farmcpu.get('p_threshold', None)
            fc_qtn = options.farmcpu.get('QTN_threshold', 0.01)  # Alpha for QTN selection, e.g., 0.01
            fc_bin = options.farmcpu.get('bin_size')
            fc_method_bin = options.farmcpu.get('method_bin', 'static')
            fc_converge = options.farmcpu.get('converge', 1.0)
            completed = run_method(
                "FARMCPU", prepared,
                runner=PANICLE_FarmCPU,
                options=dict(
                    maxLoop=options.max_iterations,
                    p_threshold=fc_p,
                    QTN_threshold=fc_qtn,
                    n_eff=options.n_eff,
                    converge=fc_converge,
                    bin_size=fc_bin,
                    method_bin=fc_method_bin,
                    cpu=options.ncpus,
                    verbose=False,
                ),
            )

        elif method == 'BLINK':
            blink_kwargs = {
                key: options.blink[key]
                for key in (
                    'Prior',
                    'maxLoop',
                    'converge',
                    'ld_threshold',
                    'maf_threshold',
                    'bic_method',
                    'method_sub',
                    'p_threshold',
                    'qtn_threshold',
                    'cut_off',
                    'fdr_cut',
                    'maxLine',
                    'max_genotype_dosage',
                )
                if key in options.blink
            }
            blink_kwargs.setdefault('maxLoop', options.max_iterations)
            completed = run_method(
                "BLINK", prepared,
                runner=PANICLE_BLINK,
                options=dict(
                    cpu=options.ncpus,
                    verbose=False,
                    **blink_kwargs,
                ),
            )

        elif method == 'BAYESLOCO':
            completed = run_method(
                "BAYESLOCO", prepared,
                runner=PANICLE_BayesLOCO,
                options=dict(
                    cpu=options.ncpus,
                    verbose=False,
                    bl_config=options.bayesloco,
                ),
            )
            
        else:
            # Resampling is handled by run_analysis with trait/output context.
            return MethodRunResult(method, error=f"Unknown method {method}")

        completed.lambda_gc, completed.lambda_gc_is_approx = (
            qq_compatible_genomic_inflation_factor(completed.result.pvalues)
        )
        return completed

    except Exception as e:
        return MethodRunResult(method, error=str(e))


def _execute_single_method(
    method,
    y_sub,
    g_sub,
    cov_sub,
    k_sub,
    map_data,
    fc_params,
    blk_params,
    bl_params,
    max_iterations,
    base_threshold,
    n_markers,
    n_eff=None,
    alpha=0.05,
    mlm_loco_kinship=None,
    mlm_kwargs=None,
    ncpus: int = 1,
    mlm_mode: str = "loco",
):
    """Compatibility adapter for the former positional worker inputs."""
    prepared = PreparedTrait("", y_sub, g_sub, cov_sub, k_sub, np.arange(len(y_sub)), map_data)
    return _execute_prepared_method(method, prepared, MethodOptions(
        farmcpu=fc_params, blink=blk_params, bayesloco=bl_params,
        max_iterations=max_iterations, n_eff=n_eff, loco_kinship=mlm_loco_kinship,
        mlm=mlm_kwargs, ncpus=ncpus, mlm_mode=mlm_mode,
    ))


def _run_single_method(*args, **kwargs):
    """Compatibility wrapper for the former tuple-returning worker."""
    return _execute_single_method(*args, **kwargs).legacy_tuple()


OUTPUT_CHOICES: Tuple[str, ...] = (
    'all_marker_pvalues',
    'significant_marker_pvalues',
    'manhattan',
    'qq',
)

class GWASPipeline:
    """
    High-level pipeline for Genome-Wide Association Studies (GWAS).

    This class provides a complete workflow for GWAS analysis, handling data loading,
    sample alignment, population structure correction, association testing, and
    result visualization.

    Typical workflow:
        1. Initialize pipeline with output directory
        2. Load phenotype, genotype, and optional covariate data
        3. Align samples across datasets
        4. Compute population structure (PCs and/or kinship matrix)
        5. Run association analysis with chosen method(s)
        6. Results are automatically saved to output directory

    Attributes:
        genotype_matrix (GenotypeMatrix): Aligned genotype data (n_individuals × n_markers)
        geno_map (GenotypeMap): Genetic map with marker information (ID, CHROM, POS)
        phenotype_df (DataFrame): Aligned phenotype data with 'ID' column + trait columns
        covariate_df (DataFrame): External covariates (if loaded)
        pcs (ndarray): Principal components (n_individuals × n_pcs)
        pc_names (list): Names of PC columns ['PC1', 'PC2', ...]
        kinship (ndarray): Kinship matrix (n_individuals × n_individuals)
        output_dir (Path): Output directory for results
        effective_tests_info (dict): Effective number of independent tests (if computed)

    Example:
        >>> from panicle.pipelines.gwas import GWASPipeline
        >>>
        >>> # Initialize pipeline
        >>> pipeline = GWASPipeline(output_dir='./my_gwas')
        >>>
        >>> # Load data
        >>> pipeline.load_data(
        ...     phenotype_file='phenotypes.csv',
        ...     genotype_file='genotypes.vcf.gz'
        ... )
        >>>
        >>> # Align samples
        >>> pipeline.align_samples()
        >>>
        >>> # Compute population structure
        >>> pipeline.compute_population_structure(n_pcs=3, calculate_kinship=True)
        >>>
        >>> # Run GWAS
        >>> pipeline.run_analysis(
        ...     traits=['Height', 'FloweringTime'],
        ...     methods=['GLM', 'MLM']
        ... )
        >>>
        >>> # Results saved to ./my_gwas/

    See Also:
        docs/quickstart.md: Quick start guide
        docs/api_reference.md: Complete API documentation
        examples/: Example scripts
    """

    def __init__(self, output_dir: str = "./GWAS_results"):
        """
        Initialize the GWAS Pipeline.

        Creates output directory if it doesn't exist and initializes all data
        storage attributes to None/empty.

        Args:
            output_dir (str): Directory where results and plots will be saved.
                            Default: './GWAS_results'

        Example:
            >>> pipeline = GWASPipeline(output_dir='./my_analysis')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.phenotype_df: Optional[pd.DataFrame] = None
        self.genotype_matrix: Optional[GenotypeMatrix] = None
        self.geno_map = None  # GenotypeMap or similar object
        self.individual_ids: List[str] = []
        self._matched_indices: Optional[np.ndarray] = None
        
        self.covariate_df: Optional[pd.DataFrame] = None
        self.covariate_names: List[str] = []
        
        # QC / Metadata
        self.effective_tests_info: Optional[Dict] = None
        
        # Population Structure
        self.pcs: Optional[np.ndarray] = None
        self.pc_names: List[str] = []
        self.kinship: Optional[np.ndarray] = None
        self._structure_indices: Optional[np.ndarray] = None
        # Cache the genotype subset from compute_population_structure for reuse
        self._structure_genotype: Optional[GenotypeMatrix] = None

        self._structure_n_pcs: int = 0
        # Set when structure covariates are not plain all-marker PCs (LD-pruned
        # PCA and/or admixture Q); such covariates must never be recomputed by
        # the all-marker PCA fallback in _prepare_trait.
        self._structure_custom: bool = False
        self.structure_marker_indices: Optional[np.ndarray] = None
        self.admixture = None
        # REML h2 per trait (null model, global kinship).
        self.heritability: Dict[str, REMLHeritability] = {}
        self._h2_eigen_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self.ld_decay: Optional[LDDecay] = None

        self._trait_cache: Optional[TraitPreparation] = None

        # Cache LOCO kinship objects keyed by trait-specific sample subsets.
        self._loco_kinship_cache: Dict[Tuple[int, int], Any] = {}
        self._loco_kinship_cache_max_entries: int = 4
        self.genotype_file: Optional[str] = None
        
        # Analysis State
        self.results: Dict[str, Dict[str, Any]] = {}  # {trait: {method: result}}

    def _clear_trait_cache(self) -> None:
        self._trait_cache = None
        self._loco_kinship_cache.clear()

    @staticmethod
    def _sample_subset_cache_key(indices: np.ndarray) -> Tuple[int, int]:
        arr = np.ascontiguousarray(indices, dtype=np.int64)
        return int(arr.size), hash(arr.tobytes())

    def _get_or_create_loco_kinship(
        self,
        genotype_subset: GenotypeMatrix,
        subset_indices: np.ndarray,
        *,
        maxLine: int = 5000,
        map_data=None,
        keep_indices: Optional[np.ndarray] = None,
    ):
        """Return cached LOCO kinship for a trait sample subset, computing on miss.

        Pass map_data when markers have been filtered (e.g., MAC filter) so
        chromosome grouping aligns with the filtered genotype. Defaults to
        self.geno_map.
        """
        if map_data is None:
            map_data = self.geno_map
        if map_data is None:
            return None

        # Key by sample subset AND map identity so that filtered-vs-unfiltered
        # maps don't share a cache entry.
        key = (self._sample_subset_cache_key(subset_indices), id(map_data))
        cached = self._loco_kinship_cache.get(key)
        if cached is not None:
            return cached

        loco_kinship = self._load_or_compute_loco_kinship(
            genotype_subset,
            subset_indices,
            map_data=map_data,
            maxLine=maxLine,
            keep_indices=keep_indices,
        )
        self._loco_kinship_cache[key] = loco_kinship

        # Keep cache bounded to avoid unbounded memory growth for many traits.
        while len(self._loco_kinship_cache) > self._loco_kinship_cache_max_entries:
            first_key = next(iter(self._loco_kinship_cache.keys()))
            del self._loco_kinship_cache[first_key]

        return loco_kinship

    def _load_or_compute_loco_kinship(
        self,
        genotype_subset: GenotypeMatrix,
        subset_indices: np.ndarray,
        *,
        map_data,
        maxLine: int,
        keep_indices: Optional[np.ndarray] = None,
    ):
        """Load leave-one-group Gram objects from disk, or compute and store them."""
        from ..matrix.loco_cache import (
            load_loco_kinship,
            loco_cache_digest,
            loco_cache_path,
            resolve_chrom_order,
            save_loco_kinship,
        )

        cache_path = None
        cache_base = self.genotype_file
        if cache_base:
            digest = loco_cache_digest(
                subset_indices,
                keep_indices,
                resolve_chrom_order(map_data),
                n_markers=int(genotype_subset.n_markers),
                max_line=int(maxLine),
            )
            cache_path = loco_cache_path(cache_base, digest)
            loaded = load_loco_kinship(cache_path)
            if loaded is not None:
                self.log(f"   [Cache] Using cached LOCO kinship ({digest[:12]})")
                return loaded

        loco_kinship = PANICLE_K_VanRaden_LOCO(
            genotype_subset,
            map_data,
            maxLine=maxLine,
            verbose=False,
        )
        if cache_path is not None:
            try:
                save_loco_kinship(cache_path, loco_kinship)
            except OSError as exc:
                self.log(f"   [Cache] Could not write LOCO kinship cache: {exc}")
        return loco_kinship

    @staticmethod
    def _association_genotype(
        genotype_view: GenotypeMatrix,
        keep_indices: Optional[np.ndarray],
    ) -> GenotypeMatrix:
        """Pack a (possibly lazy) aligned view into the scan matrix.

        When ``keep_indices`` is set this run-length-compacts those columns
        and the selected rows into a dense ``(n_valid, n_keep)`` buffer.
        """
        return association_genotype(genotype_view, keep_indices)

    @staticmethod
    def _ensure_gwas_eager_genotype(genotype_subset: GenotypeMatrix) -> GenotypeMatrix:
        return association_genotype(genotype_subset)
        
    def log(self, message: str):
        """Internal logger (can be replaced with standard logging later)"""
        print(message)
        
    def log_step(self, step_name: str, start_time: Optional[float] = None):
        """Log a pipeline step with optional timing"""
        if start_time is not None:
            elapsed = time.time() - start_time
            self.log(f"{step_name} completed in {elapsed:.2f} seconds")
        else:
            self.log(f"{step_name}...")

    def load_data(self,
                  phenotype_file: str,
                  genotype_file: str,
                  map_file: Optional[str] = None,
                  genotype_format: Optional[str] = None,
                  trait_columns: Optional[List[str]] = None,
                  covariate_file: Optional[str] = None,
                  covariate_columns: Optional[List[str]] = None,
                  covariate_id_column: str = 'ID',
                  loader_kwargs: Optional[Dict[str, Any]] = None,
                  phenotype_id_column: str = 'ID'):
        """
        Load and validate phenotype, genotype, and optional covariate data.

        This method loads all input files and performs basic validation. It sets the
        genotype_matrix, phenotype_df, geno_map, and optionally covariate_df attributes.

        Args:
            phenotype_file (str): Path to phenotype CSV file. Must include an ID column.
            genotype_file (str): Path to genotype file. Supported formats:
                                - VCF/VCF.GZ: Standard variant call format
                                - HapMap: TASSEL HapMap format
                                - Plink: Binary plink (.bed/.bim/.fam)
            map_file (str, optional): Path to genetic map file to override map from
                                    genotype file. Default: None (use genotype file map)
            genotype_format (str, optional): Genotype file format ('vcf', 'hapmap', 'plink').
                                           Auto-detected if None. Default: None
            trait_columns (list, optional): Which phenotype columns to load. If None,
                                          loads all numeric columns. Default: None
            covariate_file (str, optional): Path to external covariate CSV file with
                                          'ID' column. Default: None
            covariate_columns (list, optional): Which covariate columns to use. If None,
                                              uses all columns except ID. Default: None
            covariate_id_column (str): Column name for individual IDs in covariate file.
                                     Default: 'ID'
            loader_kwargs (dict, optional): Additional arguments for genotype loader:
                - compute_effective_tests (bool): Calculate M_eff (Li et al. 2012)
                - effective_test_kwargs (dict): Parameters for effective test calculation
            phenotype_id_column (str): Column name for sample IDs in phenotype file.

        Sets:
            self.phenotype_df: DataFrame with 'ID' column + trait columns
            self.genotype_matrix: GenotypeMatrix (n_individuals × n_markers)
            self.geno_map: GenotypeMap with marker information
            self.individual_ids: List of individual IDs from genotype file
            self.covariate_df: DataFrame with covariates (if covariate_file provided)
            self.effective_tests_info: Dict with M_eff if computed

        Raises:
            ValueError: If files cannot be loaded or validated

        Example:
            >>> pipeline.load_data(
            ...     phenotype_file='phenos.csv',
            ...     genotype_file='genos.vcf.gz'
            ... )

            >>> # With covariates
            >>> pipeline.load_data(
            ...     phenotype_file='phenos.csv',
            ...     genotype_file='genos.vcf.gz',
            ...     covariate_file='fields.csv',
            ...     covariate_columns=['Field', 'Year']
            ... )

            >>> # With effective tests
            >>> pipeline.load_data(
            ...     phenotype_file='phenos.csv',
            ...     genotype_file='genos.vcf.gz',
            ...     loader_kwargs={'compute_effective_tests': True}
            ... )

        Note:
            Call align_samples() after load_data() to match individuals across datasets.
        """
        step_start = time.time()
        self.log_step("Step 1: Loading and validating input data")
        self._clear_trait_cache()
        self._matched_indices = None
        self._structure_indices = None
        self.genotype_file = str(genotype_file)

        # 1. Phenotype
        try:
            self.phenotype_df = load_phenotype_file(
                phenotype_file,
                trait_columns=trait_columns,
                id_column=phenotype_id_column,
            )
            self.log(f"   Loaded {len(self.phenotype_df)} individuals with {len(self.phenotype_df.columns) - 1} traits")
        except Exception as e:
            raise ValueError(f"Error loading phenotype file: {e}")

        # 2. Genotype
        if genotype_format is None:
            genotype_format = detect_file_format(genotype_file)
            self.log(f"   Detected genotype format: {genotype_format}")

        loader_kwargs = dict(loader_kwargs or {})
        compute_effective = loader_kwargs.pop('compute_effective_tests', False)
        effective_kwargs = loader_kwargs.pop('effective_test_kwargs', None)

        try:
            self.genotype_matrix, self.individual_ids, self.geno_map = load_genotype_file(
                genotype_file,
                file_format=genotype_format,
                compute_effective_tests=compute_effective,
                effective_test_kwargs=effective_kwargs,
                **loader_kwargs,
            )
            self.log(f"   Loaded {self.genotype_matrix.n_individuals} individuals x {self.genotype_matrix.n_markers} markers")
            
            # VCF warning handler. Chromosome order is already cached on the
            # map; do not call to_dataframe() here (that materializes every
            # marker row just to inspect a handful of contig labels).
            if genotype_format == 'vcf' and _map_has_non_numeric_chrom_labels(self.geno_map):
                self.log("   Note: htslib may print [W::vcf_parse] warnings for non-contig lines.")

            self.effective_tests_info = self.geno_map.metadata.get("effective_tests")
            if self.effective_tests_info:
                me_value = int(self.effective_tests_info.get("Me", 0))
                total_snps = self.effective_tests_info.get("total_snps", self.geno_map.n_markers)
                self.log(f"   Effective tests (Li et al. 2012): {me_value:,} across {total_snps:,} markers")

        except Exception as e:
            raise ValueError(f"Error loading genotype file: {e}")

        # 3. Covariates
        if covariate_file:
            try:
                self.covariate_df = load_covariate_file(
                    covariate_file,
                    covariate_columns=covariate_columns,
                    id_column=covariate_id_column,
                )
                self.covariate_names = [c for c in self.covariate_df.columns if c != 'ID']
                self.log(f"   Loaded {len(self.covariate_df)} individuals with {len(self.covariate_names)} covariate columns")
            except Exception as e:
                raise ValueError(f"Error loading covariate file: {e}")

        # 4. Map File (Optional override)
        if map_file:
            try:
                supplied_map = load_map_file(map_file)
                self.log(f"   Loaded map for {supplied_map.n_markers} markers")
                
                if supplied_map.n_markers != self.geno_map.n_markers:
                    raise ValueError(f"Map marker count ({supplied_map.n_markers}) != genotype marker count ({self.geno_map.n_markers})")
                genotype_marker_ids = self.geno_map.marker_ids.astype(str).to_numpy()
                supplied_marker_ids = supplied_map.marker_ids.astype(str).to_numpy()
                if not np.array_equal(genotype_marker_ids, supplied_marker_ids):
                    # Lengths already match (checked above); names differ. Treat the
                    # supplied map as a positional override (legitimate for CSV/TSV
                    # genotypes whose marker IDs come from column headers) and warn.
                    mismatch = np.flatnonzero(genotype_marker_ids != supplied_marker_ids)
                    first_idx = int(mismatch[0]) if mismatch.size else 0
                    self.log(
                        "   Warning: Map marker IDs differ from genotype marker IDs "
                        f"(e.g. position {first_idx}: genotype {genotype_marker_ids[first_idx]!r}, "
                        f"map {supplied_marker_ids[first_idx]!r}). Lengths match; applying map "
                        "positions by order (positional override)."
                    )
                
                self.geno_map = supplied_map
                
                # Re-compute effective tests if strictly needed on new map (logic simplified from original script)
                if compute_effective:
                    effective_info = estimate_effective_tests_from_genotype(
                        self.genotype_matrix,
                        self.geno_map,
                        **(effective_kwargs or {}),
                    )
                    self.geno_map.metadata["effective_tests"] = effective_info
                    self.effective_tests_info = effective_info
                    self.log(f"   Effective tests re-calculated with new map: {effective_info.get('Me', 0):,}")
            except Exception as e:
                raise ValueError(f"Error loading map file: {e}")

        self.log_step("Data loading", step_start)

    def align_samples(self):
        """
        Match and align individuals between phenotype, genotype, and covariate datasets.

        This method identifies the intersection of individuals present in all loaded
        datasets and subsets each dataset to include only common individuals. This
        ensures all subsequent analyses use the same set of individuals.

        Requires:
            - load_data() must have been called first

        Sets:
            Updates self.phenotype_df, self.genotype_matrix, and self.covariate_df
            to contain only matched individuals in the same order.

        Raises:
            ValueError: If load_data() hasn't been called
            ValueError: If no common individuals found between datasets

        Example:
            >>> pipeline.load_data('phenos.csv', 'genos.vcf.gz')
            >>> pipeline.align_samples()
            # Output:
            #    Original phenotypes: 1000
            #    Original genotypes: 800
            #    Matched Intersection: 750

        Note:
            - Sample matching is case-sensitive and requires exact ID matches
            - If no overlap exists, an error is raised
            - This is a required step before running association analyses
        """
        if self.phenotype_df is None or self.genotype_matrix is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        step_start = time.time()
        self.log_step("Step 2: Matching individuals between datasets")
        self._clear_trait_cache()

        matched_phenotype, matched_covariate, matched_indices, summary = match_individuals(
            self.phenotype_df,
            self.individual_ids,
            covariate_df=self.covariate_df,
        )

        # Update Pipeline State
        self.phenotype_df = matched_phenotype
        self._matched_indices = np.asarray(matched_indices, dtype=int)
        self.pcs = None
        self.pc_names = []
        self.kinship = None
        self._structure_indices = None
        
        self.log(f"   Original phenotypes: {summary['n_phenotype_original']}")
        self.log(f"   Original genotypes: {summary['n_genotype_original']}")
        self.log(f"   Matched Intersection: {summary['n_common']}")
        
        if matched_covariate is not None:
             self.covariate_df = matched_covariate
             # Re-verify names
             matched_cov_names = [c for c in matched_covariate.columns if c != 'ID']
             if self.covariate_names and len(self.covariate_names) == len(matched_cov_names):
                 pass # keep names
             else:
                 self.covariate_names = matched_cov_names
             self.log(f"   Covariates matched: {summary['n_covariate_matched']}")
        else:
             self.log("   No external covariates used (or all unmatched).")

        self.log_step("Individual matching", step_start)

    def compute_population_structure(
        self,
        n_pcs: int = 3,
        calculate_kinship: bool = True,
        *,
        ld_prune_pca: bool = False,
        prune_r2: float = 0.2,
        prune_window_kb: float = 500.0,
        prune_max_window_snps: int = 500,
        prune_min_maf: float = 0.01,
        admixture_k: int = 0,
        admixture_params: Optional[Dict] = None,
    ):
        """
        Calculate principal components and/or kinship matrix for population structure correction.

        This method computes:
        1. Principal components (PCs) from genotype data for use as covariates
        2. Kinship matrix using VanRaden (2008) method for random effects

        Principal components help control for population stratification by including
        them as fixed-effect covariates. The kinship matrix accounts for sample
        relatedness in mixed linear models.

        Args:
            n_pcs (int): Number of principal components to compute. Set to 0 to skip PCA.
                       Default: 3
            calculate_kinship (bool): Whether to calculate a global kinship matrix.
                                    Only required for non-LOCO MLM. Default: True
                                    (but can skip for map-backed MLM/LOCO, GLM,
                                    FarmCPU, or BLINK)
            ld_prune_pca (bool): Compute PCs on an LD-pruned marker subset
                       (in-sample r^2; no reference panel needed). Prevents
                       long LD blocks (inversions, introgressions, sweeps) from
                       dominating the top PCs. Requires a genetic map.
            prune_r2, prune_window_kb, prune_max_window_snps, prune_min_maf:
                       Pruning parameters (pairs within the window with
                       r^2 > prune_r2 lose their lower-MAF member).
            admixture_k (int): If >= 2, fit the ADMIXTURE model with K
                       ancestral populations on the LD-pruned markers and add
                       K-1 ancestry proportions (Q1..Q{K-1}) as covariates,
                       after any PCs ("Q + K" model). Requires a genetic map.
            admixture_params (dict): Passed to ``fit_admixture`` (max_iter,
                       tol, n_restarts, seed).

        Sets:
            self.pcs: ndarray of shape (n_individuals, n_pcs) if n_pcs > 0
            self.pc_names: List of PC column names ['PC1', 'PC2', ...]
            self.kinship: ndarray of shape (n_individuals, n_individuals) if calculate_kinship=True

        Raises:
            ValueError: If genotype data not loaded (call load_data() and align_samples() first)

        Example:
            >>> # Compute 5 PCs and kinship
            >>> pipeline.compute_population_structure(n_pcs=5, calculate_kinship=True)

            >>> # Only compute kinship (no PCA)
            >>> pipeline.compute_population_structure(n_pcs=0, calculate_kinship=True)

            >>> # Only compute PCs (for GLM with covariate correction)
            >>> pipeline.compute_population_structure(n_pcs=3, calculate_kinship=False)

        Note:
            - PCs are automatically used as covariates in subsequent run_analysis() calls
            - PCs are combined with any external covariates loaded via covariate_file
            - Kinship calculation uses VanRaden (2008) method: K = XX' / m
            - Only non-LOCO MLM requires the global kinship matrix; map-backed MLM uses LOCO
              kinships during run_analysis(), and GLM, FarmCPU, and BLINK do not use it
            - If you skip kinship here, run_analysis() will auto-compute it when MLM is requested
              and no map data is available
            - Recommended to use 3-10 PCs depending on population structure complexity
        """
        if self.genotype_matrix is None:
             raise ValueError("Genotype data missing.")
        if self._matched_indices is None:
             raise ValueError("Samples not aligned. Call align_samples() first.")

        self._structure_n_pcs = int(n_pcs)
        self._clear_trait_cache()
        self._h2_eigen_cache.clear()
        self.ld_decay = None

        step_start = time.time()
        self.log_step("Step 3: Calculating population structure")
        structure_indices = np.asarray(self._matched_indices, dtype=int)
        is_full_geno = (
            structure_indices.size == self.genotype_matrix.n_individuals
            and np.array_equal(structure_indices, np.arange(self.genotype_matrix.n_individuals))
        )
        already_eager = (
            is_full_geno
            and not getattr(self.genotype_matrix, "is_memmap", False)
            and not getattr(self.genotype_matrix, "has_row_subset", False)
        )
        if already_eager:
            geno_for_structure = self.genotype_matrix
        else:
            self.log(
                f"   Materializing aligned genotype "
                f"({structure_indices.size} × {self.genotype_matrix.n_markers})"
            )
            geno_for_structure = self.genotype_matrix.subset_individuals(
                structure_indices,
                materialize=True,
            )
        self._structure_indices = structure_indices
        # Eager aligned buffer: PCA and per-mask compact read from this, not the mmap.
        self._structure_genotype = geno_for_structure

        admixture_k = int(admixture_k or 0)
        if admixture_k == 1 or admixture_k < 0:
            raise ValueError("admixture_k must be 0 (off) or >= 2")
        need_pruned = (ld_prune_pca and n_pcs > 0) or admixture_k >= 2
        self._structure_custom = bool(need_pruned)
        self.structure_marker_indices = None
        self.admixture = None
        if need_pruned:
            if self.geno_map is None:
                raise ValueError("LD pruning / admixture require a genetic map (chromosome, position)")
            self.log(
                f"   LD pruning markers in-sample (r2>{prune_r2}, window {prune_window_kb:g} kb, "
                f"<= {prune_max_window_snps} SNPs, MAF>={prune_min_maf})..."
            )
            self.structure_marker_indices = ld_prune(
                geno_for_structure,
                self.geno_map.chromosomes.to_numpy(),
                self.geno_map.positions.to_numpy(),
                r2_threshold=prune_r2,
                window_kb=prune_window_kb,
                max_window_snps=prune_max_window_snps,
                min_maf=prune_min_maf,
            )
            n_kept = int(self.structure_marker_indices.size)
            self.log(f"   Retained {n_kept}/{geno_for_structure.n_markers} markers after pruning")
            if n_kept < 10:
                raise ValueError(f"Only {n_kept} markers survived LD pruning; relax prune settings")
            pd.DataFrame({
                "SNP": self.geno_map.marker_ids.to_numpy()[self.structure_marker_indices],
                "CHROM": self.geno_map.chromosomes.to_numpy()[self.structure_marker_indices],
                "POS": self.geno_map.positions.to_numpy()[self.structure_marker_indices],
            }).to_csv(self.output_dir / "structure_pruned_markers.csv", index=False)

        # PCA
        if n_pcs > 0:
            try:
                if ld_prune_pca:
                    self.log(f"   Calculating {n_pcs} PCs on LD-pruned markers...")
                    pruned = geno_for_structure.get_columns_imputed(
                        self.structure_marker_indices, dtype=np.float32,
                    )
                    self.pcs = PANICLE_PCA(M=pruned, pcs_keep=n_pcs, verbose=False)
                else:
                    self.log(f"   Calculating {n_pcs} PCs...")
                    self.pcs = PANICLE_PCA(M=geno_for_structure, pcs_keep=n_pcs, verbose=False)
                self.pc_names = [f'PC{i + 1}' for i in range(self.pcs.shape[1])]
            except Exception as e:
                raise ValueError(f"Error calculating PCs: {e}")
        else:
            self.pcs = np.zeros((geno_for_structure.n_individuals, 0))
            self.pc_names = []
            self.log("   WARNING: Running with 0 PCs. This may lead to inflated p-values if")
            self.log("            your samples have population structure. Consider using --n-pcs 3")
            self.log("            or higher to control for population stratification.")

        # Admixture Q covariates
        if admixture_k >= 2:
            params = dict(admixture_params or {})
            self.log(f"   Fitting admixture model (K={admixture_k}) on pruned markers...")
            self.admixture = fit_admixture(
                geno_for_structure, admixture_k,
                marker_indices=self.structure_marker_indices, **params,
            )
            self.log(
                f"   Admixture log-likelihood {self.admixture.log_likelihood:.1f} "
                f"({self.admixture.n_iter} EM steps, converged={self.admixture.converged})"
            )
            q_cov = self.admixture.covariates()
            self.pcs = np.column_stack([self.pcs, q_cov]) if self.pcs.size else q_cov
            self.pc_names = self.pc_names + [f"Q{i + 1}" for i in range(q_cov.shape[1])]

        if self.pc_names:
            ids = self.phenotype_df["ID"].astype(str).to_numpy() if self.phenotype_df is not None else None
            structure_df = pd.DataFrame(self.pcs, columns=self.pc_names)
            if self.admixture is not None:
                for k in range(self.admixture.Q.shape[1]):
                    structure_df[f"Q{k + 1}_full"] = self.admixture.Q[:, k]
            if ids is not None and len(ids) == structure_df.shape[0]:
                structure_df.insert(0, "ID", ids)
            structure_df.to_csv(self.output_dir / "population_structure_covariates.csv", index=False)
        self._structure_n_pcs = int(self.pcs.shape[1])

        # Kinship
        if calculate_kinship:
            try:
                self.log("   Calculating Kinship matrix...")
                self.kinship = PANICLE_K_VanRaden(geno_for_structure, verbose=False)
                self.log(f"   Kinship shape: {self.kinship.shape}")
            except Exception as e:
                raise ValueError(f"Error calculating kinship: {e}")
        
        self.log_step("Population structure", step_start)

    def run_analysis(self,
                     traits: Optional[List[str]] = None,
                     methods: List[str] = ['GLM', 'MLM', 'FARMCPU', 'BLINK'],
                     max_iterations: int = 10,
                     ncpus: int = 1,
                     parallel_mode: str = "auto",
                     significance: Optional[float] = None,
                     alpha: float = 0.05,
                     n_eff: Optional[int] = None,
                     use_effective_tests: bool = True,
                     max_genotype_dosage: float = 2.0,
                     min_mac: int = 10,
                     mlm_mode: str = "loco",
                     farmcpu_params: Optional[Dict] = None,
                     blink_params: Optional[Dict] = None,
                     bayesloco_params: Optional[Dict] = None,
                     outputs: List[str] = list(OUTPUT_CHOICES),
                     include_standard_errors: bool = False,
                     identify_loci: bool = True,
                     heritability: bool = True,
                     post_gwas_params: Optional[Dict] = None):
        """
        Run GWAS analysis for specified traits and methods.

        Parameters
        ----------
        identify_loci : bool, default True
            Clump significant markers into independent loci using in-sample
            r^2 and write ``GWAS_<trait>_<method>_loci.csv``.
        heritability : bool, default True
            Report SNP heritability (``H2_REML``, ``H2_REML_SE``) from the
            REML null model y = Xb + g + e with the global (all-chromosome)
            VanRaden kinship and the scan's fixed covariates, and write
            ``GWAS_<trait>_heritability.csv``.
        post_gwas_params : dict, optional
            Overrides for ``PostGWASOptions`` (clump_kb, clump_r2,
            clump_p_secondary, merge_gap_kb).
        mlm_mode : {'loco', 'global'}, default 'loco'
            How MLM handles relatedness. ``'loco'`` uses leave-one-chromosome-out
            kinship when a genetic map is available (falls back to global
            kinship without a map). ``'global'`` always uses a single
            VanRaden kinship for all markers (computed via
            ``compute_population_structure(calculate_kinship=True)`` or
            auto-computed when needed).
        """
        if self.phenotype_df is None:
            raise ValueError("Data not loaded.")

        mlm_mode_norm = normalize_mlm_mode(mlm_mode)
        post_options = PostGWASOptions.from_dict(identify_loci, heritability, post_gwas_params)

        self.log_step("Step 4: Running GWAS analysis")
        method_cpus = _resolve_method_cpu(ncpus=ncpus, parallel_mode=parallel_mode)
        self.log(f"   Method CPU setting: {method_cpus} (ncpus={ncpus}, parallel_mode={parallel_mode})")
        self.log(f"   BLAS: {format_blas_runtime()}")

        # 1. Trait Selection — with case-insensitive / whitespace-stripped fallback
        available_traits = [c for c in self.phenotype_df.columns if c != 'ID' and pd.api.types.is_numeric_dtype(self.phenotype_df[c])]
        if traits:
             lower_map = {t.lower(): t for t in available_traits}
             stripped_map = {t.strip().lower(): t for t in available_traits}

             selected_traits = []
             still_missing = []
             for t in traits:
                 if t in available_traits:
                     selected_traits.append(t)
                 elif t.lower() in lower_map:
                     actual = lower_map[t.lower()]
                     selected_traits.append(actual)
                     self.log(f"   Note: trait '{t}' matched '{actual}' (case-insensitive)")
                 elif t.strip().lower() in stripped_map:
                     actual = stripped_map[t.strip().lower()]
                     selected_traits.append(actual)
                     self.log(f"   Note: trait '{t}' matched '{actual}' (after stripping whitespace)")
                 else:
                     still_missing.append(t)

             if still_missing:
                 self.log(f"   Warning: Traits not found or non-numeric: {still_missing}")
                 self.log(f"   Available traits: {available_traits}")
        else:
             selected_traits = available_traits

        if not selected_traits:
            if available_traits:
                raise ValueError(f"No valid traits found to analyze. Available traits: {available_traits}")
            else:
                all_cols = [c for c in self.phenotype_df.columns if c != 'ID']
                raise ValueError(
                    "No valid traits found to analyze. "
                    "The phenotype file has no numeric trait columns loaded. "
                    "Check that the --traits names match the column names in "
                    f"your phenotype file. Loaded columns (non-ID): {all_cols}"
                )

        methods_upper_check = [m.upper() for m in methods]
        if 'BAYESLOCO' in methods_upper_check:
            if self.geno_map is None:
                raise ValueError("BAYESLOCO requires map_data with chromosome labels")
            try:
                bl_cfg = BayesLocoConfig.from_object(bayesloco_params)
                bl_cfg.validate()
            except Exception as exc:
                raise ValueError(f"Invalid BAYESLOCO configuration: {exc}") from exc
            if bl_cfg.calibrate_stat_scale == "unrelated_subset" and bl_cfg.unrelated_subset_indices is None:
                raise ValueError(
                    "BAYESLOCO unrelated_subset calibration requires unrelated_subset_indices in bayesloco_params"
                )

        # Global kinship is required for global MLM, or LOCO MLM when no map.
        need_kinship = (
            'MLM' in methods_upper_check
            and (mlm_mode_norm == "global" or self.geno_map is None)
        )
        use_loco_mlm = (
            'MLM' in methods_upper_check
            and mlm_mode_norm == "loco"
            and self.geno_map is not None
        )
        if 'MLM' in methods_upper_check:
            if use_loco_mlm:
                self.log("   MLM mode: loco (leave-one-chromosome-out kinship)")
            else:
                reason = (
                    "mlm_mode=global"
                    if mlm_mode_norm == "global"
                    else "no genetic map; falling back to global kinship"
                )
                self.log(f"   MLM mode: global ({reason})")
        structure_n_pcs = self._structure_n_pcs

        # 2. Bonferroni / Thresholding Logic
        n_markers = self.genotype_matrix.n_markers
        estimated_me = (self.effective_tests_info or {}).get("Me")
        base_policy = resolve_base_threshold(
            n_markers=n_markers, alpha=alpha, significance=significance,
            n_eff=n_eff, use_effective_tests=use_effective_tests, estimated_me=estimated_me,
        )
        if significance is not None:
            self.log(f"   Using fixed significance threshold: {base_policy.value}")
        else:
            if not n_eff and use_effective_tests and estimated_me:
                self.log(f"   Using effective tests (Me={base_policy.n_tests}) for Bonferroni.")
            self.log(f"   Calculated Bonferroni threshold: {base_policy.value:.2e} (alpha={alpha}, n={base_policy.n_tests})")

        execution_order = ordered_pipeline_methods(methods)
        report_options = ReportOptions(outputs, alpha, max_genotype_dosage, include_standard_errors)

        # 3. Main Loop over Traits
        summary_rows = []
        prepared_traits: List[PreparedTrait] = []

        mac_dropped_total = 0
        for trait_name in selected_traits:
            trait_data = self._prepare_trait(
                trait_name,
                n_pcs=structure_n_pcs,
                need_kinship=need_kinship,
                min_mac=min_mac,
                max_dosage=max_genotype_dosage,
            )
            if not trait_data:
                continue
            trait_keep_indices = trait_data.keep_indices
            if trait_keep_indices is not None:
                n_drop = int(n_markers - trait_keep_indices.size)
                if n_drop > 0:
                    mac_dropped_total += n_drop
                    self.log(
                        f"   [{trait_name}] MAC filter (min_mac={int(min_mac)}) "
                        f"dropped {n_drop}/{n_markers} markers "
                        f"({100.0*n_drop/max(n_markers,1):.1f}%)"
                    )
            prepared_traits.append(trait_data)

        grouped_glm_results: Dict[str, AssociationResults] = {}
        grouped_mlm_results: Dict[str, AssociationResults] = {}
        packed_by_subset: Dict[Tuple[int, int], GenotypeMatrix] = {}
        remaining_pack_uses: Dict[Tuple[int, int], int] = {}
        for item in prepared_traits:
            subset_key = self._sample_subset_cache_key(item.sample_indices)
            remaining_pack_uses[subset_key] = remaining_pack_uses.get(subset_key, 0) + 1
        subset_groups = {
            key: [prepared_traits[i] for i in positions]
            for key, positions in group_sample_indices([t.sample_indices for t in prepared_traits]).items()
        }

        for group_items in subset_groups.values():
            if len(group_items) < 2:
                continue

            group_trait_names = [item.name for item in group_items]
            group_indices = group_items[0].sample_indices
            group_map = group_items[0].geno_map
            group_keep_indices = group_items[0].keep_indices
            group_geno = self._association_genotype(group_items[0].genotype, group_keep_indices)
            packed_by_subset[self._sample_subset_cache_key(group_indices)] = group_geno

            if "GLM" in methods_upper_check:
                self.log(
                    "   Running grouped GLM for "
                    f"{len(group_items)} traits sharing {group_indices.size} samples"
                )
                raw_glm = run_trait_group(
                    group_items, group_geno, runner=PANICLE_GLM_MULTI,
                    options=dict(maxLine=5000, cpu=method_cpus, verbose=False),
                )
                for tname, tres in raw_glm.items():
                    grouped_glm_results[tname] = pad_association_results(
                        tres.result, group_keep_indices, n_markers, full_map=self.geno_map,
                    )

            if "MLM" in methods_upper_check and use_loco_mlm:
                self.log(
                    "   Running grouped MLM LOCO for "
                    f"{len(group_items)} traits sharing {group_indices.size} samples"
                )
                group_loco_kinship = self._get_or_create_loco_kinship(
                    group_geno,
                    group_indices,
                    maxLine=5000,
                    map_data=group_map,
                    keep_indices=group_keep_indices,
                )
                raw_mlm = run_trait_group(
                    group_items, group_geno, runner=PANICLE_MLM_LOCO_MULTI,
                    options=dict(map_data=group_map, loco_kinship=group_loco_kinship,
                                 cpu=method_cpus, verbose=False),
                )
                for tname, tres in raw_mlm.items():
                    grouped_mlm_results[tname] = pad_association_results(
                        tres.result, group_keep_indices, n_markers, full_map=self.geno_map,
                    )

        for trait in prepared_traits:
            trait_name = trait.name
            y_sub, g_view, cov_sub, k_sub = trait.phenotype, trait.genotype, trait.covariates, trait.kinship
            trait_geno_idx, trait_geno_map, trait_keep_indices = trait.sample_indices, trait.geno_map, trait.keep_indices
            self.log(f"\n-- Analyzing Trait: {trait_name} --")
            trait_start_time = time.time()
            n_samples_trait = y_sub.shape[0]
            subset_key = self._sample_subset_cache_key(trait_geno_idx)
            g_sub = packed_by_subset.get(subset_key)
            if g_sub is None:
                g_sub = self._association_genotype(g_view, trait_keep_indices)
                packed_by_subset[subset_key] = g_sub

            prepared = PreparedTrait(trait_name, y_sub, g_sub, cov_sub, k_sub, trait_geno_idx,
                                     trait_geno_map, trait_keep_indices)

            # Run methods in deterministic order. Method engines may still use
            # internal threading based on `cpu`.
            method_reports: Dict[str, MethodReport] = {}
            fc_params = farmcpu_params or {}
            thresholds = trait_thresholds(
                base=base_policy, methods=methods, n_tested=g_sub.n_markers,
                mac_filtered=trait_keep_indices is not None, significance=significance,
                alpha=alpha, n_eff=n_eff, use_effective_tests=use_effective_tests,
                estimated_me=estimated_me, farmcpu_params=fc_params,
            )
            ordered_methods = execution_order
            run_resampling = 'FARMCPURESAMPLING' in methods_upper_check

            mlm_loco_kinship = None
            mlm_kwargs = {"cpu": method_cpus}
            use_grouped_glm = "GLM" in ordered_methods and trait_name in grouped_glm_results
            use_grouped_mlm = "MLM" in ordered_methods and trait_name in grouped_mlm_results
            if "MLM" in ordered_methods and use_loco_mlm and not use_grouped_mlm:
                # Keep MLM in-process so LOCO kinship/eigens can be cached across traits.
                self.log("   Running MLM in main process (reusing LOCO cache)")
                mlm_loco_kinship = self._get_or_create_loco_kinship(
                    g_sub,
                    trait_geno_idx,
                    maxLine=5000,
                    map_data=trait_geno_map,
                    keep_indices=trait_keep_indices,
                )

            method_options = MethodOptions(
                farmcpu=fc_params, blink=blink_params or {},
                bayesloco=bayesloco_params or {}, max_iterations=max_iterations,
                n_eff=thresholds.effective_n, loco_kinship=mlm_loco_kinship,
                mlm=mlm_kwargs, ncpus=method_cpus, mlm_mode=mlm_mode_norm,
            )

            if ordered_methods:
                self.log(f"   Running analysis for: {ordered_methods}")
                for method in ordered_methods:
                    grouped = (
                        grouped_glm_results if method == "GLM" and use_grouped_glm
                        else grouped_mlm_results if method == "MLM" and use_grouped_mlm
                        else None
                    )
                    if grouped is not None:
                        completed = MethodRunResult(method, grouped[trait_name])
                        completed.lambda_gc, completed.lambda_gc_is_approx = (
                            qq_compatible_genomic_inflation_factor(completed.result.pvalues)
                        )
                    else:
                        completed = _execute_prepared_method(
                            method, prepared, method_options,
                        )
                        completed.result = pad_association_results(
                            completed.result, trait_keep_indices, n_markers, full_map=self.geno_map,
                        )
                    if completed.error:
                        self.log(f"   {method} Failed: {completed.error}")
                        continue
                    method_reports[completed.name] = MethodReport(
                        completed, thresholds.methods[completed.name],
                    )
                    if completed.lambda_gc is not None:
                        lambda_label = "Lambda (GC, approx)" if completed.lambda_gc_is_approx else "Lambda (GC)"
                        self.log(f"   {completed.name} {lambda_label}: {completed.lambda_gc:.3f}")
                        if completed.lambda_gc > 1.3:
                            self.log(f"   WARNING: Genomic inflation factor ({completed.lambda_gc:.3f}) > 1.3 for {completed.name}.")
                            self.log(f"            This suggests population stratification or other confounding.")
                            self.log(f"            Consider using MLM or adding more PCs to control inflation.")

            # Specific handling for Resampling (Sequential)
            if run_resampling:
                 try:
                     self.log("   Running FarmCPU Resampling (Sequential)...")
                     runs = fc_params.get('resampling_runs', 100)
                     sig_thresh = thresholds.methods['FarmCPUResampling'].value
                     mask_prop = fc_params.get('resampling_mask_proportion', 0.1)
                     cluster = fc_params.get('resampling_cluster_markers', False)
                     ld_thresh = fc_params.get('resampling_ld_threshold', 0.7)
                     progress_callback = fc_params.get('resampling_progress_callback')
                     if progress_callback is None and fc_params.get('resampling_progress'):
                         progress_callback = _FarmCPUResamplingProgressReporter(self.log, trait_name)
                     res = run_method(
                         "FARMCPURESAMPLING", prepared,
                         runner=PANICLE_FarmCPUResampling,
                         options=dict(
                             runs=runs,
                             significance_threshold=sig_thresh,
                             mask_proportion=mask_prop,
                             cluster_markers=cluster,
                             ld_threshold=ld_thresh,
                             trait_name=trait_name,
                             progress_callback=progress_callback,
                             verbose=False,
                         ),
                     ).result
                     method_reports['FarmCPUResampling'] = MethodReport(
                         MethodRunResult('FarmCPUResampling', res),
                         thresholds.methods['FarmCPUResampling'],
                     )
                     self.log(f"   Resampling identified {len(res.entries)} markers.")
                 except Exception as e:
                     self.log(f"   FarmCPU Resampling Failed: {e}")

            # Save and Report for this trait
            trait_runtime = time.time() - trait_start_time
            trait_summary = self._write_trait_report(
                TraitReport(
                    name=trait_name, methods=method_reports, base_threshold=thresholds.base,
                    n_samples=n_samples_trait, n_markers=n_markers, runtime_seconds=trait_runtime,
                    genotype_for_maf=g_sub, maf_keep_indices=trait_keep_indices,
                ),
                report_options,
            )
            self._post_gwas_trait(prepared, method_reports, trait_summary, post_options, genotype=g_view)
            summary_rows.extend(trait_summary)
            self.log(f"Trait {trait_name} completed in {trait_runtime:.2f} seconds")
            remaining_pack_uses[subset_key] = remaining_pack_uses.get(subset_key, 1) - 1
            if remaining_pack_uses[subset_key] <= 0:
                packed_by_subset.pop(subset_key, None)
                if self._trait_cache is not None and self._trait_cache.key.samples == np.asarray(trait_geno_idx, dtype=np.int64).tobytes():
                    self._trait_cache = None

        # Final Summary
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            sum_path = self.output_dir / "GWAS_summary_by_traits_methods.csv"
            summary_df.to_csv(sum_path, index=False)
            self.log(f"\nSaved global summary to {sum_path}")

        self.log("\nGWAS Analysis Completed Successfully.")

    def _trait_heritability(self, trait: PreparedTrait, genotype: GenotypeMatrix) -> REMLHeritability:
        """REML h2 for one trait: null model with the global kinship.

        The kinship covers all chromosomes (not LOCO), uses the trait's
        retained samples, and the eigendecomposition is cached per sample set.
        """
        idx = np.asarray(trait.sample_indices, dtype=np.int64)
        key = self._sample_subset_cache_key(idx)
        eigen = self._h2_eigen_cache.get(key)
        if eigen is None:
            K = None
            if trait.kinship is not None:
                K = np.asarray(trait.kinship.to_numpy() if hasattr(trait.kinship, "to_numpy") else trait.kinship)
            elif (
                self.kinship is not None
                and self._structure_indices is not None
                and np.array_equal(self._structure_indices, idx)
            ):
                K = np.asarray(self.kinship.to_numpy() if hasattr(self.kinship, "to_numpy") else self.kinship)
            if K is None:
                # LOCO-MLM already holds the per-chromosome Gram sums; their
                # all-chromosome total is the global VanRaden kinship.
                for (sample_key, _), loco in self._loco_kinship_cache.items():
                    if sample_key == key and hasattr(loco, "get_full"):
                        K = np.asarray(loco.get_full().to_numpy())
                        break
            if K is None:
                K = PANICLE_K_VanRaden(genotype, verbose=False)
                K = np.asarray(K.to_numpy() if hasattr(K, "to_numpy") else K)
            s, U = np.linalg.eigh(gower_scale(K))
            eigen = (s, U)
            if len(self._h2_eigen_cache) >= 4:
                self._h2_eigen_cache.pop(next(iter(self._h2_eigen_cache)))
            self._h2_eigen_cache[key] = eigen
        return reml_heritability(trait.phenotype[:, 1], None, trait.covariates, eigen=eigen)

    def _auto_clump_kb(self, r2_threshold: float) -> float:
        """Clumping window from the panel's in-sample LD decay (computed once)."""
        if self.ld_decay is None:
            geno = self._structure_genotype
            if geno is None:
                geno = self.genotype_matrix.subset_individuals(
                    np.asarray(self._matched_indices, dtype=int), materialize=True,
                )
            self.ld_decay = ld_decay(
                geno, self.geno_map.chromosomes.to_numpy(), self.geno_map.positions.to_numpy(),
            )
            self.ld_decay.to_frame().to_csv(self.output_dir / "LD_decay.csv", index=False)
        window = self.ld_decay.decay_distance_kb(r2_threshold)
        if not np.isfinite(window):
            window = 250.0
        return float(window)

    def _post_gwas_trait(self, trait: PreparedTrait, method_reports, trait_summary,
                         options: PostGWASOptions, *, genotype: GenotypeMatrix):
        """REML h2, Lambda_1000 and locus clumping for one trait.

        ``genotype`` is the trait's sample subset with every marker (not the
        MAC-filtered association matrix), aligned with the padded p-values.
        """
        h2 = None
        if options.heritability:
            try:
                h2 = self._trait_heritability(trait, genotype)
                self.heritability[trait.name] = h2
                write_heritability(trait.name, h2, self.output_dir)
                self.log(
                    f"   REML h2 (global kinship): {h2.h2:.3f} (SE {h2.h2_se:.3f}), "
                    f"LRT p={h2.lrt_p:.2e}"
                )
            except Exception as e:
                self.log(f"   REML heritability failed for {trait.name}: {e}")
        try:
            if options.loci and options.clump_kb == "auto" and self.geno_map is not None:
                window = self._auto_clump_kb(options.clump_r2)
                self.log(
                    f"   Clumping window from LD decay: {window:.0f} kb "
                    f"(Q{int(round(self.ld_decay.quantile * 100))} r2 < {options.clump_r2} beyond it)"
                )
                options = dataclasses.replace(options, clump_kb=window)
            run_post_gwas(
                trait_name=trait.name, method_reports=method_reports,
                summary_rows=trait_summary, phenotype=trait.phenotype[:, 1],
                genotype=genotype, geno_map=self.geno_map,
                output_dir=self.output_dir, options=options, log=self.log,
                heritability=h2,
            )
        except Exception as e:
            self.log(f"   Post-GWAS step failed for {trait.name}: {e}")

    def _prepare_trait(
        self,
        trait_name,
        n_pcs: int = 0,
        need_kinship: bool = False,
        min_mac: int = 0,
        max_dosage: float = 2.0,
    ) -> Optional[PreparedTrait]:
        """
        Handle missing data removal (phenotype & covariates), then apply the
        optional per-trait MAC filter (post sample-subset).

        Return a PreparedTrait, or None if no samples remain. The map is shared
        when no marker filter applies. Otherwise keep_indices selects columns
        from the sample-subset genotype; materialization happens at dispatch.
        """
        if self._matched_indices is None:
             raise ValueError("Samples not aligned. Call align_samples() first.")
        # 1. Phenotype subset
        y_vals = pd.to_numeric(self.phenotype_df[trait_name], errors='coerce').to_numpy()
        mask = retained_samples(y_vals)
        
        # 2. Covariate subset (External + PCs)
        ext_covs = None
        if self.covariate_df is not None:
             ext_covs = self.covariate_df[self.covariate_names].to_numpy(dtype=float)
             mask = retained_samples(y_vals, ext_covs)

        if mask.sum() == 0:
             self.log(f"   Skipping {trait_name}: No valid samples after QC.")
             return None

        n_excluded = int((~mask).sum())
        if n_excluded:
             excluded_ids = self.phenotype_df.loc[~mask, "ID"].astype(str).to_numpy()
             shown = ", ".join(str(x) for x in excluded_ids[:10])
             if n_excluded > 10:
                  shown += f", ... and {n_excluded - 10} more"
             self.log(
                  f"   {trait_name}: excluded {n_excluded} sample(s) with missing/non-finite phenotype or covariate values: {shown}"
             )

        idx = np.where(mask)[0]
        base_indices = np.asarray(self._matched_indices, dtype=int)
        geno_idx = base_indices[idx]
        is_full_geno = (
            geno_idx.size == self.genotype_matrix.n_individuals
            and np.array_equal(geno_idx, np.arange(self.genotype_matrix.n_individuals))
        )

        # Every parameter that changes the keep set belongs here. A false
        # hit is a silent wrong result, not a crash: same sample mask with a
        # different min_mac or max_dosage must not reuse keep_indices.
        cache_key = TraitCacheKey.create(geno_idx, n_pcs, need_kinship, min_mac, max_dosage)
        if self._trait_cache is not None and self._trait_cache.key == cache_key:
            cached = self._trait_cache
            g_final, pcs, k_final = cached.genotype, cached.pcs, cached.kinship
            keep_indices, trait_geno_map = cached.keep_indices, cached.geno_map
        else:
            # Try to reuse the genotype subset from compute_population_structure
            if is_full_geno:
                g_final = self.genotype_matrix
            elif (
                self._structure_genotype is not None
                and self._structure_indices is not None
                and np.array_equal(self._structure_indices, geno_idx)
            ):
                # Exact match - reuse cached structure genotype directly
                g_final = self._structure_genotype
            elif (
                self._structure_genotype is not None
                and self._structure_indices is not None
                and len(geno_idx) < len(self._structure_indices)
            ):
                # Trait indices are likely a subset of structure indices (due to missing phenotypes)
                # Keep a lazy row view of the aligned in-RAM buffer. Compact
                # reads those rows directly into the packed keep-set.
                structure_set = set(self._structure_indices)
                if all(idx in structure_set for idx in geno_idx):
                    idx_map = {v: i for i, v in enumerate(self._structure_indices)}
                    local_indices = np.array([idx_map[idx] for idx in geno_idx], dtype=int)
                    g_final = self._structure_genotype.subset_individuals(
                        local_indices,
                        materialize=False,
                    )
                else:
                    # Fallback: some trait indices not in structure (shouldn't happen normally)
                    g_final = self.genotype_matrix.subset_individuals(
                        geno_idx,
                        materialize=True,
                    )
            else:
                g_final = self.genotype_matrix.subset_individuals(
                    geno_idx,
                    materialize=True,
                )

            # Check if we can subset PCs/kinship from structure cache
            # (when trait indices are a subset of structure indices)
            can_subset_from_structure = (
                self._structure_indices is not None
                and len(geno_idx) <= len(self._structure_indices)
            )
            local_indices_for_structure = None
            if can_subset_from_structure and not np.array_equal(self._structure_indices, geno_idx):
                structure_set = set(self._structure_indices)
                if all(idx in structure_set for idx in geno_idx):
                    idx_map = {v: i for i, v in enumerate(self._structure_indices)}
                    local_indices_for_structure = np.array([idx_map[idx] for idx in geno_idx], dtype=int)

            if n_pcs > 0:
                if (
                    self._structure_indices is not None
                    and np.array_equal(self._structure_indices, geno_idx)
                    and self.pcs is not None
                    and self.pcs.shape[0] == geno_idx.size
                    and self.pcs.shape[1] == n_pcs
                ):
                    # Exact match - use cached PCs directly
                    pcs = self.pcs
                elif (
                    local_indices_for_structure is not None
                    and self.pcs is not None
                    and self.pcs.shape[1] == n_pcs
                ):
                    # Subset PCs from structure cache (fast, avoids recomputing PCA)
                    pcs = self.pcs[local_indices_for_structure, :]
                elif self._structure_custom:
                    raise ValueError(
                        "Trait samples are not covered by the structure covariates "
                        "(LD-pruned PCs / admixture Q); rerun compute_population_structure()."
                    )
                else:
                    pcs = PANICLE_PCA(M=g_final, pcs_keep=n_pcs, verbose=False)
            else:
                pcs = np.zeros((idx.size, 0))

            k_final = None
            if need_kinship:
                if (
                    self._structure_indices is not None
                    and np.array_equal(self._structure_indices, geno_idx)
                    and self.kinship is not None
                    and self.kinship.shape[0] == geno_idx.size
                ):
                    # Exact match - use cached kinship directly
                    k_final = self.kinship
                elif (
                    local_indices_for_structure is not None
                    and self.kinship is not None
                ):
                    # Subset kinship from structure cache (fast, avoids recomputing)
                    k_final = self.kinship[np.ix_(local_indices_for_structure, local_indices_for_structure)]
                else:
                    k_final = PANICLE_K_VanRaden(g_final, verbose=False)

            # Apply per-trait MAC filter to the association genotype matrix.
            # Kinship and PCs above are computed (or reused) on the unfiltered
            # marker set; this filter only reshapes what the marker scan sees,
            # guarding against singleton-driven spurious hits in the subset.
            selected = select_markers(
                g_final, self.geno_map, min_mac, max_dosage,
                filter_fn=compute_mac_keep_indices,
            )
            keep_indices, trait_geno_map = selected.keep_indices, selected.geno_map
            self._trait_cache = TraitPreparation(
                cache_key, g_final, pcs, k_final, trait_geno_map, keep_indices,
            )

        y_final = np.column_stack([np.arange(mask.sum()), y_vals[mask]])

        cov_parts = []
        if ext_covs is not None:
            cov_parts.append(ext_covs[mask, :])
        if pcs is not None and pcs.size > 0:
            cov_parts.append(pcs)

        cov_final = np.column_stack(cov_parts) if cov_parts else None

        return PreparedTrait(trait_name, y_final, g_final, cov_final, k_final, geno_idx, trait_geno_map, keep_indices)

    def _prepare_trait_data(self, trait_name, n_pcs=0, need_kinship=False, min_mac=0, max_dosage=2.0):
        """Compatibility wrapper for callers of the former tuple-returning helper."""
        trait = self._prepare_trait(trait_name, n_pcs, need_kinship, min_mac, max_dosage)
        return None if trait is None else trait.legacy_tuple()

    def _write_trait_report(self, report: TraitReport, options: ReportOptions):
        context = TraitOutputContext(
            self.output_dir, self.geno_map, self.genotype_matrix, self.log, PANICLE_Report,
        )
        return write_trait_report(context, report, options)

    def _save_trait_results(
        self,
        trait_name,
        results,
        threshold,
        alpha,
        n_tests,
        max_dosage,
        outputs,
        threshold_source,
        include_standard_errors: bool = False,
        method_thresholds=None,
        method_threshold_sources=None,
        method_lambda_gc=None,
        method_lambda_gc_is_approx=None,
        n_samples=None,
        n_markers=None,
        runtime_seconds=None,
        geno_for_maf: Optional[GenotypeMatrix] = None,
        maf_keep_indices: Optional[np.ndarray] = None,
    ):
        """Compatibility adapter; reporting owns tables, metadata, and plots."""
        context = TraitOutputContext(self.output_dir, self.geno_map, self.genotype_matrix, self.log, PANICLE_Report)
        return write_trait_results(
            context,
            trait_name=trait_name,
            results=results,
            threshold=threshold,
            alpha=alpha,
            n_tests=n_tests,
            max_dosage=max_dosage,
            outputs=outputs,
            threshold_source=threshold_source,
            include_standard_errors=include_standard_errors,
            method_thresholds=method_thresholds,
            method_threshold_sources=method_threshold_sources,
            method_lambda_gc=method_lambda_gc,
            method_lambda_gc_is_approx=method_lambda_gc_is_approx,
            n_samples=n_samples,
            n_markers=n_markers,
            runtime_seconds=runtime_seconds,
            geno_for_maf=geno_for_maf,
            maf_keep_indices=maf_keep_indices,
        )
