"""
GWAS Pipeline Module

This module enables a modular, object-oriented approach to running GWAS analyses.
It encapsulates data loading, sample alignment, population structure correction,
association testing, and result reporting into a reusable pipeline class.
"""

import concurrent.futures
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
    calculate_maf_from_genotypes,
    genomic_inflation_factor
)
from ..utils.data_types import GenotypeMatrix, AssociationResults
from ..utils.effective_tests import estimate_effective_tests_from_genotype
from ..association.farmcpu_resampling import (
    PANICLE_FarmCPUResampling,
    FarmCPUResamplingResults,
)
from ..association.glm import PANICLE_GLM
from ..association.mlm import PANICLE_MLM
from ..association.mlm_loco import PANICLE_MLM_LOCO
from ..association.farmcpu import PANICLE_FarmCPU
from ..association.blink import PANICLE_BLINK
from ..matrix.pca import PANICLE_PCA
from ..matrix.kinship import PANICLE_K_VanRaden
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

# Helper function for parallel execution
def _run_single_method(method, y_sub, g_sub, cov_sub, k_sub, map_data, fc_params, blk_params, max_iterations, base_threshold, n_markers, n_eff=None, alpha=0.05):
    """Worker function to run a single GWAS method in a separate process."""
    try:
        if method == 'GLM':
            res = PANICLE_GLM(phe=y_sub, geno=g_sub, CV=cov_sub, verbose=False)
            lambda_gc = genomic_inflation_factor(res.pvalues)
            return ('GLM', res, lambda_gc, None)

        elif method == 'MLM':
            if map_data is None:
                if k_sub is None:
                    return ('MLM', None, None, "Kinship matrix missing")
                res = PANICLE_MLM(phe=y_sub, geno=g_sub, CV=cov_sub, K=k_sub, verbose=False)
            else:
                res = PANICLE_MLM_LOCO(
                    phe=y_sub,
                    geno=g_sub,
                    map_data=map_data,
                    CV=cov_sub,
                    verbose=False,
                )
            lambda_gc = genomic_inflation_factor(res.pvalues)
            return ('MLM', res, lambda_gc, None)

        elif method == 'FARMCPU':
            # Pass alpha values (uncorrected) - FarmCPU applies multiple testing correction internally
            fc_p = fc_params.get('p_threshold', alpha)  # Alpha level, e.g., 0.05
            fc_qtn = fc_params.get('QTN_threshold', 0.01)  # Alpha for QTN selection, e.g., 0.01
            fc_bin = fc_params.get('bin_size')
            fc_method_bin = fc_params.get('method_bin', 'static')
            fc_converge = fc_params.get('converge', 1.0)
            res = PANICLE_FarmCPU(
                phe=y_sub, geno=g_sub, map_data=map_data, CV=cov_sub,
                maxLoop=max_iterations,
                p_threshold=fc_p,
                QTN_threshold=fc_qtn,
                n_eff=n_eff,  # Pass effective tests for multiple testing correction
                converge=fc_converge,
                bin_size=fc_bin,
                method_bin=fc_method_bin,
                verbose=False
            )
            lambda_gc = genomic_inflation_factor(res.pvalues)
            return ('FarmCPU', res, lambda_gc, None)

        elif method == 'BLINK':
            res = PANICLE_BLINK(
                phe=y_sub, geno=g_sub, map_data=map_data, CV=cov_sub,
                maxLoop=max_iterations, verbose=False
            )
            lambda_gc = genomic_inflation_factor(res.pvalues)
            return ('BLINK', res, lambda_gc, None)
            
        elif method == 'FarmCPUResampling':
            # Resampling is usually heavy and might output files directly or need special handling
            runs = fc_params.get('resampling_runs', 100)
            # trait_name is needed? Not passed here.
            # We skip this in parallel worker for now or pass trait_name?
            # It's better to keep resampling sequential if complex, or pass trait_name.
            # Let's support it if trivial.
            # PANICLE_FarmCPUResampling requires trait_name.
            # PANICLE_FarmCPUResampling requires trait_name.
            pass

        return (method, None, None, f"Unknown method {method}")

    except Exception as e:
        return (method, None, None, str(e))


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
        geno_map (GenotypeMap): Genetic map with SNP information (ID, CHROM, POS)
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

        self._trait_cache_indices: Optional[np.ndarray] = None
        self._trait_cache_n_pcs: Optional[int] = None
        self._trait_cache_need_kinship: Optional[bool] = None
        self._trait_cache_genotype: Optional[GenotypeMatrix] = None
        self._trait_cache_pcs: Optional[np.ndarray] = None
        self._trait_cache_kinship: Optional[np.ndarray] = None
        
        # Analysis State
        self.results: Dict[str, Dict[str, Any]] = {}  # {trait: {method: result}}

    def _clear_trait_cache(self) -> None:
        self._trait_cache_indices = None
        self._trait_cache_n_pcs = None
        self._trait_cache_need_kinship = None
        self._trait_cache_genotype = None
        self._trait_cache_pcs = None
        self._trait_cache_kinship = None
        
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
            self.geno_map: GenotypeMap with SNP information
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
            
            # VCF warning handler
            if genotype_format == 'vcf':
                try:
                    chrom_labels = self.geno_map.to_dataframe()['CHROM'].astype(str).unique()
                    non_numeric_chroms = [label for label in chrom_labels if not label.isdigit()]
                    if non_numeric_chroms:
                        self.log("   Note: htslib may print [W::vcf_parse] warnings for non-contig lines.")
                except:
                    pass

            self.effective_tests_info = self.geno_map.metadata.get("effective_tests")
            if self.effective_tests_info:
                me_value = int(self.effective_tests_info.get("Me", 0))
                total_snps = self.effective_tests_info.get("total_snps", self.geno_map.n_markers)
                self.log(f"   Effective tests (Li et al. 2012): {me_value:,} across {total_snps:,} SNPs")

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

    def compute_population_structure(self, n_pcs: int = 3, calculate_kinship: bool = True):
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
            calculate_kinship (bool): Whether to calculate kinship matrix. Only required
                                    for MLM. Default: True (but can skip if only using
                                    GLM, FarmCPU, or BLINK)

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
            - Only MLM requires the kinship matrix; GLM, FarmCPU, and BLINK do not use it
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

        step_start = time.time()
        self.log_step("Step 3: Calculating population structure")
        structure_indices = np.asarray(self._matched_indices, dtype=int)
        is_full_geno = (
            structure_indices.size == self.genotype_matrix.n_individuals
            and np.array_equal(structure_indices, np.arange(self.genotype_matrix.n_individuals))
        )
        if is_full_geno:
            geno_for_structure = self.genotype_matrix
        else:
            geno_for_structure = self.genotype_matrix.subset_individuals(structure_indices)
        self._structure_indices = structure_indices
        # Cache the genotype subset for reuse in _prepare_trait_data
        self._structure_genotype = geno_for_structure

        # PCA
        if n_pcs > 0:
            try:
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
                     significance: Optional[float] = None,
                     alpha: float = 0.05,
                     n_eff: Optional[int] = None,
                     use_effective_tests: bool = True,
                     max_genotype_dosage: float = 2.0,
                     farmcpu_params: Optional[Dict] = None,
                     blink_params: Optional[Dict] = None,
                     outputs: List[str] = list(OUTPUT_CHOICES)):
        """
        Run GWAS analysis for specified traits and methods.
        """
        if self.phenotype_df is None:
            raise ValueError("Data not loaded.")

        self.log_step("Step 4: Running GWAS analysis")

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
        need_kinship = 'MLM' in methods_upper_check
        structure_n_pcs = self._structure_n_pcs

        # 2. Bonferroni / Thresholding Logic
        n_markers = self.genotype_matrix.n_markers
        bonferroni_denom = float(n_markers)
        threshold_source = "Bonferroni (markers)"
        
        if significance is not None:
             base_threshold = significance
             self.log(f"   Using fixed significance threshold: {base_threshold}")
             effective_tests_count = float('nan') # User override
             threshold_source = "Fixed p-value"
        else:
             # Logic to choose denominator
             if n_eff:
                 bonferroni_denom = float(n_eff)
                 threshold_source = "Bonferroni (effective tests)"
             elif use_effective_tests and self.effective_tests_info and self.effective_tests_info.get("Me"):
                 bonferroni_denom = float(self.effective_tests_info["Me"])
                 self.log(f"   Using effective tests (Me={bonferroni_denom}) for Bonferroni.")
                 threshold_source = "Bonferroni (effective tests)"
             
             base_threshold = alpha / max(bonferroni_denom, 1.0)
             effective_tests_count = bonferroni_denom
             self.log(f"   Calculated Bonferroni threshold: {base_threshold:.2e} (alpha={alpha}, n={bonferroni_denom})")

        # 3. Main Loop over Traits
        summary_rows = []
        
        for trait_name in selected_traits:
            self.log(f"\n-- Analyzing Trait: {trait_name} --")
            trait_start_time = time.time()

            # Prepare Trait-Specific Data (remove missing pheno/covs)
            trait_data = self._prepare_trait_data(
                trait_name,
                n_pcs=structure_n_pcs,
                need_kinship=need_kinship,
            )
            if not trait_data:
                continue

            y_sub, g_sub, cov_sub, k_sub = trait_data
            n_samples_trait = y_sub.shape[0]

            # Run Methods (Parallel Execution)
            method_results = {}
            method_lambda_gc = {}  # Track lambda GC for each method
            
            # Setup params for FarmCPU/BLINK
            fc_params = farmcpu_params or {}
            blk_params = blink_params or {}
            method_thresholds: Dict[str, float] = {}
            method_threshold_sources: Dict[str, str] = {}

            # Determine effective tests for multiple testing correction (needed for FarmCPU thresholds)
            effective_n = None
            if use_effective_tests and self.effective_tests_info and self.effective_tests_info.get("Me"):
                effective_n = int(self.effective_tests_info["Me"])
            elif n_eff:
                effective_n = n_eff

            fc_qtn_alpha = fc_params.get('QTN_threshold', 0.01)
            if fc_params.get('QTN_threshold_is_corrected'):
                fc_qtn_corrected = fc_qtn_alpha
                fc_qtn_source = 'FarmCPU QTN threshold (corrected)'
            else:
                fc_n_tests = effective_n if effective_n else n_markers
                fc_qtn_corrected = fc_qtn_alpha / fc_n_tests
                fc_qtn_source = 'FarmCPU QTN threshold'

            # Identify parallelizable methods (case-insensitive matching)
            # Note: method names passed to worker must match worker's expected case
            methods_upper = [m.upper() for m in methods]
            parallel_methods = []
            if 'GLM' in methods_upper: parallel_methods.append('GLM')
            if 'MLM' in methods_upper: parallel_methods.append('MLM')
            if 'FARMCPU' in methods_upper: parallel_methods.append('FARMCPU')
            if 'BLINK' in methods_upper: parallel_methods.append('BLINK')

            # Track method-specific thresholds for plotting/reporting
            # Note: Keys must match the result names returned from parallel workers
            # (e.g., 'FarmCPU' not 'FARMCPU', 'BLINK' not 'blink')
            if 'FARMCPU' in methods_upper:
                # FarmCPU applies multiple testing correction internally
                # Use the same denominator for reporting consistency
                method_thresholds['FarmCPU'] = fc_qtn_corrected  # Match worker return name
                method_threshold_sources['FarmCPU'] = fc_qtn_source
            if 'BLINK' in methods_upper:
                method_thresholds['BLINK'] = base_threshold
                method_threshold_sources['BLINK'] = threshold_source
            if 'GLM' in methods_upper:
                method_thresholds['GLM'] = base_threshold
                method_threshold_sources['GLM'] = threshold_source
            if 'MLM' in methods_upper:
                method_thresholds['MLM'] = base_threshold
                method_threshold_sources['MLM'] = threshold_source
            if 'FARMCPURESAMPLING' in methods_upper:
                if 'resampling_significance_threshold' in fc_params:
                    resampling_thresh = fc_params['resampling_significance_threshold']
                    resampling_source = 'Resampling significance threshold'
                else:
                    resampling_thresh = fc_qtn_corrected
                    resampling_source = 'FarmCPU QTN threshold (default)'
                method_thresholds['FarmCPUResampling'] = resampling_thresh
                method_threshold_sources['FarmCPUResampling'] = resampling_source

            # Resampling usually handled separately or sequentially due to complexity
            run_resampling = 'FARMCPURESAMPLING' in methods_upper

            # Only run parallel executor if there are parallel methods to run
            if parallel_methods:
                if len(parallel_methods) > 1:
                    self.log(f"   Running parallel analysis for: {parallel_methods}")
                else:
                    self.log(f"   Running analysis for: {parallel_methods}")

                if len(parallel_methods) == 1:
                    method = parallel_methods[0]
                    res_name, res_obj, lambda_gc, error = _run_single_method(
                        method,
                        y_sub, g_sub, cov_sub, k_sub,
                        self.geno_map,
                        fc_params, blk_params,
                        max_iterations, base_threshold, n_markers,
                        effective_n, alpha,
                    )
                    if error:
                        self.log(f"   {method} Failed: {error}")
                    else:
                        method_results[res_name] = res_obj
                        if lambda_gc is not None:
                            method_lambda_gc[res_name] = lambda_gc
                            self.log(f"   {res_name} Lambda (GC): {lambda_gc:.3f}")
                            if lambda_gc > 1.3:
                                self.log(f"   WARNING: Genomic inflation factor ({lambda_gc:.3f}) > 1.3 for {res_name}.")
                                self.log(f"            This suggests population stratification or other confounding.")
                                self.log(f"            Consider using MLM or adding more PCs to control inflation.")
                else:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=min(4, len(parallel_methods))) as executor:
                        future_to_method = {
                            executor.submit(
                                _run_single_method,
                                method,
                                y_sub, g_sub, cov_sub, k_sub,
                                self.geno_map,
                                fc_params, blk_params,
                                max_iterations, base_threshold, n_markers,
                                effective_n, alpha  # Pass n_eff and alpha for FarmCPU
                            ): method for method in parallel_methods
                        }

                        for future in concurrent.futures.as_completed(future_to_method):
                            m_name = future_to_method[future]
                            try:
                                res_name, res_obj, lambda_gc, error = future.result()
                                if error:
                                    self.log(f"   {m_name} Failed: {error}")
                                else:
                                    method_results[res_name] = res_obj
                                    if lambda_gc is not None:
                                        method_lambda_gc[res_name] = lambda_gc
                                        self.log(f"   {res_name} Lambda (GC): {lambda_gc:.3f}")
                                        if lambda_gc > 1.3:
                                            self.log(f"   WARNING: Genomic inflation factor ({lambda_gc:.3f}) > 1.3 for {res_name}.")
                                            self.log(f"            This suggests population stratification or other confounding.")
                                            self.log(f"            Consider using MLM or adding more PCs to control inflation.")
                            except Exception as exc:
                                self.log(f"   {m_name} generated an exception: {exc}")

            # Specific handling for Resampling (Sequential)
            if run_resampling:
                 try:
                     self.log("   Running FarmCPU Resampling (Sequential)...")
                     runs = fc_params.get('resampling_runs', 100)
                     sig_thresh = method_thresholds.get('FarmCPUResampling', fc_qtn_corrected)
                     mask_prop = fc_params.get('resampling_mask_proportion', 0.1)
                     cluster = fc_params.get('resampling_cluster_markers', False)
                     ld_thresh = fc_params.get('resampling_ld_threshold', 0.7)
                     progress_callback = fc_params.get('resampling_progress_callback')
                     if progress_callback is None and fc_params.get('resampling_progress'):
                         progress_callback = _FarmCPUResamplingProgressReporter(self.log, trait_name)
                     res = PANICLE_FarmCPUResampling(
                         phe=y_sub, geno=g_sub, map_data=self.geno_map, CV=cov_sub,
                         runs=runs,
                         significance_threshold=sig_thresh,
                         mask_proportion=mask_prop,
                         cluster_markers=cluster,
                         ld_threshold=ld_thresh,
                         trait_name=trait_name,
                         progress_callback=progress_callback,
                         verbose=False
                     )
                     method_results['FarmCPUResampling'] = res
                     self.log(f"   Resampling identified {len(res.entries)} markers.")
                 except Exception as e:
                     self.log(f"   FarmCPU Resampling Failed: {e}")

            # Save and Report for this trait
            trait_runtime = time.time() - trait_start_time
            trait_summary = self._save_trait_results(
                trait_name, method_results,
                base_threshold, alpha, effective_tests_count,
                max_genotype_dosage, outputs, threshold_source,
                method_thresholds=method_thresholds,
                method_threshold_sources=method_threshold_sources,
                method_lambda_gc=method_lambda_gc,
                n_samples=n_samples_trait,
                n_markers=n_markers,
                runtime_seconds=trait_runtime,
                geno_for_maf=g_sub
            )
            summary_rows.extend(trait_summary)
            self.log(f"Trait {trait_name} completed in {trait_runtime:.2f} seconds")

        # Final Summary
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            sum_path = self.output_dir / "GWAS_summary_by_traits_methods.csv"
            summary_df.to_csv(sum_path, index=False)
            self.log(f"\nSaved global summary to {sum_path}")

        self.log("\nGWAS Analysis Completed Successfully.")

    def _prepare_trait_data(self, trait_name, n_pcs: int = 0, need_kinship: bool = False):
        """
        Handle missing data removal (phenotype & covariates).
        Returns (y_sub, g_sub, cov_sub, k_sub) or None if empty.
        """
        if self._matched_indices is None:
             raise ValueError("Samples not aligned. Call align_samples() first.")
        # 1. Phenotype subset
        y_vals = pd.to_numeric(self.phenotype_df[trait_name], errors='coerce').to_numpy()
        mask = np.isfinite(y_vals)
        
        # 2. Covariate subset (External + PCs)
        ext_covs = None
        if self.covariate_df is not None:
             ext_covs = self.covariate_df[self.covariate_names].to_numpy(dtype=float)
             cov_mask = np.isfinite(ext_covs).all(axis=1)
             mask = mask & cov_mask

        if mask.sum() == 0:
             self.log(f"   Skipping {trait_name}: No valid samples after QC.")
             return None

        idx = np.where(mask)[0]
        base_indices = np.asarray(self._matched_indices, dtype=int)
        geno_idx = base_indices[idx]
        is_full_geno = (
            geno_idx.size == self.genotype_matrix.n_individuals
            and np.array_equal(geno_idx, np.arange(self.genotype_matrix.n_individuals))
        )

        cache_hit = (
            self._trait_cache_indices is not None
            and self._trait_cache_n_pcs == n_pcs
            and self._trait_cache_need_kinship == need_kinship
            and np.array_equal(self._trait_cache_indices, geno_idx)
        )

        if cache_hit:
            g_final = self._trait_cache_genotype
            pcs = self._trait_cache_pcs
            k_final = self._trait_cache_kinship
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
                # Subset from the in-memory cached genotype (fast) instead of memmap (slow)
                # Create mapping: structure_indices -> local position
                structure_set = set(self._structure_indices)
                if all(idx in structure_set for idx in geno_idx):
                    idx_map = {v: i for i, v in enumerate(self._structure_indices)}
                    local_indices = np.array([idx_map[idx] for idx in geno_idx], dtype=int)
                    g_final = self._structure_genotype.subset_individuals(local_indices)
                else:
                    # Fallback: some trait indices not in structure (shouldn't happen normally)
                    g_final = self.genotype_matrix.subset_individuals(geno_idx)
            else:
                g_final = self.genotype_matrix.subset_individuals(geno_idx)

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

            self._trait_cache_indices = geno_idx
            self._trait_cache_n_pcs = n_pcs
            self._trait_cache_need_kinship = need_kinship
            self._trait_cache_genotype = g_final
            self._trait_cache_pcs = pcs
            self._trait_cache_kinship = k_final
        
        # Apply mask
        y_final = np.column_stack([
             np.arange(mask.sum()), # Dummy IDs for internal solvers usually ok, or use real strings?
             # Solvers expect [ID, Val] usually. 
             # PANICLE_GLM expects n x 2.
             y_vals[mask]
        ])
        
        # ID column is often ignored by solvers but good to be consistent
        # For simplicity, passing string IDs if available
        # But MVP solvers might expect float or string. Let's stick to simple index or values.
        # Actually MVP solvers usually take the values column.
        # Let's fix y_final to match expectation: [ID, Value]
        # Using simple numeric IDs 0..N-1 is safest for internal matrix math unless IDs are used for output
        y_final[:, 0] = np.arange(mask.sum())

        cov_parts = []
        if ext_covs is not None:
            cov_parts.append(ext_covs[mask, :])
        if pcs is not None and pcs.size > 0:
            cov_parts.append(pcs)

        cov_final = np.column_stack(cov_parts) if cov_parts else None

        return y_final, g_final, cov_final, k_final

    def _save_trait_results(self, trait_name, results, threshold, alpha, n_tests, max_dosage, outputs, threshold_source, method_thresholds=None, method_threshold_sources=None, method_lambda_gc=None, n_samples=None, n_markers=None, runtime_seconds=None, geno_for_maf: Optional[GenotypeMatrix] = None):
        """Internal helper to save tables and plots"""
        
        summary_data = []
        base_df = self.geno_map.to_dataframe().copy() if hasattr(self.geno_map, 'to_dataframe') else pd.DataFrame()

        base_columns = base_df.columns.tolist()

        def _insert_maf_column(df: pd.DataFrame, maf_values: np.ndarray) -> None:
            if 'MAF' in df.columns:
                df['MAF'] = maf_values
                return
            insert_at = len(df.columns)
            if 'ALT' in df.columns:
                insert_at = df.columns.get_loc('ALT') + 1
            df.insert(insert_at, 'MAF', maf_values)

        def _ordered_base_columns(df: pd.DataFrame) -> List[str]:
            cols = list(base_columns)
            if 'MAF' in df.columns and 'MAF' not in cols:
                if 'ALT' in cols:
                    cols.insert(cols.index('ALT') + 1, 'MAF')
                else:
                    cols.append('MAF')
            return cols

        def _maf_for_indices(indices: np.ndarray) -> np.ndarray:
            if indices.size == 0:
                return np.array([], dtype=float)
            geno_source = geno_for_maf or self.genotype_matrix
            if hasattr(geno_source, 'get_columns_imputed'):
                geno_subset = geno_source.get_columns_imputed(indices, dtype=np.float64)
                allele_freq = geno_subset.mean(axis=0) / max(max_dosage, 1e-12)
            else:
                geno_subset = np.asarray(geno_source)[:, indices]
                allele_freq = np.nanmean(geno_subset, axis=0) / max(max_dosage, 1e-12)
            return np.minimum(allele_freq, 1.0 - allele_freq)

        all_res_df = base_df.copy()
        sig_snps = []
        hits_by_method = {}
        resampling_hit_snps = set()
        rmip_hit_threshold = 0.1

        preferred_order = ['GLM', 'MLM', 'FarmCPU', 'BLINK', 'FarmCPUResampling']
        ordered_methods = [m for m in preferred_order if m in results]
        ordered_methods.extend([m for m in results if m not in ordered_methods])

        output_prefix_base = str(self.output_dir / f"GWAS_{trait_name}")

        for method in ordered_methods:
            Res = results[method]
            method_threshold = (method_thresholds or {}).get(method, threshold)
            method_source = (method_threshold_sources or {}).get(method, threshold_source)
            method_alpha = alpha if method_threshold == threshold else None
            method_n_tests = n_tests if method_threshold == threshold else float('nan')

            if isinstance(Res, FarmCPUResamplingResults):
                # Handle Resampling
                res_file = self.output_dir / f"GWAS_{trait_name}_{method}_RMIP.csv"
                df = Res.to_dataframe()
                if 'Chr' in df.columns or 'Pos' in df.columns:
                    df = df.rename(columns={'Chr': 'CHROM', 'Pos': 'POS'})
                if 'RMIP' in df.columns:
                    resampling_hit_snps = set(
                        df.loc[df['RMIP'] >= rmip_hit_threshold, 'SNP'].astype(str)
                    )
                df.to_csv(res_file, index=False)
                summary_data.append({
                    'Trait': trait_name, 'Method': method,
                    'Significant_Hits': len(resampling_hit_snps),
                    'Threshold': rmip_hit_threshold,
                    'Lambda_GC': float('nan'),  # Not applicable for resampling
                    'N_Samples': n_samples,
                    'N_Markers': n_markers,
                    'Runtime_Seconds': round(runtime_seconds, 2) if runtime_seconds else None,
                    'Info': f"{method_source}; RMIP>={rmip_hit_threshold}; Runs={Res.total_runs}; Clustered={Res.cluster_mode}"
                })

                # Generate RMIP Manhattan plot
                if 'manhattan' in outputs:
                    try:
                        report = PANICLE_Report(
                            results=Res,
                            map_data=self.geno_map,
                            output_prefix=output_prefix_base,
                            plot_types=['manhattan'],
                            verbose=False,
                            save_plots=True
                        )
                        # Cleanup figures
                        import matplotlib.pyplot as plt
                        for m_plots in report.get('plots', {}).values():
                            for fig in m_plots.values():
                                plt.close(fig)
                    except Exception as e:
                        self.log(f"   RMIP plotting error {method}: {e}")

                continue

            # Standard Results
            # Add to big table
            all_res_df[f'{method}_P'] = Res.pvalues
            all_res_df[f'{method}_Effect'] = Res.effects
            
            # Check significance
            hits = Res.pvalues <= method_threshold
            hits_by_method[method] = hits
            n_sig = int(hits.sum())
            
            # Get lambda GC for this method
            lambda_gc_value = (method_lambda_gc or {}).get(method, float('nan'))

            summary_data.append({
                'Trait': trait_name, 'Method': method,
                'Significant_Hits': n_sig,
                'Threshold': method_threshold,
                'Lambda_GC': round(lambda_gc_value, 3) if not np.isnan(lambda_gc_value) else float('nan'),
                'N_Samples': n_samples,
                'N_Markers': n_markers,
                'Runtime_Seconds': round(runtime_seconds, 2) if runtime_seconds else None,
                'Info': method_source
            })
            
            # Plots
            if 'manhattan' in outputs or 'qq' in outputs:
                try:
                    plot_types = []
                    if 'manhattan' in outputs: plot_types.append('manhattan')
                    if 'qq' in outputs: plot_types.append('qq')
                    
                    report = PANICLE_Report(
                        results={method: Res}, map_data=self.geno_map,
                        output_prefix=output_prefix_base,
                        plot_types=plot_types,
                        threshold=method_threshold,
                        threshold_alpha=method_alpha,
                        threshold_n_tests=method_n_tests,
                        threshold_source=method_source,
                        verbose=False,
                        save_plots=True
                    )
                    
                    # Cleanup figures to avoid "More than 20 figures opened" warning
                    import matplotlib.pyplot as plt
                    for m_plots in report.get('plots', {}).values():
                        for fig in m_plots.values():
                             plt.close(fig)
                except Exception as e:
                    self.log(f"   Plotting error {method}: {e}")

        # Save merged tables
        method_columns = []
        for method in ordered_methods:
            if method == 'FarmCPUResampling':
                continue
            method_columns.extend([f'{method}_P', f'{method}_Effect'])

        if 'all_marker_pvalues' in outputs:
            maf_source = geno_for_maf or self.genotype_matrix
            maf_all = calculate_maf_from_genotypes(maf_source, max_dosage=max_dosage)
            _insert_maf_column(all_res_df, maf_all)
            ordered_base = _ordered_base_columns(all_res_df)
            if method_columns:
                all_res_df = all_res_df[ordered_base + method_columns]
            all_res_df.to_csv(self.output_dir / f"GWAS_{trait_name}_all_results.csv", index=False)
            
        if resampling_hit_snps:
            resampling_mask = all_res_df['SNP'].astype(str).isin(resampling_hit_snps).to_numpy()
            hits_by_method['FarmCPUResampling'] = resampling_mask

        if hits_by_method and 'significant_marker_pvalues' in outputs:
            n_markers = all_res_df.shape[0]
            method_labels = [[] for _ in range(n_markers)]
            for method in ordered_methods:
                hits = hits_by_method.get(method)
                if hits is None or not np.any(hits):
                    continue
                for idx in np.where(hits)[0]:
                    method_labels[idx].append(method)

            any_hits = np.array([bool(labels) for labels in method_labels], dtype=bool)
            if np.any(any_hits):
                sig_df = all_res_df.loc[any_hits].copy()
                if 'MAF' not in sig_df.columns:
                    sig_indices = np.where(any_hits)[0]
                    maf_subset = _maf_for_indices(sig_indices)
                    _insert_maf_column(sig_df, maf_subset)
                sig_df['Method'] = [
                    "|".join(method_labels[idx]) for idx in np.where(any_hits)[0]
                ]
                ordered_base = _ordered_base_columns(sig_df)
                if method_columns:
                    sig_df = sig_df[ordered_base + method_columns + ['Method']]
                sig_df.to_csv(self.output_dir / f"GWAS_{trait_name}_significant.csv", index=False)
            
        return summary_data
