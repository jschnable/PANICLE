"""
Data loading utilities for various file formats
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List, Any
import warnings
import time
import sys
import os

logger = logging.getLogger(__name__)

from ..utils.data_types import GenotypeMatrix, GenotypeMap, impute_major_allele_inplace
from ..utils.memmap_utils import load_full_from_metadata
from ..utils.effective_tests import estimate_effective_tests_from_genotype
try:
    # Local VCF loader (builtin parser by default, cyvcf2 optional)
    from .load_genotype_vcf import load_genotype_vcf as _load_genotype_vcf
except Exception:
    _load_genotype_vcf = None
try:
    # HapMap loader
    from .load_genotype_hapmap import load_genotype_hapmap as _load_genotype_hapmap
except Exception:
    _load_genotype_hapmap = None
try:
    # PLINK bed loader via bed-reader
    from .load_genotype_plink import load_genotype_plink as _load_genotype_plink
except Exception:
    _load_genotype_plink = None

NON_BINARY_FORMATS = {
    'vcf', 'csv', 'tsv', 'numeric', 'hapmap'
}

LOAD_TIME_WARNING_THRESHOLD = 300.0  # seconds (5 minutes)

# Canonical list of common ID column names recognized across loaders.
_KNOWN_ID_COLUMNS = [
    'ID', 'id', 'IID',
    'sample', 'Sample',
    'Taxa', 'taxa',
    'Genotype', 'genotype',
    'Accession', 'accession',
]


def _detect_and_rename_id_column(df: pd.DataFrame, id_column: str = 'ID',
                                  verbose: bool = True) -> pd.DataFrame:
    """Detect and rename the ID column to 'ID' in a DataFrame.

    Checks for *id_column* first, then falls back to well-known column names
    (see ``_KNOWN_ID_COLUMNS``), and finally uses the first column.

    Args:
        df: DataFrame to inspect (modified in-place via rename).
        id_column: The expected ID column name supplied by the caller.
        verbose: Whether to print auto-detection messages.

    Returns:
        The (possibly renamed) DataFrame.
    """
    if id_column in df.columns:
        if verbose:
            logger.info("Using specified ID column: '%s'", id_column)
        if id_column != 'ID':
            df = df.rename(columns={id_column: 'ID'})
        return df

    present_candidates = [c for c in df.columns if c in _KNOWN_ID_COLUMNS]
    if present_candidates:
        if len(present_candidates) > 1:
            warnings.warn(
                "Multiple potential ID columns found: {}. Selecting leftmost '{}' as ID.".format(
                    present_candidates, present_candidates[0]
                )
            )
        detected = present_candidates[0]
        if verbose:
            logger.info("Auto-detected ID column: '%s'", detected)
        df = df.rename(columns={detected: 'ID'})
        return df

    first_col = df.columns[0]
    warnings.warn(
        "No recognized ID column found; using first column '{}' as ID.".format(first_col)
    )
    if verbose:
        logger.info("Using first column as ID: '%s'", first_col)
    df = df.rename(columns={first_col: 'ID'})
    return df


def detect_file_format(filepath: Union[str, Path]) -> str:
    """Detect file format based on extension and content
    
    Args:
        filepath: Path to file
        
    Returns:
        Detected format: 'csv', 'tsv', 'vcf', 'plink', 'hmp', 'numeric'
    """
    filepath = Path(filepath)
    
    # Check extension first (handle multi-suffix like .vcf.gz)
    name_lower = filepath.name.lower()
    if (
        name_lower.endswith('.vcf')
        or name_lower.endswith('.vcf.gz')
        or name_lower.endswith('.vcf.bgz')
        or name_lower.endswith('.bcf')
    ):
        return 'vcf'
    elif filepath.suffix.lower() in ['.bed', '.bim', '.fam']:
        return 'plink'
    elif name_lower.endswith('.hmp') or name_lower.endswith('.hmp.txt') or name_lower.endswith('.hmp.gz') or name_lower.endswith('.hmp.txt.gz'):
        return 'hapmap'
    elif filepath.suffix.lower() in ['.tsv', '.txt']:
        return 'tsv'
    elif filepath.suffix.lower() == '.csv':
        return 'csv'
    elif filepath.suffix.lower() == '.npz':
        try:
            with np.load(filepath, allow_pickle=True) as meta:
                if 'memmap_file' in meta:
                    return 'memmap'
        except Exception:
            return 'unknown'

    # Try to detect by content / neighboring files
    try:
        # Check for PLINK triad if a bare prefix is given
        if filepath.suffix == '':
            pref = filepath
            if pref.with_suffix('.bed').exists() and pref.with_suffix('.bim').exists() and pref.with_suffix('.fam').exists():
                return 'plink'
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        if first_line.startswith('##fileformat=VCF'):
            return 'vcf'
        # Detect HapMap header
        if first_line.split('\t')[:11] == ['rs#','alleles','chrom','pos','strand','assembly#','center','protLSID','assayLSID','panelLSID','QCcode']:
            return 'hapmap'
        elif '\t' in first_line and ',' not in first_line:
            return 'tsv'
        elif ',' in first_line:
            return 'csv'
        else:
            return 'numeric'
    except:
        return 'unknown'


def _detect_numeric_separator(filepath: Union[str, Path]) -> str:
    """Heuristically determine the delimiter for numeric genotype matrices."""
    filepath = Path(filepath)
    try:
        with filepath.open('r') as handle:
            # Inspect up to the first 10 non-empty header lines to decide.
            for _ in range(10):
                line = handle.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                comma_count = line.count(',')
                tab_count = line.count('\t')
                if tab_count or comma_count:
                    # Prefer the character that appears more frequently.
                    return '\t' if tab_count >= comma_count else ','
    except OSError:
        pass

    # Default to comma if the structure was ambiguous; this avoids collapsing
    # every marker into a single column when a comma-separated file is treated
    # as tab-delimited.
    return ','

def load_phenotype_file(filepath: Union[str, Path], 
                       trait_columns: Optional[List[str]] = None,
                       id_column: str = 'ID') -> pd.DataFrame:
    """Load phenotype file in various formats
    
    Args:
        filepath: Path to phenotype file
        trait_columns: List of trait column names (if None, auto-detect)
        id_column: Name of ID column
        
    Returns:
        DataFrame with ID and trait columns
    """
    filepath = Path(filepath)
    file_format = detect_file_format(filepath)
    
    # Load based on format
    # Robust NA handling: recognize common missing tokens
    na_values = [
        '', 'NA', 'NaN', 'nan', 'NAN', 'na', 'N/A', 'n/a', 'Null', 'NULL',
        '.', '-', '--'
    ]
    read_kwargs = dict(na_values=na_values, keep_default_na=True)
    if file_format == 'csv':
        df = pd.read_csv(filepath, **read_kwargs)
    elif file_format == 'tsv':
        df = pd.read_csv(filepath, sep='\t', **read_kwargs)
    else:
        # Try both comma and tab separation
        try:
            df = pd.read_csv(filepath, **read_kwargs)
        except:
            df = pd.read_csv(filepath, sep='\t', **read_kwargs)
    
    # Standardize column names
    df = _detect_and_rename_id_column(df, id_column)

    # Auto-detect trait columns if not specified
    if trait_columns is None:
        # Attempt to coerce non-ID columns to numeric to improve detection
        numeric_probe = {}
        for col in df.columns:
            if col == 'ID':
                continue
            s = pd.to_numeric(df[col], errors='coerce')
            numeric_probe[col] = s
        # Choose columns that are numeric after coercion
        trait_columns = [col for col, s in numeric_probe.items() if pd.api.types.is_numeric_dtype(s)]
        # Ensure stored df has numeric dtype for selected traits
        if trait_columns:
            df[trait_columns] = df[trait_columns].apply(pd.to_numeric, errors='coerce')
    else:
        # Resolve user-specified trait names to actual column names,
        # falling back to case-insensitive and whitespace-stripped matching.
        available = [c for c in df.columns if c != 'ID']
        lower_map = {c.lower(): c for c in available}
        stripped_map = {c.strip().lower(): c for c in available}

        resolved_cols = []
        unresolved = []
        for t in trait_columns:
            # 1. Exact match
            if t in available:
                resolved_cols.append(t)
                continue
            # 2. Case-insensitive match
            if t.lower() in lower_map:
                actual = lower_map[t.lower()]
                resolved_cols.append(actual)
                logger.info("Note: trait '%s' matched column '%s' (case-insensitive)", t, actual)
                continue
            # 3. Stripped + case-insensitive match
            key = t.strip().lower()
            if key in stripped_map:
                actual = stripped_map[key]
                resolved_cols.append(actual)
                logger.info("Note: trait '%s' matched column '%s' (after stripping whitespace)", t, actual)
                continue
            unresolved.append(t)

        if unresolved:
            logger.warning("Traits not found in phenotype file: %s", unresolved)
            logger.warning("Available columns: %s", available)

        # Coerce resolved trait columns to numeric
        trait_columns = resolved_cols
        if trait_columns:
            df[trait_columns] = df[trait_columns].apply(pd.to_numeric, errors='coerce')

    # Keep only ID and specified trait columns
    columns_to_keep = ['ID'] + [col for col in trait_columns if col in df.columns]
    df = df[columns_to_keep]

    # Deduplicate phenotype IDs by aggregating trait means (skip NaNs)
    if df['ID'].duplicated().any():
        n_dups = int(df['ID'].duplicated().sum())
        if len(columns_to_keep) > 1:
            # Aggregate by ID using mean for each trait column, NaN excluded by default
            agg_cols = {col: 'mean' for col in columns_to_keep if col != 'ID'}
            df = df.groupby('ID', as_index=False).agg(agg_cols)
            warnings.warn(
                f"Detected {n_dups} duplicated phenotype records by ID; deduplicated by computing per-ID mean of trait columns (missing values ignored)."
            )
        else:
            # No trait columns found; just keep first occurrence per ID
            df = df.drop_duplicates(subset=['ID'], keep='first')
            warnings.warn(
                f"Detected {n_dups} duplicated phenotype records by ID; no numeric trait columns detected, so retained only the first record per ID."
            )

    return df

def load_covariate_file(filepath: Union[str, Path],
                        covariate_columns: Optional[List[str]] = None,
                        id_column: str = 'ID') -> pd.DataFrame:
    """Load covariate file and ensure numeric covariate columns."""
    filepath = Path(filepath)
    file_format = detect_file_format(filepath)

    na_values = [
        '', 'NA', 'NaN', 'nan', 'NAN', 'na', 'N/A', 'n/a', 'Null', 'NULL',
        '.', '-', '--'
    ]
    read_kwargs = dict(na_values=na_values, keep_default_na=True)

    if file_format == 'csv':
        cov_df = pd.read_csv(filepath, **read_kwargs)
    elif file_format == 'tsv':
        cov_df = pd.read_csv(filepath, sep='	', **read_kwargs)
    else:
        try:
            cov_df = pd.read_csv(filepath, **read_kwargs)
        except Exception:
            cov_df = pd.read_csv(filepath, sep='	', **read_kwargs)

    cov_df = _detect_and_rename_id_column(cov_df, id_column)

    cov_df['ID'] = cov_df['ID'].astype(str)

    available_covariates = [c for c in cov_df.columns if c != 'ID']
    if covariate_columns is not None:
        requested = [c for c in covariate_columns if c not in ('ID', id_column)]
        missing = [c for c in requested if c not in cov_df.columns]
        if missing:
            raise ValueError(
                "Requested covariate columns missing from file '{}': {}".format(
                    filepath, missing
                )
            )
        selected_cols = [c for c in covariate_columns if c in cov_df.columns and c not in ('ID', id_column)]
    else:
        selected_cols = available_covariates

    if not selected_cols:
        raise ValueError(
            "No covariate columns found in file '{}'. Provide at least one numeric covariate column.".format(
                filepath
            )
        )

    covariate_data = cov_df[['ID'] + selected_cols].copy()
    dropped_for_nan: List[str] = []

    for col in list(selected_cols):
        series = covariate_data[col]
        converted = pd.to_numeric(series, errors='coerce')
        invalid_mask = series.notna() & converted.isna()
        if invalid_mask.any():
            invalid_examples = sorted(series[invalid_mask].astype(str).unique()[:5])
            example_str = ', '.join(invalid_examples)
            raise ValueError(
                "Covariate column '{}' contains non-numeric values (e.g. {}).".format(
                    col, example_str
                )
            )
        covariate_data[col] = converted
        if converted.notna().sum() == 0:
            dropped_for_nan.append(col)

    if dropped_for_nan:
        warnings.warn(
            "Dropping covariate columns with no numeric values after conversion: {}".format(
                dropped_for_nan
            )
        )
        covariate_data = covariate_data.drop(columns=dropped_for_nan)

    covariate_columns_present = [c for c in covariate_data.columns if c != 'ID']
    if not covariate_columns_present:
        raise ValueError(
            "No usable covariate columns remained after processing file '{}'.".format(
                filepath
            )
        )

    if covariate_data['ID'].duplicated().any():
        n_dups = int(covariate_data['ID'].duplicated().sum())
        agg_cols = {col: 'mean' for col in covariate_columns_present}
        covariate_data = covariate_data.groupby('ID', as_index=False).agg(agg_cols)
        warnings.warn(
            f"Detected {n_dups} duplicated covariate records by ID; deduplicated by computing per-ID mean of covariate columns (missing values ignored)."
        )
    else:
        covariate_data = covariate_data.reset_index(drop=True)

    covariate_columns_present = [c for c in covariate_data.columns if c != 'ID']
    covariate_data = covariate_data[['ID'] + covariate_columns_present]

    return covariate_data


def _wrap_loaded_genotype(
    geno_np: np.ndarray,
    individual_ids: List[str],
    geno_map_df: "pd.DataFrame",
    precompute_alleles: bool = True,
    compute_effective_tests: bool = False,
    effective_test_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[GenotypeMatrix, List[str], GenotypeMap]:
    """Shared post-processing for format-specific genotype loaders.

    Deduplicates sample IDs, wraps raw arrays into ``GenotypeMatrix`` /
    ``GenotypeMap`` objects, and optionally computes effective tests.
    """
    individual_ids, geno_np = _deduplicate_genotype_samples(individual_ids, geno_np)
    is_imputed = getattr(geno_map_df, 'attrs', {}).get('is_imputed', False)
    geno_matrix = GenotypeMatrix(geno_np, precompute_alleles=precompute_alleles, is_imputed=is_imputed)
    geno_map = GenotypeMap(
        geno_map_df if isinstance(geno_map_df, pd.DataFrame) else pd.DataFrame(geno_map_df)
    )
    if compute_effective_tests:
        et_kwargs = effective_test_kwargs or {}
        geno_map.metadata["effective_tests"] = estimate_effective_tests_from_genotype(
            geno_matrix,
            geno_map,
            **et_kwargs,
        )
    return geno_matrix, individual_ids, geno_map


def _deduplicate_genotype_samples(individual_ids: List[str], geno_np: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Deduplicate genotype sample IDs by retaining the first occurrence per ID.

    Emits a warning when duplicates are found. Returns (new_ids, new_geno).
    """
    seen = set()
    keep_indices: List[int] = []
    for i, sid in enumerate(individual_ids):
        if sid not in seen:
            seen.add(sid)
            keep_indices.append(i)
    if len(keep_indices) != len(individual_ids):
        dropped = len(individual_ids) - len(keep_indices)
        warnings.warn(
            f"Detected {dropped} duplicated genotype records by sample ID; retaining only the first occurrence for each duplicated ID."
        )
        new_ids = [individual_ids[i] for i in keep_indices]
        new_geno = geno_np[keep_indices, :]
        return new_ids, new_geno
    return individual_ids, geno_np

def load_genotype_file(filepath: Union[str, Path],
                      file_format: Optional[str] = None,
                      compute_effective_tests: bool = False,
                      effective_test_kwargs: Optional[Dict[str, Any]] = None,
                      **kwargs) -> Tuple[GenotypeMatrix, List[str], GenotypeMap]:
    """Load genotype file in various formats
    
    Args:
        filepath: Path to genotype file
        file_format: File format ('csv', 'tsv', 'numeric' or None for auto-detect)
        
    Returns:
        Tuple of (GenotypeMatrix, individual_ids, GenotypeMap)
    """
    filepath = Path(filepath)
    precompute_alleles = kwargs.pop('precompute_alleles', True)
    load_start = time.time()

    if file_format is None:
        file_format = detect_file_format(filepath)

    def _maybe_warn(elapsed_seconds: float) -> None:
        if file_format in NON_BINARY_FORMATS and elapsed_seconds >= LOAD_TIME_WARNING_THRESHOLD:
            logger.warning(
                "Loading genotype file '%s' took %.1f minutes. "
                "Consider caching it with `panicle-cache-genotype` for faster future runs.",
                filepath, elapsed_seconds / 60,
            )

    if file_format == 'memmap':
        genotype, individual_ids, geno_map = load_full_from_metadata(
            filepath,
            precompute_alleles=precompute_alleles,
        )
        if geno_map is None:
            raise ValueError("Metadata file does not contain genotype map information")
        individual_ids = list(individual_ids)
        elapsed = time.time() - load_start
        _maybe_warn(elapsed)
        if compute_effective_tests:
            effective_tests_kwargs = effective_test_kwargs or {}
            geno_map.metadata["effective_tests"] = estimate_effective_tests_from_genotype(
                genotype,
                geno_map,
                **effective_tests_kwargs,
            )
        return genotype, individual_ids, geno_map

    if file_format == 'vcf':
        if _load_genotype_vcf is None:
            raise ImportError("VCF loading requires 'load_genotype_vcf' module."
                              " Ensure panicle/data/load_genotype_vcf.py is present.")
        geno_np, individual_ids, geno_map_df = _load_genotype_vcf(str(filepath), **kwargs)
        result = _wrap_loaded_genotype(geno_np, individual_ids, geno_map_df,
                                       precompute_alleles, compute_effective_tests, effective_test_kwargs)
        elapsed = time.time() - load_start
        _maybe_warn(elapsed)
        return result

    if file_format == 'plink':
        if _load_genotype_plink is None:
            raise ImportError("PLINK loading requires 'bed-reader' and 'load_genotype_plink'."
                              " pip install bed-reader.")
        geno_np, individual_ids, geno_map_df = _load_genotype_plink(str(filepath), **kwargs)
        result = _wrap_loaded_genotype(geno_np, individual_ids, geno_map_df,
                                       precompute_alleles, compute_effective_tests, effective_test_kwargs)
        elapsed = time.time() - load_start
        _maybe_warn(elapsed)
        return result

    if file_format == 'hapmap':
        if _load_genotype_hapmap is None:
            raise ImportError("HapMap loading requires 'load_genotype_hapmap' module.")
        geno_np, individual_ids, geno_map_df = _load_genotype_hapmap(str(filepath), **kwargs)
        result = _wrap_loaded_genotype(geno_np, individual_ids, geno_map_df,
                                       precompute_alleles, compute_effective_tests, effective_test_kwargs)
        elapsed = time.time() - load_start
        _maybe_warn(elapsed)
        return result

    if file_format in ['csv', 'tsv', 'numeric']:
        cache_base = str(filepath)
        cache_geno = cache_base + '.panicle.v2.geno.npy'
        cache_ind = cache_base + '.panicle.v2.ind.txt'
        cache_map = cache_base + '.panicle.v2.map.csv'
        force_recache = bool(kwargs.get('force_recache', False))

        min_maf = float(kwargs.get('min_maf', 0.0))
        max_missing = float(kwargs.get('max_missing', 1.0))
        drop_monomorphic = bool(kwargs.get('drop_monomorphic', False))
        geno_np = None
        individual_ids_csv: Optional[List[str]] = None
        geno_map_df = None
        loaded_from_cache = False
        try:
            if not force_recache:
                if os.path.exists(cache_geno) and os.path.exists(cache_ind) and os.path.exists(cache_map):
                    src_mtime = os.path.getmtime(filepath)
                    if (os.path.getmtime(cache_geno) > src_mtime and
                        os.path.getmtime(cache_ind) > src_mtime and
                        os.path.getmtime(cache_map) > src_mtime):
                        logger.info("[Cache] Loading binary cache for %s...", filepath)
                        geno_np = np.load(cache_geno, mmap_mode='r')
                        with open(cache_ind, 'r') as f:
                            individual_ids_csv = [line.strip() for line in f]
                        geno_map_df = pd.read_csv(cache_map)
                        geno_map_df.attrs["is_imputed"] = True
                        loaded_from_cache = True
                        if min_maf > 0.0 or max_missing < 1.0 or drop_monomorphic:
                            logger.warning(
                                "[Cache] Cached genotype data loaded; "
                                "min_maf/max_missing/drop_monomorphic filters are not re-applied "
                                "(min_maf=%s, max_missing=%s, drop_monomorphic=%s). "
                                "Use --force-recache or delete the cache files to rebuild.",
                                min_maf, max_missing, drop_monomorphic,
                            )
        except Exception as e:
            logger.warning("[Cache] Failed to load cache: %s", e)

        # Load as CSV/TSV with numeric genotypes
        if file_format == 'csv':
            separator = ','
        elif file_format == 'tsv':
            separator = '\t'
        else:
            separator = _detect_numeric_separator(filepath)

        # Use fast, consistent parsing and avoid mixed-type inference
        if not loaded_from_cache:
            df = pd.read_csv(filepath, sep=separator, low_memory=False)

            # First column should be individual IDs
            individual_ids_csv = df.iloc[:, 0].astype(str).tolist()

            # Remaining columns are markers
            marker_names = df.columns[1:].tolist()
            data_df = df.iloc[:, 1:]

            # Vectorized conversion to numeric with robust NA handling
            # Coerce any non-numeric (e.g., 'NA', 'N', '.') to NaN, then fill with sentinel -9
            data_df = data_df.apply(pd.to_numeric, errors='coerce')
            data_df = data_df.fillna(-9)

            # Cast to compact integer type; -9 and 0/1/2 fit in int8
            try:
                geno_np = data_df.to_numpy(dtype=np.int8, copy=False)
            except Exception:
                # Fallback path in case of unexpected wide range; use int16
                geno_np = data_df.to_numpy(dtype=np.int16, copy=False)

            impute_major_allele_inplace(geno_np, missing_value=-9)

            # Create basic GenotypeMap (assumes no map file provided)
            geno_map_df = pd.DataFrame({
                'SNP': marker_names,
                'CHROM': [1] * len(marker_names),  # Default to chromosome 1
                'POS': list(range(1, len(marker_names) + 1))  # Sequential positions
            })
            geno_map_df.attrs["is_imputed"] = True

            try:
                logger.info("[Cache] Saving binary cache to %s.panicle.v2.*", cache_base)
                np.save(cache_geno, geno_np)
                with open(cache_ind, 'w') as f:
                    for ind in individual_ids_csv:
                        f.write(f"{ind}\n")
                geno_map_df.to_csv(cache_map, index=False)
            except Exception as e:
                logger.warning("[Cache] Failed to save cache: %s", e)

        if geno_np is None or individual_ids_csv is None or geno_map_df is None:
            raise ValueError("Failed to load genotype data from CSV/TSV")

        # CSV/TSV data is already imputed above; mark it so the wrapper preserves the flag.
        geno_map_df.attrs["is_imputed"] = True
        result = _wrap_loaded_genotype(geno_np, individual_ids_csv, geno_map_df,
                                       precompute_alleles, compute_effective_tests, effective_test_kwargs)

        elapsed = time.time() - load_start
        _maybe_warn(elapsed)
        return result
    
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def load_genotype_vcf(filepath: Union[str, Path], **kwargs) -> Tuple[GenotypeMatrix, List[str], GenotypeMap]:
    """Convenience wrapper to load VCF files using the optimized loader.

    Accepts the same keyword args as ``load_genotype_vcf`` in
    ``panicle.data.load_genotype_vcf``, including ``backend``
    ('auto'|'cyvcf2'|'builtin'), ``split_multiallelic``, and filters.
    """
    if _load_genotype_vcf is None:
        raise ImportError("VCF loading requires 'load_genotype_vcf' module.")
    geno_np, individual_ids, geno_map_df = _load_genotype_vcf(str(filepath), **kwargs)
    return _wrap_loaded_genotype(geno_np, individual_ids, geno_map_df)

def load_genotype_plink(filepath: Union[str, Path], **kwargs) -> Tuple[GenotypeMatrix, List[str], GenotypeMap]:
    """Convenience wrapper to load PLINK .bed files.

    Accepts ``prefix_or_bed`` and optional ``bim``/``fam`` in kwargs, plus
    filters matching the VCF loader (drop_monomorphic, max_missing, min_maf).
    """
    if _load_genotype_plink is None:
        raise ImportError("PLINK loading requires 'bed-reader' and 'load_genotype_plink'."
                          " pip install bed-reader.")
    geno_np, individual_ids, geno_map_df = _load_genotype_plink(str(filepath), **kwargs)
    return _wrap_loaded_genotype(geno_np, individual_ids, geno_map_df)

def load_genotype_hapmap(filepath: Union[str, Path], **kwargs) -> Tuple[GenotypeMatrix, List[str], GenotypeMap]:
    """Convenience wrapper to load HapMap .hmp(.txt) files.

    Supports plain and gzipped files. QC filters align with other loaders.
    """
    if _load_genotype_hapmap is None:
        raise ImportError("HapMap loading requires 'load_genotype_hapmap' module.")
    geno_np, individual_ids, geno_map_df = _load_genotype_hapmap(str(filepath), **kwargs)
    return _wrap_loaded_genotype(geno_np, individual_ids, geno_map_df)

def load_map_file(filepath: Union[str, Path]) -> GenotypeMap:
    """Load genetic map file
    
    Args:
        filepath: Path to map file
        
    Returns:
        GenotypeMap object
    """
    filepath = Path(filepath)
    file_format = detect_file_format(filepath)
    
    if file_format == 'csv':
        df = pd.read_csv(filepath)
    elif file_format == 'tsv':
        df = pd.read_csv(filepath, sep='\t')
    else:
        try:
            df = pd.read_csv(filepath)
        except:
            df = pd.read_csv(filepath, sep='\t')
    
    # Standardize column names
    col_mapping = {
        'Chr': 'CHROM', 'chr': 'CHROM', 'chromosome': 'CHROM',
        'Pos': 'POS', 'pos': 'POS', 'position': 'POS', 'bp': 'POS',
        'snp': 'SNP', 'marker': 'SNP', 'rs': 'SNP'
    }
    
    for old_name, new_name in col_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    return GenotypeMap(df)


def match_individuals(phenotype_df: pd.DataFrame,
                     individual_ids: List[str],
                     covariate_df: Optional[pd.DataFrame] = None
                     ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[int], Dict]:
    """Match individuals across phenotype, genotype, and optional covariates."""
    phenotype_df = phenotype_df.copy()
    if 'ID' not in phenotype_df.columns:
        raise ValueError("Phenotype dataframe must contain an 'ID' column.")
    phenotype_df['ID'] = phenotype_df['ID'].astype(str)

    phe_ids = set(phenotype_df['ID'])
    genotype_ids = [str(ind_id) for ind_id in individual_ids]
    geno_ids = set(genotype_ids)

    common_ids = phe_ids & geno_ids

    summary: Dict[str, int] = {
        'n_phenotype_original': len(phe_ids),
        'n_genotype_original': len(geno_ids),
        'n_common': len(common_ids),
        'n_phenotype_dropped': len(phe_ids - common_ids),
        'n_genotype_dropped': len(geno_ids - common_ids),
    }

    if len(common_ids) == 0:
        raise ValueError("No common individuals found between phenotype and genotype data")

    matched_phenotype = phenotype_df[phenotype_df['ID'].isin(common_ids)].copy()
    matched_phenotype = matched_phenotype.sort_values('ID').reset_index(drop=True)
    sorted_ids = matched_phenotype['ID'].tolist()

    id_to_index: Dict[str, int] = {}
    for idx, raw_id in enumerate(genotype_ids):
        if raw_id not in id_to_index:
            id_to_index[raw_id] = idx

    try:
        matched_indices = [id_to_index[sid] for sid in sorted_ids]
    except KeyError as exc:
        missing_id = exc.args[0]
        raise RuntimeError(
            f"Internal inconsistency while aligning genotype indices; ID '{missing_id}' was expected but not found."
        ) from None

    matched_covariate: Optional[pd.DataFrame] = None
    if covariate_df is not None:
        if 'ID' not in covariate_df.columns:
            raise ValueError("Covariate dataframe must contain an 'ID' column.")

        covariate_df = covariate_df.copy()
        covariate_df['ID'] = covariate_df['ID'].astype(str)
        cov_ids = set(covariate_df['ID'])

        summary['n_covariate_provided'] = len(cov_ids)

        missing_covariate_ids = sorted(common_ids - cov_ids)
        if missing_covariate_ids:
            preview = ', '.join(missing_covariate_ids[:5])
            if len(missing_covariate_ids) > 5:
                preview += ', ...'
            raise ValueError(
                "Covariate file is missing {} individuals required after phenotype/genotype matching: {}".format(
                    len(missing_covariate_ids), preview
                )
            )

        unused_covariate_ids = cov_ids - common_ids
        summary['n_covariate_unused'] = len(unused_covariate_ids)

        if covariate_df['ID'].duplicated().any():
            n_dups = int(covariate_df['ID'].duplicated().sum())
            agg_cols = {col: 'mean' for col in covariate_df.columns if col != 'ID'}
            covariate_df = covariate_df.groupby('ID', as_index=False).agg(agg_cols)
            warnings.warn(
                f"Detected {n_dups} duplicated covariate records by ID; deduplicated by computing per-ID mean of covariate columns (missing values ignored)."
            )

        matched_covariate = covariate_df.set_index('ID').loc[sorted_ids].reset_index()
        cov_cols = [c for c in matched_covariate.columns if c != 'ID']
        matched_covariate = matched_covariate[['ID'] + cov_cols]
        summary['n_covariate_matched'] = len(sorted_ids)
    else:
        summary['n_covariate_provided'] = 0
        summary['n_covariate_unused'] = 0
        summary['n_covariate_matched'] = 0

    return matched_phenotype, matched_covariate, matched_indices, summary
