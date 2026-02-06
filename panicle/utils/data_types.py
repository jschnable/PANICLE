"""
Core data structures for PANICLE package
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Dict, Any
from pathlib import Path

class Phenotype:
    """Phenotype data structure compatible with R rMVP format
    
    Expected format: n × 2 matrix where:
    - Column 1: Individual IDs 
    - Column 2: Trait values
    """
    
    def __init__(self, data: Union[np.ndarray, pd.DataFrame, str, Path]):
        if isinstance(data, (str, Path)):
            # Load from file - try with header first
            try:
                self.data = pd.read_csv(data, header=0)
            except:
                self.data = pd.read_csv(data, header=None)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, np.ndarray):
            # Convert to DataFrame
            self.data = pd.DataFrame(data)
        else:
            raise ValueError("Data must be array, DataFrame, or file path")
            
        # Validate structure
        if self.data.shape[1] != 2:
            raise ValueError(f"Phenotype must have 2 columns, got {self.data.shape[1]}")
            
        # Set standard column names
        self.data.columns = ['ID', 'Trait']
        
    @property
    def ids(self) -> pd.Series:
        """Individual IDs"""
        return self.data['ID']
    
    @property 
    def values(self) -> pd.Series:
        """Trait values"""
        return self.data['Trait']
    
    @property
    def n_individuals(self) -> int:
        """Number of individuals"""
        return len(self.data)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return self.data.values


class GenotypeMap:
    """SNP map information compatible with R rMVP format
    
    Expected columns: [SNP_ID, Chr, Pos, REF, ALT]
    """
    
    def __init__(self, data: Union[pd.DataFrame, str, Path], metadata: Optional[Dict[str, Any]] = None):
        if isinstance(data, (str, Path)):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError("Data must be DataFrame or file path")

        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}
            
        # Validate required columns
        required_cols = ['SNP', 'CHROM', 'POS']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
    
    @property
    def snp_ids(self) -> pd.Series:
        """SNP identifiers"""
        return self.data['SNP']
    
    @property
    def chromosomes(self) -> pd.Series:
        """Chromosome numbers"""
        return self.data['CHROM']
    
    @property
    def positions(self) -> pd.Series:
        """Physical positions"""
        return self.data['POS']
    
    @property
    def n_markers(self) -> int:
        """Number of markers"""
        return len(self.data)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return self.data.copy()

    def with_metadata(self, **metadata: Any) -> "GenotypeMap":
        """Return a new GenotypeMap with merged metadata dictionary."""
        merged = dict(self.metadata)
        merged.update(metadata)
        new_map = GenotypeMap(self.data.copy(), metadata=merged)
        return new_map


def impute_major_allele_inplace(geno: np.ndarray, missing_value: int = -9) -> int:
    """Fill missing values in-place with the per-marker major allele (0/1/2)."""
    if geno.size == 0:
        return 0
    missing_mask = geno == missing_value
    n_missing = int(missing_mask.sum())
    if n_missing == 0:
        return 0
    counts_0 = np.sum(geno == 0, axis=0)
    counts_1 = np.sum(geno == 1, axis=0)
    counts_2 = np.sum(geno == 2, axis=0)
    major_alleles = np.argmax(np.stack([counts_0, counts_1, counts_2], axis=0), axis=0)
    major_alleles = major_alleles.astype(geno.dtype, copy=False)
    geno[missing_mask] = np.broadcast_to(major_alleles, geno.shape)[missing_mask]
    return n_missing


class GenotypeMatrix:
    """Memory-efficient genotype matrix with lazy loading support

    Handles large genotype matrices that may not fit in memory.
    Compatible with R rMVP memory-mapped format.
    Includes pre-computed major alleles for efficient missing data imputation.

    Supports column-major (transposed) storage for efficient individual subsetting.
    When transposed=True, internal data is stored as (n_markers, n_individuals)
    but the API still presents it as (n_individuals, n_markers).
    """

    def __init__(self, data: Union[np.ndarray, str, Path],
                 shape: Optional[Tuple[int, int]] = None,
                 dtype: np.dtype = np.int8,
                 precompute_alleles: bool = True,
                 is_imputed: bool = False,
                 transposed: bool = False):
        """Initialize GenotypeMatrix.

        Args:
            data: Numpy array, memmap, or path to memory-mapped file
            shape: Shape for memory-mapped files (n_individuals, n_markers)
                   or (n_markers, n_individuals) if transposed=True
            dtype: Data type for memory-mapped files
            precompute_alleles: Pre-compute major alleles for imputation
            is_imputed: Whether data has been pre-imputed (no -9 values)
            transposed: If True, internal storage is (n_markers, n_individuals)
                       for fast individual subsetting on memmaps
        """
        self._transposed = transposed

        if isinstance(data, np.memmap):
            self._data = data
            self._is_memmap = True
        elif isinstance(data, np.ndarray):
            self._data = data
            self._is_memmap = False
        elif isinstance(data, (str, Path)):
            # Memory-mapped file
            if shape is None:
                raise ValueError("Shape required for memory-mapped files")
            self._data = np.memmap(data, dtype=dtype, mode='r', shape=shape)
            self._is_memmap = True
        else:
            raise ValueError("Data must be array or file path")

        # Track if data has been pre-imputed (no -9 values)
        # This allows downstream code to skip -9 checks for faster processing
        self._is_imputed = is_imputed

        # Pre-compute major alleles for efficient imputation
        self._major_alleles = None
        self._missing_masks = None
        # Skip precompute when data is already imputed (no -9 values).
        if precompute_alleles and not self._is_imputed:
            self._precompute_major_alleles()
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape (n_individuals, n_markers)"""
        if self._transposed:
            # Internal storage is (n_markers, n_individuals)
            return (self._data.shape[1], self._data.shape[0])
        return self._data.shape

    @property
    def n_individuals(self) -> int:
        """Number of individuals"""
        return self.shape[0]

    @property
    def n_markers(self) -> int:
        """Number of markers"""
        return self.shape[1]

    @property
    def is_imputed(self) -> bool:
        """Whether missing values (-9) have been pre-imputed.

        When True, downstream code can skip -9 checks for faster processing.
        """
        return self._is_imputed

    @property
    def is_transposed(self) -> bool:
        """Whether internal storage is transposed (n_markers, n_individuals).

        When True, individual subsetting is fast (column slice on memmap).
        """
        return self._transposed

    def __getitem__(self, key):
        """Support array indexing - returns data in (individuals, markers) order"""
        if self._transposed:
            # Transpose the key for internal access
            if isinstance(key, tuple) and len(key) == 2:
                return self._data[key[1], key[0]].T if hasattr(self._data[key[1], key[0]], 'T') else self._data[key[1], key[0]]
            # For simple indexing, return transposed view
            return self._data.T[key]
        return self._data[key]

    def get_marker(self, marker_idx: int) -> np.ndarray:
        """Get genotypes for a specific marker"""
        if self._transposed:
            return self._data[marker_idx, :]
        return self._data[:, marker_idx]

    def get_individual(self, ind_idx: int) -> np.ndarray:
        """Get genotypes for a specific individual"""
        if self._transposed:
            return self._data[:, ind_idx]
        return self._data[ind_idx, :]

    def get_batch(self, marker_start: int, marker_end: int) -> np.ndarray:
        """Get batch of markers for efficient processing.

        Returns array of shape (n_individuals, n_markers_in_batch).
        """
        if self._transposed:
            # Internal: (markers, individuals), need to return (individuals, markers)
            return self._data[marker_start:marker_end, :].T
        return self._data[:, marker_start:marker_end]

    def subset_individuals(
        self,
        indices: Union[np.ndarray, list],
        *,
        precompute_alleles: Optional[bool] = None,
    ) -> "GenotypeMatrix":
        """Return a GenotypeMatrix restricted to a subset of individuals.

        Preserves the is_imputed flag so downstream code can use fast paths
        when genotypes are already imputed.

        When data is transposed (column-major), this is a fast column slice.
        The returned GenotypeMatrix is NOT transposed (standard row-major).
        """
        if isinstance(indices, list):
            indices = np.asarray(indices)
        if isinstance(indices, np.ndarray) and indices.dtype == bool:
            indexer = indices
        else:
            indexer = np.asarray(indices, dtype=int)

        if self._transposed:
            # Fast path: column slice on transposed data
            # Internal: (n_markers, n_individuals) -> slice columns
            # Result needs to be (n_individuals_subset, n_markers) for standard format
            subset = self._data[:, indexer].T.copy()
        else:
            # Slow path: row slice on row-major data
            subset = self._data[indexer, :]

        if precompute_alleles is None:
            precompute_alleles = not self._is_imputed
        return GenotypeMatrix(
            subset,
            precompute_alleles=precompute_alleles,
            is_imputed=self._is_imputed,
            transposed=False,  # Result is always standard format
        )
    
    def calculate_allele_frequencies(
        self,
        batch_size: int = 1000,
        max_dosage: float = 2.0,
    ) -> np.ndarray:
        """Calculate allele frequencies for all markers.

        Args:
            batch_size: Number of markers to process per batch.
            max_dosage: Maximum genotype dosage used when normalising to an
                allele frequency (default 2.0 for diploids).
        """
        n_markers = self.n_markers
        frequencies = np.zeros(n_markers)

        for start in range(0, n_markers, batch_size):
            end = min(start + batch_size, n_markers)
            batch = self.get_batch(start, end)
            # Frequency of alt allele = mean(genotype) / max_dosage
            frequencies[start:end] = np.mean(batch, axis=0) / max(max_dosage, 1e-12)

        return frequencies

    def calculate_maf(
        self,
        batch_size: int = 1000,
        max_dosage: float = 2.0,
    ) -> np.ndarray:
        """Calculate minor allele frequencies.

        Args:
            batch_size: Number of markers to process per batch.
            max_dosage: Maximum genotype dosage used when normalising to an
                allele frequency (default 2.0 for diploids).
        """
        frequencies = self.calculate_allele_frequencies(
            batch_size=batch_size,
            max_dosage=max_dosage,
        )
        return np.minimum(frequencies, 1 - frequencies)
    
    def _precompute_major_alleles(self, batch_size: int = 1000):
        """Pre-compute major alleles for all markers to optimize missing data imputation
        
        This matches rMVP's missing data imputation strategy exactly.
        """
        n_markers = self.n_markers
        self._major_alleles = np.zeros(n_markers, dtype=self._data.dtype)
        
        # Process in batches to handle large datasets
        for start in range(0, n_markers, batch_size):
            end = min(start + batch_size, n_markers)
            batch = self.get_batch(start, end)

            if batch.size == 0:
                continue

            # Missing mask aligns with rMVP sentinel handling
            missing_mask = (batch == -9) | np.isnan(batch)
            non_missing_counts = (~missing_mask).sum(axis=0)
            completely_missing = non_missing_counts == 0

            if np.all(completely_missing):
                # All markers in this block are entirely missing
                self._major_alleles[start:end] = 0
                continue

            valid_values = batch[~missing_mask]
            if valid_values.size == 0:
                self._major_alleles[start:end] = 0
                continue

            unique_vals = np.unique(valid_values)
            if unique_vals.size == 0:
                self._major_alleles[start:end] = 0
                continue

            unique_vals = unique_vals.astype(self._data.dtype, copy=False)
            counts = np.zeros((unique_vals.size, end - start), dtype=np.int32)
            for idx, val in enumerate(unique_vals):
                counts[idx, :] = np.sum(batch == val, axis=0)

            major_indices = np.argmax(counts, axis=0)
            major_vals = unique_vals[major_indices]
            if np.any(completely_missing):
                major_vals = major_vals.astype(self._data.dtype, copy=False)
                major_vals[completely_missing] = 0
            self._major_alleles[start:end] = major_vals
    
    def get_marker_imputed(self,
                           marker_idx: int,
                           *,
                           fill_value: Optional[float] = None,
                           dtype: np.dtype = np.float64) -> np.ndarray:
        """Get genotypes for a specific marker with optional missing-data imputation."""
        out_dtype = np.dtype(dtype)
        if self._is_imputed:
            # Fast path: data contains no missing values, so skip mask checks.
            if self._transposed:
                return self._data[marker_idx, :].astype(out_dtype, copy=True)
            return self._data[:, marker_idx].astype(out_dtype, copy=True)

        if self._transposed:
            marker = self._data[marker_idx, :].astype(out_dtype, copy=True)
        else:
            marker = self._data[:, marker_idx].astype(out_dtype, copy=True)

        missing_mask = (marker == -9) | np.isnan(marker)
        if not missing_mask.any():
            return marker

        if fill_value is not None:
            marker[missing_mask] = out_dtype.type(fill_value)
        elif self._major_alleles is not None:
            marker[missing_mask] = out_dtype.type(self._major_alleles[marker_idx])
        else:
            marker[missing_mask] = out_dtype.type(0.0)

        return marker

    def get_batch_imputed(self,
                          marker_start: int,
                          marker_end: int,
                          *,
                          fill_value: Optional[float] = None,
                          dtype: np.dtype = np.float64) -> np.ndarray:
        """Get batch of markers with missing data imputed.

        Args:
            marker_start: Inclusive start index
            marker_end: Exclusive end index
            fill_value: Optional constant to impute missing genotypes.
                If None, the pre-computed major allele is used (rMVP default).
            dtype: Output dtype for the returned array (default float64).

        Returns:
            Array of shape (n_individuals, n_markers_in_batch).
        """
        out_dtype = np.dtype(dtype)
        if self._is_imputed:
            # Fast path: pre-imputed data, no missing checks needed.
            if self._transposed:
                return self._data[marker_start:marker_end, :].T.astype(out_dtype, copy=True)
            return self._data[:, marker_start:marker_end].astype(out_dtype, copy=True)

        if self._transposed:
            batch = self._data[marker_start:marker_end, :].T.astype(out_dtype, copy=True)
        else:
            batch = self._data[:, marker_start:marker_end].astype(out_dtype, copy=True)

        if batch.size == 0:
            return batch

        missing_mask = (batch == -9) | np.isnan(batch)
        if not missing_mask.any():
            return batch

        if fill_value is not None:
            batch[missing_mask] = out_dtype.type(fill_value)
        elif self._major_alleles is not None:
            fill_vals = self._major_alleles[marker_start:marker_end].astype(out_dtype, copy=False)
            batch[missing_mask] = np.broadcast_to(fill_vals, batch.shape)[missing_mask]
        else:
            batch[missing_mask] = out_dtype.type(0.0)

        return batch

    def get_columns_imputed(self,
                             indices: Union[np.ndarray, list],
                             *,
                             fill_value: Optional[float] = None,
                             dtype: np.dtype = np.float64) -> np.ndarray:
        """Get arbitrary marker columns with missing data imputed.

        Optimized for fetching non-contiguous SNPs (e.g., pseudo-QTNs) in one call.
        Returns an array of shape (n_individuals, len(indices)) with requested dtype.
        """
        if isinstance(indices, list):
            indices = np.array(indices, dtype=int)
        out_dtype = np.dtype(dtype)
        if self._is_imputed:
            # Fast path: pre-imputed data, no missing checks needed.
            if self._transposed:
                return self._data[indices, :].T.astype(out_dtype, copy=True)
            return self._data[:, indices].astype(out_dtype, copy=True)

        # Slice and copy to ensure we don't mutate underlying storage
        if self._transposed:
            batch = self._data[indices, :].T.astype(out_dtype, copy=True)
        else:
            batch = self._data[:, indices].astype(out_dtype, copy=True)

        if batch.size == 0:
            return batch

        missing_mask = (batch == -9) | np.isnan(batch)
        if not missing_mask.any():
            return batch

        if fill_value is not None:
            batch[missing_mask] = out_dtype.type(fill_value)
        elif self._major_alleles is not None:
            fill_vals = self._major_alleles[indices].astype(out_dtype, copy=False)
            batch[missing_mask] = np.broadcast_to(fill_vals, batch.shape)[missing_mask]
        else:
            batch[missing_mask] = out_dtype.type(0.0)

        return batch
    
    @property
    def major_alleles(self) -> Optional[np.ndarray]:
        """Pre-computed major alleles for all markers"""
        return self._major_alleles


class AssociationResults:
    """GWAS association results structure
    
    Standard format: [Effect, SE, P-value] for each marker
    """
    
    def __init__(self, effects: np.ndarray, se: np.ndarray, pvalues: np.ndarray,
                 snp_map: Optional[GenotypeMap] = None):
        
        # Validate lengths (first dimension must match)
        n = len(effects)
        if hasattr(se, '__len__') and len(se) != n:
             raise ValueError(f"SE array length {len(se)} does not match effects length {n}")
        if hasattr(pvalues, '__len__') and len(pvalues) != n:
             raise ValueError(f"P-value array length {len(pvalues)} does not match effects length {n}")
            
        self.effects = effects
        self.se = se  
        self.pvalues = pvalues
        self.snp_map = snp_map
    
    @property
    def n_markers(self) -> int:
        """Number of markers"""
        return len(self.effects)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        df = pd.DataFrame({
            'Effect': self.effects,
            'SE': self.se,
            'P-value': self.pvalues
        })
        
        if self.snp_map is not None:
            df['SNP'] = self.snp_map.snp_ids.values
            df['Chr'] = self.snp_map.chromosomes.values  
            df['Pos'] = self.snp_map.positions.values
            df = df[['SNP', 'Chr', 'Pos', 'Effect', 'SE', 'P-value']]
            
        return df
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array [Effect, SE, P-value]
        
        If results contain multiple columns (e.g. for covariates),
        only the first column (marker results) is returned by default
        to maintain backward compatibility with 2D array expectations.
        """
        eff = self.effects if self.effects.ndim == 1 else self.effects[:, 0]
        se = self.se if self.se.ndim == 1 else self.se[:, 0]
        pv = self.pvalues if self.pvalues.ndim == 1 else self.pvalues[:, 0]
        
        return np.column_stack([eff, se, pv])


class KinshipMatrix:
    """Kinship matrix with validation and properties
    
    Must be symmetric positive semi-definite matrix
    """
    
    def __init__(self, data: Union[np.ndarray, str, Path]):
        if isinstance(data, (str, Path)):
            # Load from file - handle CSV with headers
            try:
                df = pd.read_csv(data, header=0)
                self._data = df.values.astype(float)
            except:
                # Try without headers  
                self._data = np.loadtxt(data, delimiter=',', skiprows=1)
        elif isinstance(data, np.ndarray):
            self._data = data.copy()
        else:
            raise ValueError("Data must be array or file path")
            
        # Validate properties
        if self._data.ndim != 2:
            raise ValueError("Kinship matrix must be 2D")
        if self._data.shape[0] != self._data.shape[1]:
            raise ValueError("Kinship matrix must be square")
        if not np.allclose(self._data, self._data.T, atol=1e-10):
            raise ValueError("Kinship matrix must be symmetric")
            
        self.n = self._data.shape[0]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape"""
        return self._data.shape
    
    def __getitem__(self, key):
        """Support array indexing"""
        return self._data[key]
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return self._data.copy()
    
    def eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigendecomposition for MLM"""
        eigenvals, eigenvecs = np.linalg.eigh(self._data)
        return eigenvals, eigenvecs


def load_validation_data(data_path: Union[str, Path]) -> Dict[str, Any]:
    """Load validation data for testing
    
    Returns dictionary with test data matching R rMVP outputs
    """
    data_path = Path(data_path)
    
    result = {}
    
    # Load test data
    if (data_path / "test_phenotype.csv").exists():
        result['phenotype'] = Phenotype(data_path / "test_phenotype.csv")
    
    if (data_path / "test_genotype_full.csv").exists():
        geno_data = pd.read_csv(data_path / "test_genotype_full.csv").values
        result['genotype'] = GenotypeMatrix(geno_data)
    
    if (data_path / "test_map.csv").exists():
        result['map'] = GenotypeMap(data_path / "test_map.csv")
    
    # Load expected results
    if (data_path / "test_glm_results.csv").exists():
        glm_data = pd.read_csv(data_path / "test_glm_results.csv").values
        result['expected_glm'] = AssociationResults(
            glm_data[:, 0], glm_data[:, 1], glm_data[:, 2]
        )
    
    if (data_path / "test_mlm_results.csv").exists():
        mlm_data = pd.read_csv(data_path / "test_mlm_results.csv").values  
        result['expected_mlm'] = AssociationResults(
            mlm_data[:, 0], mlm_data[:, 1], mlm_data[:, 2]
        )
    
    if (data_path / "test_kinship.csv").exists():
        result['expected_kinship'] = KinshipMatrix(data_path / "test_kinship.csv")
        
    if (data_path / "test_pca_results.csv").exists():
        result['expected_pca'] = pd.read_csv(data_path / "test_pca_results.csv").values
    
    return result
