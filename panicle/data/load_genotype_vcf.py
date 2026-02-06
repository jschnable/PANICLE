#!/usr/bin/env python
"""
VCF loader for GWAS: builds (geno_matrix, individual_ids, geno_map).

Key features:
- Streaming parsing of VCF text (supports .vcf and .vcf.gz)
- Supports arbitrary ploidy by counting ALT alleles in each GT field
- Optional multi-allelic splitting into per-ALT pseudo-biallelic markers
- GT-based coding with DS fallback; missing as -9 (int8)
- Optional basic QC filters: monomorphic, missingness, MAF

Return signature:
    (geno_matrix: np.ndarray[int8], individual_ids: List[str], geno_map: DataFrame-like)

The geno_map is a pandas DataFrame if pandas is installed; otherwise a list of dict rows.

This module avoids external VCF libraries for portability. For very large VCFs,
consider replacing the parser with cyvcf2/pysam keeping the same coding logic.
"""
from __future__ import print_function
import sys
import gzip
import io
import os
import tempfile
from typing import Dict, Optional, Tuple
from typing import Dict, Optional, Tuple

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise ImportError("NumPy is required: pip install numpy")
from panicle.utils.data_types import impute_major_allele_inplace


MISSING = -9

_GT_TOKEN_CACHE_SENTINEL = object()
_GT_TOKEN_CACHE: Dict[str, Optional[Tuple[str, ...]]] = {}
_BIALLELIC_DOSAGE_CACHE: Dict[Tuple[str, ...], Tuple[int, int]] = {}
_BIALLELIC_GT_CACHE: Dict[str, Tuple[int, int]] = {}
_BIALLELIC_GT_DIRECT: Dict[str, Tuple[int, int]] = {
    '0/0': (0, 2),
    '0|0': (0, 2),
    '0/1': (1, 2),
    '1/0': (1, 2),
    '0|1': (1, 2),
    '1|0': (1, 2),
    '1/1': (2, 2),
    '1|1': (2, 2),
    '0': (0, 1),
    '1': (1, 1),
    './.': (MISSING, 0),
    '.|.': (MISSING, 0),
    '.': (MISSING, 0),
}

_FORMAT_CACHE: Dict[str, Tuple[Tuple[str, ...], Dict[str, int]]] = {}


class _DynamicInt8MatrixWriter:
    """Append-only int8 matrix builder backed by a temporary memmap."""

    def __init__(self, n_rows, initial_capacity=4096):
        self.n_rows = int(n_rows)
        if self.n_rows <= 0:
            raise ValueError("Writer requires a positive number of rows")
        self.capacity = max(int(initial_capacity), 1)
        tmp = tempfile.NamedTemporaryFile(prefix="panicle_geno_", suffix=".tmp", delete=False)
        self.path = tmp.name
        tmp.close()
        self.memmap = np.memmap(self.path, dtype=np.int8, mode='w+', shape=(self.n_rows, self.capacity))
        self.count = 0

    def _grow(self, min_capacity):
        new_capacity = self.capacity
        while new_capacity < min_capacity:
            new_capacity = max(new_capacity * 2, min_capacity)
        old_columns = self.count
        if old_columns > 0:
            preserved = np.empty((self.n_rows, old_columns), dtype=np.int8)
            np.copyto(preserved, self.memmap[:, :old_columns])
        else:
            preserved = None
        self.memmap.flush()
        del self.memmap
        with open(self.path, 'r+b') as fh:
            fh.truncate(self.n_rows * new_capacity)
        self.memmap = np.memmap(self.path, dtype=np.int8, mode='r+', shape=(self.n_rows, new_capacity))
        if preserved is not None:
            self.memmap[:, :old_columns] = preserved
        self.capacity = new_capacity

    def append(self, column):
        if column.shape != (self.n_rows,):
            raise ValueError(f"Column shape mismatch: expected ({self.n_rows},), got {column.shape}")
        if self.count >= self.capacity:
            self._grow(self.count + 1)
        self.memmap[:, self.count] = column
        self.count += 1

    def finalize(self):
        mm = self.memmap
        if mm is None:
            return np.zeros((self.n_rows, 0), dtype=np.int8)
        total_cols = self.count
        mm.flush()
        if total_cols == 0:
            result = np.zeros((self.n_rows, 0), dtype=np.int8)
        else:
            result = np.array(mm[:, :total_cols], dtype=np.int8, copy=True, order='C')
        try:
            os.remove(self.path)
        except OSError:
            pass
        self.memmap = None
        del mm
        return result


def _open_text(path):
    """Open VCF text transparently from plain or gzip-compressed files.

    Accepts string or Path-like, and handles .vcf, .vcf.gz, and .vcf.bgz.
    """
    p = str(path)
    pl = p.lower()
    if pl.endswith('.gz') or pl.endswith('.bgz'):
        return io.TextIOWrapper(gzip.open(p, 'rb'))
    return open(p, 'r')


def _parse_samples(header_line):
    # header line starts with #CHROM
    cols = header_line.strip().split('\t')
    if len(cols) < 9 or cols[0] != '#CHROM':
        raise ValueError('Malformed VCF header line: missing #CHROM ... FORMAT ...')
    samples = cols[9:]
    return samples


def _build_snp_id(chrom, pos, vid, ref, alt):
    if vid and vid != '.':
        return vid
    return "%s:%s:%s:%s" % (chrom, pos, ref, alt)


def _parse_format_keys(fmt_str: str) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    cached = _FORMAT_CACHE.get(fmt_str)
    if cached is not None:
        return cached
    if not fmt_str:
        result: Tuple[Tuple[str, ...], Dict[str, int]] = (tuple(), {})
    else:
        keys = tuple(fmt_str.split(':'))
        key_to_idx = {k: i for i, k in enumerate(keys)}
        result = (keys, key_to_idx)
    _FORMAT_CACHE[fmt_str] = result
    return result


def _split_gt_tokens(gt):
    # Accept phased or unphased; return list of allele indices as strings
    if gt is None or gt == '.' or gt == './.' or gt == '.|.':
        return None
    cached = _GT_TOKEN_CACHE.get(gt, _GT_TOKEN_CACHE_SENTINEL)
    if cached is not _GT_TOKEN_CACHE_SENTINEL:
        return cached
    sep = '/' if '/' in gt else '|' if '|' in gt else None
    if sep is None:
        _GT_TOKEN_CACHE[gt] = None
        return None
    toks = tuple(gt.split(sep))
    if any(token == '' for token in toks):
        _GT_TOKEN_CACHE[gt] = None
        return None
    _GT_TOKEN_CACHE[gt] = toks
    return toks


def _code_dosage_biallelic(gt_tokens):
    # gt_tokens like ['0','1', ...]; returns (dosage, ploidy)
    if not gt_tokens:
        return MISSING, 0
    if isinstance(gt_tokens, tuple):
        key = gt_tokens
    else:
        key = tuple(gt_tokens)
    cached = _BIALLELIC_DOSAGE_CACHE.get(key)
    if cached is not None:
        return cached
    alt_count = 0
    ploidy = 0
    for token in gt_tokens:
        if token == '.':
            result = (MISSING, 0)
            _BIALLELIC_DOSAGE_CACHE[key] = result
            return result
        try:
            allele = int(token)
        except ValueError:
            result = (MISSING, 0)
            _BIALLELIC_DOSAGE_CACHE[key] = result
            return result
        if allele not in (0, 1):
            result = (MISSING, 0)
            _BIALLELIC_DOSAGE_CACHE[key] = result
            return result
        ploidy += 1
        if allele == 1:
            alt_count += 1
    result = (alt_count, ploidy)
    _BIALLELIC_DOSAGE_CACHE[key] = result
    return result


def _code_dosage_split(gt_tokens, alt_index):
    # alt_index is the 1-based index of ALT for the split
    if not gt_tokens:
        return MISSING, 0
    allowed = {0, alt_index}
    alt_count = 0
    ploidy = 0
    for token in gt_tokens:
        if token == '.':
            return MISSING, 0
        try:
            allele = int(token)
        except ValueError:
            return MISSING, 0
        if allele not in allowed:
            return MISSING, 0
        ploidy += 1
        if allele == alt_index:
            alt_count += 1
    return alt_count, ploidy


def _decode_biallelic_gt(gt: Optional[str]) -> Tuple[int, int]:
    if gt is None:
        return MISSING, 0
    direct = _BIALLELIC_GT_DIRECT.get(gt)
    if direct is not None:
        return direct
    cached = _BIALLELIC_GT_CACHE.get(gt)
    if cached is not None:
        return cached
    alt_count = 0
    ploidy = 0
    result: Tuple[int, int]
    reading_digit = False
    allele_value = 0
    for ch in gt:
        if ch == '.':
            result = (MISSING, 0)
            _BIALLELIC_GT_CACHE[gt] = result
            return result
        if ch in '/|':
            if reading_digit:
                if allele_value == 1:
                    alt_count += 1
                elif allele_value not in (0,):
                    result = (MISSING, 0)
                    _BIALLELIC_GT_CACHE[gt] = result
                    return result
                ploidy += 1
                reading_digit = False
                allele_value = 0
            continue
        if '0' <= ch <= '9':
            reading_digit = True
            allele_value = allele_value * 10 + (ord(ch) - 48)
        else:
            result = (MISSING, 0)
            _BIALLELIC_GT_CACHE[gt] = result
            return result
    if reading_digit:
        if allele_value == 1:
            alt_count += 1
        elif allele_value not in (0,):
            result = (MISSING, 0)
            _BIALLELIC_GT_CACHE[gt] = result
            return result
        ploidy += 1
    if ploidy == 0:
        result = (MISSING, 0)
    else:
        result = (alt_count, ploidy)
    _BIALLELIC_GT_CACHE[gt] = result
    return result


def _ds_to_int(ds_val):
    try:
        x = float(ds_val)
    except Exception:
        return MISSING
    if np.isnan(x):
        return MISSING
    xi = int(round(x))
    if xi < 0:
        return MISSING
    return xi


def load_genotype_vcf(
    vcf_path,
    split_multiallelic=True,
    include_indels=True,
    drop_monomorphic=False,
    max_missing=1.0,
    min_maf=0.0,
    return_pandas=True,
    backend='auto',  # 'auto', 'cyvcf2', 'builtin'
    force_recache=False,
):
    """
    Load a VCF file and return (geno_matrix, individual_ids, geno_map).

    Parameters
    - vcf_path: path to .vcf or .vcf.gz
    - split_multiallelic: if True, split multi-ALT variants into separate entries
    - include_indels: include biallelic indels (if False, only include SNPs)
    - drop_monomorphic: drop variants with all non-missing 0 or all 2
    - max_missing: drop variants with missing rate > threshold (0..1]
    - min_maf: drop variants with minor allele frequency < threshold
      (missing calls treated as major allele for filtering)
    - force_recache: if True, ignore any existing cache and overwrite it
    - return_pandas: return geno_map as pandas.DataFrame if pandas is available
    - backend: 'auto' (defaults to builtin for VCF/VCF.GZ), 'cyvcf2', or 'builtin'
    """
    import re
    import os
    import numpy as np
    import pandas as pd

    # Backend selection: prefer builtin for VCF text, require cyvcf2 for BCF.
    # --- CACHING LOGIC START ---
    # Cache version 2: pre-imputes missing values (-9) at cache time for faster downstream
    cache_base = str(vcf_path)
    cache_geno = cache_base + '.panicle.v2.geno.npy'
    cache_ind = cache_base + '.panicle.v2.ind.txt'
    cache_map = cache_base + '.panicle.v2.map.csv'

    # Check if cache exists and is fresh
    try:
        if not force_recache:
            if os.path.exists(cache_geno) and os.path.exists(cache_ind) and os.path.exists(cache_map):
                vcf_mtime = os.path.getmtime(vcf_path)
                if (os.path.getmtime(cache_geno) > vcf_mtime and
                    os.path.getmtime(cache_ind) > vcf_mtime and
                    os.path.getmtime(cache_map) > vcf_mtime):

                    print(f"   [Cache] Loading binary cache for {vcf_path}...")
                    if min_maf > 0.0 or max_missing < 1.0 or drop_monomorphic:
                        print(
                            "   [Cache] Warning: cached genotype data loaded; "
                            "min_maf/max_missing/drop_monomorphic filters are not re-applied "
                            f"(min_maf={min_maf}, max_missing={max_missing}, "
                            f"drop_monomorphic={drop_monomorphic}). "
                            "Use --force-recache or delete the cache files to rebuild."
                        )

                    # Load Genotypes (memmap for speed/memory efficiency)
                    geno_matrix = np.load(cache_geno, mmap_mode='r')

                    # Load Individuals
                    with open(cache_ind, 'r') as f:
                        individual_ids = [line.strip() for line in f]

                    # Load Map
                    geno_map = pd.read_csv(cache_map)

                    # Mark that this data is from v2 cache (pre-imputed, no -9 values)
                    geno_map.attrs['is_imputed'] = True

                    # If memmapped, we return it as is. GenotypeMatrix handles it.
                    return geno_matrix, individual_ids, geno_map
    except Exception as e:
        print(f"   [Cache] Failed to load cache: {e}")
    # --- CACHING LOGIC END ---

    # Standard loading proceeds...
    vcf_lower = str(vcf_path).lower()
    is_bcf = vcf_lower.endswith('.bcf')

    if backend == 'auto':
        try:
            import cyvcf2  # type: ignore
            use_cyvcf2 = True
        except ImportError:
            use_cyvcf2 = False
            # Check if BCF (which strictly requires cyvcf2)
            if is_bcf:
                raise ImportError(
                    'Loading .bcf requires cyvcf2. Install with "pip install cyvcf2" or convert to .vcf/.vcf.gz.'
                )
    elif backend == 'cyvcf2':
        try:
            import cyvcf2  # type: ignore
            use_cyvcf2 = True
        except Exception:
            raise ImportError('cyvcf2 requested but not available')
    elif backend == 'builtin':
        use_cyvcf2 = False
    else:
        raise ValueError("backend must be one of {'auto', 'cyvcf2', 'builtin'}")

    # Determine thread count (default to 4 or CPU count)
    import multiprocessing
    n_threads = min(4, multiprocessing.cpu_count())
    
    # Initialize VCF reader based on backend and file type
    vcf = None
    if is_bcf:
        # BCF requires cyvcf2
        try:
            from cyvcf2 import VCF
            vcf = VCF(vcf_path, threads=n_threads)
        except ImportError:
            raise ImportError('cyvcf2 is required for BCF files')
    elif use_cyvcf2:
        # Optimized VCF path
        from cyvcf2 import VCF
        vcf = VCF(vcf_path, threads=n_threads)
    else:
        # Builtin path (no threads)
        # Guard: .bcf is binary and not supported by builtin parser
        if is_bcf: # This case should have been caught by the `if is_bcf` block above
            raise ImportError('Loading .bcf requires cyvcf2. Install with "pip install cyvcf2" or convert to .vcf/.vcf.gz.')
        # vcf reader not needed here; builtin path uses _open_text() directly below

    # Initialize
    individual_ids = None
    writer = None  # lazy initialised streaming writer
    map_rows = []  # dict rows

    # Helper to finalize a candidate variant column with QC
    def consider_variant(col, chrom, pos, vid, ref, alt, ploidy):
        nonlocal writer, map_rows
        col = np.asarray(col, dtype=np.int16)  # temp safe range
        # If requested, restrict to SNPs
        if not include_indels:
            if len(ref) != 1 or len(alt) != 1:
                return
        # Skip if all missing
        valid = col != MISSING
        if not np.any(valid):
            return
        # Optional monomorphic filter
        if drop_monomorphic:
            vals = np.unique(col[valid])
            if vals.size == 1 and (vals[0] == 0 or vals[0] == 2):
                return
        # Missingness filter
        miss_rate = 1.0 - (np.count_nonzero(valid) / float(col.size))
        if miss_rate > max_missing:
            return
        # MAF filter
        if min_maf > 0.0:
            n_total = col.size
            n_valid = int(np.count_nonzero(valid))
            if n_valid > 0:
                total_alleles = max(ploidy, 1) * n_total
                valid_alleles = max(ploidy, 1) * n_valid
                sum_dos = float(np.sum(col[valid]))
                minor_count = min(sum_dos, valid_alleles - sum_dos)
                maf = minor_count / max(total_alleles, 1.0)
                if maf < min_maf:
                    return
        # Finalize dtype and append to streaming writer
        col = col.astype(np.int8, copy=False)
        if writer is None:
            writer = _DynamicInt8MatrixWriter(len(individual_ids))
        writer.append(col)
        map_rows.append({
            'SNP': _build_snp_id(chrom, pos, vid, ref, alt),
            'CHROM': str(chrom),
            'POS': int(pos),
            'REF': ref,
            'ALT': alt,
        })

    if use_cyvcf2:
        # Fast path using cyvcf2 (reuse threaded reader created above)
        if vcf is None:
            from cyvcf2 import VCF  # type: ignore
            vcf = VCF(vcf_path, threads=n_threads)
        individual_ids = list(vcf.samples)
        n = len(individual_ids)
        if n == 0:
            raise ValueError('VCF contains no sample columns')

        def consider_variant(col, chrom, pos, vid, ref, alt, ploidy):
            nonlocal writer, map_rows
            col = np.asarray(col, dtype=np.int16)
            if not include_indels and (len(ref) != 1 or len(alt) != 1):
                return
            valid = col != MISSING
            if not np.any(valid):
                return
            if drop_monomorphic:
                vals = np.unique(col[valid])
                if vals.size == 1 and (vals[0] == 0 or vals[0] == 2):
                    return
            miss_rate = 1.0 - (np.count_nonzero(valid) / float(col.size))
            if miss_rate > max_missing:
                return
            if min_maf > 0.0:
                n_total = col.size
                n_valid = int(np.count_nonzero(valid))
                if n_valid > 0:
                    total_alleles = max(ploidy, 1) * n_total
                    valid_alleles = max(ploidy, 1) * n_valid
                    sum_dos = float(np.sum(col[valid]))
                    minor_count = min(sum_dos, valid_alleles - sum_dos)
                    maf = minor_count / max(total_alleles, 1.0)
                    if maf < min_maf:
                        return
            col = col.astype(np.int8, copy=False)
            if writer is None:
                writer = _DynamicInt8MatrixWriter(len(individual_ids))
            writer.append(col)
            map_rows.append({
                'SNP': _build_snp_id(chrom, pos, vid, ref, alt),
                'CHROM': str(chrom),
                'POS': int(pos),
                'REF': ref,
                'ALT': alt,
            })

        # Fast iteration over variants
        for var in vcf:
            chrom = var.CHROM
            pos = int(var.POS)
            vid = var.ID if var.ID else '.'
            ref = var.REF
            alts = var.ALT or []
            if not alts:
                continue

            # Check if biallelic SNP (most common, optimize this path)
            if len(alts) == 1:
                # Fast path using genotype.array() 
                # Benchmarks show this is robust and fast (~4.6s vs 36s builtin)
                # We avoid gt_types because it can return 3 (Unknown) for valid HomAlt indels in some files.
                
                try:
                    # Returns (N, 3) for diploid phased [a, b, phase]
                    gt_arr = np.array(var.genotype.array())
                except Exception:
                    continue

                # Strip phase -> (N, 2)
                alleles = gt_arr[:, :-1]
                
                # Check for missing (-1)
                # If any allele is missing, treat call as missing
                missing_mask = np.any(alleles < 0, axis=1)
                
                # Check for non-biallelic codes (shouldn't happen if len(alts)==1 and VCF is valid)
                # But if we see 2, 3.. it means multi-allelic site encoded weirdly?
                # For safety, mask them or rely on simple sum if we trust the file
                # Simple sum matches biallelic expectation: 0+0=0, 0+1=1, 1+1=2.
                # If we have 2 (allele 2), sum is > 2, which logic below might clamp or accept? 
                # PANICLE expects 0,1,2.
                
                # Compute dosage directly; missing handled by mask below
                dosages = np.sum(alleles, axis=1)
                
                # Cast to int16 for consider_variant
                col = dosages.astype(np.int16)
                
                # Apply missing mask
                col[missing_mask] = MISSING
                
                # Final integrity check: if sum > 2, treat as missing (unexpected allele index)
                # This handles cases where a site is marked biallelic but has allele index 2
                col[dosages > 2] = MISSING
                
                consider_variant(col, chrom, pos, vid, ref, alts[0], 2)
                
            else:
                if not split_multiallelic:
                    continue
                
                # Multi-allelic: slightly slower path using genotype.array() or manual parsing
                # cyvcf2 usually handles this by iterating, but we can do better with genotype.array()
                # genotype.array() returns (N, 3) for diploid: n_alleles=2 + phased_bool
                # The values are 0, 1, ... index of allele. -1 for missing.
                
                try:
                    # Shape (N, P+1) where P is max ploidy + phase bit
                    gt_arr = np.array(var.genotype.array())
                except Exception:
                    # Fallback to slow loop if array access fails
                    continue
                
                # Strip last column (phasing) -> Shape (N, P)
                alleles = gt_arr[:, :-1]
                # Filter out -2 (pad)? cyvcf2 uses -2 for pad, -1 for missing
                
                # Iterate over ALTs
                for ai, alt_base in enumerate(alts, start=1):
                    # We want count of allele `ai`
                    # Mask for valid calls: neither allele is missing (-1) and logic for allowed alleles
                    # Builtin logic: "allowed = {0, ai}". If any allele is not 0 or ai, set to missing.
                    
                    # 1. Mask where any allele is NOT (0 or ai or -1 or -2)
                    # This is equivalent to: (allele != 0) & (allele != ai) & (allele >= 0)
                    invalid_alleles = (alleles != 0) & (alleles != ai) & (alleles >= 0)
                    # If any allele in a genotype is invalid, the whole call is missing
                    row_invalid_mask = np.any(invalid_alleles, axis=1)
                    
                    # 2. Count `ai` in valid rows
                    # (alleles == ai).sum(axis=1)
                    counts = np.sum(alleles == ai, axis=1)
                    
                    # 3. Determine PLOIDY (count of non-pad alleles)
                    # Pad is -2
                    # n_alleles per sample
                    non_pad = (alleles != -2)
                    sample_ploidy = np.sum(non_pad, axis=1)
                    # Max ploidy for this variant
                    var_ploidy = int(np.max(sample_ploidy)) if len(sample_ploidy) > 0 else 2
                    
                    # 4. Construct final col
                    col = counts.astype(np.int16)
                    
                    # Apply missingness
                    # Missing if: row_invalid OR any allele is -1 (missing)
                    # Note: cyvcf2 uses -1 for missing.
                    # If any allele is -1, is the whole call missing? PyMVP logic says yes.
                    has_missing = np.any(alleles == -1, axis=1)
                    
                    final_mask = row_invalid_mask | has_missing
                    col[final_mask] = MISSING
                    
                    consider_variant(col, chrom, pos, vid, ref, alt_base, var_ploidy)
    else:
        # Built-in text parser
        # Guard: .bcf is binary and not supported by builtin parser
        if is_bcf:
            raise ImportError('Builtin VCF parser does not support .bcf. Please install cyvcf2 or use .vcf/.vcf.gz.')
        sanity_checked = False
        fh = _open_text(vcf_path)
        try:
            for line in fh:
                if not line:
                    continue
                if line.startswith('##'):
                    continue
                if line.startswith('#CHROM'):
                    individual_ids = _parse_samples(line)
                    n = len(individual_ids)
                    # Edge: no samples
                    if n == 0:
                        raise ValueError('VCF contains no sample columns')
                    continue
                # Data line
                if individual_ids is None:
                    raise ValueError('VCF header not found before data lines')
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 8:
                    continue  # malformed
                chrom, pos_str, vid, ref, alt_str = parts[0], parts[1], parts[2], parts[3], parts[4]
                pos = int(pos_str)

                # Determine ALT alleles
                alt_alleles = alt_str.split(',') if alt_str and alt_str != '.' else []
                if not alt_alleles:
                    continue  # no ALT

                fmt = parts[8] if len(parts) >= 9 else ''
                sample_fields = parts[9:] if len(parts) >= 10 else []
                fmt_keys, key_to_idx = _parse_format_keys(fmt)

                gt_index = key_to_idx.get('GT')
                ds_index = key_to_idx.get('DS')
                gt_primary = gt_index == 0

                ds_array: Optional[np.ndarray]
                if ds_index is None:
                    ds_array = None
                else:
                    ds_array = np.full(len(individual_ids), MISSING, dtype=np.int16)
                    if ds_index == 0:
                        for si, field in enumerate(sample_fields):
                            token = field.partition(':')[0]
                            if token:
                                ds_array[si] = _ds_to_int(token)
                    elif ds_index == 1 and gt_primary:
                        for si, field in enumerate(sample_fields):
                            head, sep, tail = field.partition(':')
                            if sep:
                                token, _, _ = tail.partition(':')
                                if token:
                                    ds_array[si] = _ds_to_int(token)
                    else:
                        for si, field in enumerate(sample_fields):
                            toks = field.split(':')
                            if ds_index < len(toks):
                                token = toks[ds_index]
                                if token:
                                    ds_array[si] = _ds_to_int(token)

                is_biallelic = len(alt_alleles) == 1

                if gt_index is not None:
                    if gt_primary:
                        gt_values = [
                            field.partition(':')[0] if field else '' for field in sample_fields
                        ]
                    else:
                        gt_values = []
                        for field in sample_fields:
                            if not field:
                                gt_values.append('')
                                continue
                            toks = field.split(':')
                            gt_values.append(toks[gt_index] if gt_index < len(toks) else '')
                else:
                    gt_values = [''] * len(sample_fields)
                gt_array = np.array(gt_values, dtype='<U8') if gt_values else np.empty(len(sample_fields), dtype='<U8')

                split_tokens = _split_gt_tokens

                # Helper: build column(s) for this site
                def build_columns_for_alt(alt_index, alt_base):
                    nonlocal sanity_checked
                    col = np.full(len(individual_ids), MISSING, dtype=np.int16)
                    missing_mask = np.ones(len(individual_ids), dtype=bool)
                    variant_ploidy = 0
                    if is_biallelic and gt_index is not None:
                        if not sanity_checked:
                            if len(alt_alleles) != 1:
                                raise ValueError(
                                    "Multi-allelic variants are not supported by the fast builtin loader. "
                                    "Please switch to the cyvcf2 backend."
                                )
                            subset = gt_array[: min(10, gt_array.size)]
                            if subset.size:
                                subset = subset[(subset != '') & (np.char.find(subset, '.') == -1)]
                                if subset.size:
                                    cleaned_subset = np.char.replace(np.char.replace(subset, '/', ''), '|', '')
                                    if np.any(np.char.find(cleaned_subset, '2') != -1) or np.any(np.char.find(cleaned_subset, '3') != -1):
                                        raise ValueError(
                                            "Detected genotype allele codes greater than 1. "
                                            "Polyploid genotypes require the cyvcf2 backend."
                                        )
                                    lengths = np.char.str_len(cleaned_subset)
                                    if np.any(lengths > 2):
                                        raise ValueError(
                                            "Detected genotypes with ploidy greater than diploid. "
                                            "Please use the cyvcf2 backend for polyploid datasets."
                                        )
                            sanity_checked = True

                        unique_gts = np.unique(gt_array)
                        for gt_code in unique_gts:
                            mask = gt_array == gt_code
                            if not gt_code:
                                if ds_array is not None:
                                    ds_mask = mask & (ds_array != MISSING)
                                    if np.any(ds_mask):
                                        col[ds_mask] = ds_array[ds_mask]
                                        missing_mask[ds_mask] = False
                                continue
                            dosage, ploidy = _decode_biallelic_gt(gt_code)
                            if dosage != MISSING:
                                col[mask] = dosage
                                missing_mask[mask] = False
                                variant_ploidy = max(variant_ploidy, ploidy)
                            elif ds_array is not None:
                                ds_mask = mask & (ds_array != MISSING)
                                if np.any(ds_mask):
                                    col[ds_mask] = ds_array[ds_mask]
                                    missing_mask[ds_mask] = False
                    else:
                        for si, gt in enumerate(gt_values):
                            ds_val = ds_array[si] if ds_array is not None else MISSING
                            gt_tokens = split_tokens(gt) if gt else None
                            if gt_tokens is not None:
                                dosage, ploidy = _code_dosage_split(gt_tokens, alt_index)
                                if dosage != MISSING:
                                    col[si] = dosage
                                    variant_ploidy = max(variant_ploidy, ploidy)
                                    missing_mask[si] = False
                                elif ds_val != MISSING:
                                    col[si] = ds_val
                                    missing_mask[si] = False
                            elif ds_val != MISSING:
                                col[si] = ds_val
                                missing_mask[si] = False
                    if ds_array is not None:
                        ds_mask = missing_mask & (ds_array != MISSING)
                        if np.any(ds_mask):
                            col[ds_mask] = ds_array[ds_mask]
                            missing_mask[ds_mask] = False
                    return col, variant_ploidy

                if len(alt_alleles) == 1:
                    col, ploidy = build_columns_for_alt(1, alt_alleles[0])
                    if ploidy == 0:
                        ploidy = 2
                    consider_variant(col, chrom, pos, vid, ref, alt_alleles[0], ploidy)
                else:
                    if not split_multiallelic:
                        # Skip multi-allelic sites entirely in non-split mode
                        continue
                    for ai, alt_base in enumerate(alt_alleles, start=1):
                        col, ploidy = build_columns_for_alt(ai, alt_base)
                        if ploidy == 0:
                            ploidy = 2
                        consider_variant(col, chrom, pos, vid, ref, alt_base, ploidy)
        finally:
            try:
                fh.close()
            except Exception:
                pass

    if writer is None:
        # No variants passed filters
        geno = np.zeros((len(individual_ids or []), 0), dtype=np.int8)
    else:
        geno = writer.finalize()

    # Build geno_map output
    if return_pandas:
        try:
            import pandas as pd  # type: ignore
            geno_map = pd.DataFrame(map_rows, columns=['SNP', 'CHROM', 'POS', 'REF', 'ALT'])
        except Exception:
            geno_map = map_rows
    else:
        geno_map = map_rows

    # Integrity checks
    if individual_ids is None:
        raise ValueError('No header line found; invalid VCF')
    if geno.shape[0] != len(individual_ids):
        raise AssertionError('Row count mismatch: %d vs %d' % (geno.shape[0], len(individual_ids)))
    n_markers = geno.shape[1]
    if isinstance(geno_map, list):
        if len(geno_map) != n_markers:
            raise AssertionError('Map length mismatch: %d vs %d' % (len(geno_map), n_markers))
    else:
        if int(getattr(geno_map, 'shape', (0, 0))[0]) != n_markers:
            raise AssertionError('Map length mismatch: %d vs %d' % (int(geno_map.shape[0]), n_markers))

    # --- CACHING LOGIC SAVE START ---
    try:
        # Impute missing values (-9) before caching
        # This avoids repeated -9 checks in downstream kinship/MLM code
        n_missing = impute_major_allele_inplace(geno, missing_value=MISSING)
        if n_missing > 0:
            print(f"   [Cache] Imputed {n_missing:,} missing values ({100*n_missing/geno.size:.2f}%)")
        if hasattr(geno_map, "attrs"):
            geno_map.attrs["is_imputed"] = True

        # Save only if successful
        print(f"   [Cache] Saving binary cache to {cache_base}.panicle.v2.*")
        np.save(cache_geno, geno)

        with open(cache_ind, 'w') as f:
            for ind in individual_ids:
                f.write(f"{ind}\n")

        # Save Map
        if isinstance(geno_map, list):
             pd.DataFrame(geno_map).to_csv(cache_map, index=False)
        else:
             geno_map.to_csv(cache_map, index=False)

    except Exception as e:
        print(f"   [Cache] Warning: Failed to save cache: {e}")
    # --- CACHING LOGIC SAVE END ---

    return geno, individual_ids, geno_map


def _main(argv):  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description='Load VCF into genotype matrix for GWAS')
    p.add_argument('vcf')
    p.add_argument('--no-split', action='store_true', help='Do not split multi-allelic sites')
    p.add_argument('--snps-only', action='store_true', help='Restrict to SNPs only')
    p.add_argument('--drop-monomorphic', action='store_true')
    p.add_argument('--max-missing', type=float, default=1.0)
    p.add_argument('--min-maf', type=float, default=0.0)
    p.add_argument('--force-recache', action='store_true', help='Rebuild and overwrite cache files')
    p.add_argument('--no-pandas', action='store_true', help='Return map as list instead of DataFrame')
    p.add_argument('--backend', choices=['auto','cyvcf2','builtin'], default='auto',
                   help='Choose parsing backend (auto prefers builtin for VCF, cyvcf2 for BCF)')
    args = p.parse_args(argv)

    geno, ids, gmap = load_genotype_vcf(
        args.vcf,
        split_multiallelic=not args.no_split,
        include_indels=not args.snps_only,
        drop_monomorphic=args.drop_monomorphic,
        max_missing=args.max_missing,
        min_maf=args.min_maf,
        return_pandas=not args.no_pandas,
        backend=args.backend,
        force_recache=args.force_recache,
    )
    print('Samples:', len(ids))
    print('Markers:', geno.shape[1])
    print('Genotype dtype:', geno.dtype)
    # Spot-check MAF for first few markers
    if geno.shape[1] > 0:
        col = geno[:, 0]
        mask = col != MISSING
        if mask.any():
            n_total = col.size
            n_valid = int(np.count_nonzero(mask))
            total_alleles = 2 * n_total
            valid_alleles = 2 * n_valid
            sum_dos = float(np.sum(col[mask]))
            minor_count = min(sum_dos, valid_alleles - sum_dos)
            maf = minor_count / max(total_alleles, 1.0)
            print('First marker MAF ~', round(maf, 4))
    # Print first few map rows
    if hasattr(gmap, 'head'):
        print(gmap.head())
    else:
        print(gmap[:3])


if __name__ == '__main__':  # pragma: no cover
    _main(sys.argv[1:])
