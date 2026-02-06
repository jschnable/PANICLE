#!/usr/bin/env python
"""
HapMap (.hmp or .hmp.txt) loader: builds (geno_matrix, individual_ids, geno_map).

Spec highlights:
- Tab-delimited text, fixed 11 metadata columns, samples start at column 12
- Header columns: rs#, alleles, chrom, pos, strand, assembly#, center, protLSID, assayLSID, panelLSID, QCcode, <sample...> (# characters are optional)
- Genotypes: single IUPAC character per cell: A,C,G,T (hom), M,R,W,S,Y,K (hets), N (missing). Lowercase allowed.
- Alleles field is REF/ALT, e.g., A/C. Use the right allele (ALT) to define dosage: REF=0, HET=1, ALT=2.

Compressed files (.gz/.bgz) are supported by extension.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import os
import io
import gzip

logger = logging.getLogger(__name__)

try:
    import numpy as np
except Exception:
    raise ImportError("NumPy is required: pip install numpy")
from panicle.utils.data_types import impute_major_allele_inplace

MISSING = -9


_HAPMAP_HEADER = [
    'rs#','alleles','chrom','pos','strand','assembly#','center','protLSID','assayLSID','panelLSID','QCcode'
]

# Heterozygote IUPAC codes for unordered allele pairs
_HET_IUPAC = {
    ('A','C'): 'M', ('C','A'): 'M',
    ('A','G'): 'R', ('G','A'): 'R',
    ('A','T'): 'W', ('T','A'): 'W',
    ('C','G'): 'S', ('G','C'): 'S',
    ('C','T'): 'Y', ('T','C'): 'Y',
    ('G','T'): 'K', ('T','G'): 'K',
}


def _open_text(path: Union[str, Path]):
    p = str(path)
    pl = p.lower()
    if pl.endswith('.gz') or pl.endswith('.bgz'):
        return io.TextIOWrapper(gzip.open(p, 'rb'))
    return open(p, 'r')


def _parse_header(line: str) -> List[str]:
    cols = [c.strip() for c in line.rstrip('\n').split('\t')]
    if len(cols) < 12:
        raise ValueError('HapMap header must have at least 12 columns (11 metadata + >=1 sample)')
    # Validate first 11 headers with optional # characters and panelLSID/panel variation
    normalized_cols = [col.rstrip('#') for col in cols[:11]]
    normalized_expected = [col.rstrip('#') for col in _HAPMAP_HEADER]

    # Handle the panelLSID/panel variation at position 9
    if len(normalized_cols) > 9 and normalized_cols[9] == 'panel':
        normalized_cols[9] = 'panelLSID'

    if normalized_cols != normalized_expected:
        raise ValueError('Invalid HapMap header columns. Expected first 11 columns: ' + ','.join(_HAPMAP_HEADER))
    return cols


def _code_cell(geno_char: str, ref: str, alt: str) -> int:
    if not geno_char:
        return MISSING
    g = geno_char.strip().upper()
    if g in ('N', '-', '?', 'NN'):
        return MISSING

    # Handle diploid format (e.g., "AA", "CC", "AC")
    if len(g) == 2:
        allele1, allele2 = g[0], g[1]
        # Count occurrences of ref and alt alleles
        ref_count = (allele1 == ref) + (allele2 == ref)
        alt_count = (allele1 == alt) + (allele2 == alt)

        if ref_count == 2:
            return 0  # Homozygous reference
        elif alt_count == 2:
            return 2  # Homozygous alternate
        elif ref_count == 1 and alt_count == 1:
            return 1  # Heterozygous
        else:
            return MISSING  # Contains alleles not in ref/alt

    # Handle single character format (original IUPAC)
    # Allowed direct homozygote calls
    if g == ref:
        return 0
    if g == alt:
        return 2
    # Heterozygote IUPAC only for SNP alleles (A/C/G/T)
    if ref in 'ACGT' and alt in 'ACGT':
        het_code = _HET_IUPAC.get((ref, alt))
        if het_code and g == het_code:
            return 1
    # Otherwise treat as missing (e.g., indel sites with nonstandard encoding)
    return MISSING


def load_genotype_hapmap(
    hapmap_path: Union[str, Path],
    include_indels: bool = True,
    drop_monomorphic: bool = False,
    max_missing: float = 1.0,
    min_maf: float = 0.0,
    return_pandas: bool = True,
    force_recache: bool = False,
) -> Tuple[np.ndarray, List[str], object]:
    """
    Load HapMap .hmp(.txt) file and return (geno_matrix, individual_ids, geno_map).

    - HapMap is biallelic by design; we code 0/1/2 as REF/het/ALT by alleles field.
    - Indel handling is conservative: only A/C/G/T base codes are recognized; all others (except N) → missing,
      unless the cell is an exact REF or ALT base.
    - force_recache: ignore any existing cache and rebuild it.
    """
    # Cache version 2: pre-imputed, matches VCF cache behavior
    cache_base = str(hapmap_path)
    cache_geno = cache_base + '.panicle.v2.geno.npy'
    cache_ind = cache_base + '.panicle.v2.ind.txt'
    cache_map = cache_base + '.panicle.v2.map.csv'

    try:
        if not force_recache:
            if os.path.exists(cache_geno) and os.path.exists(cache_ind) and os.path.exists(cache_map):
                src_mtime = os.path.getmtime(hapmap_path)
                if (os.path.getmtime(cache_geno) > src_mtime and
                    os.path.getmtime(cache_ind) > src_mtime and
                    os.path.getmtime(cache_map) > src_mtime):
                    logger.info("[Cache] Loading binary cache for %s...", hapmap_path)
                    if min_maf > 0.0 or max_missing < 1.0 or drop_monomorphic:
                        logger.warning(
                            "[Cache] Cached genotype data loaded; "
                            "min_maf/max_missing/drop_monomorphic filters are not re-applied "
                            "(min_maf=%s, max_missing=%s, drop_monomorphic=%s). "
                            "Use --force-recache or delete the cache files to rebuild.",
                            min_maf, max_missing, drop_monomorphic,
                        )
                    geno_matrix = np.load(cache_geno, mmap_mode='r')
                    with open(cache_ind, 'r') as f:
                        individual_ids = [line.strip() for line in f]
                    import pandas as pd  # type: ignore
                    geno_map = pd.read_csv(cache_map)
                    geno_map.attrs["is_imputed"] = True
                    return geno_matrix, individual_ids, geno_map
    except Exception as e:
        logger.warning("[Cache] Failed to load cache: %s", e)

    individual_ids: Optional[List[str]] = None
    columns: List[np.ndarray] = []
    map_rows: List[dict] = []

    with _open_text(hapmap_path) as fh:
        header = fh.readline()
        if not header:
            raise ValueError('Empty HapMap file')
        cols = _parse_header(header)
        individual_ids = cols[11:]
        n = len(individual_ids)
        if n == 0:
            raise ValueError('HapMap contains no sample columns')

        for line in fh:
            if not line:
                continue
            parts = [c.strip() for c in line.rstrip('\n').split('\t')]
            if len(parts) != 11 + n:
                # Enforce strict column count
                continue
            rsid, alleles, chrom, pos_str = parts[0], parts[1], parts[2], parts[3]
            # Basic integrity
            if not alleles or '/' not in alleles:
                continue
            a_ref, a_alt = [a.strip().upper() for a in alleles.split('/')[:2]]
            # Indel inclusion filter
            if not include_indels:
                if a_ref not in 'ACGT' or a_alt not in 'ACGT':
                    continue
            try:
                pos = int(float(pos_str))
                if pos <= 0:
                    continue
            except Exception:
                continue

            geno = np.full(n, MISSING, dtype=np.int16)
            sample_cells = parts[11:]
            for i, cell in enumerate(sample_cells):
                geno[i] = _code_cell(cell, a_ref, a_alt)

            # QC filters
            valid = (geno != MISSING)
            if not np.any(valid):
                continue
            if drop_monomorphic:
                vals = np.unique(geno[valid])
                if vals.size == 1 and (vals[0] == 0 or vals[0] == 2):
                    continue
            miss_rate = 1.0 - (np.count_nonzero(valid) / float(n))
            if miss_rate > max_missing:
                continue
            if min_maf > 0.0:
                mean_dos = float(np.mean(geno[valid]))
                maf = min(mean_dos / 2.0, 1.0 - (mean_dos / 2.0))
                if maf < min_maf:
                    continue

            columns.append(geno.astype(np.int8, copy=False))
            map_rows.append({
                'SNP': rsid,
                'CHROM': str(chrom),
                'POS': int(pos),
                'REF': a_ref,
                'ALT': a_alt,
            })

    if not columns:
        geno_mat = np.zeros((len(individual_ids or []), 0), dtype=np.int8)
    else:
        geno_mat = np.column_stack(columns).astype(np.int8, copy=False)
    impute_major_allele_inplace(geno_mat, missing_value=MISSING)

    # Build geno_map
    if return_pandas:
        try:
            import pandas as pd  # type: ignore
            geno_map = pd.DataFrame(map_rows, columns=['SNP','CHROM','POS','REF','ALT'])
        except Exception:
            geno_map = map_rows
    else:
        geno_map = map_rows
    if hasattr(geno_map, "attrs"):
        geno_map.attrs["is_imputed"] = True

    try:
        logger.info("[Cache] Saving binary cache to %s.panicle.v2.*", cache_base)
        np.save(cache_geno, geno_mat)
        with open(cache_ind, 'w') as f:
            for ind in individual_ids:
                f.write(f"{ind}\n")
        import pandas as pd  # type: ignore
        if isinstance(geno_map, list):
            pd.DataFrame(geno_map).to_csv(cache_map, index=False)
        else:
            geno_map.to_csv(cache_map, index=False)
    except Exception as e:
        logger.warning("[Cache] Failed to save cache: %s", e)

    if individual_ids is None:
        raise ValueError('Invalid HapMap: missing header')
    if geno_mat.shape[0] != len(individual_ids):
        raise AssertionError('Row count mismatch')

    return geno_mat, individual_ids, geno_map
