"""On-disk cache for leave-one-group-out Gram (LOCO kinship) objects.

The objects are small (ten 730×730 float64 matrices ≈ 40 MB) and depend only
on the row mask, the kept-column set, and the column-group labels — not on
the response, batch width, or BLAS thread count. A hit skips the convert+GEMM
pass.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .kinship_exact import UncenteredGram
from .kinship_loco import LocoKinship

logger = logging.getLogger(__name__)

CACHE_FORMAT = 2
CACHE_DIR_SUFFIX = ".panicle.v2.loco"


def loco_cache_digest(
    row_indices: np.ndarray,
    keep_indices: Optional[np.ndarray],
    chrom_order: Sequence[str],
    *,
    n_markers: int,
    max_line: int = 5000,
) -> str:
    """Stable hex digest for a LOCO Gram cache entry.

    ``max_line`` is accepted for call-site compatibility but is not hashed:
    the exact 0/1/2 Gram does not depend on batch width.
    """
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(b"loco-v2\0")
    hasher.update(np.int64(n_markers).tobytes())
    rows = np.ascontiguousarray(row_indices, dtype=np.int64)
    hasher.update(np.int64(rows.size).tobytes())
    hasher.update(rows.tobytes())
    if keep_indices is None:
        hasher.update(b"nokeep\0")
    else:
        keep = np.ascontiguousarray(keep_indices, dtype=np.int64)
        hasher.update(np.int64(keep.size).tobytes())
        hasher.update(keep.tobytes())
    hasher.update(b"chroms\0")
    for chrom in chrom_order:
        hasher.update(str(chrom).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def loco_cache_path(cache_base: Union[str, Path], digest: str) -> Path:
    """Return `{cache_base}.panicle.v2.loco/{digest}.npz`."""
    return Path(str(cache_base) + CACHE_DIR_SUFFIX) / f"{digest}.npz"


def save_loco_kinship(path: Union[str, Path], loco: LocoKinship) -> Path:
    """Write a LocoKinship to an npz sidecar (atomic replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chroms = list(loco.chromosomes)
    arrays = {
        "format_version": np.asarray(CACHE_FORMAT, dtype=np.int16),
        "n_individuals": np.asarray(loco.n_individuals, dtype=np.int64),
        "n_chroms": np.asarray(len(chroms), dtype=np.int64),
        "chrom_order": np.asarray(chroms, dtype="U"),
    }
    if loco._exact_chroms:
        arrays["exact"] = np.asarray(1, dtype=np.int16)
        for idx, chrom in enumerate(chroms):
            gram = loco._exact_chroms[chrom]
            arrays[f"chrom_S_{idx}"] = np.ascontiguousarray(gram.S)
            arrays[f"chrom_ss_{idx}"] = np.asarray(gram.ss, dtype=np.int64)
    else:
        arrays["exact"] = np.asarray(0, dtype=np.int16)
        arrays["total_raw"] = np.ascontiguousarray(loco._total_raw)
        arrays["total_diag"] = np.ascontiguousarray(loco._total_diag)
        for idx, chrom in enumerate(chroms):
            arrays[f"chrom_raw_{idx}"] = np.ascontiguousarray(loco._chrom_raw[chrom])
            arrays[f"chrom_diag_{idx}"] = np.ascontiguousarray(loco._chrom_diag[chrom])
    # np.savez appends .npz unless the name already ends with it.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return path


def load_loco_kinship(path: Union[str, Path]) -> Optional[LocoKinship]:
    """Load a LocoKinship from disk, or None if missing/invalid."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if int(np.asarray(archive["format_version"]).item()) != CACHE_FORMAT:
                return None
            chroms = [str(c) for c in archive["chrom_order"]]
            n_ind = int(np.asarray(archive["n_individuals"]).item())
            is_exact = "exact" in archive and int(np.asarray(archive["exact"]).item()) == 1
            if is_exact:
                exact_chroms = {}
                for idx, chrom in enumerate(chroms):
                    gram = UncenteredGram(n_ind)
                    gram.S = np.array(archive[f"chrom_S_{idx}"], copy=True)
                    gram.ss = np.int64(np.asarray(archive[f"chrom_ss_{idx}"]).item())
                    if gram.S.shape != (n_ind, n_ind):
                        return None
                    exact_chroms[chrom] = gram
                return LocoKinship(chrom_order=chroms, exact_chroms=exact_chroms)
            total_raw = np.array(archive["total_raw"], copy=True)
            total_diag = np.array(archive["total_diag"], copy=True)
            if total_raw.shape != (n_ind, n_ind) or total_diag.shape != (n_ind,):
                return None
            chrom_raw = {}
            chrom_diag = {}
            for idx, chrom in enumerate(chroms):
                raw = np.array(archive[f"chrom_raw_{idx}"], copy=True)
                diag = np.array(archive[f"chrom_diag_{idx}"], copy=True)
                if raw.shape != (n_ind, n_ind) or diag.shape != (n_ind,):
                    return None
                chrom_raw[chrom] = raw
                chrom_diag[chrom] = diag
        return LocoKinship(
            total_raw=total_raw,
            total_diag=total_diag,
            chrom_raw=chrom_raw,
            chrom_diag=chrom_diag,
            chrom_order=chroms,
        )
    except (OSError, KeyError, ValueError, TypeError) as exc:
        logger.debug("Ignoring unreadable LOCO cache %s: %s", path, exc)
        return None


def resolve_chrom_order(map_data) -> Iterable[str]:
    """Chromosome labels in map order, empty if unavailable."""
    if map_data is None:
        return []
    if hasattr(map_data, "get_chromosome_order"):
        return list(map_data.get_chromosome_order())
    return []
