"""Result store for many-trait LOCO MLM scans (eQTL scale).

`PANICLE_MLM_LOCO_MULTI` produces three (traits x markers) float64 arrays when
it keeps everything in memory. At 20k genes x 500k markers that is ~240 GB, so
for large trait sets results are streamed here block by block as they are
computed:

- ``keep="hits"``: only cells with p <= p_threshold are kept, as a flat table
  (trait, marker, effect, SE, p). Written incrementally to ``hits.csv`` when an
  output directory is given, otherwise held in memory (hits are small).
- ``keep="all"``: every cell is written to on-disk ``.npy`` memmaps
  (``effects.npy``, ``se.npy``, ``pvalues.npy``; shape traits x markers) so a
  trait's full scan can be reloaded later, e.g. for SuSiE fine mapping. The hit
  table is written as well.

Both modes also record each trait's strongest marker in ``trait_summary.csv``.
A finished store can be reopened with :meth:`MultiTraitResultStore.open`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd

from ..utils.data_types import (
    AssociationResults,
    GenotypeMap,
    CHROM_COLUMN,
    LEGACY_MARKER_ID_COLUMN,
    MARKER_ID_COLUMN,
    POS_COLUMN,
)

HITS_FILE = "hits.csv"
SUMMARY_FILE = "trait_summary.csv"
META_FILE = "metadata.json"
MARKERS_FILE = "markers.csv"
DENSE_FILES = {"effects": "effects.npy", "se": "se.npy", "pvalues": "pvalues.npy"}
HIT_COLUMNS = ["Trait", "MARKER", "CHROM", "POS", "Effect", "SE", "P_value"]


def marker_table(map_data, n_markers: int) -> pd.DataFrame:
    """MARKER/CHROM/POS table for ``n_markers`` markers, from whatever map form is given.

    Falls back to the marker index as MARKER when the map has no identifiers.
    """
    marker = chrom = pos = None
    if isinstance(map_data, GenotypeMap):
        marker = map_data.marker_ids.to_numpy()
        chrom = map_data.chromosomes.to_numpy()
        pos = map_data.positions.to_numpy()
    else:
        df = None
        if isinstance(map_data, pd.DataFrame):
            df = map_data
        elif hasattr(map_data, "to_dataframe"):
            df = map_data.to_dataframe()
        if df is not None:
            for col in (MARKER_ID_COLUMN, LEGACY_MARKER_ID_COLUMN):
                if col in df.columns:
                    marker = df[col].to_numpy()
                    break
            if CHROM_COLUMN in df.columns:
                chrom = df[CHROM_COLUMN].to_numpy()
            if POS_COLUMN in df.columns:
                pos = df[POS_COLUMN].to_numpy()
        elif map_data is not None:
            # Bare sequence of chromosome labels.
            arr = np.asarray(map_data)
            if arr.ndim == 1 and arr.size == n_markers:
                chrom = arr

    def _fit(values, default):
        if values is None or len(values) != n_markers:
            return default
        return values

    idx = np.arange(n_markers)
    return pd.DataFrame(
        {
            MARKER_ID_COLUMN: _fit(marker, idx),
            CHROM_COLUMN: _fit(chrom, np.full(n_markers, "", dtype=object)),
            POS_COLUMN: _fit(pos, idx),
        }
    )


class MultiTraitResultStore:
    """Streaming sink and reader for multi-trait association results.

    Behaves like a read-only mapping of trait name -> :class:`AssociationResults`
    when every cell was kept (``keep="all"``); in ``keep="hits"`` mode use
    :attr:`hits`, :meth:`trait_hits` and :attr:`trait_summary`.
    """

    def __init__(
        self,
        trait_names: List[str],
        n_markers: int,
        *,
        keep: str = "hits",
        p_threshold: float,
        output_dir: Optional[Union[str, Path]] = None,
        markers: Optional[pd.DataFrame] = None,
        dense_dtype=np.float32,
    ):
        keep = str(keep).lower()
        if keep not in {"hits", "all"}:
            raise ValueError("keep must be 'hits' or 'all'")
        if keep == "all" and output_dir is None:
            raise ValueError("keep='all' streaming requires output_dir")
        self.trait_names = [str(t) for t in trait_names]
        self.n_markers = int(n_markers)
        self.keep = keep
        self.p_threshold = float(p_threshold)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.dense_dtype = np.dtype(dense_dtype)
        self.markers = (
            markers
            if markers is not None
            else marker_table(None, self.n_markers)
        )
        self.complete = False
        self._trait_index = {name: i for i, name in enumerate(self.trait_names)}

        T = len(self.trait_names)
        self._min_p = np.ones(T, dtype=np.float64)
        self._min_idx = np.zeros(T, dtype=np.int64)
        self._min_eff = np.zeros(T, dtype=np.float64)
        self._min_se = np.full(T, np.nan, dtype=np.float64)
        self._n_hits = np.zeros(T, dtype=np.int64)
        self._hit_parts: List[pd.DataFrame] = []
        self._hits: Optional[pd.DataFrame] = None
        self._hits_handle = None
        self._summary_cache: Optional[pd.DataFrame] = None
        self._dense: Dict[str, np.memmap] = {}

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._hits_handle = open(self.output_dir / HITS_FILE, "w", newline="")
            self._hits_handle.write(",".join(HIT_COLUMNS) + "\n")
            if self.keep == "all":
                for key, fname in DENSE_FILES.items():
                    dtype = np.float64 if key == "pvalues" else self.dense_dtype
                    self._dense[key] = np.lib.format.open_memmap(
                        self.output_dir / fname, mode="w+", dtype=dtype, shape=(T, self.n_markers)
                    )
            self._write_metadata()

    # ------------------------------------------------------------------ writing
    def write_block(
        self,
        trait_start: int,
        marker_indices: np.ndarray,
        effects: np.ndarray,
        se: np.ndarray,
        pvalues: np.ndarray,
    ) -> int:
        """Record a (traits x markers) block. Returns the number of hits recorded.

        ``marker_indices`` are the global marker positions of the block columns;
        rows are traits ``trait_start`` onwards.
        """
        if self.complete:
            raise RuntimeError("store already finalized")
        B, mb = pvalues.shape
        trait_end = trait_start + B
        marker_indices = np.asarray(marker_indices, dtype=np.int64)

        if self._dense:
            contiguous = mb > 0 and marker_indices[-1] - marker_indices[0] == mb - 1 and np.all(
                np.diff(marker_indices) == 1
            )
            cols = slice(marker_indices[0], marker_indices[0] + mb) if contiguous else marker_indices
            self._dense["effects"][trait_start:trait_end, cols] = effects
            self._dense["se"][trait_start:trait_end, cols] = se
            self._dense["pvalues"][trait_start:trait_end, cols] = pvalues

        # Per-trait strongest marker.
        if mb > 0:
            safe_p = np.where(np.isfinite(pvalues), pvalues, np.inf)
            local_best = safe_p.argmin(axis=1)
            rows = np.arange(B)
            best_p = safe_p[rows, local_best]
            better = best_p < self._min_p[trait_start:trait_end]
            t_sel = rows[better]
            self._min_p[trait_start + t_sel] = best_p[better]
            self._min_idx[trait_start + t_sel] = marker_indices[local_best[better]]
            self._min_eff[trait_start + t_sel] = effects[t_sel, local_best[better]]
            self._min_se[trait_start + t_sel] = se[t_sel, local_best[better]]

        mask = pvalues <= self.p_threshold
        t_idx, m_idx = np.nonzero(mask)
        n_hits = int(t_idx.size)
        if n_hits:
            np.add.at(self._n_hits, trait_start + t_idx, 1)
            g_idx = marker_indices[m_idx]
            part = pd.DataFrame(
                {
                    "Trait": np.asarray(self.trait_names, dtype=object)[trait_start + t_idx],
                    MARKER_ID_COLUMN: self.markers[MARKER_ID_COLUMN].to_numpy()[g_idx],
                    CHROM_COLUMN: self.markers[CHROM_COLUMN].to_numpy()[g_idx],
                    POS_COLUMN: self.markers[POS_COLUMN].to_numpy()[g_idx],
                    "Effect": np.asarray(effects[t_idx, m_idx], dtype=np.float64),
                    "SE": np.asarray(se[t_idx, m_idx], dtype=np.float64),
                    "P_value": np.asarray(pvalues[t_idx, m_idx], dtype=np.float64),
                }
            )
            self._hit_parts.append(part)
            if self._hits_handle is not None:
                part.to_csv(self._hits_handle, header=False, index=False)
                self._hits_handle.flush()
        return n_hits

    def finalize(self) -> "MultiTraitResultStore":
        """Close files, write the per-trait summary and metadata."""
        if self.complete:
            return self
        if self._hit_parts:
            self._hits = pd.concat(self._hit_parts, ignore_index=True)
        else:
            self._hits = pd.DataFrame({c: pd.Series(dtype=object if c in ("Trait", MARKER_ID_COLUMN, CHROM_COLUMN) else float) for c in HIT_COLUMNS})
        self._hit_parts = []
        if self._hits_handle is not None:
            self._hits_handle.close()
            self._hits_handle = None
        for mm in self._dense.values():
            mm.flush()
        self.complete = True
        if self.output_dir is not None:
            self.trait_summary.to_csv(self.output_dir / SUMMARY_FILE, index=False)
            self.markers.to_csv(self.output_dir / MARKERS_FILE, index=False)
            self._write_metadata()
        return self

    def _write_metadata(self) -> None:
        meta = {
            "format": "panicle-multi-trait-results-v1",
            "keep": self.keep,
            "p_threshold": self.p_threshold,
            "n_traits": len(self.trait_names),
            "n_markers": self.n_markers,
            "dense_dtype": self.dense_dtype.name,
            "complete": self.complete,
            "trait_names": self.trait_names,
            "files": {
                "hits": HITS_FILE,
                "trait_summary": SUMMARY_FILE,
                "markers": MARKERS_FILE,
                **({"dense": DENSE_FILES} if self.keep == "all" else {}),
            },
        }
        with open(self.output_dir / META_FILE, "w") as fh:
            json.dump(meta, fh, indent=2)

    # ------------------------------------------------------------------ reading
    @property
    def hits(self) -> pd.DataFrame:
        """All cells with p <= p_threshold (Trait, MARKER, CHROM, POS, Effect, SE, P_value)."""
        if self._hits is None:
            if self._hit_parts:
                return pd.concat(self._hit_parts, ignore_index=True)
            return pd.DataFrame(columns=HIT_COLUMNS)
        return self._hits

    def trait_hits(self, name: str) -> pd.DataFrame:
        hits = self.hits
        return hits[hits["Trait"] == str(name)].reset_index(drop=True)

    @property
    def trait_summary(self) -> pd.DataFrame:
        """One row per trait: hit count and the strongest marker."""
        if self._summary_cache is not None:
            return self._summary_cache
        idx = self._min_idx
        return pd.DataFrame(
            {
                "Trait": self.trait_names,
                "n_hits": self._n_hits,
                "top_" + MARKER_ID_COLUMN: self.markers[MARKER_ID_COLUMN].to_numpy()[idx],
                "top_" + CHROM_COLUMN: self.markers[CHROM_COLUMN].to_numpy()[idx],
                "top_" + POS_COLUMN: self.markers[POS_COLUMN].to_numpy()[idx],
                "top_P_value": self._min_p,
                "top_Effect": self._min_eff,
                "top_SE": self._min_se,
            }
        )

    @property
    def dense(self) -> bool:
        return bool(self._dense)

    @property
    def effects(self) -> np.ndarray:
        return self._require_dense()["effects"]

    @property
    def se(self) -> np.ndarray:
        return self._require_dense()["se"]

    @property
    def pvalues(self) -> np.ndarray:
        return self._require_dense()["pvalues"]

    def _require_dense(self) -> Dict[str, np.memmap]:
        if not self._dense:
            raise RuntimeError(
                "this store kept only hits (keep='hits'); per-marker arrays are not available. "
                "Use .hits / .trait_hits(name), or rerun with keep='all' and an output_dir."
            )
        return self._dense

    def __getitem__(self, name: str) -> AssociationResults:
        dense = self._require_dense()
        t = self._trait_index[str(name)]
        return AssociationResults(
            effects=dense["effects"][t],
            se=dense["se"][t],
            pvalues=dense["pvalues"][t],
            metadata={"trait": self.trait_names[t], "store": str(self.output_dir)},
        )

    def __contains__(self, name) -> bool:
        return str(name) in self._trait_index

    def __len__(self) -> int:
        return len(self.trait_names)

    def __iter__(self) -> Iterator[str]:
        return iter(self.trait_names)

    def keys(self):
        return list(self.trait_names)

    def items(self):
        return ((name, self[name]) for name in self.trait_names)

    def __repr__(self) -> str:
        where = str(self.output_dir) if self.output_dir is not None else "memory"
        return (
            f"MultiTraitResultStore(keep={self.keep!r}, traits={len(self.trait_names)}, "
            f"markers={self.n_markers}, p_threshold={self.p_threshold:g}, "
            f"hits={int(self._n_hits.sum())}, at={where})"
        )

    @classmethod
    def open(cls, output_dir: Union[str, Path], markers: Optional[pd.DataFrame] = None) -> "MultiTraitResultStore":
        """Reopen a finished store from disk (dense arrays are memory-mapped read-only)."""
        output_dir = Path(output_dir)
        with open(output_dir / META_FILE) as fh:
            meta = json.load(fh)
        if not meta.get("complete", False):
            raise RuntimeError(f"store at {output_dir} was not finalized (run did not complete)")
        store = cls.__new__(cls)
        store.trait_names = list(meta["trait_names"])
        store.n_markers = int(meta["n_markers"])
        store.keep = meta["keep"]
        store.p_threshold = float(meta["p_threshold"])
        store.output_dir = output_dir
        store.dense_dtype = np.dtype(meta.get("dense_dtype", "float32"))
        store.complete = True
        store._trait_index = {name: i for i, name in enumerate(store.trait_names)}
        store._hit_parts = []
        store._hits_handle = None
        store._hits = pd.read_csv(output_dir / HITS_FILE)
        summary = pd.read_csv(output_dir / SUMMARY_FILE)
        store._n_hits = summary["n_hits"].to_numpy(dtype=np.int64)
        store._min_p = summary["top_P_value"].to_numpy(dtype=np.float64)
        store._min_eff = summary["top_Effect"].to_numpy(dtype=np.float64)
        store._min_se = summary["top_SE"].to_numpy(dtype=np.float64)
        store._dense = {}
        if store.keep == "all":
            for key, fname in DENSE_FILES.items():
                store._dense[key] = np.load(output_dir / fname, mmap_mode="r")
        if markers is None and (output_dir / MARKERS_FILE).exists():
            markers = pd.read_csv(output_dir / MARKERS_FILE)
        store.markers = markers if markers is not None else marker_table(None, store.n_markers)
        store._min_idx = np.zeros(len(store.trait_names), dtype=np.int64)
        store._summary_cache = summary
        return store
