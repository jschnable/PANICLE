"""Data pipeline utilities for BAYESLOCO."""

from __future__ import annotations

from typing import Optional, Union, Dict, List
import numpy as np

from ...utils.data_types import GenotypeMatrix, impute_numpy_batch_major_allele
from ...matrix.kinship_loco import _extract_chromosomes, _group_markers_by_chrom
from .config import BayesLocoConfig


class BayesLocoData:
    """Shared data/context for BAYESLOCO fitting and testing."""

    def __init__(
        self,
        *,
        phe: np.ndarray,
        geno: Union[GenotypeMatrix, np.ndarray],
        map_data,
        CV: Optional[np.ndarray],
        cfg: BayesLocoConfig,
    ) -> None:
        self.phe = phe
        self.geno = geno
        self.map_data = map_data
        self.CV = CV
        self.cfg = cfg

        self.n: int
        self.m: int
        if isinstance(geno, GenotypeMatrix):
            self.n = int(geno.n_individuals)
            self.m = int(geno.n_markers)
        elif isinstance(geno, np.ndarray):
            self.n, self.m = map(int, geno.shape)
        else:
            raise ValueError("geno must be GenotypeMatrix or numpy.ndarray")

        self.chrom_values = _extract_chromosomes(map_data, self.m)
        self.chrom_groups = _group_markers_by_chrom(self.chrom_values)
        self.chrom_order: List[str] = list(self.chrom_groups.keys())

        self.X, self.Q, self.r = self._build_fwl()
        self.var_r = float(np.var(self.r, ddof=1)) if self.n > 1 else float(np.var(self.r))

        self.marker_mean = np.zeros(self.m, dtype=np.float64)
        self.marker_scale = np.ones(self.m, dtype=np.float64)
        self.marker_d_std = np.zeros(self.m, dtype=np.float64)
        self.marker_d_unstd = np.zeros(self.m, dtype=np.float64)
        self.marker_missing_rate = np.zeros(self.m, dtype=np.float64)
        self.marker_maf = np.zeros(self.m, dtype=np.float64)
        self.fit_mask = np.zeros(self.m, dtype=bool)
        self.fit_indices: np.ndarray = np.array([], dtype=np.int64)
        self.m_effective: int = 0

        self._precompute_marker_stats()

    def _build_fwl(self):
        y = np.asarray(self.phe[:, 1], dtype=np.float64)
        if self.CV is None:
            X = np.ones((self.n, 1), dtype=np.float64)
        else:
            CV = np.asarray(self.CV, dtype=np.float64)
            if CV.ndim == 1:
                CV = CV.reshape(-1, 1)
            X = np.column_stack([np.ones(self.n, dtype=np.float64), CV])
        Q, _ = np.linalg.qr(X, mode="reduced")
        # np.dot avoids noisy BLAS matmul warnings observed on some platforms.
        r = y - np.dot(Q, np.dot(Q.T, y))
        return X, Q, r

    def _raw_batch(self, start: int, end: int) -> np.ndarray:
        if isinstance(self.geno, GenotypeMatrix):
            return self.geno.get_batch(start, end).astype(np.float64, copy=False)
        return np.asarray(self.geno[:, start:end], dtype=np.float64)

    def _imputed_batch(self, start: int, end: int) -> np.ndarray:
        if isinstance(self.geno, GenotypeMatrix):
            return self.geno.get_batch_imputed(start, end, fill_value=None, dtype=np.float64)
        return impute_numpy_batch_major_allele(
            self.geno[:, start:end],
            fill_value=None,
            dtype=np.float64,
        )

    def _imputed_columns(self, indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            return np.zeros((self.n, 0), dtype=np.float64)
        if isinstance(self.geno, GenotypeMatrix):
            return self.geno.get_columns_imputed(indices, fill_value=None, dtype=np.float64)
        return impute_numpy_batch_major_allele(
            self.geno[:, indices],
            fill_value=None,
            dtype=np.float64,
        )

    def _residualize_block(self, G: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            Z = G - np.dot(self.Q, np.dot(self.Q.T, G))
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    def _precompute_marker_stats(self) -> None:
        cfg = self.cfg
        eps = cfg.eps
        batch = max(1, min(cfg.batch_markers_fit, self.m))

        keep = np.ones(self.m, dtype=bool)
        for start in range(0, self.m, batch):
            end = min(start + batch, self.m)
            raw = self._raw_batch(start, end)
            miss = (~np.isfinite(raw)) | (raw == -9)
            miss_rate = miss.mean(axis=0)

            G = self._imputed_batch(start, end)
            af = np.clip(G.mean(axis=0) / 2.0, 0.0, 1.0)
            maf = np.minimum(af, 1.0 - af)
            Z = self._residualize_block(G)
            mean_z = Z.mean(axis=0)
            std_z = Z.std(axis=0, ddof=0)
            scale = np.where(std_z > eps, std_z, 1.0)
            Zc = Z - mean_z[np.newaxis, :]
            Zstd = Zc / scale[np.newaxis, :]
            d_unstd = np.einsum("ij,ij->j", Z, Z)
            d_std = np.einsum("ij,ij->j", Zstd, Zstd)

            idx = slice(start, end)
            self.marker_missing_rate[idx] = miss_rate
            self.marker_maf[idx] = maf
            self.marker_mean[idx] = mean_z
            self.marker_scale[idx] = scale
            self.marker_d_unstd[idx] = d_unstd
            self.marker_d_std[idx] = d_std

            keep_batch = np.ones(end - start, dtype=bool)
            keep_batch &= (maf >= cfg.maf_min)
            keep_batch &= (miss_rate <= cfg.marker_missing_max)
            if cfg.drop_monomorphic:
                keep_batch &= (d_unstd > eps)
            keep[idx] = keep_batch

        self.fit_mask = keep
        self.fit_indices = np.where(keep)[0].astype(np.int64)
        self.m_effective = int(self.fit_indices.size)
        if self.m_effective == 0:
            raise ValueError("BAYESLOCO marker filtering removed all markers; relax QC thresholds")

    def get_standardized_block(
        self,
        indices: np.ndarray,
        *,
        row_index: Optional[np.ndarray] = None,
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        G = self._imputed_columns(indices)
        Z = self._residualize_block(G)
        Z = (Z - self.marker_mean[indices][np.newaxis, :]) / self.marker_scale[indices][np.newaxis, :]
        if row_index is not None:
            Z = Z[row_index, :]
        return np.asarray(Z, dtype=dtype)

    def get_unstandardized_block(
        self,
        indices: np.ndarray,
        *,
        row_index: Optional[np.ndarray] = None,
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        G = self._imputed_columns(indices)
        Z = self._residualize_block(G)
        if row_index is not None:
            Z = Z[row_index, :]
        return np.asarray(Z, dtype=dtype)

    def compute_d_std_subset(self, row_index: np.ndarray, marker_indices: np.ndarray) -> np.ndarray:
        """Compute ||z_std||^2 in a subset of samples for selected markers."""
        if marker_indices.size == 0:
            return np.array([], dtype=np.float64)
        out = np.zeros(marker_indices.size, dtype=np.float64)
        batch = max(1, min(self.cfg.batch_markers_fit, marker_indices.size))
        for start in range(0, marker_indices.size, batch):
            end = min(start + batch, marker_indices.size)
            idx = marker_indices[start:end]
            Z = self.get_standardized_block(idx, row_index=row_index, dtype=np.float64)
            out[start:end] = np.einsum("ij,ij->j", Z, Z)
        return out

    def split_train_val(self) -> Dict[str, np.ndarray]:
        """Deterministic train/validation split in FWL space."""
        rng = np.random.default_rng(self.cfg.random_seed)
        perm = rng.permutation(self.n)
        n_val = max(1, int(round(self.n * self.cfg.prior_tune_val_fraction)))
        n_val = min(n_val, self.n - 1) if self.n > 1 else 0
        val_idx = np.sort(perm[:n_val]) if n_val > 0 else np.array([], dtype=np.int64)
        train_idx = np.sort(perm[n_val:]) if n_val > 0 else np.arange(self.n, dtype=np.int64)
        return {"train_idx": train_idx.astype(np.int64), "val_idx": val_idx.astype(np.int64)}
