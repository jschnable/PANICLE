"""Public API for BAYESLOCO."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np
import pandas as pd

from ...utils.data_types import GenotypeMatrix, AssociationResults, ensure_eager_genotype
from .config import BayesLocoConfig
from .engine import run_bayes_loco


def PANICLE_BayesLOCO(
    phe: np.ndarray,
    geno: Union[GenotypeMatrix, np.ndarray],
    map_data,
    CV: Optional[np.ndarray] = None,
    cpu: int = 1,
    verbose: bool = True,
    bl_config: Optional[Union[BayesLocoConfig, dict]] = None,
) -> AssociationResults:
    """Run BAYESLOCO association testing.

    Args:
        phe: Phenotype array shape (n, 2) with [ID, trait_value].
        geno: Genotype matrix (GenotypeMatrix or numpy array) shape (n, m).
        map_data: Map with CHROM column aligned to markers.
        CV: Optional covariates shape (n, p).
        cpu: Reserved for future parallel backends.
        verbose: Print progress.
        bl_config: Optional BayesLocoConfig or dict override.
    """
    cfg = BayesLocoConfig.from_object(bl_config)
    cfg.validate()

    if map_data is None:
        raise ValueError("map_data is required for BAYESLOCO")

    if not isinstance(phe, np.ndarray) or phe.ndim != 2 or phe.shape[1] != 2:
        raise ValueError("Phenotype must be numpy array with 2 columns [ID, trait_value]")
    y = np.asarray(phe[:, 1], dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise ValueError("Phenotype contains missing/non-finite values; filter before BAYESLOCO")

    # v1 is quantitative only.
    finite_y = y[np.isfinite(y)]
    if finite_y.size > 0:
        uniq = np.unique(finite_y)
        if uniq.size <= 2 and np.all(np.isin(uniq, [0.0, 1.0])):
            raise NotImplementedError("BAYESLOCO binary-trait model is not implemented in v1")

    if CV is not None:
        CV = np.asarray(CV)
        if CV.ndim == 1:
            CV = CV.reshape(-1, 1)
        if CV.ndim != 2:
            raise ValueError("Covariates must be 1D or 2D array-like")
        if CV.shape[0] != phe.shape[0]:
            raise ValueError("Covariates must have same number of rows as phenotype")
        CV = CV.astype(np.float64, copy=False)
        if not np.all(np.isfinite(CV)):
            raise ValueError("Covariates contain missing/non-finite values; filter before BAYESLOCO")

    geno = ensure_eager_genotype(geno)

    if isinstance(geno, GenotypeMatrix):
        n = geno.n_individuals
    elif isinstance(geno, np.ndarray):
        n = geno.shape[0]
    else:
        raise ValueError("Genotype must be GenotypeMatrix or numpy array")
    if n != phe.shape[0]:
        raise ValueError("Phenotype and genotype row counts must match for BAYESLOCO")

    if isinstance(map_data, pd.DataFrame) and "CHROM" not in map_data.columns:
        raise ValueError("map_data DataFrame must contain 'CHROM'")

    return run_bayes_loco(
        phe=phe,
        geno=geno,
        map_data=map_data,
        CV=CV,
        cpu=cpu,
        verbose=verbose,
        cfg=cfg,
    )
