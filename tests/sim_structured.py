"""Simulators with known truth for structure, LD, loci and heritability tests.

LD is produced by thresholding a latent AR(1) Gaussian process along each
chromosome, so r^2 decays with marker distance as in real genomes.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm


@dataclass
class SimData:
    genotypes: np.ndarray          # (n, m) 0/1/2
    chrom: np.ndarray              # (m,) labels
    pos: np.ndarray                # (m,) bp
    Q: Optional[np.ndarray] = None           # true ancestry proportions
    inversion_dosage: Optional[np.ndarray] = None
    inversion_markers: Optional[np.ndarray] = None


def _ar1_latent(rng, n_hap, m, rho):
    u = np.empty((n_hap, m))
    u[:, 0] = rng.standard_normal(n_hap)
    s = np.sqrt(1 - rho ** 2)
    e = rng.standard_normal((n_hap, m))
    for j in range(1, m):
        u[:, j] = rho * u[:, j - 1] + s * e[:, j]
    return u


def simulate_structured(
    *,
    n=600,
    n_pops=3,
    n_chrom=5,
    snps_per_chrom=3000,
    spacing_bp=1000,
    fst=0.05,
    rho=0.9,
    dirichlet_alpha=0.3,
    inversion=None,          # (chrom_idx, start_marker, n_markers)
    seed=1,
) -> SimData:
    rng = np.random.default_rng(seed)
    m = n_chrom * snps_per_chrom
    p_anc = rng.uniform(0.1, 0.9, m)
    a = p_anc * (1 - fst) / fst
    b = (1 - p_anc) * (1 - fst) / fst
    p_pop = np.clip(rng.beta(a, b, size=(n_pops, m)), 0.01, 0.99)
    Q = rng.dirichlet(np.full(n_pops, dirichlet_alpha), size=n)

    G = np.zeros((n, m), dtype=np.int8)
    for c in range(n_chrom):
        cols = slice(c * snps_per_chrom, (c + 1) * snps_per_chrom)
        for h in range(2):
            u = _ar1_latent(rng, n, snps_per_chrom, rho)
            # Each haplotype carries one ancestry per chromosome (drawn from Q).
            anc = np.array([rng.choice(n_pops, p=q) for q in Q])
            thr = norm.ppf(p_pop[anc][:, cols])
            G[:, cols] += (u < thr).astype(np.int8)

    inv_dosage, inv_markers = None, None
    if inversion is not None:
        ci, start, length = inversion
        j0 = ci * snps_per_chrom + start
        inv_markers = np.arange(j0, j0 + length)
        pattern = rng.integers(0, 2, size=length)   # alleles on the inverted arrangement
        # Arrangement frequency is the same in every population: pure LD, no ancestry signal.
        arr = rng.random((n, 2)) < 0.5
        inv_dosage = arr.sum(axis=1)
        block = np.zeros((n, length), dtype=np.int8)
        for h in range(2):
            hap = np.where(arr[:, h:h + 1], pattern[None, :], 1 - pattern[None, :])
            flip = rng.random(hap.shape) < 0.02
            block += np.where(flip, 1 - hap, hap).astype(np.int8)
        G[:, inv_markers] = block

    chrom = np.repeat([f"Chr{c + 1:02d}" for c in range(n_chrom)], snps_per_chrom)
    pos = np.tile(np.arange(1, snps_per_chrom + 1) * spacing_bp, n_chrom)
    return SimData(G, chrom, pos, Q=Q, inversion_dosage=inv_dosage, inversion_markers=inv_markers)


def simulate_unstructured(*, n=2000, n_chrom=10, snps_per_chrom=1000, spacing_bp=1000,
                          rho=0.9, seed=2) -> SimData:
    rng = np.random.default_rng(seed)
    m = n_chrom * snps_per_chrom
    p = rng.uniform(0.05, 0.95, m)
    G = np.zeros((n, m), dtype=np.int8)
    for c in range(n_chrom):
        cols = slice(c * snps_per_chrom, (c + 1) * snps_per_chrom)
        thr = norm.ppf(p[cols])
        for _ in range(2):
            G[:, cols] += (_ar1_latent(rng, n, snps_per_chrom, rho) < thr).astype(np.int8)
    chrom = np.repeat([f"Chr{c + 1:02d}" for c in range(n_chrom)], snps_per_chrom)
    pos = np.tile(np.arange(1, snps_per_chrom + 1) * spacing_bp, n_chrom)
    return SimData(G, chrom, pos)


def standardize(G):
    X = G.astype(float)
    X -= X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return X / sd


def polygenic_phenotype(G, h2, rng, causal=None):
    """y = X beta + e on standardised genotypes, with Var(X beta) = h2."""
    X = standardize(G)
    m = X.shape[1]
    beta = np.zeros(m)
    idx = np.arange(m) if causal is None else np.asarray(causal)
    beta[idx] = rng.standard_normal(idx.size)
    g = X @ beta
    g = (g - g.mean()) / g.std() * np.sqrt(h2)
    e = rng.standard_normal(G.shape[0]) * np.sqrt(1 - h2)
    return g + e


def marginal_pvalues(G, y):
    """Simple-regression p-values (what GLM without covariates reports)."""
    from scipy.stats import t as tdist
    X = standardize(G)
    yc = (y - y.mean()) / y.std()
    n = len(y)
    r = X.T @ yc / n
    r = np.clip(r, -0.999999, 0.999999)
    tstat = r * np.sqrt((n - 2) / (1 - r ** 2))
    return 2 * tdist.sf(np.abs(tstat), n - 2)
