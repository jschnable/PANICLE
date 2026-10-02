# LD-aware population structure, locus identification, lambda_1000 and REML h2

Many crop and wild plant panels have long linkage disequilibrium (selfing,
inversions, introgressions, selective sweeps) and no external LD reference
panel. This page describes options that estimate everything they need from
the analysed genotypes themselves ("in-sample" LD).

## Why prune before PCA?

PCA on all markers weights every marker equally, so a region with many
markers in near-perfect LD acts like one locus counted hundreds of times. A
segregating inversion or a large introgression can then become PC1, even
though it says nothing about genome-wide ancestry. Using that PC as a
covariate does two harmful things:

1. ancestry is captured less well, so stratification control is weaker;
2. every marker in or linked to the block is regressed against itself,
   removing real associations there (over-correction).

In the bundled simulation (3 ancestral populations, Fst = 0.05, plus a
1,200-marker inversion whose frequency is the same in every population):

| PCs computed on | \|corr(PC1, inversion)\| | ancestry R² explained by PC1–2 |
|---|---|---|
| all markers | 1.00 | 0.42 |
| LD-pruned markers (r² ≤ 0.2, 500 kb) | 0.05 | 0.92 |

## Options

### Python

```python
pipeline.compute_population_structure(
    n_pcs=3,
    calculate_kinship=False,
    ld_prune_pca=True,        # PCs from LD-pruned markers
    prune_r2=0.2,             # drop the lower-MAF marker of any pair with r2 > 0.2 ...
    prune_window_kb=500,      # ... closer than 500 kb ...
    prune_max_window_snps=500,  # ... and at most 500 markers apart
    prune_min_maf=0.01,
    admixture_k=3,            # optional: ADMIXTURE-model Q (K-1 columns) as covariates
)

pipeline.run_analysis(
    methods=["GLM", "MLM"],
    identify_loci=True,       # default
    heritability=True,        # default: REML h2 + SE, null model with global kinship
    post_gwas_params={"clump_kb": "auto", "clump_r2": 0.1},  # "auto" = LD-decay window
)
```

### CLI

```
panicle-gwas ... --pca-ld-prune --prune-r2 0.2 --prune-window-kb 500 \
                 --admixture-k 3 \
                 --clump-kb auto --clump-r2 0.1 [--clump-p2 1e-4] [--merge-gap-kb 0] [--no-loci] \
                 [--no-h2]
```

Pick the pruning window from your species' LD decay: for long-LD selfers,
increase `--prune-window-kb` until r² has decayed to background.

## Methods

**LD pruning** (`panicle.matrix.ld.ld_prune`). Markers below `prune_min_maf`
are removed first. Within each chromosome, in position order, any pair inside
the window with r² above the threshold loses its lower-MAF member. This is the
step-1 limit of PLINK `--indep-pairwise`. r² is computed in blocks from
mean-imputed, standardised genotypes, so memory does not grow with marker
count.

**Admixture** (`panicle.matrix.admixture.fit_admixture`). The binomial
admixture likelihood used by ADMIXTURE/FRAPPE, `p_ij = Σ_k q_ik f_kj`, fitted
by EM with SQUAREM acceleration and random restarts. Missing calls are masked
out. It runs on the pruned markers, as the model assumes linkage equilibrium.
K−1 Q columns are used as covariates, because Q rows sum to 1 and all K
columns would be collinear with the intercept. All K columns are written to
`population_structure_covariates.csv` (`Q*_full`). Choose K from prior
knowledge or by comparing fits; there is no automatic K selection.

**Locus identification** (`panicle.postgwas.identify_loci`). PLINK
`--clump`-style. Markers at or below the method's significance threshold are
taken as index markers in order of p-value. Each index marker claims the
unclaimed markers within `clump_kb` that have `p ≤ clump_p2` and in-sample
`r² ≥ clump_r2`. A locus spans its claimed markers. Loci whose spans overlap
(or lie within `merge_gap_kb`) are merged, keeping the strongest lead. A
merged locus is a *region*; it can contain more than one independent signal.

**Clumping window = LD decay (`--clump-kb auto`, default).** PLINK's 250 kb
default suits outbred species. In selfing crops LD extends over megabases, and
a fixed 250 kb window splits one LD block into many "loci". In the sorghum
example data, it split the Chr09 height block into 17 loci whose lead SNPs
are in LD with each other (r² 0.35–0.88). By default the window is
the distance beyond which the 90th percentile of in-sample r² stays below
`clump_r2` (`panicle.matrix.ld.ld_decay`). Pairs are binned on a log distance
scale up to 10 Mb, with up to 4,000 markers (MAF ≥ 0.05) sampled per
chromosome. The curve is written to `LD_decay.csv`, the standard LD-decay
figure for plant GWAS papers. Pass a number to fix the window.

**lambda_1000** (`panicle.postgwas.lambda_1000`). Under polygenicity,
λ_GC − 1 grows linearly with N, so raw λ is not comparable between panels.
For a quantitative trait, λ1000 = 1 + (λ − 1)·1000/N. For a 0/1 or 1/2 trait
it is scaled to 1000 cases + 1000 controls:
1 + (λ − 1)(1/n_cases + 1/n_controls)·500.

**SNP heritability (REML)** (`panicle.postgwas.reml_heritability`). This is
the MLM null model y = Xb + g + e, with g ~ N(0, Vg·K) and e ~ N(0, Ve·I),
fitted by REML. h² = Vg / (Vg + Ve). Details:

* **K is the global VanRaden kinship over all chromosomes** for the trait's
  retained samples, not the LOCO matrices, which each leave a chromosome out.
  LOCO-MLM runs reuse `LocoKinship.get_full()`, so no extra kinship pass is
  needed. Global-MLM runs reuse their K.
* **K is Gower-scaled** (tr(K)/n − mean(K) = 1), so Vg is the genetic
  variance among the analysed lines, for outbred and inbred panels alike.
  PANICLE's VanRaden K already satisfies this.
* **X holds the same fixed effects as the scan:** intercept, external
  covariates, PCs, and admixture Q. Collinear columns are dropped. h² is
  therefore heritability *conditional on* those covariates. Genetic variance
  aligned with ancestry is absorbed by the PCs/Q, so h² is typically lower
  than in a model without them.
* **The SE comes from the inverse REML Fisher information** of (Vg, Ve),
  tr(P·Vi·P·Vj)/2, mapped to h² by the delta method, as in GCTA. It is
  evaluated in the eigenbasis of K in O(n·p²).
* An LRT of h² > 0 against the h² = 0 boundary (50:50 χ²₀/χ²₁ mixture) is
  written to the per-trait file. Near the 0/1 boundaries the
  information-based SE is approximate (`at_boundary` flags this).
* For 0/1 traits the estimate is on the observed scale.

## Real data: sorghum example (`examples/`, SbDiv, 738 lines, 6,533 SNPs)

The example set is curated: it contains every Bonferroni-significant marker
from an MLM run on PlantHeight plus 5,000 random markers. So λ over all
markers is not interpretable. Chr01–05, 08 and 10 carry only random
markers, so λ on those chromosomes measures residual stratification.

| | plain PCs (3) | LD-pruned PCs (3) |
|---|---|---|
| Chr09 share of PC1 loadings | 69% | 19% |
| \|corr(PC1, Chr09 lead SNP)\| / corr(PC1, PlantHeight) | 0.72 / 0.33 | 0.10 / 0.01 |
| Background λ, PlantHeight GLM / MLM | 4.30 / 1.04 | 2.32 / 1.05 |
| Background λ, DaysToFlower GLM / MLM | 5.55 / 1.41 | 2.60 / 1.42 |
| PlantHeight MLM hits on Chr09 / Chr07 | 70 / 41 | 1,173 / 289 |
| Lead p, Chr09 height block | 2e-13 | 2e-55 |

Plain PC1 is essentially the Chr09 height haplotype block, so using it as a
covariate removes most of the height signal there. Pruned PCs keep that
signal, keep MLM background inflation unchanged, and halve GLM background
inflation. LD decay gives a 2.4 Mb clumping window: PlantHeight MLM gives 7
loci (35 with 250 kb), and DaysToFlower gives 3 (8 with 250 kb). REML h²
(global kinship, intercept + 3 PCs) is 0.75 ± 0.04 for both traits. This is
inflated upward by the hit-enriched marker set, so don't read it as a
population estimate.

## Validation (simulations in `tests/`)

| check | result |
|---|---|
| Pruned set: any within-window pair above threshold | none |
| Pruned PCA under an ancestry-independent inversion | PC1 tracks ancestry, not the inversion (table above) |
| Admixture Q vs true Q | mean matched r ≈ 0.96 (≥ 0.85 with 5% missing calls) |
| Locus clumping, 3 QTLs | each QTL in exactly one locus, lead ≤ 20 kb away |
| LD decay vs brute-force all-pairs r² by distance bin | identical pair counts; mean r² to 1e-4 |
| Auto window, one QTL in a long-LD chromosome | fixed 250 kb splits it into several loci; auto window gives one locus containing the QTL |
| λ1000 | removes the linear N dependence of λ − 1 |
| REML h² vs brute-force REML (dense V, Nelder–Mead) | identical (to 1e-4) |
| REML SE vs observed-information (numerical Hessian) SE | agree (0.102 vs 0.101) |
| REML h², true 0.4, N = 500, 30 replicates | outbred 0.385 ± 0.016; inbred (homozygous) lines 0.393 ± 0.009. Mean SE 0.085 vs empirical SD 0.089 (outbred); conservative for inbred lines (0.087 vs 0.048) |
| REML h², null trait with collinear covariates | small ĥ², LRT p > 0.01, duplicates dropped |
| LOCO `get_full()` vs global VanRaden K | identical |

## Output files

| file | content |
|---|---|
| `structure_pruned_markers.csv` | markers retained by LD pruning (SNP, CHROM, POS) |
| `population_structure_covariates.csv` | ID, PCs, Q covariates used, all K Q columns (`Q*_full`) |
| `LD_decay.csv` | r² by distance bin (mean, 90th percentile, number of pairs); written when `--clump-kb auto` |
| `GWAS_<trait>_<method>_loci.csv` | Locus, CHROM, Start, End, Lead_SNP, Lead_POS, Lead_P, Lead_Effect, N_Significant, N_Clumped, Clumped_SNPs |
| `GWAS_<trait>_heritability.csv` | h2, h2_se, vg, ve, lrt_p, n, n_fixed, at_boundary |
| `GWAS_summary_by_traits_methods.csv` | new columns: `H2_REML`, `H2_REML_SE` (one value per trait, repeated per method), `Lambda_1000`, `N_Loci` |
