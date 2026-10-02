# Changelog

All notable changes to PANICLE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **LD-aware population structure for panels without an LD reference.**
  `compute_population_structure(ld_prune_pca=True, ...)` / `--pca-ld-prune`
  computes PCs on markers pruned with in-sample r² (`panicle.matrix.ld.ld_prune`),
  so long LD blocks such as inversions or introgressions no longer become top PCs.
  `admixture_k=K` / `--admixture-k K` fits the ADMIXTURE/FRAPPE likelihood
  (`panicle.matrix.admixture.fit_admixture`, SQUAREM-accelerated EM) on the
  pruned markers and adds K-1 ancestry proportions as covariates (Q + K model).
- **Locus identification**: significant markers are clumped into independent
  loci with in-sample r² (`panicle.postgwas.identify_loci`), written to
  `GWAS_<trait>_<method>_loci.csv`; `N_Loci` is added to the summary. On by
  default; `--no-loci` disables it. The clumping window defaults to the panel's
  in-sample LD-decay distance (`--clump-kb auto`; `panicle.matrix.ld.ld_decay`,
  written to `LD_decay.csv`), so long LD blocks in selfing crops are not split
  into many loci.
- **lambda_1000** (`Lambda_1000` summary column): lambda_GC rescaled to
  1000 samples, or to 1000 cases + 1000 controls for binary traits.
- **SNP heritability (REML)**: `H2_REML` and `H2_REML_SE` summary columns and
  `GWAS_<trait>_heritability.csv` (Vg, Ve, LRT p). The MLM null model is fitted
  with the global (all-chromosome) VanRaden kinship and the scan's fixed
  covariates; the SE comes from the REML information matrix with the delta
  method. On by default; `--no-h2` disables it. LOCO-MLM runs reuse the LOCO
  object's full kinship.
- See `docs/ld_aware_structure_and_postgwas.md` for methods, validation, and caveats.

### Fixed
- **LOCO MLM LRT refinement no longer stalls on traits with two-bracket markers.** Markers whose profile-ML derivative changes sign on both sides of the null h2 (caused by the clamped near-zero kinship eigenvalue making the derivative negative at h2 = 0.999, combined with an alternative-model h2 below the null h2) were sent to the ~2-3 ms/marker scalar exact solver. Both brackets are now root-solved inside the compiled LRT kernel (`_lrt_grid_solve_numba`) and the best state is kept. P-values are unchanged (< 1e-10 in -log10 p across 242k refined markers); refinement is 17-44x faster for affected traits (e.g. 47 s -> 1 s for 21,880 candidates). The numba-less NumPy path is unchanged.
- Close VCF readers and discard temporary genotype matrices after decoding
  failures, including errors in the final bulk batch.
- Significant-only output now handles NumPy-backed map columns correctly during
  selective row extraction, while retaining lazy decoding for cached columns.
- Global MLM in `PANICLE(...)` now subsets the kinship matrix correctly when
  missing/non-finite phenotype or covariate values exclude samples. Regression
  tests compare per-trait results with direct MLM using the retained samples.

### Changed
- Use bounded buffered writes for direct VCF cache construction, removing the
  mapped-output writer and its selection option. Prioritize shorter large-panel
  loads over the small-file overhead. Cache format and small/uncached RAM paths
  are unchanged. ISA-L gzip decoding (`panicle[vcf-fast]`) remains opt-in;
  standard gzip remains the default.
- Add opt-in VCF stage timing, logical storage counters, normalized memory and
  resource reporting, with verified macOS/ARM64 and Linux/x86-64 production
  baselines. Loader algorithms and tuning defaults remain unchanged.
- Bulk-decode diploid GT from extra FORMAT fields (including GT:DP and reordered
  GT), and resume general decoding in the same input stream instead of restarting
  after unsupported records. Preserve existing GT/DS, QC, and imputation rules.
- Finalize VCF matrices in bounded tiles directly into large v2 caches, avoiding
  a separate full-size heap matrix and cache copy. Fresh large loads return
  writable copy-on-write memmaps; edits do not modify cached data.
- Separate VCF backend decoding from scalar QC/map accumulation and temporary
  matrix storage. Keep vectorized bulk QC outside the record-level path.
- Decode simple-GT batches into marker-major storage, avoiding per-batch
  transposed writes while preserving C-order sample-major returned arrays.
- Extract pipeline threshold policy with explicit precedence, centralize method
  identities/CLI aliases/input contracts, and use named execution/reporting
  inputs while preserving legacy adapters and interface-specific ordering.
- Share trait selection, grouping, MAC preparation, and solver dispatch between
  the one-call API and pipeline, using named preparation/cache/result objects.
- Consolidate VCF/PLINK/HapMap cache handling and extract map serialization and
  result reporting into focused modules; retain interfaces and cache formats.
- Cast small int8 genotype batches directly to C-order float32, avoiding an
  intermediate int8 copy while retaining the parallel converter for large blocks.
- BLINK reuses the input matrix when its MAF filter retains all markers and uses
  pre-imputed genotype metadata for MAF calculation.
- The top-level `PANICLE(...)` API groups GLM traits with identical retained
  samples, sharing sample/MAC preparation and the genotype scan. Per-trait GLM
  runtimes apportion the shared scan time equally; the pipeline already grouped
  compatible traits. Documented the existing GLM prefetch controls.
- Growing the VCF loader's temporary matrix extends and remaps the file without
  copying and rewriting its existing prefix. Mappings are explicitly closed
  before resizing/removal, and failed remaps remain cleanable.

## [0.5.0] - 2026-08-19

### Changed
- **VanRaden / LOCO kinship for complete 0/1/2 dosages is now an exact Gram, and faster.** The uncentered `ZZᵀ` is accumulated from float32 batch GEMMs (exact integers at the default 5k width) into float64; centering is algebraic (`G = ZZᵀ − 1sᵀ − s1ᵀ + (μ·μ)11ᵀ`, with `s` recovered as the row-sum of `ZZᵀ`). VanRaden scaling is a scalar divide. `K` no longer depends on BLAS build, thread count, or batch width. On the maize panel this is ~15 s vs ~20–23 s for the old float32-center-then-GEMM path. This does **not** reproduce those previous matrices; association results that use kinship will change. Missing / non-{0,1,2} dosages keep the older path.
- LOCO on-disk cache format is now v2 (float64 Grams). v1 sidecars are ignored and rebuilt. The digest no longer includes `maxLine`.
- `GWASPipeline.run_analysis` logs the runtime BLAS library, version, and thread count.
- Faster LOCO MLM scan: fused `UsWUs` cross-product, strided GEMM views, reused `U'G` buffer, and `assume_no_missing` on the already-imputed path. Single-trait `PANICLE_MLM_LOCO` now routes through the chromosome-major multi-trait kernel so a trait analysed alone and in a group share the same numbers (float32 rounding only vs the previous single-trait path).

### Added
- Parallel `int8→float32` convert for large C-contiguous blocks (bit-identical to `astype`; used on whole-chromosome scans, not 5k kinship batches).
- Prepare-cache key includes `max_dosage` as well as `min_mac`.

## [0.4.0] - 2026-07-08

### Added
- **Packaged GWAS CLI:** `panicle-gwas` console script and `python -m panicle` after `pip install panicle`. Implementation lives in `panicle.cli.gwas`; `scripts/run_GWAS.py` is a thin compatibility wrapper.
- `--version` on the GWAS CLI.
- `mlm_mode` / `--mlm-mode` (`loco` default, or `global`) selects LOCO vs full-kinship MLM. Global kinship is always computed from genotypes when needed; external kinship inputs are not accepted on the high-level API or CLI.
- PR/push CI workflow (`.github/workflows/ci.yml`) runs pytest across Python 3.9–3.13.

### Fixed
- Genotype binary caches (VCF/PLINK/HapMap/numeric) now fingerprint QC filter settings in a `*.panicle.v2.filters.json` sidecar. Changing `min_maf`, `max_missing`, `drop_monomorphic`, `include_indels`, or `split_multiallelic` rebuilds the cache instead of silently reusing a prior marker set. Legacy caches without a sidecar are rebuilt once.
- `GWASPipeline` FarmCPU defaults no longer force `p_threshold=alpha` (0.05). Unset thresholds now use the library rMVP-style early-stop (`0.01/n_tests`) and uncorrected QTN threshold of 0.01.
- `compute_mac_keep_indices` excludes missing genotype sentinels (`-9`) and non-finite values from allele counts so unimputed matrices are not mis-filtered.
- `PANICLE()` no longer eagerly computes an unused VanRaden kinship matrix when FarmCPU is requested (FarmCPU does not consume kinship).
- GLM residual-df < 50 now uses Student-t p-values (with a one-time console warning) instead of a normal approximation that understates p-values on small cohorts.

## [0.3.5] - 2026-06-22

### Changed
- Faster LRT-based MLM: the per-marker likelihood-ratio refinement now runs through a dedicated Numba kernel (verified to reproduce the prior implementation to ~1e-15). CPU-budget parallelism is now affinity/cgroup-aware via `available_cpu_count()`, and small jobs are guarded against thread oversubscription.
- Optimized builtin VCF parsing path (bulk/numba fast paths) for faster first-load; output encoding is unchanged.
- `load_genotype_vcf(backend='auto')` uses the builtin text parser for `.vcf`/`.vcf.gz` and reserves cyvcf2 for `.bcf` (reverses the 0.3.3 note that `auto` preferred cyvcf2 for VCF text). Outputs are identical across paths.
- Removed the LOCO-kinship per-chromosome threading in favor of the sequential batched path (performance only; results unchanged).
- Tutorial notebooks moved from `docs/` to `examples/`; README opener reworked.

### Fixed
- A supplied `--map` whose marker IDs differ from the genotype's but match in count and order is now treated as a positional override (logged as a warning) instead of being rejected; a length mismatch still raises.
- Near-singular (very low-MAF) designs in the LRT solver now route to the exact fallback via a relative pivot tolerance, instead of returning large finite (garbage) standard errors.
- `PANICLE_MLM_LOCO_MULTI` now expands `cpu=0` via the affinity-aware `available_cpu_count()`, matching the other LOCO paths and avoiding oversubscription on cgroup/SLURM-limited nodes.
- Excluded sample IDs are stringified before logging, so numeric IDs no longer raise a `TypeError` on the exclusion path.
- Hardened marker-map alignment checks and guarded lazy genotype-subset storage access against reading the unsubsetted parent array.

### Internal
- `panicle.utils.effective_tests` now reports the pre-prune block size for `n_snps` consistently across the shortcut and eigendecomposition paths (`Me`/total counts were already correct).

## [0.3.4] - 2026-04-22

### Fixed
- **Manhattan plot chromosome misassignment when MAC filter (or any other path that produces NaN-padded p-values) was active in 0.3.3.** `PANICLE_Report` pre-filtered NaN p-values before calling `create_manhattan_plot`, then `create_manhattan_plot` aligned the surviving p-values against `map_df['CHROM'].values[:len(pvalues)]` — slicing the *first N* markers of the map rather than the markers that actually survived filtering. This silently shifted peaks onto neighboring chromosomes (e.g., a real chr8 peak rendering on chr7) and could drop the last chromosome entirely from the plot. The merged `GWAS_<trait>_all_results.csv` was unaffected (its assignment used the full-length padded results against the full map). Per-method PNG Manhattan plots are now correctly aligned. `create_manhattan_plot` and `create_multi_panel_manhattan` now require `len(map_data) == len(pvalues)` and apply the finite-pvalue mask jointly to both arrays.

## [0.3.3] - 2026-04-21

### Added
- Per-trait minor allele count (MAC) filter applied *after* sample subsetting, guarding against spurious p-values driven by singleton/very-rare variants when missing phenotypes or covariates reduce the cohort. Exposed as `min_mac=` on `GWASPipeline.run_analysis()` and `PANICLE()` (default 10 — twice the common PLINK `--mac 5` convention, appropriate for inbred cohorts where effective sample size per allele is roughly half) and as `--min-mac` on the CLI. Set to 0 to disable.
- `GenotypeMatrix.subset_markers()` and `GenotypeMap.subset_markers()` helpers.
- Shared `compute_mac_keep_indices()` and `pad_association_results()` utilities in `panicle.utils.stats`.
- Per-trait Bonferroni denominator now uses the post-MAC marker count so the significance threshold reflects the number of tests actually performed.
- `threads=` parameter on `load_genotype_vcf()` and matching `--threads` CLI flag for tuning cyvcf2/htslib decompression workers (0 = all detected CPUs, default = min(4, cpu_count)).

### Changed
- Default behavior: `min_mac=10` is now applied by default in the high-level GWAS APIs. Pass `min_mac=0` to restore pre-0.3.3 behavior.
- VCF first-load ingestion now writes the dynamic int8 matrix in marker-major order so each appended marker is a contiguous write, then transposes once on finalize. Speeds up VCF first-load without affecting the cached output layout.
- `load_genotype_vcf(backend='auto')` now prefers cyvcf2 when installed (previously defaulted to the builtin text parser for VCF/VCF.GZ).

## [0.3.2] - 2026-04-14

### Added
- High-level `PANICLE()` support for internal PCA via `n_pcs`, with computed PCs appended after any external covariates.
- Regression coverage for `PANICLE()` MLM runs with NA-padded phenotypes and trait-specific LOCO sample subsetting.

### Fixed
- Guarded LOCO MLM against reusing kinship matrices built on the wrong sample subset after phenotype filtering.
- Updated user-facing documentation so the high-level API consistently documents internal PCA support.

## [0.3.1] - 2026-04-13

### Added
- Optional CLI and pipeline support to export GWAS standard errors.
- Grouped multi-trait GLM execution paths with pipeline auto-dispatch.
- eQTL multi-trait acceleration tutorial documentation.
- Test coverage for effective tests, stats utilities, visualization, CLI utilities, and expanded GWAS pipeline paths.

### Changed
- Optimized LOCO MLM multi-trait execution and genotype alignment/subsetting paths.
- Optimized effective marker number calculations and added CPU control wiring.
- Improved PCA and kinship-related data flow for large analyses.

### Fixed
- Corrected phenotype parsing and BLINK option forwarding behavior.
- Aligned reported genomic inflation lambda with the QQ plot lambda computation.

## [0.1.0] - 2026-01-25

### Changed
- **Package rebranded from pyMVP to PANICLE** (Python Algorithms for Nucleotide-phenotype Inference and Chromosome-wide Locus Evaluation)
- All `MVP_*` functions renamed to `PANICLE_*` (e.g., `MVP_GLM` → `PANICLE_GLM`)
- Package name changed from `pymvp` to `panicle` in imports
- Cache file extensions changed from `.pymvp.*` to `.panicle.*`
- CLI command renamed from `pymvp-cache-genotype` to `panicle-cache-genotype`

### Added
- Initial public release of PANICLE
- Core GWAS methods: GLM, MLM, FarmCPU, BLINK
- **Hybrid MLM method** combining Wald test screening with LRT refinement
  - 2-3% runtime overhead vs standard MLM
  - Orders of magnitude p-value improvement for significant associations
- High-level `GWASPipeline` API for streamlined workflows
- Multiple genotype format support:
  - VCF/BCF with automatic binary caching (~26x faster loading)
  - PLINK binary format (.bed/.bim/.fam)
  - HapMap format
  - CSV/TSV matrices
- Automatic population structure correction:
  - Step-wise PCA calculation
  - VanRaden kinship matrix computation
- Effective tests calculation for accurate Bonferroni correction
- Parallel execution of multiple GWAS methods
- Comprehensive visualization:
  - Manhattan plots with decimated rendering
  - QQ plots with genomic inflation factor
  - Results comparison plots
- Command-line interface via `scripts/run_GWAS.py`
- Binary genotype caching tool: `panicle-cache-genotype`

### Documentation
- Complete API reference for all classes and functions
- Quick start guide with 6 common scenarios
- Output file format specifications
- 5 runnable example scripts demonstrating different workflows
- Interactive Jupyter notebook for Hybrid MLM demonstration
- PDF report generator for publication-ready results
- Detailed algorithm documentation for Hybrid MLM method

### Performance
- Vectorized VCF loading with cyvcf2 optimization
- Binary caching for instant subsequent loads (~1.5s)
- Numba JIT acceleration for computationally intensive operations
- 2-4x faster than R-based rMVP implementation

### Dependencies
- Core: numpy, scipy, pandas, h5py, tables, statsmodels, scikit-learn, matplotlib, seaborn, tqdm, numba
- Optional: cyvcf2 (VCF support), bed-reader (PLINK support)

[0.5.0]: https://github.com/jschnable/PANICLE/releases/tag/v0.5.0
[0.4.0]: https://github.com/jschnable/PANICLE/releases/tag/v0.4.0
[0.3.5]: https://github.com/jschnable/PANICLE/releases/tag/v0.3.5
[0.3.4]: https://github.com/jschnable/PANICLE/releases/tag/v0.3.4
[0.3.3]: https://github.com/jschnable/PANICLE/releases/tag/v0.3.3
[0.3.2]: https://github.com/jschnable/PANICLE/releases/tag/v0.3.2
[0.3.1]: https://github.com/jschnable/PANICLE/releases/tag/v0.3.1
[0.1.0]: https://github.com/jschnable/PANICLE/releases/tag/v0.1.0
