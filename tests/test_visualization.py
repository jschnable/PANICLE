import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panicle.association.farmcpu_resampling import FarmCPUResamplingEntry, FarmCPUResamplingResults
from panicle.utils.data_types import AssociationResults, GenotypeMap
from panicle.visualization import manhattan


def _make_genotype_map(n: int = 3) -> GenotypeMap:
    chroms = [str((i % 2) + 1) for i in range(n)]
    return GenotypeMap(
        pd.DataFrame(
            {
                "SNP": [f"s{i}" for i in range(n)],
                "CHROM": chroms,
                "POS": np.arange(1, n + 1) * 10,
            }
        )
    )


def test_create_manhattan_plot_with_map_and_thresholds() -> None:
    pvalues = np.array([0.05, 0.5, 1e-8])
    geno_map = _make_genotype_map(3)

    fig = manhattan.create_manhattan_plot(
        pvalues,
        map_data=geno_map,
        threshold=5e-8,
        suggestive_threshold=1e-5,
        title="Manhattan",
        point_size=5.0,
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_create_manhattan_plot_sequential_and_map_error(monkeypatch) -> None:
    # Sequential path when no map provided
    pvalues = np.array([0.1, 0.2, 0.3])
    fig_seq = manhattan.create_manhattan_plot(pvalues, map_data=None, threshold=0, title="", point_size=2.0)
    assert isinstance(fig_seq, matplotlib.figure.Figure)
    plt.close(fig_seq)

    # Map error fallback
    class BadMap:
        def to_dataframe(self):
            raise ValueError("boom")

    fig_err = manhattan.create_manhattan_plot(pvalues, map_data=BadMap(), threshold=0.05, suggestive_threshold=0.01, title="BadMap")
    assert isinstance(fig_err, matplotlib.figure.Figure)
    plt.close(fig_err)


def test_panicle_report_multi_panel_and_density(tmp_path) -> None:
    pvals = np.array([0.05, 0.01, 1e-6, 0.2])
    effects = np.array([0.1, -0.2, 0.3, 0.0])
    ses = np.full_like(effects, 0.1)
    res_a = AssociationResults(effects, ses, pvals)
    res_b = AssociationResults(effects * 2, ses * 2, pvals * 0.5)
    geno_map = _make_genotype_map(4)

    report = manhattan.PANICLE_Report(
        {"MethodA": res_a, "MethodB": res_b},
        map_data=geno_map,
        plot_types=["manhattan", "qq", "density"],
        save_plots=False,
        verbose=False,
        multi_panel=True,
        output_prefix=str(tmp_path / "out"),
    )

    assert "multi_panel_manhattan" in report["plots"]
    assert "MethodA" in report["summary"]
    assert report["summary"]["MethodA"]["n_markers"] == 4

    # Close figures to avoid resource warnings
    plt.close(report["plots"]["multi_panel_manhattan"])
    for plots in report["plots"].values():
        if isinstance(plots, dict):
            for fig in plots.values():
                plt.close(fig)


def test_create_manhattan_plot_aligns_chrom_with_padded_pvalues(monkeypatch) -> None:
    """Regression: NaN-padded pvalues (from MAC filter etc.) must not shift
    surviving markers onto the wrong chromosome.

    Previously the plot path called ``map_df['CHROM'].values[:len(pvalues)]``,
    which silently truncated the map when pre-filtered pvalues were passed.
    """
    # Map: 6 markers spread across chr1, chr2, chr3 (2 each).
    map_df = pd.DataFrame(
        {
            "SNP": [f"s{i}" for i in range(6)],
            "CHROM": ["1", "1", "2", "2", "3", "3"],
            "POS": [10, 20, 30, 40, 50, 60],
        }
    )
    geno_map = GenotypeMap(map_df)

    # Strong signal on chr3 marker s5; rest are noise. NaN holes mimic the
    # MAC filter dropping markers s0 and s2 (chr1 and chr2).
    pvalues = np.array([np.nan, 0.5, np.nan, 0.4, 0.3, 1e-9])

    captured: dict = {}

    def fake_plot_with_positions(ax, chromosomes, positions, log_pvalues, **kwargs):
        captured["chromosomes"] = np.asarray(chromosomes).copy()
        captured["positions"] = np.asarray(positions).copy()
        captured["log_pvalues"] = np.asarray(log_pvalues).copy()

    monkeypatch.setattr(
        manhattan, "plot_manhattan_with_positions", fake_plot_with_positions
    )

    fig = manhattan.create_manhattan_plot(pvalues, map_data=geno_map, threshold=5e-8)
    plt.close(fig)

    chroms = captured["chromosomes"]
    positions = captured["positions"]
    log_pvals = captured["log_pvalues"]

    # The NaN holes must be excluded (4 surviving markers).
    assert len(chroms) == 4
    assert list(chroms) == ["1", "2", "3", "3"]
    assert list(positions) == [20, 40, 50, 60]

    # The strongest peak (-log10(1e-9)) must land on chr3, not shifted earlier.
    peak_idx = int(np.argmax(log_pvals))
    assert chroms[peak_idx] == "3"
    assert positions[peak_idx] == 60


def test_create_manhattan_plot_rejects_map_length_mismatch() -> None:
    """Length mismatch between pvalues and map must raise (no silent slicing)."""
    geno_map = _make_genotype_map(5)
    pvalues = np.array([0.5, 0.1, 1e-6])  # 3 pvalues, 5 markers

    import pytest

    with pytest.raises(ValueError, match="lengths must match"):
        manhattan.create_manhattan_plot(pvalues, map_data=geno_map, threshold=5e-8)


def test_create_manhattan_plot_filters_marker_names_with_padded_pvalues(monkeypatch) -> None:
    """True-QTN marker names must follow the same finite-pvalue mask as positions."""
    map_df = pd.DataFrame(
        {
            "SNP": [f"s{i}" for i in range(6)],
            "CHROM": ["1", "1", "2", "2", "3", "3"],
            "POS": [10, 20, 30, 40, 50, 60],
        }
    )
    geno_map = GenotypeMap(map_df)
    pvalues = np.array([np.nan, 0.5, np.nan, 0.4, 0.3, 1e-9])
    captured: dict = {}

    def fake_plot_with_positions(ax, chromosomes, positions, log_pvalues, **kwargs):
        captured["marker_names"] = np.asarray(kwargs["marker_names"]).copy()

    monkeypatch.setattr(
        manhattan, "plot_manhattan_with_positions", fake_plot_with_positions
    )

    fig = manhattan.create_manhattan_plot(
        pvalues,
        map_data=geno_map,
        threshold=5e-8,
        true_qtns=["s5"],
    )
    plt.close(fig)

    assert list(captured["marker_names"]) == ["s1", "s3", "s4", "s5"]


def test_create_rmip_manhattan_plot_with_counts_and_fallback() -> None:
    entries = [
        FarmCPUResamplingEntry(marker_index=0, snp="s0", chrom="1", pos=10, rmip=0.5),
        FarmCPUResamplingEntry(marker_index=2, snp="s2", chrom="2", pos=30, rmip=0.1),
    ]
    result = FarmCPUResamplingResults(
        entries=entries,
        trait_name="Trait",
        total_runs=10,
        cluster_mode=False,
        per_marker_counts={0: 5, 2: 1},
    )
    geno_map = _make_genotype_map(3)

    fig = manhattan.create_rmip_manhattan_plot(result, map_data=geno_map, title="RMIP")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    empty_result = FarmCPUResamplingResults([], "Trait", total_runs=0, cluster_mode=False)
    fig_empty = manhattan.create_rmip_manhattan_plot(empty_result, map_data=None, title="Empty")
    assert isinstance(fig_empty, matplotlib.figure.Figure)
    plt.close(fig_empty)


def test_plot_manhattan_with_positions_true_qtns_and_highlight() -> None:
    pvals = np.array([0.01, 0.02, 0.5, 1e-6])
    log_p = -np.log10(pvals)
    map_df = pd.DataFrame(
        {"SNP": ["s0", "s1", "s2", "s3"], "CHROM": ["1", "1", "2", "2"], "POS": [1, 2, 3, 4]}
    )
    geno_map = GenotypeMap(map_df)
    fig, ax = plt.subplots()
    manhattan.plot_manhattan_with_positions(
        ax,
        chromosomes=map_df["CHROM"].to_numpy(),
        positions=map_df["POS"].to_numpy(),
        log_pvalues=log_p,
        colors=["blue", "green"],
        point_size=4.0,
        map_data=geno_map,
        true_qtns=["s1", "s3"],
        highlight_mask=np.array([True, False, True, True]),
        highlight_kwargs={"edgecolors": "red"},
    )
    plt.close(fig)


def test_create_pvalue_density_and_summary_branches() -> None:
    fig = manhattan.create_pvalue_density_plot(np.array([np.nan, -1.0, 2.0]), title="None")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    fig_valid = manhattan.create_pvalue_density_plot(np.array([0.1, 0.2, 0.3]), title="Valid")
    assert isinstance(fig_valid, matplotlib.figure.Figure)
    plt.close(fig_valid)

    summary_empty = manhattan.calculate_gwas_summary(np.array([np.nan, -1.0]), np.array([0.0, 0.0]))
    assert summary_empty["n_markers"] == 0
    assert np.isnan(summary_empty["min_pvalue"])

    pvalues = np.array([0.01, 0.5, 1e-6])
    effects = np.array([0.2, -0.1, 0.5])
    summary = manhattan.calculate_gwas_summary(pvalues, effects, threshold=1e-4, suggestive_threshold=0.05)
    assert summary["n_significant"] == 1
    assert summary["n_suggestive"] >= 1


def test_create_multi_panel_manhattan_colors_and_true_qtns() -> None:
    pvals = np.array([0.05, 0.01, 0.2])
    effects = np.array([0.1, -0.2, 0.3])
    ses = np.full_like(effects, 0.1)
    res_dict = {"A": AssociationResults(effects, ses, pvals), "B": AssociationResults(effects * 2, ses, pvals)}
    geno_map = _make_genotype_map(3)

    fig = manhattan.create_multi_panel_manhattan(
        results_dict=res_dict,
        map_data=geno_map,
        threshold=0.01,
        true_qtns=["s1"],
        colors=["#1f77b4", "#ff7f0e"],
        point_size=5.0,
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_association_results_plot_helpers() -> None:
    pvals = np.array([0.05, 1e-6, 0.2])
    effects = np.array([0.1, -0.2, 0.3])
    ses = np.array([0.01, 0.02, 0.03])
    geno_map = _make_genotype_map(3)
    results = AssociationResults(effects, ses, pvals, snp_map=geno_map)

    manhattan_fig = results.manhattan_plot()
    qq_fig = results.qq_plot()

    assert isinstance(manhattan_fig, matplotlib.figure.Figure)
    assert isinstance(qq_fig, matplotlib.figure.Figure)
    plt.close(manhattan_fig)
    plt.close(qq_fig)


def test_create_qq_plot_respects_lambda_override() -> None:
    pvalues = np.linspace(1e-6, 1.0, 1000)

    fig = manhattan.create_qq_plot(
        pvalues=pvalues,
        title="Q-Q Plot - Override",
        lambda_gc_override=1.234,
        lambda_is_approx_override=True,
    )

    title = fig.axes[0].get_title()
    assert "λ≈ 1.234" in title
    plt.close(fig)
