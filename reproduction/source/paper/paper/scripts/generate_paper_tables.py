#!/usr/bin/env python
"""Generate LaTeX tables for the state-normalized innovation paper.

The script is intentionally read-only with respect to experiment artifacts. It
summarizes existing JSON evidence into paper-ready tables under
``paper/generated_tables``.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "generated_tables"
GRAPH_TRI_SCOPE = ROOT / "results" / "block_ar" / "768a_astgi_graph_triscope"
CURRENT_PANEL = ROOT / "results" / "baselines_current_panel"

BASELINE_ORDER = [
    "random_walk",
    "bootstrap",
    "historical_sim",
    "pca_var",
    "garch_ccc",
    "filtered_hs",
    "deepvar",
    "timegrad",
    "path_diffusion",
    "cond_wgan",
]

BASELINE_DISPLAY = {
    "random_walk": "Random walk",
    "bootstrap": "Bootstrap",
    "historical simulation": "Historical simulation",
    "historical_sim": "Historical simulation",
    "pca_var": "PCA-VAR",
    "garch_ccc": "GARCH-CCC",
    "filtered_hs": "Filtered historical simulation",
    "deepvar": "DeepVAR",
    "timegrad": "TimeGrad",
    "path_diffusion": "CSDI",
    "cond_wgan": "Cond-WGAN",
}


def load_json(path: str) -> dict[str, Any]:
    with (ROOT / path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_json_path(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def current_panel_rows() -> list[tuple[str, dict[str, Any]]]:
    """Return proposed-model and baseline rows for the current 39-channel panel."""
    summary = load_json("results/baselines_current_panel/summary.json")
    rows: list[tuple[str, dict[str, Any]]] = []
    sni_paths = [
        ("SNI (ours)", CURRENT_PANEL / "sni_joint" / "results_current_panel.json"),
        (
            "SNI + adaptive graph (ours)",
            CURRENT_PANEL / "sni_graph_residual" / "results_current_panel.json",
        ),
    ]
    for display_name, path in sni_paths:
        if path.exists():
            rows.append((display_name, load_json_path(path)))
    for name in BASELINE_ORDER:
        d = summary["baselines"].get(name)
        if d:
            rows.append((BASELINE_DISPLAY[name], d))
    return rows


def tex_escape(value: Any) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return tex_escape(value)
    if math.isnan(x):
        return "--"
    return f"{x:.{digits}f}"


def best_row_indices(
    rows: list[tuple[str, dict[str, Any]]],
    key: str,
    *,
    mode: str,
    target: float | None = None,
) -> set[int]:
    values: list[tuple[int, float]] = []
    for i, (_name, metrics) in enumerate(rows):
        value = metrics.get(key)
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(x):
            continue
        values.append((i, x))
    if not values:
        return set()
    if mode == "min":
        best = min(x for _i, x in values)
        return {
            i for i, x in values if math.isclose(x, best, rel_tol=1e-12, abs_tol=1e-12)
        }
    if mode == "max":
        best = max(x for _i, x in values)
        return {
            i for i, x in values if math.isclose(x, best, rel_tol=1e-12, abs_tol=1e-12)
        }
    if mode == "closest":
        if target is None:
            raise ValueError("closest mode requires target")
        best = min(abs(x - target) for _i, x in values)
        return {
            i
            for i, x in values
            if math.isclose(abs(x - target), best, rel_tol=1e-12, abs_tol=1e-12)
        }
    raise ValueError(f"unknown best-value mode {mode!r}")


def fmt_best(value: Any, digits: int, is_best: bool) -> str:
    text = fmt(value, digits)
    if is_best and text != "--":
        return r"\textbf{" + text + "}"
    return text


def is_operational_failure(key: str, value: Any) -> bool:
    """Flag metrics that would make a scenario deck operationally unusable.

    The thresholds are intentionally coarse. They are not the optimization
    target and they are not meant to crown a column winner. They mark obvious
    failure modes: collapsed mean reversion, broken dependence, excessive
    calibration error, unrealistic jumps, or severe tail/marginal mismatch.
    """
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    if math.isnan(x):
        return False
    if key == "crps":
        return x > 4.5
    if key == "factor_crps":
        return x > 12.0
    if key == "energy":
        return x > 10.0
    if key == "variogram":
        return x > 1.0
    if key == "corr":
        return x < 0.30
    if key == "factor_ks":
        return x > 0.15
    if key == "tail_q99_ratio":
        return x < 0.50 or x > 1.50
    if key == "cov90":
        return x < 0.75 or x > 0.98
    if key == "calerr":
        return x > 0.20
    if key == "level_ks":
        return x > 0.20
    if key == "width_rho":
        return x < 0.20
    if key == "mr_ratio":
        return x < 0.25 or x > 1.75
    if key == "econ_ratio":
        return x < 0.75 or x > 1.30
    if key == "jump_ks":
        return x > 0.50
    if key == "kurtosis":
        return x < 0.50 or x > 1.50
    if key == "acf_corr":
        return x < 0.95
    return False


def fmt_marked(value: Any, digits: int, is_best: bool, is_bad: bool) -> str:
    text = fmt(value, digits)
    if is_best and text != "--":
        text = r"\textbf{" + text + "}"
    if is_bad and text != "--":
        text = r"\textcolor{red}{" + text + "}"
    return text


def pass_count(value: dict[str, Any], key_pass: str, key_total: str) -> str:
    return f"{int(value[key_pass])}/{int(value[key_total])}"


def flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numbers(item))
        return out
    try:
        x = float(value)
    except (TypeError, ValueError):
        return []
    if math.isnan(x):
        return []
    return [x]


def median_abs_deviation_from_half(value: Any) -> float | None:
    vals = flatten_numbers(value)
    if not vals:
        return None
    return statistics.median(abs(x - 0.5) for x in vals)


def panel_metrics(panel: dict[str, Any]) -> dict[str, Any]:
    s = panel["summary"]
    return {
        "finite": s["finite_rate"],
        "delta_ks": s["factor_delta_ks_mean"],
        "ks_pass": f"{s['factor_delta_ks_pass_020']}/{s['n_factors']}",
        "tail_ratio": s["factor_tail_q99_ratio_median"],
        "tail_pass": f"{s['factor_tail_q99_pass_05_20']}/{s['n_factors']}",
        "factor_corr": s["factor_factor_corr"]["upper_corr"],
        "iv_factor_corr": s["iv_factor_corr"].get("matrix_corr"),
        "cond_reduction": s["conditional_panel"]["median_mae_reduction_vs_rolled_pct"],
        "width_rho": s["conditional_panel"]["history_activity_width_spearman"],
    }


def structured_metrics(path: str) -> dict[str, Any]:
    return structured_metrics_from_result(load_json(path))


def structured_metrics_from_result(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "cov90": d["coverage"]["overall"]["0.9"],
        "calerr": d["coverage"]["calibration_error"],
        "level_ks_pass": d["distributional_fidelity"]["ks_level_test"]["n_pass"],
        "level_ks_median": d["distributional_fidelity"]["ks_level_test"]["median_stat"],
        "level_ks_worst": d["distributional_fidelity"]["ks_level_test"]["worst_stat"],
        "median_pass": d["distributional_fidelity"]["median_bias"]["n_pass"],
        "median_abs_dev": median_abs_deviation_from_half(
            d["distributional_fidelity"]["median_bias"].get("above_frac")
        ),
        "risk_history_rho": d["risk_state_allocation"]["history_width_spearman"],
        "risk_future_rho": d["risk_state_allocation"]["future_width_spearman"],
        "risk_width_ratio": d["risk_state_allocation"]["history_low_high_width_ratio"],
        "mr_ratio": d["mean_reversion"]["mr_gt_ratio"],
        "econ_ratio": d.get("iv_ewma_economic_link", {}).get(
            "slope_ratio",
            d["cointegration"].get(
                "gen_gt_ratio_legacy", d["cointegration"].get("gen_gt_ratio")
            ),
        ),
        "acf_corr": d["time_series"]["acf"]["acf_correlation"],
        "kurtosis_ratio": d["time_series"]["kurtosis"]["kurtosis_ratio"],
        "jump_ks": d["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"],
    }


def write(path: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / path).write_text(text, encoding="utf-8")


def make_panel_quality_table() -> None:
    structured_result = load_json_path(GRAPH_TRI_SCOPE / "768a_iv_val_full11_s64.json")
    hetero_result = load_json_path(GRAPH_TRI_SCOPE / "768a_anchor_val_panel_s64.json")
    combined_result = load_json_path(
        CURRENT_PANEL / "sni_graph_residual" / "results_current_panel.json"
    )
    structured = structured_metrics_from_result(structured_result)
    hetero = panel_metrics(hetero_result)
    combined_struct = combined_result["current_panel_iv_metrics"]
    combined_panel = combined_result["joint_panel_quality"]

    rows = [
        ("Structured bounded variables", structured, None),
        ("Heterogeneous non-surface variables", None, hetero),
        (
            "Combined heterogeneous panel",
            {
                "cov90": combined_struct["cov90"],
                "calerr": combined_struct["calibration_error"],
                "level_ks_median": combined_struct["level_ks_median"],
                "median_abs_dev": combined_struct["median_abs_dev"],
                "risk_history_rho": combined_struct["width_rho"],
                "mr_ratio": combined_struct["mr_ratio"],
                "econ_ratio": combined_struct["economic_ratio"],
                "acf_corr": combined_struct["acf_corr"],
                "kurtosis_ratio": combined_struct["kurtosis_ratio"],
                "jump_ks": combined_struct["jump_ks"],
            },
            {
                "delta_ks": combined_panel["factor_delta_ks_mean"],
                "tail_ratio": combined_panel["factor_tail_q99_ratio_median"],
                "factor_corr": combined_panel["factor_factor_corr"]["upper_corr"],
                "iv_factor_corr": combined_panel["iv_factor_corr"]["matrix_corr"],
                "width_rho": combined_struct["width_rho"],
            },
        ),
    ]

    body = []
    for name, sm, fm in rows:
        body.append(
            " & ".join(
                [
                    tex_escape(name),
                    fmt(sm["cov90"]) if sm else "--",
                    fmt(sm["calerr"]) if sm else "--",
                    fmt(sm["level_ks_median"]) if sm else "--",
                    fmt(sm["median_abs_dev"]) if sm else "--",
                    fmt(sm["risk_history_rho"]) if sm else fmt(fm["width_rho"]),
                    fmt(sm["mr_ratio"]) if sm else "--",
                    fmt(sm["econ_ratio"]) if sm else "--",
                    fmt(sm["acf_corr"]) if sm else "--",
                    fmt(sm["kurtosis_ratio"]) if sm else "--",
                    fmt(sm["jump_ks"]) if sm else "--",
                    fmt(fm["delta_ks"]) if fm else "--",
                    fmt(fm["tail_ratio"]) if fm else "--",
                    fmt(fm["factor_corr"]) if fm else "--",
                    fmt(fm["iv_factor_corr"]) if fm else "--",
                ]
            )
            + r" \\"
        )

    text = (
        r"""\begin{table}[t]
\centering
\scriptsize
\caption{Three-panel validation diagnostics for the adaptive-graph \sni{} framework. Rows use the same state-normalized innovation generation mechanism with panel-specific input/output dimensionality. The structured row uses the 25-channel surface panel, the non-surface row uses the 14-factor market panel, and the combined row uses the 39-channel panel with real VIX. A dash means that the metric is not defined for that panel type.}
\label{tab:panel-quality}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccccccccc}
\toprule
Panel & Cov90 & CalErr & Level KS med. & Median dev. & Width $\rho$ & MR ratio & Econ ratio & ACF corr & Kurtosis & Jump KS & Factor KS & Tail q99 ratio & Factor corr & Cross corr \\
\midrule
"""
        + "\n".join(body)
        + r"""
\bottomrule
\end{tabular}
}
\end{table}
"""
    )
    write("table_panel_quality.tex", text)


def make_iv_diagnostic_table() -> None:
    runs = [
        (
            "734a structured panel",
            "results/block_ar/766a_evidence_reset/734a_iv_val_full11_s64.json",
        ),
        (
            "734a combined-panel slice",
            "results/block_ar/766a_evidence_reset/734a_joint_iv_val_full11_s64.json",
        ),
        (
            "755a structured-panel ablation",
            "results/block_ar/766a_evidence_reset/755a_iv_val_full11_s64.json",
        ),
        (
            "755a train-tail diagnostic",
            "results/block_ar/766a_evidence_reset/755a_iv_train_tail_full11_s64.json",
        ),
        (
            "766a combined-panel ablation",
            "results/block_ar/766a_evidence_reset/766a_joint_iv_val_full11_s64.json",
        ),
    ]

    body = []
    for name, path in runs:
        m = structured_metrics(path)
        body.append(
            " & ".join(
                [
                    tex_escape(name),
                    fmt(m["cov90"]),
                    fmt(m["calerr"]),
                    fmt(m["level_ks_median"]),
                    fmt(m["median_abs_dev"]),
                    fmt(m["risk_history_rho"]),
                    fmt(m["mr_ratio"]),
                    fmt(m["econ_ratio"]),
                    fmt(m["acf_corr"]),
                    fmt(m["kurtosis_ratio"]),
                    fmt(m["jump_ks"]),
                ]
            )
            + r" \\"
        )

    text = (
        r"""\begin{table}[t]
\centering
\small
\caption{Structured-surface ablation diagnostics. The short-prefix variant improves some structured-variable metrics, but it is treated as an ablation because it does not dominate the combined heterogeneous panel.}
\label{tab:structured-diagnostics}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccccc}
\toprule
Run & Cov90 & CalErr & Level KS med. & Median dev. & Width $\rho$ & MR ratio & Econ ratio & ACF corr & Kurtosis & Jump KS \\
\midrule
"""
        + "\n".join(body)
        + r"""
\bottomrule
\end{tabular}
}
\end{table}
"""
    )
    write("table_structured_diagnostics.tex", text)


def make_classical_baseline_table(frame: str = "insample") -> None:
    # The cross-model comparison tables are the only tables wired to the
    # held-out frame. The baseline rows are read from the per-frame panel
    # summary; the two SNI rows are read from their dedicated per-frame result
    # files. For the held-out frame the SNI rows MUST come from the dedicated
    # ``baselines_heldout/`` checkpoints, not from the ``sni_joint`` row cached
    # inside ``baselines_heldout_panel/summary.json`` (a known validation-cache
    # artifact). All other tables keep their in-sample / fixed-diagnostic
    # sources, so the conditional-responsiveness flagship stays on the frame
    # where it is valid.
    if frame == "heldout":
        summary = load_json("results/baselines_heldout_panel/summary.json")
        sni_paths = [
            (
                "SNI (ours)",
                ROOT
                / "results"
                / "baselines_heldout"
                / "sni_joint_734a"
                / "results_current_panel.json",
            ),
            (
                "SNI + adaptive graph (ours)",
                ROOT
                / "results"
                / "baselines_heldout"
                / "sni_graph_residual_768a"
                / "results_current_panel.json",
            ),
        ]
    else:
        summary = load_json("results/baselines_current_panel/summary.json")
        sni_paths = [
            (
                "SNI (ours)",
                ROOT
                / "results"
                / "baselines_current_panel"
                / "sni_joint"
                / "results_current_panel.json",
            ),
            (
                "SNI + adaptive graph (ours)",
                ROOT
                / "results"
                / "baselines_current_panel"
                / "sni_graph_residual"
                / "results_current_panel.json",
            ),
        ]
    names = [
        "random_walk",
        "bootstrap",
        "historical_sim",
        "pca_var",
        "garch_ccc",
        "filtered_hs",
        "deepvar",
        "timegrad",
        "path_diffusion",
        "cond_wgan",
    ]
    display = {
        "random_walk": "Random walk",
        "bootstrap": "Bootstrap",
        "historical_sim": "Historical simulation",
        "pca_var": "PCA-VAR",
        "garch_ccc": "GARCH-CCC",
        "filtered_hs": "Filtered historical simulation",
        "deepvar": "DeepVAR",
        "timegrad": "TimeGrad",
        "path_diffusion": "CSDI",
        "cond_wgan": "Cond-WGAN",
    }

    rows: list[tuple[str, dict[str, Any]]] = []
    for display_name, path in sni_paths:
        if path.exists():
            rows.append((display_name, json.loads(path.read_text(encoding="utf-8"))))
    for name in names:
        d = summary["baselines"].get(name)
        if d:
            rows.append((display[name], d))

    metric_rows: list[tuple[str, dict[str, Any]]] = []
    for display_name, d in rows:
        crps = d["panel_b_crps"]
        iv = d["current_panel_iv_metrics"]
        joint = d["joint_panel_quality"]
        metric_rows.append(
            (
                display_name,
                {
                    "crps": crps["overall"],
                    "iv_crps": crps["iv_crps"],
                    "factor_crps": crps["factor_crps"],
                    "energy": d["panel_b_energy_score"]["overall"],
                    "variogram": d["panel_b_variogram_score"]["variogram_score"],
                    "corr": d["panel_b_correlation"]["corr_score"],
                    "factor_ks": d["panel_b_factor_ks"]["mean_statistic"],
                    "tail_q99_ratio": joint["factor_tail_q99_ratio_median"],
                    "cov90": iv["cov90"],
                    "calerr": iv["calibration_error"],
                    "level_ks": iv["level_ks_median"],
                    "median_dev": iv["median_abs_dev"],
                    "width_rho": iv["width_rho"],
                    "mr_ratio": iv["mr_ratio"],
                    "econ_ratio": iv["economic_ratio"],
                    "jump_ks": iv["jump_ks"],
                    "kurtosis": iv["kurtosis_ratio"],
                    "acf_corr": iv["acf_corr"],
                },
            )
        )

    dist_columns = [
        ("CRPS", "crps", 4, "min", None),
        ("IV CRPS", "iv_crps", 4, "min", None),
        ("Factor CRPS", "factor_crps", 4, "min", None),
        ("Energy", "energy", 3, "min", None),
        ("Variogram", "variogram", 4, "min", None),
        ("Corr score", "corr", 3, "max", None),
        ("Factor KS", "factor_ks", 3, "min", None),
        ("Tail q99 ratio", "tail_q99_ratio", 3, "closest", 1.0),
    ]
    risk_columns = [
        ("Cov90", "cov90", 3, "closest", 0.90),
        ("CalErr", "calerr", 3, "min", None),
        ("Level KS med.", "level_ks", 3, "min", None),
        ("Median dev.", "median_dev", 3, "min", None),
        (r"Width $\rho$", "width_rho", 3, "max", None),
        ("MR ratio", "mr_ratio", 3, "closest", 1.0),
        ("Econ ratio", "econ_ratio", 3, "closest", 1.0),
        ("Jump KS", "jump_ks", 3, "min", None),
        ("Kurtosis", "kurtosis", 3, "closest", 1.0),
        ("ACF corr", "acf_corr", 3, "max", None),
    ]

    dist_best = {
        key: best_row_indices(metric_rows, key, mode=mode, target=target)
        for _label, key, _digits, mode, target in dist_columns
    }
    risk_best = {
        key: best_row_indices(metric_rows, key, mode=mode, target=target)
        for _label, key, _digits, mode, target in risk_columns
    }

    dist_body = []
    risk_body = []
    for row_idx, (display_name, metrics) in enumerate(metric_rows):
        dist_body.append(
            " & ".join(
                [tex_escape(display_name)]
                + [
                    fmt_marked(
                        metrics[key],
                        digits,
                        row_idx in dist_best[key],
                        is_operational_failure(key, metrics[key]),
                    )
                    for _label, key, digits, _mode, _target in dist_columns
                ]
            )
            + r" \\"
        )
        risk_body.append(
            " & ".join(
                [tex_escape(display_name)]
                + [
                    fmt_marked(
                        metrics[key],
                        digits,
                        row_idx in risk_best[key],
                        is_operational_failure(key, metrics[key]),
                    )
                    for _label, key, digits, _mode, _target in risk_columns
                ]
            )
            + r" \\"
        )

    dist_text = (
        r"""\begin{table}[t]
\centering
\small
\caption{Distributional comparison on the current real-VIX 39-state panel. Lower is better for CRPS, Energy Score, Variogram Score, and Factor KS; higher is better for correlation score; values closer to one are better for the tail q99 ratio. Bold marks the best value in each column. Red marks an operationally unacceptable failure mode under the paper's scenario-realism thresholds. Both SNI rows are evaluated with the same current-panel metric code as the baseline comparison models.}
\label{tab:baseline-distributional}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
Model & CRPS & IV CRPS & Factor CRPS & Energy & Variogram & Corr score & Factor KS & Tail q99 ratio \\
\midrule
"""
        + "\n".join(dist_body)
        + r"""
\bottomrule
\end{tabular}
}
\end{table}
"""
    )
    risk_text = (
        r"""\begin{table}[t]
\centering
\small
\caption{Risk-facing diagnostics on the current real-VIX 39-state panel. Coverage, calibration, level KS, mean reversion, economic-link ratio, jump realism, kurtosis, and ACF are reported for the structured bounded variables; factor KS and tail ratio are reported in Table~\ref{tab:baseline-distributional}. Bold marks the best value in each column: coverage is judged by closeness to 0.90, ratios by closeness to one, KS/error metrics by lower values, and correlation/width-response metrics by higher values. Red marks an operationally unacceptable failure mode. The intended comparison is not column-wise dominance, but whether a model avoids catastrophic failures across the diagnostic profile.}
\label{tab:baseline-risk}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccccc}
\toprule
Model & Cov90 & CalErr & Level KS med. & Median dev. & Width $\rho$ & MR ratio & Econ ratio & Jump KS & Kurtosis & ACF corr \\
\midrule
"""
        + "\n".join(risk_body)
        + r"""
\bottomrule
\end{tabular}
}
\end{table}
"""
    )
    write("table_baseline_distributional.tex", dist_text)
    write("table_baseline_risk.tex", risk_text)

    # Keep the historical filename as a small compatibility wrapper for older
    # Makefile dependencies and external links.
    write("table_baseline_comparison.tex", dist_text + "\n" + risk_text)


def make_sni_graph_ablation_table() -> None:
    def load_current_panel_result(name: str) -> dict[str, Any]:
        path = (
            ROOT
            / "results"
            / "baselines_current_panel"
            / name
            / "results_current_panel.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def factor_abs_corr_ratio(joint: dict[str, Any]) -> float | None:
        corr = joint["factor_factor_corr"]
        if "abs_corr_ratio" in corr:
            return corr["abs_corr_ratio"]
        gt = corr.get("gt_mean_abs")
        gen = corr.get("gen_mean_abs")
        if gt in (None, 0):
            return None
        return float(gen) / float(gt)

    rows: list[tuple[str, dict[str, Any]]] = []
    for label, artifact in [
        ("SNI flow", "sni_joint"),
        ("SNI + graph residual", "sni_graph_residual"),
    ]:
        d = load_current_panel_result(artifact)
        iv = d["current_panel_iv_metrics"]
        joint = d["joint_panel_quality"]
        rows.append(
            (
                label,
                {
                    "crps": d["panel_b_crps"]["overall"],
                    "corr": d["panel_b_correlation"]["corr_score"],
                    "iv_factor_corr": joint["iv_factor_corr"]["matrix_corr"],
                    "iv_factor_mae": joint["iv_factor_corr"]["mae"],
                    "factor_abs_corr_ratio": factor_abs_corr_ratio(joint),
                    "factor_ks": d["panel_b_factor_ks"]["mean_statistic"],
                    "cov90": iv["cov90"],
                    "level_ks": iv["level_ks_median"],
                    "mr_ratio": iv["mr_ratio"],
                    "jump_ks": iv["jump_ks"],
                },
            )
        )

    columns = [
        ("CRPS", "crps", 4, "min", None),
        ("Corr score", "corr", 3, "max", None),
        ("IV-factor corr", "iv_factor_corr", 3, "max", None),
        ("IV-factor MAE", "iv_factor_mae", 3, "min", None),
        ("Factor abs-corr ratio", "factor_abs_corr_ratio", 3, "closest", 1.0),
        ("Factor KS", "factor_ks", 3, "min", None),
        ("Cov90", "cov90", 3, "closest", 0.90),
        ("Level KS med.", "level_ks", 3, "min", None),
        ("MR ratio", "mr_ratio", 3, "closest", 1.0),
        ("Jump KS", "jump_ks", 3, "min", None),
    ]
    best = {
        key: best_row_indices(rows, key, mode=mode, target=target)
        for _label, key, _digits, mode, target in columns
    }
    body = []
    for row_idx, (name, metrics) in enumerate(rows):
        body.append(
            " & ".join(
                [tex_escape(name)]
                + [
                    fmt_best(metrics[key], digits, row_idx in best[key])
                    for _label, key, digits, _mode, _target in columns
                ]
            )
            + r" \\"
        )

    text = (
        r"""\begin{table}[t]
\centering
\small
\caption{Ablation of the adaptive channel-graph residual mixer inside the \sni{} velocity head on the same combined 39-channel validation panel. The graph residual improves the direct cross-variable dependency diagnostics, especially IV--factor correlation and factor--factor correlation magnitude, while retaining the same shared stochastic generator and state-normalized innovation interface. The marginal and calibration metrics reveal the tradeoff: the graph mixer is a dependency-routing module, not a replacement for distributional calibration.}
\label{tab:sni-graph-ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccccc}
\toprule
Model & CRPS & Corr score & IV-factor corr & IV-factor MAE & Factor abs-corr ratio & Factor KS & Cov90 & Level KS med. & MR ratio & Jump KS \\
\midrule
"""
        + "\n".join(body)
        + r"""
\bottomrule
\end{tabular}
}
\end{table}
"""
    )
    write("table_sni_graph_ablation.tex", text)


def make_conditional_responsiveness_table() -> None:
    summary = load_json("results/baselines_current_panel/summary.json")
    sni_paths = [
        (
            "SNI (ours)",
            ROOT
            / "results"
            / "baselines_current_panel"
            / "sni_joint"
            / "results_current_panel.json",
        ),
        (
            "SNI + adaptive graph (ours)",
            ROOT
            / "results"
            / "baselines_current_panel"
            / "sni_graph_residual"
            / "results_current_panel.json",
        ),
    ]
    names = [
        "random_walk",
        "bootstrap",
        "historical_sim",
        "pca_var",
        "garch_ccc",
        "filtered_hs",
        "deepvar",
        "timegrad",
        "path_diffusion",
        "cond_wgan",
    ]
    display = {
        "random_walk": "Random walk",
        "bootstrap": "Bootstrap",
        "historical_sim": "Historical simulation",
        "pca_var": "PCA-VAR",
        "garch_ccc": "GARCH-CCC",
        "filtered_hs": "Filtered historical simulation",
        "deepvar": "DeepVAR",
        "timegrad": "TimeGrad",
        "path_diffusion": "CSDI",
        "cond_wgan": "Cond-WGAN",
    }

    rows: list[tuple[str, dict[str, Any]]] = []
    for display_name, path in sni_paths:
        if path.exists():
            rows.append((display_name, json.loads(path.read_text(encoding="utf-8"))))
    for name in names:
        d = summary["baselines"].get(name)
        if d:
            rows.append((display[name], d))

    metric_rows: list[tuple[str, dict[str, Any]]] = []
    for display_name, d in rows:
        cond = d["current_panel_iv"]["conditionality"]
        state = d["current_panel_iv"]["risk_state_allocation"]
        per_h = cond["per_horizon_conditionality"]
        h1_width = float(per_h["1"]["cond_width"])
        h30_width = float(per_h["30"]["cond_width"])
        cell_mae = flatten_numbers(cond["per_cell_mae_reduction"])
        panel_cond = d["joint_panel_quality"]["conditional_panel"]
        metric_rows.append(
            (
                display_name,
                {
                    "surface_width_rho": state["history_width_spearman"],
                    "surface_low_high": state["history_low_high_width_ratio"],
                    "surface_bucket_mono": state["history_bucket_width_monotonicity"],
                    "surface_h30_h1": (
                        h30_width / h1_width if h1_width > 1e-12 else None
                    ),
                    "surface_cell_positive": (
                        sum(1 for x in cell_mae if x > 0.0) / len(cell_mae)
                        if cell_mae
                        else None
                    ),
                    "surface_mae_reduction": cond["mae_reduction_pct"],
                    "panel_width_rho": panel_cond["history_activity_width_spearman"],
                },
            )
        )

    columns = [
        (r"Surface width $\rho$", "surface_width_rho", 3, "max", None),
        ("Surface low/high width", "surface_low_high", 3, "max", None),
        ("Surface bucket mono.", "surface_bucket_mono", 2, "max", None),
        ("H30/H1 width", "surface_h30_h1", 2, None, None),
        ("Cell MAE+ \\%", "surface_cell_positive", 0, "max", None),
        ("Surface MAE red. \\%", "surface_mae_reduction", 1, "max", None),
        (r"Panel width $\rho$", "panel_width_rho", 3, "max", None),
    ]
    best: dict[str, set[int]] = {}
    for _label, key, _digits, mode, target in columns:
        if mode is None:
            best[key] = set()
        else:
            best[key] = best_row_indices(metric_rows, key, mode=mode, target=target)

    body = []
    for row_idx, (name, metrics) in enumerate(metric_rows):
        body.append(
            " & ".join(
                [tex_escape(name)]
                + [
                    fmt_best(
                        (
                            100.0 * metrics[key]
                            if key == "surface_cell_positive"
                            else metrics[key]
                        ),
                        digits,
                        row_idx in best[key],
                    )
                    for _label, key, digits, _mode, _target in columns
                ]
            )
            + r" \\"
        )

    text = (
        r"""\begin{table}[t]
\centering
\small
\caption{Conditional-responsiveness diagnostics on the current real-VIX 39-channel validation panel. These metrics isolate whether generated uncertainty changes with the conditioning history, horizon, and channel rather than only whether the unconditional distribution looks plausible. Higher is better for all columns except H30/H1 width, which is descriptive of horizon-dependent dispersion. Surface metrics use the structured 25-channel surface slice; panel width $\rho$ uses the full reconstructed 39-channel panel.}
\label{tab:conditional-responsiveness}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
Model & Surface width $\rho$ & Surface low/high width & Surface bucket mono. & H30/H1 width & Cell MAE+ \% & Surface MAE red. \% & Panel width $\rho$ \\
\midrule
"""
        + "\n".join(body)
        + r"""
\bottomrule
\end{tabular}
}
\end{table}
"""
    )
    write("table_conditional_responsiveness.tex", text)


def make_reconstruction_table() -> None:
    manifest = load_json(
        "results/block_ar/management_report_734a739a_stress_ready/manifest.json"
    )
    structured_channels = 25
    factor_channels = 14
    structured_mean = manifest["iv"]["reconstruction_error"]["mean_abs_error"]
    factor_mean = manifest["anchor"]["reconstruction_error"]["mean_abs_error"]
    combined_mean = (
        structured_channels * structured_mean + factor_channels * factor_mean
    ) / (structured_channels + factor_channels)
    rows = [
        (
            "Structured bounded variables",
            manifest["iv"]["n_windows"],
            manifest["n_samples"],
            manifest["iv"]["reconstruction_error"]["max_abs_error"],
            manifest["iv"]["reconstruction_error"]["mean_abs_error"],
        ),
        (
            "Heterogeneous non-surface variables",
            manifest["anchor"]["n_windows"],
            manifest["n_samples"],
            manifest["anchor"]["reconstruction_error"]["max_abs_error"],
            manifest["anchor"]["reconstruction_error"]["mean_abs_error"],
        ),
        (
            "Combined heterogeneous panel",
            min(manifest["iv"]["n_windows"], manifest["anchor"]["n_windows"]),
            manifest["n_samples"],
            max(
                manifest["iv"]["reconstruction_error"]["max_abs_error"],
                manifest["anchor"]["reconstruction_error"]["max_abs_error"],
            ),
            combined_mean,
        ),
    ]
    body = [
        " & ".join([tex_escape(name), str(nw), str(ns), f"{maxe:.2e}", f"{meane:.2e}"])
        + r" \\"
        for name, nw, ns, maxe, meane in rows
    ]
    text = (
        r"""\begin{table}[t]
\centering
\small
\caption{Oracle reconstruction audit for the management-report data path. The near-zero errors confirm that generated normalized innovations are integrated back to raw levels using the intended inverse map. The combined row is the channel-concatenated structured plus non-surface audit.}
\label{tab:reconstruction-audit}
\begin{tabular}{lrrrr}
\toprule
Panel & Windows & Samples & Max abs. error & Mean abs. error \\
\midrule
"""
        + "\n".join(body)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    write("table_reconstruction_audit.tex", text)


def make_operational_red_thresholds_table() -> None:
    rows = [
        (
            "Cov90",
            "$<0.75$ or $>0.98$",
            "Severe undercoverage misses realized paths; severe overcoverage is too wide to be a calibrated law.",
        ),
        (
            "CalErr",
            "$>0.20$",
            "Interval calibration error is too large for probability interpretation.",
        ),
        (
            "Level KS med.",
            "$>0.20$",
            "Structured-surface level distribution is materially mismatched.",
        ),
        (
            "Width $\\rho$",
            "$<0.20$",
            "Generated uncertainty barely responds to observed state activity.",
        ),
        (
            "MR ratio",
            "$<0.25$ or $>1.75$",
            "Mean reversion is materially under- or over-stated relative to realized behavior.",
        ),
        (
            "Econ ratio",
            "outside $[0.75,1.30]$",
            "Economic link strength is materially under- or over-stated.",
        ),
        ("Jump KS", "$>0.50$", "Pathwise maximum-jump distribution is unrealistic."),
        (
            "Kurtosis",
            "$<0.50$ or $>1.50$",
            "Daily-change tails are materially too thin or too extreme.",
        ),
        ("Corr score", "$<0.30$", "Cross-variable dependence is materially broken."),
        (
            "Factor KS",
            "$>0.15$",
            "Non-surface factor daily-change marginal mismatch is large.",
        ),
        (
            "Tail q99 ratio",
            "$>1.50$ or $<0.50$",
            "Factor tail scale is materially over- or under-stated.",
        ),
    ]
    body = "\n".join(
        f"{metric} & {rule} & {reason} \\\\" for metric, rule, reason in rows
    )
    text = (
        r"""\begin{table}[t]
\centering
\small
\caption{Operational red-cell thresholds used in the comparison tables. These thresholds are not optimized training gates; they are fixed manuscript reporting rules for flagging failure modes that make a scenario deck unsuitable as a risk artifact.}
\label{tab:operational-red-thresholds}
\begin{tabularx}{\textwidth}{llX}
\toprule
Metric & Red-cell rule & Interpretation \\
\midrule
"""
        + body
        + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
    )
    write("table_operational_red_thresholds.tex", text)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frame",
        choices=["insample", "heldout"],
        default="insample",
        help=(
            "Evaluation frame for the cross-model comparison tables. "
            "'insample' (default) reproduces the original behavior; 'heldout' "
            "wires only the comparison tables to the held-out frame and leaves "
            "all other tables untouched."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Override the output directory (default: paper/generated_tables). "
            "Used to write the camera-ready ICAIF copy without touching the "
            "originals."
        ),
    )
    args = parser.parse_args()

    global OUT
    if args.out_dir:
        OUT = Path(args.out_dir).resolve()

    if args.frame == "heldout":
        # Held-out re-wire: regenerate ONLY the cross-model comparison tables
        # (held-out frame) and the operational-threshold table (frame-independent
        # text). Every other table keeps its existing in-sample / fixed-diagnostic
        # source and is intentionally not regenerated, so e.g. the
        # conditional-responsiveness table stays byte-identical.
        make_classical_baseline_table(frame="heldout")
        make_operational_red_thresholds_table()
    else:
        make_panel_quality_table()
        make_iv_diagnostic_table()
        make_classical_baseline_table(frame="insample")
        make_sni_graph_ablation_table()
        make_conditional_responsiveness_table()
        make_reconstruction_table()
        make_operational_red_thresholds_table()
    print(f"Wrote LaTeX tables to {OUT} (frame={args.frame})")


if __name__ == "__main__":
    main()
