"""Publication report generation from authenticated SOP-16 matrix artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.contracts import SCHEMA_VERSION
from src.evaluation.experiment_matrix import (
    EXPERIMENT_MATRIX_VERSION,
    ExperimentMatrixResult,
    load_experiment_matrix_result,
)
from src.evaluation.result_registry import RegisteredResult, load_result
from src.utils.atomic_publish import atomic_rename_noreplace


REPORT_VERSION = "sop16_publication_report_v1"
REPORT_LAYOUT_VERSION = "sop16_publication_report_layout_v1"
_REPORT_STATIC_FILES = frozenset({"report_manifest.json", "COMPLETE.json"})
_METRIC_FIELDS = (
    "matrix_name",
    "experiment_id",
    "category",
    "risk_method",
    "value_method",
    "strategy",
    "metric",
    "seed_count",
    "mean",
    "std",
    "seed_values_json",
)
_CLOSED_LOOP_METRICS = (
    "collision_rate",
    "near_miss_rate",
    "false_safe_execution_rate",
    "verification_count_mean",
    "unnecessary_verification_rate",
    "reject_rate",
    "success_rate",
    "successful_completion_time_mean_s",
    "extra_path_length_mean_m",
    "extra_time_mean_s",
)
_STRATEGY_COLORS = {
    "never": "#0072B2",
    "always": "#56B4E9",
    "visible": "#009E73",
    "swept": "#E69F00",
    "entropy": "#CC79A7",
    "learned": "#D55E00",
    "oracle": "#2C2C2C",
}
_STRATEGY_MARKERS = {
    "never": "o",
    "always": "s",
    "visible": "^",
    "swept": "D",
    "entropy": "v",
    "learned": "*",
    "oracle": "P",
}
_AXIS_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
)


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
        }
    )


def _json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("report value must be finite JSON") from exc
    return (encoded + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    payload = _json_bytes(value)
    path.write_bytes(payload)
    return payload


def _read_file(path: Path, *, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a real file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc


def _read_json(path: Path, *, name: str) -> dict[str, object]:
    payload = _read_file(path, name=name)
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain an object")
    return value


def _csv_bytes(
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _save_figure(fig: plt.Figure, root: Path, stem: str) -> tuple[str, str]:
    pdf_name = f"{stem}.pdf"
    png_name = f"{stem}.png"
    fig.savefig(
        root / pdf_name,
        format="pdf",
        metadata={
            "Creator": REPORT_VERSION,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        root / png_name,
        format="png",
        dpi=300,
        metadata={"Software": REPORT_VERSION},
    )
    plt.close(fig)
    return pdf_name, png_name


def _matrix_name(matrix: ExperimentMatrixResult) -> str:
    value = matrix.summary.get("matrix_name")
    if not isinstance(value, str) or not value:
        raise ValueError("matrix summary is missing matrix_name")
    return value


def _metric_rows(
    matrices: Sequence[ExperimentMatrixResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for matrix in matrices:
        matrix_name = _matrix_name(matrix)
        experiments = matrix.summary.get("experiments")
        if not isinstance(experiments, Mapping):
            raise ValueError("matrix experiments summary is invalid")
        for experiment_id, experiment in sorted(experiments.items()):
            if not isinstance(experiment, Mapping):
                raise ValueError("matrix experiment summary is invalid")
            aggregates = experiment.get("aggregates")
            if not isinstance(aggregates, Mapping):
                raise ValueError("matrix aggregates are invalid")
            for strategy, aggregate in sorted(aggregates.items()):
                if not isinstance(aggregate, Mapping):
                    raise ValueError("matrix strategy aggregate is invalid")
                metrics = aggregate.get("metrics")
                if not isinstance(metrics, Mapping):
                    raise ValueError("matrix metric aggregate is invalid")
                for metric_name, metric in sorted(metrics.items()):
                    if not isinstance(metric, Mapping):
                        raise ValueError("matrix metric row is invalid")
                    rows.append(
                        {
                            "matrix_name": matrix_name,
                            "experiment_id": experiment_id,
                            "category": experiment["category"],
                            "risk_method": experiment["risk_method"],
                            "value_method": experiment["value_method"],
                            "strategy": strategy,
                            "metric": metric_name,
                            "seed_count": aggregate["seed_count"],
                            "mean": metric["mean"],
                            "std": metric["std"],
                            "seed_values_json": json.dumps(
                                metric["seed_values"],
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    )
    return rows


def _metric_lookup(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str, str], Mapping[str, object]]:
    return {
        (
            str(row["matrix_name"]),
            str(row["experiment_id"]),
            str(row["strategy"]),
            str(row["metric"]),
        ): row
        for row in rows
    }


def _closed_loop_table_rows(
    metric_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    lookup = _metric_lookup(metric_rows)
    identities = sorted(
        {
            (
                str(row["matrix_name"]),
                str(row["experiment_id"]),
                str(row["category"]),
                str(row["risk_method"]),
                str(row["value_method"]),
                str(row["strategy"]),
                int(row["seed_count"]),
            )
            for row in metric_rows
        }
    )
    fields = (
        "matrix_name",
        "experiment_id",
        "category",
        "risk_method",
        "value_method",
        "strategy",
        "seed_count",
        *(
            field
            for metric in _CLOSED_LOOP_METRICS
            for field in (f"{metric}_mean", f"{metric}_std")
        ),
    )
    rows: list[dict[str, object]] = []
    for (
        matrix_name,
        experiment_id,
        category,
        risk_method,
        value_method,
        strategy,
        seed_count,
    ) in identities:
        row: dict[str, object] = {
            "matrix_name": matrix_name,
            "experiment_id": experiment_id,
            "category": category,
            "risk_method": risk_method,
            "value_method": value_method,
            "strategy": strategy,
            "seed_count": seed_count,
        }
        for metric in _CLOSED_LOOP_METRICS:
            aggregate = lookup.get(
                (matrix_name, experiment_id, strategy, metric)
            )
            row[f"{metric}_mean"] = "" if aggregate is None else aggregate["mean"]
            row[f"{metric}_std"] = "" if aggregate is None else aggregate["std"]
        rows.append(row)
    return rows, tuple(fields)


def _plot_pareto(
    matrices: Sequence[ExperimentMatrixResult],
    metric_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> tuple[str, str] | None:
    points = [
        (_matrix_name(matrix), row)
        for matrix in matrices
        for row in matrix.pareto_rows
    ]
    if not points:
        return None
    lookup = _metric_lookup(metric_rows)
    fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
    for matrix_name, point in points:
        experiment_id = str(point["experiment_id"])
        strategy = str(point["strategy"])
        x = float(point["verification_count_mean"])
        y = float(point["collision_rate_mean"])
        x_std = float(
            lookup[
                (
                    matrix_name,
                    experiment_id,
                    strategy,
                    "verification_count_mean",
                )
            ]["std"]
        )
        y_std = float(
            lookup[
                (matrix_name, experiment_id, strategy, "collision_rate")
            ]["std"]
        )
        color = _STRATEGY_COLORS.get(strategy, "#8C8C8C")
        marker = _STRATEGY_MARKERS.get(strategy, "o")
        ax.errorbar(
            x,
            y,
            xerr=x_std,
            yerr=y_std,
            fmt=marker,
            color=color,
            markeredgecolor=("#111111" if point["pareto_optimal"] else color),
            markeredgewidth=(1.2 if point["pareto_optimal"] else 0.5),
            markersize=(9 if strategy == "learned" else 6),
            capsize=2,
            alpha=(1.0 if point["pareto_optimal"] else 0.5),
            zorder=(4 if strategy == "learned" else 3),
        )
        label = (
            strategy
            if len({str(item[1]["experiment_id"]) for item in points}) == 1
            else f"{experiment_id}/{strategy}"
        )
        ax.annotate(
            label,
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
        )
    ax.set_xlabel("Mean verification count per episode ↓")
    ax.set_ylabel("Collision rate ↓")
    ax.set_title("Safety–efficiency Pareto")
    handles = [
        Line2D(
            [0],
            [0],
            marker=_STRATEGY_MARKERS[strategy],
            color="none",
            markerfacecolor=_STRATEGY_COLORS[strategy],
            markeredgecolor=_STRATEGY_COLORS[strategy],
            label=strategy,
        )
        for strategy in _STRATEGY_COLORS
        if any(str(point["strategy"]) == strategy for _, point in points)
    ]
    ax.legend(handles=handles, ncol=min(4, len(handles)), loc="best")
    return _save_figure(fig, output_dir, "pareto")


def _preferred_strategy(aggregates: Mapping[str, object]) -> str | None:
    if "learned" in aggregates:
        return "learned"
    return min(aggregates) if aggregates else None


def _plot_ablations(
    matrices: Sequence[ExperimentMatrixResult],
    output_dir: Path,
) -> tuple[str, str] | None:
    labels: list[str] = []
    collision: list[float] = []
    verification: list[float] = []
    for matrix in matrices:
        experiments = matrix.summary["experiments"]
        if not isinstance(experiments, Mapping):
            continue
        for experiment_id, experiment in sorted(experiments.items()):
            if not isinstance(experiment, Mapping) or experiment["category"] != "ablation":
                continue
            aggregates = experiment["aggregates"]
            if not isinstance(aggregates, Mapping):
                continue
            strategy = _preferred_strategy(aggregates)
            if strategy is None:
                continue
            aggregate = aggregates[strategy]
            if not isinstance(aggregate, Mapping):
                continue
            labels.append(str(experiment_id))
            collision.append(
                float(aggregate["metrics"]["collision_rate"]["mean"])
            )
            verification.append(
                float(aggregate["metrics"]["verification_count_mean"]["mean"])
            )
    if not labels:
        return None
    positions = range(len(labels))
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, max(2.8, 0.32 * len(labels) + 1.2)),
        constrained_layout=True,
    )
    axes[0].barh(positions, collision, color="#0072B2", height=0.58)
    axes[1].barh(positions, verification, color="#D55E00", height=0.58)
    for axis, title, xlabel in (
        (axes[0], "Safety", "Collision rate ↓"),
        (axes[1], "Efficiency", "Verification count ↓"),
    ):
        axis.set_yticks(list(positions))
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel(xlabel)
    return _save_figure(fig, output_dir, "ablations")


def _plot_sensitivity(
    matrices: Sequence[ExperimentMatrixResult],
    output_dir: Path,
) -> tuple[str, str] | None:
    points: list[tuple[str, str, float, float]] = []
    for matrix in matrices:
        experiments = matrix.summary["experiments"]
        if not isinstance(experiments, Mapping):
            continue
        for experiment in experiments.values():
            if not isinstance(experiment, Mapping) or experiment["category"] != "sensitivity":
                continue
            parameters = experiment["parameters"]
            aggregates = experiment["aggregates"]
            if not isinstance(parameters, Mapping) or not isinstance(
                aggregates, Mapping
            ):
                continue
            strategy = _preferred_strategy(aggregates)
            if strategy is None:
                continue
            aggregate = aggregates[strategy]
            points.append(
                (
                    str(parameters.get("axis", "unknown")),
                    str(parameters.get("value", "unknown")),
                    float(
                        aggregate["metrics"]["verification_count_mean"]["mean"]
                    ),
                    float(aggregate["metrics"]["collision_rate"]["mean"]),
                )
            )
    if not points:
        return None
    axes = sorted({point[0] for point in points})
    colors = {
        axis: _AXIS_COLORS[index % len(_AXIS_COLORS)]
        for index, axis in enumerate(axes)
    }
    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    for axis_name, value, verification, collision in points:
        ax.scatter(
            verification,
            collision,
            color=colors[axis_name],
            marker="o",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            value,
            (verification, collision),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[axis],
            label=axis,
        )
        for axis in axes
    ]
    ax.legend(handles=handles, ncol=min(3, len(handles)))
    ax.set_xlabel("Mean verification count per episode ↓")
    ax.set_ylabel("Collision rate ↓")
    ax.set_title("Sensitivity operating points")
    return _save_figure(fig, output_dir, "sensitivity")


def _case_records(
    matrices: Sequence[ExperimentMatrixResult],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    selected: list[tuple[dict[str, object], dict[str, object]]] = []
    result_cache: dict[Path, RegisteredResult] = {}
    for matrix in matrices:
        authenticated_paths = {
            path.resolve(strict=False) for path in matrix.run_paths
        }
        raw_cases = matrix.case_index.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError("matrix case index is invalid")
        for raw_case in raw_cases:
            if len(selected) == 10:
                return selected
            if not isinstance(raw_case, dict):
                raise ValueError("matrix case entry is invalid")
            relative = Path(str(raw_case["run_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("case index run path is unsafe")
            result_path = matrix.output_dir / relative
            if result_path.resolve(strict=False) not in authenticated_paths:
                raise ValueError("case index references an unauthenticated run")
            if result_path not in result_cache:
                result_cache[result_path] = load_result(result_path)
            result = result_cache[result_path]
            episode = next(
                (
                    row
                    for row in result.episodes
                    if row.get("episode_id") == raw_case["episode_id"]
                ),
                None,
            )
            if episode is None:
                raise ValueError("case index references a missing episode")
            selected.append((dict(raw_case), episode))
    return selected


def _plot_cases(
    cases: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    output_dir: Path,
) -> tuple[str, str] | None:
    if not cases:
        return None
    count = len(cases)
    fig, axes = plt.subplots(
        count,
        1,
        figsize=(7.0, max(3.0, 0.55 * count + 1.2)),
        sharex=True,
        constrained_layout=True,
    )
    if count == 1:
        axes = [axes]
    decision_colors = {
        "execute": "#0072B2",
        "verify": "#E69F00",
        "reject": "#8C8C8C",
    }
    for axis, (case, episode) in zip(axes, cases):
        steps = episode.get("steps")
        if not isinstance(steps, list):
            raise ValueError("case episode steps are invalid")
        for step in steps:
            if not isinstance(step, Mapping):
                raise ValueError("case episode step is invalid")
            start = float(step["elapsed_before_s"])
            end = float(step["elapsed_after_s"])
            duration = max(end - start, 0.025)
            decision = str(step["decision"])
            axis.broken_barh(
                [(start, duration)],
                (0.2, 0.6),
                facecolors=decision_colors.get(decision, "#8C8C8C"),
                edgecolors="white",
                linewidth=0.5,
            )
        elapsed = float(episode["elapsed_s"])
        if episode.get("collision") is True:
            axis.plot(elapsed, 0.5, marker="x", color="#D55E00", markersize=6)
        elif episode.get("success") is True:
            axis.plot(elapsed, 0.5, marker="*", color="#009E73", markersize=7)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks([0.5])
        axis.set_yticklabels(
            [f"{case['strategy']} · {case['episode_id']}"],
            fontsize=6.5,
        )
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
    axes[-1].set_xlim(0.0, 6.4)
    axes[-1].set_xlabel("Elapsed time within the original Long40 window (s)")
    axes[0].set_title("Selected success and failure episode traces")
    axes[0].legend(
        handles=[
            Patch(color=decision_colors["execute"], label="execute"),
            Patch(color=decision_colors["verify"], label="verify"),
            Patch(color=decision_colors["reject"], label="reject"),
            Line2D([0], [0], marker="x", color="#D55E00", label="collision"),
            Line2D([0], [0], marker="*", color="#009E73", label="success"),
        ],
        ncol=5,
        loc="upper right",
    )
    return _save_figure(fig, output_dir, "cases")


def _plot_offline_metrics(
    metric_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> tuple[str, str] | None:
    rows = [
        row
        for row in metric_rows
        if _is_offline_metric(str(row["metric"]))
    ][:24]
    if not rows:
        return None
    labels = [
        f"{row['experiment_id']}/{row['strategy']}/{row['metric']}" for row in rows
    ]
    means = [float(row["mean"]) for row in rows]
    stds = [float(row["std"]) for row in rows]
    positions = range(len(rows))
    colors = [
        _STRATEGY_COLORS.get(str(row["strategy"]), "#8C8C8C") for row in rows
    ]
    fig, ax = plt.subplots(
        figsize=(7.0, max(3.0, 0.28 * len(rows) + 1.2)),
        constrained_layout=True,
    )
    ax.barh(
        positions,
        means,
        xerr=stds,
        color=colors,
        height=0.58,
        error_kw={"elinewidth": 0.8, "capsize": 2},
    )
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Registered metric value")
    ax.set_title("Calibration, coverage, and verification-value metrics")
    return _save_figure(fig, output_dir, "offline_metrics")


def _is_offline_metric(metric_name: str) -> bool:
    name = metric_name.lower()
    ece_metric = name == "ece" or name.endswith("_ece") or name.startswith("ece_")
    coverage_metric = (
        name == "coverage"
        or name.startswith("coverage_")
        or name.endswith("_coverage")
        or "_coverage_" in name
    )
    return (
        "calibration" in name
        or ece_metric
        or coverage_metric
        or "regret" in name
        or "ranking" in name
        or "spearman" in name
        or "kendall" in name
        or "useful_action_f1" in name
    )


def _combined_claim_index(
    matrices: Sequence[ExperimentMatrixResult],
) -> dict[str, object]:
    claims: list[dict[str, object]] = []
    for matrix in matrices:
        source_name = _matrix_name(matrix)
        raw_claims = matrix.claim_index.get("claims")
        if not isinstance(raw_claims, list):
            raise ValueError("matrix claim index is invalid")
        for raw_claim in raw_claims:
            if not isinstance(raw_claim, dict):
                raise ValueError("matrix claim entry is invalid")
            claim = dict(raw_claim)
            claim["source_matrix"] = source_name
            claims.append(claim)
    return {
        "schema_version": SCHEMA_VERSION,
        "report_version": REPORT_VERSION,
        "claims": claims,
    }


@dataclass(frozen=True)
class EvaluationReport:
    output_dir: Path
    generated_files: tuple[str, ...]
    summary: dict[str, object]
    manifest: dict[str, object]


def build_evaluation_report(
    matrix_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path,
) -> EvaluationReport:
    """Build tables and figures only from authenticated matrix results."""

    if isinstance(matrix_dirs, (str, bytes)) or not isinstance(
        matrix_dirs, Sequence
    ):
        raise TypeError("matrix_dirs must be a sequence")
    if not matrix_dirs:
        raise ValueError("matrix_dirs must not be empty")
    matrices = tuple(
        load_experiment_matrix_result(path) for path in matrix_dirs
    )
    names = [_matrix_name(matrix) for matrix in matrices]
    if len(set(names)) != len(names):
        raise ValueError("source matrix names must be distinct")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    _apply_publication_style()
    try:
        metric_rows = _metric_rows(matrices)
        table_rows, table_fields = _closed_loop_table_rows(metric_rows)
        files: dict[str, bytes] = {}
        files["metrics_long.csv"] = _csv_bytes(
            metric_rows, fieldnames=_METRIC_FIELDS
        )
        files["closed_loop_table.csv"] = _csv_bytes(
            table_rows, fieldnames=table_fields
        )
        (staging / "metrics_long.csv").write_bytes(files["metrics_long.csv"])
        (staging / "closed_loop_table.csv").write_bytes(
            files["closed_loop_table.csv"]
        )
        claims = _combined_claim_index(matrices)
        files["claim_to_evidence.json"] = _write_json(
            staging / "claim_to_evidence.json", claims
        )
        cases = _case_records(matrices)
        generated_figures: list[str] = []
        for figure in (
            _plot_pareto(matrices, metric_rows, staging),
            _plot_ablations(matrices, staging),
            _plot_sensitivity(matrices, staging),
            _plot_cases(cases, staging),
            _plot_offline_metrics(metric_rows, staging),
        ):
            if figure is not None:
                generated_figures.extend(figure)
        for name in generated_figures:
            files[name] = (staging / name).read_bytes()
        summary = {
            "schema_version": SCHEMA_VERSION,
            "report_version": REPORT_VERSION,
            "source_matrix_count": len(matrices),
            "source_matrix_names": names,
            "all_sources_scientifically_complete": all(
                matrix.summary.get("scientifically_complete") is True
                for matrix in matrices
            ),
            "metric_row_count": len(metric_rows),
            "closed_loop_table_row_count": len(table_rows),
            "case_count": len(cases),
            "claim_count": len(claims["claims"]),
            "offline_metric_names": sorted(
                {
                    str(row["metric"])
                    for row in metric_rows
                    if _is_offline_metric(str(row["metric"]))
                }
            ),
            "generated_figures": sorted(generated_figures),
        }
        files["report_summary.json"] = _write_json(
            staging / "report_summary.json", summary
        )
        source_matrices = []
        for matrix in matrices:
            manifest_payload = (
                matrix.output_dir / "matrix_manifest.json"
            ).read_bytes()
            source_matrices.append(
                {
                    "matrix_name": _matrix_name(matrix),
                    "matrix_manifest_sha256": _sha256(manifest_payload),
                    "scientific_status": matrix.summary["scientific_status"],
                    "scientifically_complete": matrix.summary[
                        "scientifically_complete"
                    ],
                }
            )
        manifest = {
            "report_layout_version": REPORT_LAYOUT_VERSION,
            "report_version": REPORT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "source_matrices": source_matrices,
            "file_digests_sha256": {
                name: _sha256(payload) for name, payload in sorted(files.items())
            },
        }
        manifest_payload = _write_json(
            staging / "report_manifest.json", manifest
        )
        _write_json(
            staging / "COMPLETE.json",
            {
                "report_layout_version": REPORT_LAYOUT_VERSION,
                "report_manifest_sha256": _sha256(manifest_payload),
            },
        )
        atomic_rename_noreplace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return EvaluationReport(
        output_dir=destination,
        generated_files=tuple(sorted(files)),
        summary=summary,
        manifest=manifest,
    )


def load_evaluation_report(output_dir: str | Path) -> EvaluationReport:
    """Authenticate a generated report and all tables/figures."""

    root = Path(output_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("report output must be a real directory")
    manifest_payload = _read_file(
        root / "report_manifest.json", name="report_manifest.json"
    )
    try:
        manifest_value = json.loads(manifest_payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid report manifest: {exc}") from exc
    if not isinstance(manifest_value, dict):
        raise ValueError("report manifest must contain an object")
    manifest = manifest_value
    if set(manifest) != {
        "report_layout_version",
        "report_version",
        "schema_version",
        "source_matrices",
        "file_digests_sha256",
    }:
        raise ValueError("report manifest keys are invalid")
    if manifest["report_layout_version"] != REPORT_LAYOUT_VERSION:
        raise ValueError("unsupported report layout version")
    if manifest["report_version"] != REPORT_VERSION:
        raise ValueError("unsupported report version")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("report schema mismatch")
    complete = _read_json(root / "COMPLETE.json", name="COMPLETE.json")
    if complete != {
        "report_layout_version": REPORT_LAYOUT_VERSION,
        "report_manifest_sha256": _sha256(manifest_payload),
    }:
        raise ValueError("report completion marker does not authenticate manifest")
    digests = manifest["file_digests_sha256"]
    if not isinstance(digests, dict) or not digests:
        raise ValueError("report file digest inventory is invalid")
    expected_names = set(digests) | _REPORT_STATIC_FILES
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("report output layout is invalid")
    for name, digest in digests.items():
        if digest != _sha256(_read_file(root / name, name=name)):
            raise ValueError(f"report file digest mismatch: {name}")
    summary = _read_json(root / "report_summary.json", name="report_summary.json")
    if summary.get("generated_figures") != sorted(
        name for name in digests if name.endswith((".pdf", ".png"))
    ):
        raise ValueError("report figure inventory does not match summary")
    return EvaluationReport(
        output_dir=root,
        generated_files=tuple(sorted(digests)),
        summary=summary,
        manifest=manifest,
    )


__all__ = (
    "REPORT_LAYOUT_VERSION",
    "REPORT_VERSION",
    "EvaluationReport",
    "build_evaluation_report",
    "load_evaluation_report",
)
