#!/usr/bin/env python3
"""Plot eval metrics across models and embedding dimensions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from itertools import cycle
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MPLCONFIGDIR = REPO_ROOT / ".cache" / "matplotlib"
DEFAULT_XDG_CACHE_HOME = REPO_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_XDG_CACHE_HOME))
DEFAULT_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
DEFAULT_IMG_DIR = REPO_ROOT / "img"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
DEFAULT_EVAL_ROOT = REPO_ROOT / "outputs" / "eval"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_metric_name(metric_name: str) -> str:
    return metric_name.replace("@", "_at_").replace("/", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval metrics from eval result dirs.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory of eval runs, or a manifest file listing eval run dirs / ids.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to <input>/plots or <manifest_stem>_plots.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return raw


def resolve_eval_run_dirs_from_manifest(manifest_path: Path) -> list[Path]:
    run_dirs: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            candidate = Path(value)
            candidates = []
            if candidate.is_absolute():
                candidates.append(candidate)
            else:
                candidates.append((manifest_path.parent / candidate).resolve())
                candidates.append((REPO_ROOT / candidate).resolve())
                candidates.append((DEFAULT_EVAL_ROOT / candidate).resolve())

            resolved = next((path for path in candidates if path.exists() and path.is_dir()), None)
            if resolved is None:
                raise FileNotFoundError(
                    f"Manifest entry not found as eval run dir at {manifest_path}:{lineno}: {value}"
                )
            run_dirs.append(resolved)
    return run_dirs


def resolve_eval_run_dirs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted([path for path in input_path.iterdir() if path.is_dir()], key=lambda p: p.name)
    if input_path.is_file():
        return resolve_eval_run_dirs_from_manifest(input_path)
    raise FileNotFoundError(f"Input path not found: {input_path}")


def collect_leaf_report_paths(run_dir: Path) -> list[Path]:
    root_report_path = run_dir / "run_eval_report.json"
    if not root_report_path.exists():
        return []
    root_report = load_json(root_report_path)
    if isinstance(root_report.get("runs"), list):
        report_paths: list[Path] = []
        for run_entry in root_report["runs"]:
            if not isinstance(run_entry, dict):
                continue
            report_output = normalize_text(run_entry.get("report_output"))
            if not report_output:
                continue
            report_path = Path(report_output)
            if report_path.exists():
                report_paths.append(report_path)
        return sorted(report_paths, key=lambda p: p.as_posix())
    return [root_report_path]


def collect_records(run_dirs: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        for report_path in collect_leaf_report_paths(run_dir):
            report = load_json(report_path)
            metrics_raw = report.get("metrics")
            if not isinstance(metrics_raw, dict):
                continue

            config = report.get("config")
            if not isinstance(config, dict):
                config = {}
            embedding_identity = report.get("embedding_identity")
            if not isinstance(embedding_identity, dict):
                embedding_identity = {}
            index_stats = report.get("index_stats")
            if not isinstance(index_stats, dict):
                index_stats = {}
            eval_stats = report.get("eval_stats")
            if not isinstance(eval_stats, dict):
                eval_stats = {}

            model_name = normalize_text(embedding_identity.get("model_name_guess")) or normalize_text(
                embedding_identity.get("model_dir_name")
            )
            eval_run_id = normalize_text(config.get("eval_run_id")) or run_dir.name
            embedding_dim = int(index_stats["embedding_dim"])

            for at_k, metric_values in metrics_raw.items():
                if not isinstance(metric_values, dict):
                    continue
                k = normalize_text(at_k).lstrip("@")
                for metric_name, metric_value in metric_values.items():
                    records.append(
                        {
                            "model_name": model_name,
                            "embedding_dim": embedding_dim,
                            "eval_run_id": eval_run_id,
                            "run_dir": str(run_dir.resolve()),
                            "report_path": str(report_path.resolve()),
                            "eval_input": normalize_text(config.get("eval_input")),
                            "query_pooling": normalize_text(config.get("query_pooling")),
                            "query_retrieval_mode": normalize_text(config.get("query_retrieval_mode")),
                            "index_type": normalize_text(config.get("index_type")),
                            "topk_list": ",".join(str(x) for x in config.get("topk", []))
                            if isinstance(config.get("topk"), list)
                            else "",
                            "metric_group": at_k,
                            "k": k,
                            "metric_name": normalize_text(metric_name),
                            "metric_key": f"{normalize_text(metric_name)}@{k}",
                            "metric_value": float(metric_value),
                            "valid_eval_rows": int(eval_stats.get("valid_eval_rows", 0)),
                        }
                    )
    return records


def write_results_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "model_name",
        "embedding_dim",
        "eval_run_id",
        "run_dir",
        "report_path",
        "eval_input",
        "query_pooling",
        "query_retrieval_mode",
        "index_type",
        "topk_list",
        "metric_group",
        "k",
        "metric_name",
        "metric_key",
        "metric_value",
        "valid_eval_rows",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda x: (
                x["metric_key"],
                x["model_name"],
                int(x["embedding_dim"]),
                x["eval_run_id"],
            ),
        ):
            writer.writerow(record)


def build_plot_series(records: list[dict[str, Any]], metric_key: str) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, dict[int, float]] = {}
    for record in records:
        if record["metric_key"] != metric_key:
            continue
        model_name = record["model_name"]
        dim = int(record["embedding_dim"])
        value = float(record["metric_value"])
        series.setdefault(model_name, {})[dim] = value
    return {
        model_name: sorted(dim_to_value.items(), key=lambda x: x[0])
        for model_name, dim_to_value in sorted(series.items(), key=lambda x: x[0])
    }


def make_plot(metric_key: str, series: dict[str, list[tuple[int, float]]], output_path: Path) -> None:
    all_dims = sorted({dim for points in series.values() for dim, _ in points})
    all_values = [value for points in series.values() for _, value in points]
    if not all_dims or not all_values:
        return

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    x_positions = {dim: idx for idx, dim in enumerate(all_dims)}
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for model_name, points in series.items():
        xs = [x_positions[dim] for dim, _ in points]
        values = [value for _, value in points]
        ax.plot(
            xs,
            values,
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=model_name,
            color=next(color_cycle),
        )

    ax.set_title(metric_key)
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel(metric_key)
    ax.set_xticks(list(range(len(all_dims))), [str(dim) for dim in all_dims])
    min_value = min(all_values)
    max_value = max(all_values)
    if min_value == max_value:
        pad = max(abs(min_value) * 0.05, 1e-4)
    else:
        pad = max((max_value - min_value) * 0.15, 1e-4)
    lower = max(min_value - pad, 0.0) if min_value >= 0.0 else min_value - pad
    upper = min(max_value + pad, 1.0) if max_value <= 1.0 else max_value + pad
    if lower >= upper:
        upper = lower + 1e-3
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def derive_default_output_dir(input_path: Path) -> Path:
    _ = input_path
    return DEFAULT_IMG_DIR


def write_summary(records: list[dict[str, Any]], output_path: Path, input_path: Path) -> None:
    metric_keys = sorted({record["metric_key"] for record in records})
    models = sorted({record["model_name"] for record in records})
    dims = sorted({int(record["embedding_dim"]) for record in records})
    payload = {
        "input": str(input_path.resolve()),
        "num_rows": len(records),
        "num_models": len(models),
        "models": models,
        "dims": dims,
        "metric_keys": metric_keys,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else derive_default_output_dir(input_path)
    ensure_dir(output_dir)

    run_dirs = resolve_eval_run_dirs(input_path)
    records = collect_records(run_dirs)
    if not records:
        raise ValueError(f"No eval metric records found from input: {input_path}")

    write_results_csv(records, output_dir / "results.csv")
    write_summary(records, output_dir / "summary.json", input_path)

    metric_keys = sorted({record["metric_key"] for record in records})
    for metric_key in metric_keys:
        series = build_plot_series(records, metric_key)
        if not series:
            continue
        output_path = output_dir / f"{sanitize_metric_name(metric_key)}.png"
        make_plot(metric_key=metric_key, series=series, output_path=output_path)

    print(f"[plot] input={input_path}")
    print(f"[plot] output_dir={output_dir}")
    print(f"[plot] records={len(records)}")
    print(f"[plot] metrics={len(metric_keys)}")


if __name__ == "__main__":
    main()
