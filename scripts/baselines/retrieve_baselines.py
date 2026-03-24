#!/usr/bin/env python3
"""Run non-embedding retrieval baselines on eval.jsonl."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.baselines.baseline_utils import (
    ensure_parent_dir,
    file_sha256,
    find_rank,
    generate_run_id_local,
    load_items_metadata,
    load_positive_popularity,
    metric_value,
    normalize_text,
    parse_topk_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval baselines from eval.jsonl.")
    parser.add_argument(
        "--baseline",
        required=True,
        choices=["random", "global_popular", "category_random", "category_popular"],
        help="Baseline retrieval method.",
    )
    parser.add_argument(
        "--items-input",
        default="data/processed/items.jsonl",
        help="Input items jsonl path.",
    )
    parser.add_argument(
        "--interactions-input",
        default="data/processed/interactions.jsonl",
        help="Input interactions jsonl path.",
    )
    parser.add_argument(
        "--eval-input",
        default="data/processed/eval.jsonl",
        help="Input eval queries jsonl path.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/baselines",
        help="Root output directory for baseline artifacts.",
    )
    parser.add_argument(
        "--topk",
        default="10,50",
        help="Comma-separated metric K list, e.g. '10,50'.",
    )
    parser.add_argument(
        "--max-query",
        type=int,
        default=0,
        help="Maximum number of valid eval queries to evaluate. 0 means no limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic baselines.",
    )
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=4.0,
        help="Positive interaction threshold used for popularity counting.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id. Default is local timestamp YYYYMMDDHHMMSSmmm.",
    )
    args = parser.parse_args()
    if args.max_query < 0:
        parser.error("--max-query must be >= 0")
    args.topk_list = parse_topk_list(args.topk)
    return args


def sort_popular_item_ids(item_ids: list[str], popularity: dict[str, int]) -> list[str]:
    return sorted(item_ids, key=lambda item_id: (-popularity.get(item_id, 0), item_id))


def build_random_order(item_ids: list[str], seed: int) -> list[str]:
    item_ids_copy = list(item_ids)
    rng = random.Random(seed)
    rng.shuffle(item_ids_copy)
    return item_ids_copy


def take_topk_excluding(
    *,
    ordered_item_ids: list[str],
    excluded_item_ids: set[str],
    top_k: int,
) -> list[str]:
    selected: list[str] = []
    if top_k <= 0:
        return selected
    for item_id in ordered_item_ids:
        if item_id in excluded_item_ids:
            continue
        selected.append(item_id)
        if len(selected) >= top_k:
            break
    return selected


def take_combined_topk_excluding(
    *,
    primary_item_ids: list[str],
    fallback_item_ids: list[str],
    excluded_item_ids: set[str],
    top_k: int,
) -> list[str]:
    selected: list[str] = []
    selected_set: set[str] = set()
    if top_k <= 0:
        return []
    for item_ids in (primary_item_ids, fallback_item_ids):
        for item_id in item_ids:
            if item_id in excluded_item_ids or item_id in selected_set:
                continue
            selected.append(item_id)
            selected_set.add(item_id)
            if len(selected) >= top_k:
                return selected
    return selected


def build_predictions(
    *,
    ordered_item_ids: list[str],
    top_k: int,
    popularity: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    selected = ordered_item_ids[:top_k]
    return [
        {
            "rank": idx + 1,
            "item_id": item_id,
            "score": float(popularity.get(item_id, 0)) if popularity is not None else 0.0,
        }
        for idx, item_id in enumerate(selected)
    ]


def build_short_pools(
    *,
    item_ids: list[str],
    item_ids_by_category: dict[str, list[str]],
    popularity: dict[str, int],
    seed: int,
    pool_size: int,
) -> tuple[list[str], list[str], dict[str, list[str]], dict[str, list[str]]]:
    global_popular_pool = sort_popular_item_ids(item_ids, popularity)[:pool_size]
    global_random_pool = build_random_order(item_ids, seed)[:pool_size]
    category_popular_pools = {
        category: sort_popular_item_ids(category_item_ids, popularity)[:pool_size]
        for category, category_item_ids in item_ids_by_category.items()
    }
    category_random_pools = {
        category: build_random_order(category_item_ids, seed)[:pool_size]
        for category, category_item_ids in item_ids_by_category.items()
    }
    return global_popular_pool, global_random_pool, category_popular_pools, category_random_pools


def main() -> None:
    args = parse_args()

    items_input = Path(args.items_input)
    interactions_input = Path(args.interactions_input)
    eval_input = Path(args.eval_input)

    item_ids, category_by_item_id, item_ids_by_category, item_stats = load_items_metadata(items_input)
    valid_item_ids = set(item_ids)
    popularity, popularity_stats = load_positive_popularity(
        interactions_input,
        valid_item_ids=valid_item_ids,
        rating_threshold=float(args.rating_threshold),
    )

    run_id = normalize_text(args.run_id) or generate_run_id_local()
    run_output_dir = Path(args.output_root) / args.baseline / run_id
    predictions_output = run_output_dir / "predictions.jsonl"
    report_output = run_output_dir / "report.json"
    info_output = run_output_dir / "info.json"
    ensure_parent_dir(predictions_output)
    ensure_parent_dir(report_output)
    ensure_parent_dir(info_output)

    ks = args.topk_list
    max_k = max(ks)
    pool_size = max(128, max_k + 8)
    (
        global_popular_pool,
        global_random_pool,
        category_popular_pools,
        category_random_pools,
    ) = build_short_pools(
        item_ids=item_ids,
        item_ids_by_category=item_ids_by_category,
        popularity=popularity,
        seed=args.seed,
        pool_size=pool_size,
    )
    sum_hit = {k: 0.0 for k in ks}
    sum_mrr = {k: 0.0 for k in ks}
    sum_ndcg = {k: 0.0 for k in ks}

    input_rows_total = 0
    parse_error_rows = 0
    non_object_rows = 0
    rows_missing_user_id = 0
    rows_missing_target_item_id = 0
    rows_invalid_query_list = 0

    dropped_target_not_in_items = 0
    dropped_query_contains_target = 0
    dropped_query_item_not_in_items = 0
    dropped_query_empty_after_clean = 0
    dropped_no_candidates_after_filter = 0

    category_rows = 0
    category_fallback_missing_category = 0
    category_fallback_short_pool = 0
    category_rows_with_matched_category_pool = 0

    valid_eval_rows = 0

    with eval_input.open("r", encoding="utf-8") as in_f, predictions_output.open(
        "w", encoding="utf-8"
    ) as pred_f:
        for line in in_f:
            input_rows_total += 1
            line = line.strip()
            if not line:
                parse_error_rows += 1
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                parse_error_rows += 1
                continue
            if not isinstance(raw, dict):
                non_object_rows += 1
                continue

            user_id = normalize_text(raw.get("user_id"))
            target_item_id = normalize_text(raw.get("target_item_id"))
            query_item_ids_raw = raw.get("query_item_ids")

            if not user_id:
                rows_missing_user_id += 1
                continue
            if not target_item_id:
                rows_missing_target_item_id += 1
                continue
            if not isinstance(query_item_ids_raw, list):
                rows_invalid_query_list += 1
                continue

            cleaned_query_item_ids: list[str] = []
            seen_query_item_ids: set[str] = set()
            for item in query_item_ids_raw:
                item_id = normalize_text(item)
                if not item_id:
                    continue
                if item_id in seen_query_item_ids:
                    continue
                seen_query_item_ids.add(item_id)
                cleaned_query_item_ids.append(item_id)

            if not cleaned_query_item_ids:
                dropped_query_empty_after_clean += 1
                continue
            if target_item_id in seen_query_item_ids:
                dropped_query_contains_target += 1
                continue
            if target_item_id not in valid_item_ids:
                dropped_target_not_in_items += 1
                continue

            missing_query_item = False
            for query_item_id in cleaned_query_item_ids:
                if query_item_id not in valid_item_ids:
                    missing_query_item = True
                    break
            if missing_query_item:
                dropped_query_item_not_in_items += 1
                continue

            excluded_item_ids = set(cleaned_query_item_ids)

            if args.baseline == "random":
                predictions = build_predictions(
                    ordered_item_ids=take_topk_excluding(
                        ordered_item_ids=global_random_pool,
                        excluded_item_ids=excluded_item_ids,
                        top_k=max_k,
                    ),
                    top_k=max_k,
                )
            elif args.baseline == "global_popular":
                predictions = build_predictions(
                    ordered_item_ids=take_topk_excluding(
                        ordered_item_ids=global_popular_pool,
                        excluded_item_ids=excluded_item_ids,
                        top_k=max_k,
                    ),
                    top_k=max_k,
                    popularity=popularity,
                )
            else:
                category_rows += 1
                last_query_item_id = cleaned_query_item_ids[-1]
                query_category = category_by_item_id.get(last_query_item_id, "")
                if query_category:
                    if args.baseline == "category_random":
                        primary_pool = category_random_pools.get(query_category, [])
                        fallback_pool = global_random_pool
                        popularity_for_predictions = None
                    else:
                        primary_pool = category_popular_pools.get(query_category, [])
                        fallback_pool = global_popular_pool
                        popularity_for_predictions = popularity
                    primary_filtered_item_ids = take_topk_excluding(
                        ordered_item_ids=primary_pool,
                        excluded_item_ids=excluded_item_ids,
                        top_k=max_k,
                    )
                    combined_item_ids = take_combined_topk_excluding(
                        primary_item_ids=primary_pool,
                        fallback_item_ids=fallback_pool,
                        excluded_item_ids=excluded_item_ids,
                        top_k=max_k,
                    )
                    if len(primary_filtered_item_ids) == max_k:
                        category_rows_with_matched_category_pool += 1
                    else:
                        category_fallback_short_pool += 1
                    predictions = build_predictions(
                        ordered_item_ids=combined_item_ids,
                        top_k=max_k,
                        popularity=popularity_for_predictions,
                    )
                else:
                    category_fallback_missing_category += 1
                    if args.baseline == "category_random":
                        predictions = build_predictions(
                            ordered_item_ids=take_topk_excluding(
                                ordered_item_ids=global_random_pool,
                                excluded_item_ids=excluded_item_ids,
                                top_k=max_k,
                            ),
                            top_k=max_k,
                        )
                    else:
                        predictions = build_predictions(
                            ordered_item_ids=take_topk_excluding(
                                ordered_item_ids=global_popular_pool,
                                excluded_item_ids=excluded_item_ids,
                                top_k=max_k,
                            ),
                            top_k=max_k,
                            popularity=popularity,
                        )

            if not predictions:
                dropped_no_candidates_after_filter += 1
                continue

            target_rank = find_rank(predictions, target_item_id)
            valid_eval_rows += 1

            for k in ks:
                hit, mrr, ndcg = metric_value(target_rank, k)
                sum_hit[k] += hit
                sum_mrr[k] += mrr
                sum_ndcg[k] += ndcg

            pred_f.write(
                json.dumps(
                    {
                        "user_id": user_id,
                        "query_item_ids": cleaned_query_item_ids,
                        "target_item_id": target_item_id,
                        "target_rank": target_rank,
                        "predictions": predictions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if args.max_query > 0 and valid_eval_rows >= args.max_query:
                break

    metrics: dict[str, dict[str, float]] = {}
    for k in ks:
        denom = float(valid_eval_rows) if valid_eval_rows else 1.0
        metrics[f"@{k}"] = {
            "hit_rate": sum_hit[k] / denom,
            "recall": sum_hit[k] / denom,
            "mrr": sum_mrr[k] / denom,
            "ndcg": sum_ndcg[k] / denom,
        }

    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "generated_at_utc": generated_at,
        "baseline": args.baseline,
        "config": {
            "baseline": args.baseline,
            "items_input": str(items_input.resolve()),
            "interactions_input": str(interactions_input.resolve()),
            "eval_input": str(eval_input.resolve()),
            "output_root": str(Path(args.output_root).resolve()),
            "run_output_dir": str(run_output_dir.resolve()),
            "predictions_output": str(predictions_output.resolve()),
            "report_output": str(report_output.resolve()),
            "info_output": str(info_output.resolve()),
            "topk": ks,
            "max_query": args.max_query,
            "seed": args.seed,
            "rating_threshold": args.rating_threshold,
            "run_id": run_id,
        },
        "input_stats": {
            "eval_rows_total": input_rows_total,
            "eval_parse_error_rows": parse_error_rows,
            "eval_non_object_rows": non_object_rows,
            "eval_rows_missing_user_id": rows_missing_user_id,
            "eval_rows_missing_target_item_id": rows_missing_target_item_id,
            "eval_rows_invalid_query_item_ids": rows_invalid_query_list,
            "items": item_stats,
            "interactions": popularity_stats,
        },
        "filter_stats": {
            "dropped_target_not_in_items": dropped_target_not_in_items,
            "dropped_query_contains_target": dropped_query_contains_target,
            "dropped_query_item_not_in_items": dropped_query_item_not_in_items,
            "dropped_query_empty_after_clean": dropped_query_empty_after_clean,
            "dropped_no_candidates_after_filter": dropped_no_candidates_after_filter,
        },
        "baseline_stats": {
            "category_rows": category_rows,
            "category_rows_with_matched_category_pool": category_rows_with_matched_category_pool,
            "category_fallback_missing_category": category_fallback_missing_category,
            "category_fallback_short_pool": category_fallback_short_pool,
            "candidate_pool_size": pool_size,
        },
        "eval_stats": {
            "valid_eval_rows": valid_eval_rows,
            "kept_rate_over_input": (
                valid_eval_rows / input_rows_total if input_rows_total else 0.0
            ),
        },
        "metrics": metrics,
    }
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    info = {
        "generated_at_utc": generated_at,
        "baseline": args.baseline,
        "run_id": run_id,
        "inputs": {
            "items_input": {
                "path": str(items_input.resolve()),
                "sha256": file_sha256(items_input),
            },
            "interactions_input": {
                "path": str(interactions_input.resolve()),
                "sha256": file_sha256(interactions_input),
            },
            "eval_input": {
                "path": str(eval_input.resolve()),
                "sha256": file_sha256(eval_input),
            },
        },
        "config": report["config"],
        "outputs": {
            "run_output_dir": str(run_output_dir.resolve()),
            "predictions_output": str(predictions_output.resolve()),
            "report_output": str(report_output.resolve()),
            "info_output": str(info_output.resolve()),
        },
    }
    info_output.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
