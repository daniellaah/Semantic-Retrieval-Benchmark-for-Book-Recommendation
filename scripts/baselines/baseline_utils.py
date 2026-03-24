#!/usr/bin/env python3
"""Shared helpers for retrieval baselines."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_topk_list(value: str) -> list[int]:
    tokens = [x.strip() for x in value.split(",") if x.strip()]
    if not tokens:
        raise ValueError("topk list cannot be empty")
    ks: list[int] = []
    for token in tokens:
        k = int(token)
        if k <= 0:
            raise ValueError("topk values must be > 0")
        ks.append(k)
    return sorted(set(ks))


def generate_run_id_local() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_rank(predictions: list[dict[str, Any]], target_item_id: str) -> int | None:
    for pred in predictions:
        if pred["item_id"] == target_item_id:
            return int(pred["rank"])
    return None


def metric_value(rank: int | None, k: int) -> tuple[float, float, float]:
    if rank is None or rank > k:
        return 0.0, 0.0, 0.0
    recall = 1.0
    mrr = 1.0 / float(rank)
    ndcg = 1.0 / math.log2(float(rank) + 1.0)
    return recall, mrr, ndcg


def load_items_metadata(
    items_input: Path,
) -> tuple[list[str], dict[str, str], dict[str, list[str]], dict[str, int]]:
    item_ids: list[str] = []
    category_by_item_id: dict[str, str] = {}
    item_ids_by_category: dict[str, list[str]] = {}
    stats = {
        "rows_total": 0,
        "parse_error_rows": 0,
        "non_object_rows": 0,
        "rows_missing_item_id": 0,
        "duplicate_item_id_rows": 0,
        "rows_valid": 0,
    }
    seen_item_ids: set[str] = set()

    with items_input.open("r", encoding="utf-8") as f:
        for line in f:
            stats["rows_total"] += 1
            line = line.strip()
            if not line:
                stats["parse_error_rows"] += 1
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_error_rows"] += 1
                continue
            if not isinstance(raw, dict):
                stats["non_object_rows"] += 1
                continue

            item_id = normalize_text(raw.get("item_id"))
            if not item_id:
                stats["rows_missing_item_id"] += 1
                continue
            if item_id in seen_item_ids:
                stats["duplicate_item_id_rows"] += 1
                continue

            seen_item_ids.add(item_id)
            stats["rows_valid"] += 1
            item_ids.append(item_id)
            category = normalize_text(raw.get("categories"))
            category_by_item_id[item_id] = category
            if category:
                item_ids_by_category.setdefault(category, []).append(item_id)

    stats["unique_item_ids"] = len(item_ids)
    return item_ids, category_by_item_id, item_ids_by_category, stats


def parse_rating(value: Any) -> float:
    if value is None:
        raise ValueError("missing rating")
    rating = float(value)
    if rating != rating:
        raise ValueError("invalid rating nan")
    return rating


def load_positive_popularity(
    interactions_input: Path,
    valid_item_ids: set[str],
    rating_threshold: float,
) -> tuple[dict[str, int], dict[str, int]]:
    popularity: dict[str, int] = {}
    stats = {
        "rows_total": 0,
        "parse_error_rows": 0,
        "non_object_rows": 0,
        "rows_missing_item_id": 0,
        "rows_missing_rating": 0,
        "rows_invalid_rating": 0,
        "rows_item_not_in_items": 0,
        "rows_below_rating_threshold": 0,
        "positive_rows_counted": 0,
    }

    with interactions_input.open("r", encoding="utf-8") as f:
        for line in f:
            stats["rows_total"] += 1
            line = line.strip()
            if not line:
                stats["parse_error_rows"] += 1
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_error_rows"] += 1
                continue
            if not isinstance(raw, dict):
                stats["non_object_rows"] += 1
                continue

            item_id = normalize_text(raw.get("item_id"))
            raw_rating = raw.get("rating")
            if not item_id:
                stats["rows_missing_item_id"] += 1
                continue
            if raw_rating is None:
                stats["rows_missing_rating"] += 1
                continue
            try:
                rating = parse_rating(raw_rating)
            except (TypeError, ValueError):
                stats["rows_invalid_rating"] += 1
                continue
            if rating < rating_threshold:
                stats["rows_below_rating_threshold"] += 1
                continue
            if item_id not in valid_item_ids:
                stats["rows_item_not_in_items"] += 1
                continue

            stats["positive_rows_counted"] += 1
            popularity[item_id] = popularity.get(item_id, 0) + 1

    stats["unique_positive_items"] = len(popularity)
    return popularity, stats
