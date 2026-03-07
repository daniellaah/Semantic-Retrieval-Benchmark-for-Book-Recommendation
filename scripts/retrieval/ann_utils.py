from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


@dataclass(frozen=True)
class Neighbor:
    rank: int
    item_id: str
    score: float


def load_item_ids(item_ids_path: Path) -> tuple[list[str], dict[str, int]]:
    item_ids: list[str] = []
    item_id_to_row: dict[str, int] = {}
    with item_ids_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                raise ValueError(f"Empty line found in {item_ids_path}:{lineno}")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {item_ids_path}:{lineno}: {e}") from e
            if not isinstance(raw, dict):
                raise ValueError(f"Expected object JSON in {item_ids_path}:{lineno}")
            item_id = str(raw.get("item_id", "")).strip()
            if not item_id:
                raise ValueError(f"Missing item_id in {item_ids_path}:{lineno}")
            if item_id in item_id_to_row:
                raise ValueError(f"Duplicate item_id '{item_id}' in {item_ids_path}:{lineno}")
            row = len(item_ids)
            item_ids.append(item_id)
            item_id_to_row[item_id] = row
    return item_ids, item_id_to_row


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings matrix, got shape={embeddings.shape}")
    return np.asarray(embeddings, dtype=np.float32)


def normalize_rows_inplace(matrix: np.ndarray) -> None:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    matrix /= norms


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "hnsw",
    hnsw_m: int = 32,
    hnsw_ef_search: int = 64,
    hnsw_ef_construction: int = 200,
    normalize: bool = True,
) -> tuple[faiss.Index, np.ndarray]:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings matrix, got shape={embeddings.shape}")
    vectors = np.array(embeddings, dtype=np.float32, copy=True, order="C")
    if normalize:
        normalize_rows_inplace(vectors)
    dim = int(vectors.shape[1])
    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = int(hnsw_ef_search)
        index.hnsw.efConstruction = int(hnsw_ef_construction)
    elif index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError(f"Unsupported index_type='{index_type}', expected one of: hnsw, flat")
    index.add(vectors)
    return index, vectors


def search_topk_by_item_id(
    *,
    index: faiss.Index,
    vectors: np.ndarray,
    item_ids: list[str],
    item_id_to_row: dict[str, int],
    query_item_id: str,
    top_k: int,
) -> list[Neighbor]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    query_row = item_id_to_row.get(query_item_id)
    if query_row is None:
        raise KeyError(f"query_item_id not found: {query_item_id}")
    query_vec = vectors[query_row : query_row + 1]
    scores, row_ids = index.search(query_vec, top_k + 1)
    neighbors: list[Neighbor] = []
    for row_id, score in zip(row_ids[0].tolist(), scores[0].tolist()):
        if row_id < 0:
            continue
        if row_id == query_row:
            continue
        neighbors.append(
            Neighbor(
                rank=len(neighbors) + 1,
                item_id=item_ids[row_id],
                score=float(score),
            )
        )
        if len(neighbors) >= top_k:
            break
    return neighbors


def fetch_items_by_ids(items_path: Path, wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    if not wanted_ids:
        return found
    with items_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {items_path}:{lineno}: {e}") from e
            if not isinstance(raw, dict):
                raise ValueError(f"Expected object JSON in {items_path}:{lineno}")
            item_id = str(raw.get("item_id", "")).strip()
            if not item_id:
                continue
            if item_id in wanted_ids and item_id not in found:
                found[item_id] = raw
                if len(found) == len(wanted_ids):
                    break
    return found

