from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src/retrieval/ann_utils.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ann_utils", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load ann_utils module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = load_module()


class AnnUtilsTests(unittest.TestCase):
    def test_search_topk_excludes_query_item(self) -> None:
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        item_ids = ["A", "B", "C", "D"]
        item_id_to_row = {x: i for i, x in enumerate(item_ids)}
        index, vectors = mod.build_faiss_index(
            embeddings=embeddings,
            index_type="flat",
            normalize=True,
        )
        neighbors = mod.search_topk_by_item_id(
            index=index,
            vectors=vectors,
            item_ids=item_ids,
            item_id_to_row=item_id_to_row,
            query_item_id="A",
            top_k=2,
        )
        self.assertEqual([x.item_id for x in neighbors], ["B", "C"])

    def test_fetch_items_by_ids_only_returns_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            items_path = Path(tmp_dir) / "items.jsonl"
            rows = [
                {"item_id": "A", "title": "tA"},
                {"item_id": "B", "title": "tB"},
                {"item_id": "C", "title": "tC"},
            ]
            with items_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            found = mod.fetch_items_by_ids(items_path, {"B", "C"})
            self.assertEqual(set(found.keys()), {"B", "C"})
            self.assertEqual(found["B"]["title"], "tB")
            self.assertEqual(found["C"]["title"], "tC")

    def test_load_item_ids_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            item_ids_path = Path(tmp_dir) / "item_ids.jsonl"
            with item_ids_path.open("w", encoding="utf-8") as f:
                f.write('{"item_id":"A"}\n')
                f.write('{"item_id":"A"}\n')
            with self.assertRaisesRegex(ValueError, "Duplicate item_id"):
                mod.load_item_ids(item_ids_path)


if __name__ == "__main__":
    unittest.main()
