from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "src/eval/run_eval.py"


class RunEvalTests(unittest.TestCase):
    def test_run_eval_stable_and_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "BAAI__bge-m3" / "exp_bge_tac" / "20260304000000"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            embeddings_path = embedding_dir / "item_embeddings.npy"

            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"

            item_ids = ["A", "B", "C", "D"]
            with item_ids_path.open("w", encoding="utf-8") as f:
                for item_id in item_ids:
                    f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")

            embeddings = np.array(
                [
                    [1.0, 0.0],
                    [0.95, 0.05],
                    [0.9, 0.1],
                    [-1.0, 0.0],
                ],
                dtype=np.float32,
            )
            np.save(embeddings_path, embeddings)

            eval_rows = [
                {
                    "user_id": "U1",
                    "query_item_ids": ["A"],
                    "target_item_id": "B",
                }
            ]
            with eval_input_path.open("w", encoding="utf-8") as f:
                for row in eval_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            run1 = self._run_script(
                eval_input_path,
                embedding_dir,
                output_root,
                eval_run_id="run_1",
            )
            run2 = self._run_script(
                eval_input_path,
                embedding_dir,
                output_root,
                eval_run_id="run_2",
            )

            self.assertEqual(
                run1["predictions"].read_text(encoding="utf-8"),
                run2["predictions"].read_text(encoding="utf-8"),
            )

            pred_rows = [json.loads(x) for x in run1["predictions"].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(pred_rows), 1)
            self.assertEqual(pred_rows[0]["user_id"], "U1")
            self.assertEqual(pred_rows[0]["target_item_id"], "B")
            self.assertEqual(pred_rows[0]["target_rank"], 1)

            predicted_item_ids = [x["item_id"] for x in pred_rows[0]["predictions"]]
            self.assertNotIn("A", predicted_item_ids)
            self.assertEqual(predicted_item_ids[:2], ["B", "C"])

            report = json.loads(run1["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["eval_stats"]["valid_eval_rows"], 1)
            self.assertAlmostEqual(report["metrics"]["@1"]["recall"], 1.0)
            self.assertAlmostEqual(report["metrics"]["@1"]["mrr"], 1.0)
            self.assertAlmostEqual(report["metrics"]["@1"]["ndcg"], 1.0)

            info = json.loads(run1["info"].read_text(encoding="utf-8"))
            self.assertEqual(info["eval_run_id"], "run_1")
            self.assertEqual(info["embedding"]["model_name_guess"], "BAAI/bge-m3")

    def test_drops_rows_with_missing_query_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "M" / "E" / "R"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            embeddings_path = embedding_dir / "item_embeddings.npy"
            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"

            with item_ids_path.open("w", encoding="utf-8") as f:
                f.write('{"item_id":"A"}\n')
                f.write('{"item_id":"B"}\n')
            np.save(
                embeddings_path,
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )

            with eval_input_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "user_id": "U1",
                            "query_item_ids": ["A", "Z"],
                            "target_item_id": "B",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            run = self._run_script(
                eval_input_path,
                embedding_dir,
                output_root,
                eval_run_id="run_x",
            )

            self.assertEqual(run["predictions"].read_text(encoding="utf-8"), "")
            report = json.loads(run["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["filter_stats"]["dropped_query_item_not_in_index"], 1)
            self.assertEqual(report["eval_stats"]["valid_eval_rows"], 0)

    def test_max_query_limits_number_of_evaluated_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            embedding_dir = tmp_path / "emb" / "BAAI__bge-m3" / "exp_bge_tac" / "20260304000000"
            embedding_dir.mkdir(parents=True, exist_ok=True)
            item_ids_path = embedding_dir / "item_ids.jsonl"
            embeddings_path = embedding_dir / "item_embeddings.npy"
            eval_input_path = tmp_path / "eval.jsonl"
            output_root = tmp_path / "outputs" / "eval"

            with item_ids_path.open("w", encoding="utf-8") as f:
                for item_id in ["A", "B", "C", "D"]:
                    f.write(json.dumps({"item_id": item_id}, ensure_ascii=False) + "\n")
            np.save(
                embeddings_path,
                np.array(
                    [
                        [1.0, 0.0],
                        [0.9, 0.1],
                        [0.8, 0.2],
                        [-1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )

            with eval_input_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"user_id": "U1", "query_item_ids": ["A"], "target_item_id": "B"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {"user_id": "U2", "query_item_ids": ["C"], "target_item_id": "B"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            run = self._run_script(
                eval_input=eval_input_path,
                embedding_dir=embedding_dir,
                output_root=output_root,
                eval_run_id="run_limit",
                max_query=1,
            )
            pred_rows = run["predictions"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(pred_rows), 1)
            report = json.loads(run["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["eval_stats"]["valid_eval_rows"], 1)
            self.assertEqual(report["config"]["max_query"], 1)

    def _run_script(
        self,
        eval_input: Path,
        embedding_dir: Path,
        output_root: Path,
        eval_run_id: str,
        max_query: int = 0,
    ) -> dict[str, Path]:
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--eval-input",
            str(eval_input),
            "--embedding-dir",
            str(embedding_dir),
            "--output-root",
            str(output_root),
            "--eval-run-id",
            str(eval_run_id),
            "--topk",
            "1,2",
            "--index-type",
            "flat",
            "--seed",
            "42",
        ]
        if max_query > 0:
            cmd.extend(["--max-query", str(max_query)])
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        run_dir = output_root / eval_run_id
        return {
            "run_dir": run_dir,
            "predictions": run_dir / "predictions.jsonl",
            "report": run_dir / "run_eval_report.json",
            "info": run_dir / "info.json",
        }


if __name__ == "__main__":
    unittest.main()
