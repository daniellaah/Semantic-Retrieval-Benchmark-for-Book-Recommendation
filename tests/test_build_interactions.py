from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/data/build_interactions.py"


class BuildInteractionsTests(unittest.TestCase):
    def test_deterministic_and_item_membership(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            items_path = tmp_path / "items.jsonl"
            books_path = tmp_path / "Books.jsonl"
            out1 = tmp_path / "interactions_1.jsonl"
            out2 = tmp_path / "interactions_2.jsonl"
            report1 = tmp_path / "report_1.json"
            report2 = tmp_path / "report_2.json"

            items_rows = [
                {"item_id": "B1", "title": "t1"},
                {"item_id": "B2", "title": "t2"},
            ]
            books_rows = [
                {
                    "user_id": "U1",
                    "parent_asin": "B1",
                    "rating": 5,
                    "timestamp": 1001,
                },
                {
                    "user_id": "U2",
                    "parent_asin": "B2",
                    "rating": 3.5,
                    "timestamp": 1002,
                },
                {
                    "user_id": "U3",
                    "parent_asin": "B3",
                    "rating": 4,
                    "timestamp": 1003,
                },
            ]

            items_path.write_text(
                "\n".join(json.dumps(row) for row in items_rows) + "\n",
                encoding="utf-8",
            )
            books_path.write_text(
                "\n".join(json.dumps(row) for row in books_rows) + "\n",
                encoding="utf-8",
            )

            self._run_script(books_path, items_path, out1, report1)
            self._run_script(books_path, items_path, out2, report2)

            self.assertEqual(out1.read_text(encoding="utf-8"), out2.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in out1.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["item_id"] for row in rows}, {"B1", "B2"})

            report = json.loads(report1.read_text(encoding="utf-8"))
            self.assertEqual(report["books_input_stats"]["rows_item_not_in_items"], 1)
            self.assertEqual(report["output_stats"]["rows_written"], 2)

    def test_non_object_json_row_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            items_path = tmp_path / "items.jsonl"
            books_path = tmp_path / "Books.jsonl"
            output = tmp_path / "interactions.jsonl"
            report = tmp_path / "report.json"

            items_path.write_text(json.dumps({"item_id": "B1", "title": "t1"}) + "\n", encoding="utf-8")
            books_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "user_id": "U1",
                                "parent_asin": "B1",
                                "rating": 5,
                                "timestamp": 1001,
                            }
                        ),
                        "[]",
                        "1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self._run_script(books_path, items_path, output, report)

            rows = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)

            report_obj = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(report_obj["books_input_stats"]["non_object_rows"], 2)
            self.assertEqual(report_obj["output_stats"]["rows_written"], 1)

    def _run_script(
        self,
        books_path: Path,
        items_path: Path,
        output_path: Path,
        report_path: Path,
    ) -> None:
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--books-input",
            str(books_path),
            "--items-input",
            str(items_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--seed",
            "42",
        ]
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
