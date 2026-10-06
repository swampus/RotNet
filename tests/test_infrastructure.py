"""Dependency-free infrastructure checks; not a substitute for torch tests."""

import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rotnet.experiment import main, parser, source_hashes
from rotnet.reporting import render_report, write_json, write_predictions, write_report


class InfrastructureTests(unittest.TestCase):
    def pending_metrics(self):
        config = vars(parser().parse_args([]))
        return {"run_name": "test-fixture", "status": "blocked", "config": config,
                "models": {}, "command": "python scripts/run_experiment.py --device cpu",
                "error": "test fixture: missing dependencies"}

    def test_cli_defaults_and_fashion_flag(self):
        args = parser().parse_args(["--dataset", "fashion-mnist", "--model", "rotation", "--seed", "17"])
        self.assertEqual(args.dataset, "fashion-mnist")
        self.assertEqual(args.seed, 17)
        self.assertEqual(args.hidden_dim, 256)
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.thresholds, [0.7, 0.8, 0.9, 0.95, 0.99])

    def test_pending_report_has_no_fake_accuracy(self):
        report = render_report(self.pending_metrics())
        self.assertIn("Runtime results are pending", report)
        self.assertNotIn("0.00%", report)
        self.assertNotIn("| dense |", report)

    def test_json_and_prediction_schema_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_json(path / "metrics.json", {"status": "blocked", "models": {}})
            self.assertEqual(json.loads((path / "metrics.json").read_text())["models"], {})
            write_predictions(path / "predictions.csv", [])
            with (path / "predictions.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertTrue({"sample_id", "target", "prediction", "correct", "confidence",
                                 "exit_stage", "threshold", "logit_0", "logit_9"}.issubset(reader.fieldnames))
                self.assertEqual(list(reader), [])

    def test_report_written_to_run_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_report(path, self.pending_metrics(), path / "summary.md")
            self.assertEqual((path / "RESULTS.md").read_text(), (path / "summary.md").read_text())

    def test_hashes_cover_models_and_tests(self):
        hashes = source_hashes()
        self.assertIn("rotnet/models/rotation.py", hashes)
        self.assertIn("tests/test_adaptive.py", hashes)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_dependency_failure_records_pending_run_and_nonzero_status(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("rotnet.experiment.dependency_check", return_value={"torch": "test fixture: unavailable"}):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    code = main(["--results-dir", directory, "--run-name", "blocked-fixture", "--no-update-summary"])
            path = Path(directory) / "blocked-fixture"
            metrics = json.loads((path / "metrics.json").read_text())
            self.assertEqual(code, 2)
            self.assertEqual(metrics["status"], "blocked")
            self.assertEqual(metrics["models"], {})
            self.assertEqual(metrics["config"]["thresholds"], [0.7, 0.8, 0.9, 0.95, 0.99])
            self.assertTrue((path / "config.json").exists())
            self.assertTrue((path / "RESULTS.md").exists())
            self.assertFalse((path / "plots").exists())

    def test_run_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing"
            path.mkdir()
            sentinel = path / "sentinel.txt"
            sentinel.write_text("keep")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--results-dir", directory, "--run-name", "existing", "--no-update-summary"])
            self.assertEqual(sentinel.read_text(), "keep")

    def test_run_name_cannot_escape_results_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--results-dir", directory, "--run-name", "../escape", "--no-update-summary"])
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
