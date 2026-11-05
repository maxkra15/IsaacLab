# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Report contracts: matched workloads, original evidence and safe publication."""

import copy
import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("simulator_report", Path(__file__).parents[1] / "report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def measurement() -> dict:
    """Provide a minimal recorded workload without simulator dependencies."""
    return {
        "schema_version": 1,
        "status": "ok",
        "config": {
            "backend": "genesis",
            "task": "cartpole",
            "mode": "learning",
            "vision": "none",
            "num_envs": 64,
            "dt": 0.005,
            "decimation": 4,
            "solver_iterations": 50,
            "steps": 100,
            "iterations": 200,
            "rollout_steps": 24,
            "evaluation_steps": 250,
            "seed": 0,
        },
        "metrics": {"startup_s": 2.0},
        "provenance": {
            "source_revisions": {"genesis": "revision-a", "isaaclab": "revision-b"},
            "hardware": {"gpu": "example GPU"},
            "harness_sha256": "harness-a",
        },
    }


class TestReport(unittest.TestCase):
    """Protect independent contracts at the measurement-to-publication boundary."""

    def test_workloads_match_before_backends_and_seeds_are_combined(self):
        baseline = measurement()
        peer = copy.deepcopy(baseline)
        peer["config"].update(backend="isaaclab", seed=1)
        runs = [baseline, peer]
        # Each changed axis invalidates a performance comparison, including an
        # unknown future backend option. This is not a Cartesian product.
        for field, value in (
            ("task", "anymal"),
            ("decimation", 2),
            ("solver_iterations", 10),
            ("iterations", 100),
            ("rollout_steps", 12),
            ("evaluation_steps", 100),
            ("new_workload_option", True),
        ):
            changed = copy.deepcopy(baseline)
            changed["config"][field] = value
            runs.append(changed)
        different_gpu = copy.deepcopy(baseline)
        different_gpu["provenance"]["hardware"]["gpu"] = "other GPU"
        runs.append(different_gpu)
        different_harness = copy.deepcopy(baseline)
        different_harness["provenance"]["harness_sha256"] = "harness-b"
        runs.append(different_harness)
        groups = report.group_results(runs)
        self.assertEqual(sorted(len(values) for _, values in groups), [1] * 9 + [2])
        paired = next(values for _, values in groups if len(values) == 2)
        self.assertEqual({run["config"]["backend"] for run in paired}, {"genesis", "isaaclab"})

    def test_raw_measurements_remain_original_and_hash_verifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            original = json.dumps(measurement(), indent=3).encode() + b"\n\n"
            path = source / "run.result.json"
            path.write_bytes(original)
            report.build_report(source, root / "site")
            summary = json.loads((root / "site/data/measurements.json").read_text())
            published = summary["runs"][0]
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual((root / "site" / published["download"]).read_bytes(), original)
            self.assertEqual(published["sha256"], hashlib.sha256(original).hexdigest())

    def test_single_sample_bar_and_whisker_share_the_same_scale(self):
        class Chart(HTMLParser):
            def __init__(self):
                super().__init__()
                self.bars, self.whiskers, self.axes = [], [], []

            def handle_starttag(self, tag, attributes):
                values = dict(attributes)
                if tag == "rect":
                    self.bars.append(values)
                elif tag == "path":
                    self.whiskers.append(values["d"])
                elif tag == "line" and values.get("x1") == values.get("x2"):
                    self.axes.append(float(values["x1"]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            record = measurement()
            record["config"]["mode"] = "physics"
            (source / "run.result.json").write_text(json.dumps(record))
            chart = Chart()
            chart.feed(report.build_report(source, root / "site").read_text())
            self.assertEqual(len(chart.bars), 1)
            bar = chart.bars[0]
            # A single observation has equal minimum, median and maximum.
            endpoint = float(re.search(r"H\s*([\d.]+)", chart.whiskers[0]).group(1))
            self.assertAlmostEqual(float(bar["x"]) + float(bar["width"]), endpoint)
            self.assertEqual(float(bar["x"]), chart.axes[0])

    def test_failed_inputs_do_not_publish_active_html_or_unrelated_files(self):
        class Document(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tags = []
                self.links = []

            def handle_starttag(self, tag, attributes):
                self.tags.append(tag)
                self.links.extend(value for name, value in attributes if name in {"src", "href"})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            (root / "private.png").write_bytes(b"private unrelated content")
            failed = measurement()
            failed.update(
                status="failed",
                error='<script src="https://example.com/exfiltrate"></script>',
                camera_sample="../private.png",
                metrics={},
            )
            (source / "run.result.json").write_text(json.dumps(failed))
            page = report.build_report(source, root / "site")
            document = Document()
            document.feed(page.read_text())
            self.assertNotIn("script", document.tags)
            self.assertNotIn("svg", document.tags)
            self.assertFalse(list((root / "site").rglob("*.png")))
            self.assertNotIn("https://example.com/exfiltrate", document.links)
            summary = json.loads((root / "site/data/measurements.json").read_text())
            self.assertEqual(summary["status_counts"], {"failed": 1})


if __name__ == "__main__":
    unittest.main()
