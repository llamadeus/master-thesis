from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from tools.analysis import export_principal_v89_paper_results as exporter


PLAN = dict(teacher=dict(configuration_hash="training-hash"),
            validation_teacher=dict(configuration_hash="validation-hash"))


def timing(stage: str, start: float, finish: float, *, components=(), **settings) -> dict:
    origin = datetime(2026, 10, 5, tzinfo=timezone.utc)
    record = dict(stage=stage, status="completed", elapsed_seconds=finish - start,
                  started_at=(origin + timedelta(seconds=start)).isoformat(),
                  finished_at=(origin + timedelta(seconds=finish)).isoformat(),
                  settings=dict(workers=8, **settings))
    if components:
        record["components"] = list(components)
    return record


def generation() -> dict:
    folds = [timing("mpc-teacher-fold", index * 10, (index + 1) * 10,
                    fold_index=index) for index in range(5)]
    return timing("mpc", 0, 100, components=(
        timing("mpc-teacher-training", 0, 60, components=folds,
               output=r"C:\thesis\data\datasets\mpc-teacher-cem\training-hash"),
        timing("mpc-teacher-validation", 60, 90,
               output=r"C:\thesis\data\datasets\mpc-teacher-cem\validation-hash"),
    ))


def save_pipeline(root: Path, mpc_records: list[dict]) -> None:
    directory = root / "pipelines/hanoi-principal-v89/seed-17"
    directory.mkdir(parents=True)
    stages = ("forecasting", "behavior-cloning", "baseline", "baseline-evaluation",
              "fine-tuning", "fine-tuning-evaluation")
    records = [timing(stage, -100, -90) for stage in stages] + mpc_records
    (directory / "stage-timings.json").write_text(json.dumps(dict(records=records)))
    (directory / "status.json").write_text(json.dumps(dict(status="completed", seed=17)))


class PrincipalPaperTimingTests(unittest.TestCase):
    def test_direct_components_do_not_use_parent_remainder(self):
        rows = exporter.mpc_generation_costs(PLAN, generation(), 17, "training-pc")
        self.assertEqual([row["wall_time_seconds"] for row in rows[:2]], [60, 30])
        self.assertEqual([row["fold_id"] for row in rows[2:]], list(range(5)))
        self.assertEqual([row["wall_time_seconds"] for row in rows[2:]], [10] * 5)
        self.assertEqual({row["stage"] for row in rows[2:]}, {"mpc_teacher_fold_generation"})
        self.assertEqual({row["cost_scope"] for row in rows[2:]}, {"per_fold"})
        self.assertEqual(len({row["cost_id"] for row in rows}), 7)

    def test_selects_generation_among_old_interrupted_and_resume_only_runs(self):
        interrupted = deepcopy(generation())
        interrupted["status"] = "interrupted"
        records = [timing("mpc", -200, -150), generation(), interrupted,
                   timing("mpc", 200, 200.05)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_pipeline(root, records)
            stages = exporter.timed_stages(root, 17)
        self.assertEqual(stages["mpc"], [generation()])

    def test_uses_latest_completed_regeneration(self):
        earlier = generation()
        earlier["started_at"] = timing("mpc", -200, -100)["started_at"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_pipeline(root, [earlier, generation()])
            self.assertEqual(exporter.timed_stages(root, 17)["mpc"], [generation()])

    def test_aggregate_only_logs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_pipeline(root, [timing("mpc", 0, 100)])
            with self.assertRaisesRegex(ValueError, "Missing direct MPC"):
                exporter.timed_stages(root, 17)

    def test_missing_or_duplicate_folds_are_rejected(self):
        for indexes in ((0, 1, 2, 3), (0, 1, 2, 3, 3)):
            with self.subTest(indexes=indexes):
                record = generation()
                record["components"][0]["components"] = [
                    timing("mpc-teacher-fold", i * 10, (i + 1) * 10, fold_index=index)
                    for i, index in enumerate(indexes)
                ]
                with self.assertRaisesRegex(ValueError, "Missing or duplicate MPC fold"):
                    exporter.mpc_generation_costs(PLAN, record, 17, "training-pc")

    def test_dataset_identity_mismatch_is_rejected(self):
        record = generation()
        record["components"][1]["settings"]["output"] = "other-dataset"
        with self.assertRaisesRegex(ValueError, "timing dataset mismatch"):
            exporter.mpc_generation_costs(PLAN, record, 17, "training-pc")

    def test_failed_and_nonfinite_component_timings_are_rejected(self):
        for field, value in (("status", "interrupted"), ("elapsed_seconds", float("nan")),
                             ("elapsed_seconds", 0)):
            with self.subTest(field=field, value=value):
                record = generation()
                record["components"][0]["components"][0][field] = value
                with self.assertRaises(ValueError):
                    exporter.mpc_generation_costs(PLAN, record, 17, "training-pc")

    def test_components_cannot_exceed_or_overlap_parent_time(self):
        record = generation()
        record["components"][0]["elapsed_seconds"] = 40
        with self.assertRaisesRegex(ValueError, "exceed parent runtime"):
            exporter.mpc_generation_costs(PLAN, record, 17, "training-pc")
        record = generation()
        record["components"][1]["started_at"] = record["started_at"]
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            exporter.mpc_generation_costs(PLAN, record, 17, "training-pc")


if __name__ == "__main__":
    unittest.main()
