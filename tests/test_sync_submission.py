"""Submission selection, copy fidelity, and preservation of local artifacts."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.submission import sync_submission as sync


class SubmissionSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "source"
        self.target = Path(self.temporary.name) / "submission"
        self.source.mkdir()
        subprocess.run(["git", "init", "--quiet", str(self.source)], check=True)
        self.write_source(".gitignore", "data/\n*.rpt\n")
        self.write_source("README.md", "Current reproduction guide\n")
        self.write_source("training/final.py", "final = True\n")
        self.write_source("training/obsolete.py", "obsolete = True\n")
        self.write_source("packages/trainer/model.py", "model = True\n")
        self.write_source("packages/trainer/README.md", "Unselected package documentation\n")
        self.write_source("paper/main.typ", "Current paper\n")
        self.write_source("paper/report.rpt", "Ignored simulator output\n")
        self.write_source("data/large-output.txt", "Excluded generated data\n")
        self.write_selection()

    def write_source(self, relative, text):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def write_target(self, relative, text):
        path = self.target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def write_selection(self, extra_files=()):
        selection = {
            "files": [".gitignore", "README.md", "training/final.py", *extra_files],
            "exclude_components": ["AGENTS.md", "CLAUDE.md", "SKILL.md", "skills"],
            "trees": [{"path": "packages/trainer", "suffixes": [".py"]},
                      {"path": "paper"}],
        }
        self.write_source(str(sync.SELECTION_FILE), json.dumps(selection))

    def test_copy_is_exact_and_second_sync_has_no_changes(self):
        plan = sync.build_plan(self.source, self.target)
        self.assertEqual(set(plan.selected), {
            Path(".gitignore"), Path("README.md"), Path("training/final.py"),
            Path("packages/trainer/model.py"), Path("paper/main.typ"),
        })
        sync.synchronize(plan)
        for relative in plan.selected:
            self.assertEqual((self.source / relative).read_bytes(),
                             (self.target / relative).read_bytes())
        self.assertEqual(sync.build_plan(self.source, self.target).copies, ())
        self.assertFalse((self.target / "training/obsolete.py").exists())
        self.assertFalse((self.target / "paper/report.rpt").exists())

    def test_prune_keeps_git_hidden_and_ignored_contents(self):
        protected = {
            ".git/HEAD": "Existing submission history\n",
            ".venv/local.txt": "Existing environment\n",
            ".local-notes": "Private local note\n",
            "data/result.txt": "Existing experiment output\n",
            "obsolete/report.rpt": "Ignored nested simulator output\n",
        }
        for relative, text in protected.items():
            self.write_target(relative, text)
        self.write_target("training/obsolete.py", "Old experiment\n")
        self.write_target("obsolete/old.txt", "Unselected file\n")
        self.write_target("empty/.keep", "Hidden sentinel\n")
        plan = sync.build_plan(self.source, self.target, prune=True)
        self.assertEqual(set(plan.removals), {
            Path("training/obsolete.py"), Path("obsolete/old.txt"),
        })
        sync.synchronize(plan)
        for relative, text in protected.items():
            self.assertEqual((self.target / relative).read_text(), text)
        self.assertTrue((self.target / "empty/.keep").exists())
        self.assertFalse((self.target / "training/obsolete.py").exists())

    def test_destination_ignore_rules_survive_replacement(self):
        self.write_target(".gitignore", "locally-ignored/\n")
        self.write_target("locally-ignored/measurements.txt", "Preserved local measurements\n")
        self.write_target("other-old-file.txt", "Remove this\n")
        sync.synchronize(sync.build_plan(self.source, self.target, prune=True))
        self.assertEqual((self.target / ".gitignore").read_bytes(),
                         (self.source / ".gitignore").read_bytes())
        self.assertTrue((self.target / "locally-ignored/measurements.txt").exists())
        self.assertFalse((self.target / "other-old-file.txt").exists())

    def test_nested_ignore_rules_and_negation_are_respected(self):
        self.write_source("paper/.gitignore", "*.tmp\n!important.tmp\n")
        self.write_source("paper/temporary.tmp", "Ignored output\n")
        self.write_source("paper/important.tmp", "Included source\n")
        plan = sync.build_plan(self.source, self.target, prune=True)
        self.assertNotIn(Path("paper/temporary.tmp"), plan.selected)
        self.assertIn(Path("paper/important.tmp"), plan.selected)
        self.assertIn(Path("paper/.gitignore"), plan.selected)
        self.write_target("paper/temporary.tmp", "Existing ignored output\n")
        sync.synchronize(sync.build_plan(self.source, self.target, prune=True))
        self.assertEqual((self.target / "paper/temporary.tmp").read_text(),
                         "Existing ignored output\n")

    def test_copy_only_does_not_remove_extra_files(self):
        self.write_target("training/obsolete.py", "Existing extra file\n")
        plan = sync.build_plan(self.source, self.target)
        self.assertEqual(plan.removals, ())
        sync.synchronize(plan)
        self.assertTrue((self.target / "training/obsolete.py").exists())

    def test_ai_instructions_and_skill_contents_are_excluded(self):
        files = ("AGENTS.md", "paper/agents.md", "paper/CLAUDE.md",
                 "paper/SKILL.md", "paper/skills/writer/SKILL.md",
                 "paper/skills/writer/helper.py")
        for relative in files:
            self.write_source(relative, "Local AI-tool instructions\n")
        self.write_selection(["AGENTS.md"])
        plan = sync.build_plan(self.source, self.target)
        self.assertFalse(set(map(Path, files)) & set(plan.selected))
        self.assertIn(Path("paper/main.typ"), plan.selected)

    def test_prune_removes_previously_copied_ai_files_from_target_only(self):
        files = ("AGENTS.md", "paper/AGENTS.md", "paper/skills/writer/SKILL.md",
                 "paper/skills/writer/helper.py")
        for relative in files:
            self.write_source(relative, "Source AI-tool instructions\n")
            self.write_target(relative, "Previously copied instructions\n")
        plan = sync.build_plan(self.source, self.target, prune=True)
        self.assertEqual(set(plan.removals), set(map(Path, files)))
        sync.synchronize(plan)
        for relative in files:
            self.assertTrue((self.source / relative).is_file())
            self.assertFalse((self.target / relative).exists())
        self.assertFalse((self.target / "paper/skills").exists())

    def test_dry_run_creates_no_destination(self):
        with patch.object(sync, "PROJECT_ROOT", self.source), \
                patch.object(sync, "compile_paper") as compile_paper:
            result = sync.main(["--target", str(self.target), "--dry-run", "--prune"])
        self.assertEqual(result, 0)
        self.assertFalse(self.target.exists())
        compile_paper.assert_not_called()

    def test_main_compiles_after_copy_and_prune(self):
        self.write_target("obsolete.txt", "Old source\n")

        def compile_submission(target):
            self.assertEqual((target / "paper/main.typ").read_bytes(),
                             (self.source / "paper/main.typ").read_bytes())
            self.assertFalse((target / "obsolete.txt").exists())

        with patch.object(sync, "PROJECT_ROOT", self.source), \
                patch.object(sync, "compile_paper", side_effect=compile_submission) as compile_paper:
            result = sync.main(["--target", str(self.target), "--prune"])
        self.assertEqual(result, 0)
        compile_paper.assert_called_once_with(self.target.resolve())

    def test_compilation_failure_returns_error(self):
        with patch.object(sync, "PROJECT_ROOT", self.source), \
                patch.object(sync, "compile_paper",
                             side_effect=subprocess.CalledProcessError(1, ["typst", "compile"])):
            result = sync.main(["--target", str(self.target)])
        self.assertEqual(result, 2)

    def test_pdf_symlink_cannot_overwrite_outside_file(self):
        outside = Path(self.temporary.name) / "outside.pdf"
        outside.write_bytes(b"Preserved PDF")
        (self.target / "output").mkdir(parents=True)
        (self.target / "output/thesis.pdf").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symlink in the submission PDF path"):
            sync.compile_paper(self.target)
        self.assertEqual(outside.read_bytes(), b"Preserved PDF")

    def test_compile_paper_passes_deterministic_input_to_typst(self):
        with patch.object(sync.subprocess, "run") as run:
            sync.compile_paper(self.target)
        run.assert_called_once_with([
            str(sync.TYPST), "compile", "--input", "figure-boundary=true",
            "--input", "deterministic=true",
            "paper/main.typ", "output/thesis.pdf",
        ], cwd=self.target, check=True)
        self.assertTrue((self.target / "output").is_dir())

    def test_missing_required_source_stops_before_writing(self):
        self.write_selection(["training/missing.py"])
        with self.assertRaisesRegex(ValueError, "Missing or symlinked"):
            sync.build_plan(self.source, self.target, prune=True)
        self.assertFalse(self.target.exists())

    def test_ignored_required_source_is_rejected(self):
        self.write_selection(["data/large-output.txt"])
        with self.assertRaisesRegex(ValueError, "Required submission files are ignored"):
            sync.build_plan(self.source, self.target)

    def test_overlapping_source_and_target_are_rejected(self):
        for target in (self.source, self.source / "submission", self.source.parent):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "non-overlapping"):
                sync.build_plan(self.source, target, prune=True)

    def test_symlinked_destination_cannot_overwrite_outside_file(self):
        outside = Path(self.temporary.name) / "outside.txt"
        outside.write_text("Preserved outside file\n")
        self.target.mkdir()
        (self.target / "README.md").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symlink in a submission destination"):
            sync.build_plan(self.source, self.target)
        self.assertEqual(outside.read_text(), "Preserved outside file\n")

    def test_protected_selected_file_conflict_is_reported(self):
        self.write_target(".gitignore", "README.md\n")
        self.write_target("README.md", "Locally protected guide\n")
        with self.assertRaisesRegex(ValueError, "protected by the target ignore rules"):
            sync.build_plan(self.source, self.target, prune=True)
        self.assertEqual((self.target / "README.md").read_text(), "Locally protected guide\n")

    def test_directory_conflict_preserves_ignored_children(self):
        self.write_target("README.md/data/results.txt", "Preserved data\n")
        with self.assertRaisesRegex(ValueError, "destination directory occupies"):
            sync.build_plan(self.source, self.target, prune=True)
        self.assertTrue((self.target / "README.md/data/results.txt").exists())


if __name__ == "__main__":
    unittest.main()
