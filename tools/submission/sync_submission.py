#!/usr/bin/env python3
"""Copy the selected thesis sources and compile with a fixed document date.

Git supplies the ignore rules. Pruning removes only unselected, non-ignored
files. Hidden destination contents, including .git and virtual environments,
are preserved. The selected root .gitignore is copied with the other sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = Path.home() / "Code" / "master-thesis"
SELECTION_FILE = Path("tools/submission/submission_files.json")
TYPST = Path.home() / "Code" / "typst" / "target" / "release" / "typst"


def git_output(arguments: list[str], *, cwd: Path, stdin: bytes | None = None,
               accepted_codes: tuple[int, ...] = (0,)) -> bytes:
    result = subprocess.run(["git", *arguments], cwd=cwd, input=stdin,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode not in accepted_codes:
        raise RuntimeError(result.stderr.decode(errors="replace").strip()
                           or "Git could not read the project or ignore rules")
    return result.stdout


class IgnoreRules:
    def __init__(self, source: Path):
        self.git_dir = git_output(["rev-parse", "--absolute-git-dir"], cwd=source).decode().strip()

    def ignored(self, root: Path, paths: list[str]) -> set[Path]:
        if not paths or not root.exists():
            return set()
        output = git_output([
            f"--git-dir={self.git_dir}", f"--work-tree={root}",
            "check-ignore", "--no-index", "--stdin", "-z",
        ], cwd=root, stdin=b"".join(os.fsencode(path) + b"\0" for path in paths),
            accepted_codes=(0, 1))
        return {Path(os.fsdecode(path).rstrip("/")) for path in output.split(b"\0") if path}


def relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"..", ".git"} for part in path.parts):
        raise ValueError(f"Invalid submission path: {value!r}")
    return path


def hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def has_symlink(root: Path, relative: Path) -> bool:
    return any(root.joinpath(*relative.parts[:index]).is_symlink()
               for index in range(1, len(relative.parts) + 1))


def excluded(path: Path, components: frozenset[str]) -> bool:
    return any(part.casefold() in components for part in path.parts)


def selected_files(source: Path, rules: IgnoreRules) -> tuple[Path, ...]:
    configuration = json.loads((source / SELECTION_FILE).read_text(encoding="utf-8"))
    exclusions = frozenset(value.casefold() for value in configuration.get("exclude_components", []))
    required = {relative_path(value) for value in configuration["files"]}
    required = {path for path in required if not excluded(path, exclusions)}
    for relative in required:
        if not (source / relative).is_file() or has_symlink(source, relative):
            raise ValueError(f"Missing or symlinked submission source: {relative}")
    ignored_required = rules.ignored(source, [path.as_posix() for path in required])
    if ignored_required:
        raise ValueError(f"Required submission files are ignored: {sorted(map(str, ignored_required))}")

    selected = set(required)
    for tree in configuration["trees"]:
        directory = relative_path(tree["path"])
        if excluded(directory, exclusions):
            continue
        if not (source / directory).is_dir() or has_symlink(source, directory):
            raise ValueError(f"Missing or symlinked submission directory: {directory}")
        candidates = git_output([
            "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
            directory.as_posix(),
        ], cwd=source).split(b"\0")
        candidates = [relative_path(os.fsdecode(value)) for value in candidates if value]
        ignored = rules.ignored(source, [path.as_posix() for path in candidates])
        for relative in candidates:
            if excluded(relative, exclusions):
                continue
            is_ignore_file = relative.name == ".gitignore" and not hidden(relative.parent)
            if (hidden(relative) and not is_ignore_file) or relative in ignored or not (source / relative).is_file():
                continue
            if tree.get("suffixes") and relative.suffix not in tree["suffixes"]:
                continue
            if has_symlink(source, relative):
                raise ValueError(f"Symlinked submission source: {relative}")
            selected.add(relative)
    return tuple(sorted(selected))


def digest(path: Path) -> bytes:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def destination_contents(source: Path, target: Path, rules: IgnoreRules) -> tuple[list[Path], list[Path]]:
    files, directories = [], []
    if not target.exists():
        return files, directories
    for current, names, filenames in os.walk(target, followlinks=False):
        parent = Path(current).relative_to(target)
        children = [parent / name for name in names + filenames]
        requested = [path.as_posix() + ("/" if (target / path).is_dir() else "")
                     for path in children]
        # Preserve contents ignored by either the old target rules or the
        # source rules that will be installed by this synchronization.
        ignored = rules.ignored(target, requested) | rules.ignored(source, requested)
        for name in list(names):
            relative = parent / name
            if hidden(relative) or relative in ignored:
                names.remove(name)
            elif (target / relative).is_symlink():
                names.remove(name)
                files.append(relative)
            else:
                directories.append(relative)
        files.extend(parent / name for name in filenames
                     if not hidden(parent / name) and parent / name not in ignored)
    return files, directories


@dataclass(frozen=True)
class SyncPlan:
    source: Path
    target: Path
    selected: tuple[Path, ...]
    copies: tuple[Path, ...]
    removals: tuple[Path, ...]
    directories: tuple[Path, ...]


def build_plan(source: Path, target: Path, *, prune: bool = False) -> SyncPlan:
    if target.is_symlink():
        raise ValueError("The submission directory must not be a symlink")
    source, target = source.resolve(), target.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and target must be separate, non-overlapping directories")
    if target.exists() and not target.is_dir():
        raise ValueError("The submission target must be a directory")
    rules = IgnoreRules(source)
    selected = selected_files(source, rules)
    ignored = rules.ignored(target, [path.as_posix() for path in selected])
    copies = []
    for relative in selected:
        destination = target / relative
        if has_symlink(target, relative):
            raise ValueError(f"Symlink in a submission destination path: {relative}")
        if relative in ignored and destination.exists():
            raise ValueError(f"Selected file is protected by the target ignore rules: {relative}")
        if destination.exists() and not destination.is_file():
            raise ValueError(f"A destination directory occupies a selected file path: {relative}")
        for parent in destination.parents:
            if parent == target:
                break
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"A destination file occupies a required directory: {parent}")
        if not destination.exists() or digest(source / relative) != digest(destination):
            copies.append(relative)
    files, directories = destination_contents(source, target, rules) if prune else ([], [])
    return SyncPlan(source, target, selected, tuple(copies),
                    tuple(sorted(set(files) - set(selected))),
                    tuple(sorted(directories, key=lambda path: (len(path.parts), str(path)), reverse=True)))


def synchronize(plan: SyncPlan) -> None:
    plan.target.mkdir(parents=True, exist_ok=True)
    for relative in plan.copies:
        destination = plan.target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".submission-", dir=destination.parent)
        os.close(descriptor)
        try:
            shutil.copy2(plan.source / relative, temporary)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    # Remove individual unselected files, never whole directory trees.
    for relative in plan.removals:
        (plan.target / relative).unlink()
    for relative in plan.directories:
        directory = plan.target / relative
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def compile_paper(target: Path) -> None:
    output = Path("output/thesis.pdf")
    if has_symlink(target, output):
        raise ValueError(f"Symlink in the submission PDF path: {output}")
    (target / output).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(TYPST), "compile", "--input", "figure-boundary=true",
        "--input", "deterministic=true",
        "paper/main.typ", str(output),
    ], cwd=target, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--dry-run", action="store_true", help="show changes without writing")
    parser.add_argument("--prune", action="store_true",
                        help="remove unselected files, preserving hidden and ignored contents")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(PROJECT_ROOT, args.target, prune=args.prune)
        for relative in plan.copies:
            print(f"COPY   {relative}")
        for relative in plan.removals:
            print(f"REMOVE {relative}")
        print(f"Selected {len(plan.selected)} files: {len(plan.copies)} to copy, "
              f"{len(plan.removals)} to remove. Target: {plan.target}")
        print(f"COMPILE paper/main.typ -> {plan.target / 'output/thesis.pdf'}", flush=True)
        if not args.dry_run:
            synchronize(plan)
            compile_paper(plan.target)
            print("Submission synchronized and paper compiled.")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError,
            subprocess.CalledProcessError) as exc:
        print(f"Submission sync failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
