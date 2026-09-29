#!/usr/bin/env python3
"""Build the submission zip, and the branch that shows what is in it.

    uv run python scripts/make_submission.py outputs/<video stem>     # the zip
    uv run python scripts/make_submission.py --branch submission      # the branch

The zip is the committed code, plus the results of one run.

The code comes from `git archive`, so exactly what is committed ships, minus what
`.gitattributes` marks `export-ignore` (eval/, scripts/, docs/, the label-reading tests).
`answer.json` and `annotated.mp4` come from the results directory of a `run.py run`.

The zip is then checked the way a reviewer would use it: unzipped into a fresh folder, its
answer held to the task's schema, its video opened, `uv lock --check` run, its own tests
run from that folder, and `run.py --help` called. The script exits non-zero if any of that
fails, so a zip that does not pass is never handed in by accident.

The branch holds the same filtered code as the zip, without the results, so what is
submitted can be read on GitHub without unpacking anything. It is generated: each run adds
a snapshot commit when the code changed, and it is never edited by hand.

This is development tooling and is not itself part of the submission.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PHASES = ("digging", "hauling", "dumping", "swinging")
REQUIRED = ("run.py", "pyproject.toml", "uv.lock", "README.md", "answer.json", "annotated.mp4")
NOT_SHIPPED = ("eval/", "scripts/", "docs/", ".github/", "outputs/", "weights/", "data/")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def check_answer(path: Path) -> dict:
    """The task's schema, key for key. Raises ValueError with what is wrong."""
    record = json.loads(path.read_text())
    top = ["cycle_count", "average_cycle_duration_seconds", "average_phase_duration_seconds"]
    if list(record) != top:
        raise ValueError(f"answer.json keys are {list(record)}, expected {top}")
    if not isinstance(record["cycle_count"], int) or record["cycle_count"] < 0:
        raise ValueError("cycle_count must be a non-negative integer")
    if list(record["average_phase_duration_seconds"]) != list(PHASES):
        raise ValueError(f"phase keys are {list(record['average_phase_duration_seconds'])}")
    numbers = [
        record["average_cycle_duration_seconds"],
        *record["average_phase_duration_seconds"].values(),
    ]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in numbers):
        raise ValueError("every duration must be a finite, non-negative number")
    if record["cycle_count"] and not all(v > 0 for v in numbers):
        raise ValueError("cycles were counted but a duration is zero")
    return record


def check_video(path: Path) -> tuple[int, int, float]:
    """Open the video and read its first and last frame. Returns (frames, width, height)."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames <= 0:
            raise ValueError("the annotated video has no frames")
        ok, first = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, frames - 1)
        ok_last, _ = capture.read()
        if not (ok and ok_last):
            raise ValueError("the annotated video cannot be read to its end")
        return frames, first.shape[1], capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()


def build(results: Path, out: Path, tree: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    subprocess.run(
        ["git", "archive", "--format=zip", "-o", str(out), tree], cwd=REPO, check=True
    )
    with zipfile.ZipFile(out, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.write(results / "answer.json", "answer.json")
        archive.write(results / "annotated.mp4", "annotated.mp4")


def verify(out: Path) -> list[str]:
    """Unzip into a fresh folder and use it as a reviewer would. Returns the failures."""
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(out) as archive:
            archive.extractall(root)
            names = archive.namelist()

        for name in REQUIRED:
            if name not in names:
                failures.append(f"missing from the zip: {name}")
        for name in names:
            if name.startswith(NOT_SHIPPED):
                failures.append(f"should not ship: {name}")
        videos = [n for n in names if n.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        if videos != ["annotated.mp4"]:
            failures.append(
                f"the task asks for exactly one video, annotated.mp4; found {videos}"
            )

        env = {**os.environ, "PYTHONPATH": str(root / "src")}
        checks = {
            "uv lock --check": ["uv", "lock", "--check", "--offline"],
            "run.py --help": [sys.executable, "run.py", "run", "--help"],
            "its own tests": [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        }
        for label, command in checks.items():
            done = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            status = "ok" if done.returncode == 0 else "FAILED"
            summary = (done.stdout.strip().splitlines() or [""])[-1]
            print(f"  {status:6s} {label}   {summary if label == 'its own tests' else ''}")
            if done.returncode != 0:
                failures.append(f"{label} failed:\n{(done.stdout + done.stderr)[-1500:]}")
    return failures


def publish_branch(tree: str, name: str) -> str:
    """Point branch ``name`` at a snapshot of ``tree`` with the export filter applied.

    Built with git's plumbing in a scratch directory and a scratch index, so neither the
    working tree nor the real index is touched. Returns what happened."""
    source = git("rev-parse", "--short", tree)
    subject = git("log", "-1", "--format=%s", tree)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", tree], cwd=REPO, check=True, capture_output=True
    ).stdout
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "tree"
        root.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(root, filter="data")
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git_dir = ["--git-dir", str(REPO / ".git"), "--work-tree", str(root)]

        def plumbing(*args: str) -> str:
            done = subprocess.run(
                ["git", *git_dir, *args],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            return done.stdout.strip()

        plumbing("add", "-A")
        snapshot = plumbing("write-tree")

    ref = f"refs/heads/{name}"
    exists = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", ref], cwd=REPO, capture_output=True
    )
    parent = git("rev-parse", ref) if exists.returncode == 0 else None
    if parent and git("rev-parse", f"{parent}^{{tree}}") == snapshot:
        return f"{name} is already up to date with {source} ({parent[:7]})"
    message = (
        f"Submission snapshot of {source}: {subject}\n\n"
        "The committed code the submission zip is built from, without eval/, docs/, scripts/\n"
        "or CI. Generated by scripts/make_submission.py --branch; do not edit by hand.\n\n"
        "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\n"
    )
    command = ["commit-tree", snapshot, "-m", message] + (["-p", parent] if parent else [])
    commit = git(*command)
    git("update-ref", ref, commit)
    return f"{name} now at {commit[:7]}, a snapshot of {source}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, nargs="?", help="a run's results directory")
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "submission.zip")
    parser.add_argument(
        "--tree",
        default="HEAD",
        help="what to archive (default: last commit; uncommitted changes are NOT in it)",
    )
    parser.add_argument("--branch", help="also point this branch at the filtered code")
    args = parser.parse_args()
    if not args.results and not args.branch:
        parser.error("give a results directory (for the zip), --branch NAME, or both")

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and args.tree == "HEAD":
        print("WARNING: uncommitted changes are NOT in the zip:\n" + dirty, file=sys.stderr)

    if args.branch:
        print(publish_branch(args.tree, args.branch))
    if not args.results:
        return 0

    answer = args.results / "answer.json"
    video = args.results / "annotated.mp4"
    for path in (answer, video):
        if not path.exists():
            print(
                f"missing {path}: run `uv run run.py run <video> --out {args.results}` first"
            )
            return 1
    try:
        record = check_answer(answer)
        frames, width, fps = check_video(video)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"not submittable: {exc}")
        return 1
    print(f"answer.json : {json.dumps(record)}")
    print(f"video       : {frames} frames, {width} px wide, {fps:g} fps")

    build(args.results, args.out, args.tree)
    print(f"built       : {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB, {args.tree})")
    print("verifying the zip the way a reviewer would:")
    failures = verify(args.out)
    if failures:
        print("\nNOT READY:\n  " + "\n  ".join(failures))
        return 1
    print("\nREADY: the zip unpacks, matches the schema, and passes its own tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
