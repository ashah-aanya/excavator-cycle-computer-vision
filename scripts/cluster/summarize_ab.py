"""One table from the runs that `run_ab.sh` wrote: how well was the bucket held?

    python scripts/cluster/summarize_ab.py [~/ab-results]

Reads only track.json and run.json, so it needs no GPU and no models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NAN = float("nan")


def commit_of(run_dir: Path) -> str:
    """The code revision the run recorded, so a row proves which version produced it."""
    record = run_dir / "run.json"
    if not record.exists():
        return "?"
    git = json.loads(record.read_text()).get("git", {})
    return git.get("revision", "?")[:7] + ("*" if git.get("dirty") else "")


def answer_of(run_dir: Path) -> str:
    """The pipeline's answer for this run ('-' if it did not get as far as writing one)."""
    path = run_dir / "answer.json"
    if not path.exists():
        return "-"
    answer = json.loads(path.read_text())
    seconds = answer.get("average_cycle_duration_seconds")
    return f"{answer.get('cycle_count')} cycles, {seconds} s"


def main() -> None:
    out = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "ab-results"
    versions = sorted(p.name for p in out.iterdir() if p.is_dir())
    ran = {p.parent.name for p in out.glob("*/*/track.json")}
    started = {p.name[: -len(".log")] for p in out.glob("*/*.log") if ".debug" not in p.name}
    videos = sorted(ran | started)

    print(
        f"{'video':26s}{'version':16s}{'commit':9s}{'bucket cov':>11s}"
        f"{'area p90/p10':>13s}{'reseeds':>9s}  {'QA':6s}{'answer'}"
    )
    for video in videos:
        for version in versions:
            run_dir = out / version / video
            track = run_dir / "track.json"
            if not track.exists():
                if (out / version / f"{video}.log").exists():
                    print(
                        f"{video:26s}{version:16s}{'':9s}  "
                        f"NO track.json -> tail -30 {run_dir}.log"
                    )
                continue
            data = json.loads(track.read_text())
            bucket = data["qa"].get("bucket") or {}
            reseeds = data.get("bucket_reseeds", [])
            found = sum(1 for r in reseeds if r.get("seed_sample") is not None)
            qa_status = data["qa"]["status"]
            print(
                f"{video:26s}{version:16s}{commit_of(run_dir):9s}"
                f"{bucket.get('coverage', NAN):11.2f}"
                f"{bucket.get('area_p90_over_p10', NAN):13.1f}"
                f"{f'{found}/{len(reseeds)}':>9s}  {qa_status:6s}{answer_of(run_dir)}"
            )
    print("\ncommit* = working tree had uncommitted edits.  reseeds = seed found / attempted.")


if __name__ == "__main__":
    main()
