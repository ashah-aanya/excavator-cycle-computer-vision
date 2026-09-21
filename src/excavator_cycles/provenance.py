"""Recording what produced a result.

Every stage of this pipeline writes its outputs into a run directory, and beside
them a ``run.json`` describing the conditions that produced them: the exact
config, the code revision, the package versions, the hardware, the inputs and
their content hashes, and how long it took.

This is the practice that separates a defensible research result from a number
you have to take on faith. Six months from now, "which threshold produced this
answer?" has to be answerable from the artifacts alone -- and for this task in
particular, a reviewer re-running the pipeline should be able to tell whether
they reproduced our conditions or merely got a similar number.

It also gives the cache a correct key: outputs are only reusable if the input
video, the config and the code that produced them are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# Packages whose versions can change a result. Recorded on every run; a
# reproduction mismatch usually shows up here first.
_TRACKED_PACKAGES = (
    "numpy",
    "opencv-python-headless",
    "scipy",
    "torch",
    "torchvision",
    "transformers",
)

_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB: hashing a video should not load it into RAM


@dataclass(frozen=True)
class GitState:
    """Which code ran.

    ``dirty`` matters as much as the revision: a result produced from a modified
    working tree is not reproducible from the commit alone, and saying so is more
    useful than pretending otherwise.
    """

    revision: str | None
    branch: str | None
    dirty: bool | None

    @classmethod
    def capture(cls, repo_root: Path | None = None) -> GitState:
        root = repo_root or Path(__file__).resolve().parents[2]

        def run(*args: str) -> str | None:
            try:
                out = subprocess.run(
                    ["git", *args],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            return out.stdout.strip() if out.returncode == 0 else None

        revision = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        status = run("status", "--porcelain")
        return cls(
            revision=revision,
            branch=branch,
            dirty=None if status is None else bool(status),
        )


@dataclass(frozen=True)
class Environment:
    """Where the code ran. Explains results that differ across machines."""

    python: str
    platform: str
    packages: dict[str, str]
    device: str

    @classmethod
    def capture(cls) -> Environment:
        packages: dict[str, str] = {}
        for name in _TRACKED_PACKAGES:
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                continue  # optional extra not installed; absence is informative
        return cls(
            python=sys.version.split()[0],
            platform=platform.platform(),
            packages=packages,
            device=detect_device(),
        )


def detect_device() -> str:
    """Name the compute device, without requiring torch to be installed.

    The perception stage is an optional extra, so everything else must remain
    importable and testable on a machine with no deep-learning stack at all.
    """
    try:
        import torch
    except ImportError:
        return "cpu (torch not installed)"

    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "cpu"


def file_digest(path: str | Path) -> str:
    """SHA-256 of a file, streamed.

    Identifies an input video by content rather than by filename, so a renamed
    or re-downloaded file still hits the same cache entry, and two different
    videos with the same name never collide.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config_dict: dict[str, Any]) -> str:
    """Short, stable hash of a config.

    Sorted keys so the hash depends on the values, not on dict ordering. Used to
    key caches: change a threshold and stale results are invalidated instead of
    silently reused.
    """
    encoded = json.dumps(config_dict, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class RunRecord:
    """Everything needed to explain, and re-create, one run of a stage."""

    stage: str
    started_at: str
    git: GitState
    environment: Environment
    config: dict[str, Any]
    config_digest: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float | None = None
    status: str = "running"
    error: str | None = None

    def add_input(self, name: str, path: str | Path, *, hash_content: bool = True) -> None:
        path = Path(path)
        entry: dict[str, Any] = {"path": str(path)}
        if hash_content and path.is_file():
            entry["sha256"] = file_digest(path)
            entry["bytes"] = path.stat().st_size
        self.inputs[name] = entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "started_at": self.started_at,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "git": asdict(self.git),
            "environment": asdict(self.environment),
            "config_digest": self.config_digest,
            "config": self.config,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metrics": self.metrics,
        }


@contextmanager
def run_record(
    stage: str,
    output_dir: str | Path,
    config: dict[str, Any],
) -> Iterator[RunRecord]:
    """Wrap a stage so its provenance is written whether or not it succeeds.

    A failed run's record is *more* valuable than a successful one's, so the
    record is written on the way out either way, with the error attached::

        with run_record("spike", out_dir, cfg.to_dict()) as record:
            record.add_input("video", video_path)
            ...
            record.metrics["detection_rate"] = 0.98
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record = RunRecord(
        stage=stage,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        git=GitState.capture(),
        environment=Environment.capture(),
        config=config,
        config_digest=config_digest(config),
    )
    started = time.perf_counter()
    try:
        yield record
    except Exception as exc:
        record.status = "failed"
        record.error = f"{type(exc).__name__}: {exc}"
        raise
    else:
        record.status = "ok"
    finally:
        record.duration_seconds = round(time.perf_counter() - started, 3)
        path = output_dir / "run.json"
        path.write_text(json.dumps(record.to_dict(), indent=2, default=str))


def set_seeds(seed: int = 0) -> None:
    """Make a run repeatable on the same machine.

    Honest scope: this pins Python, NumPy and torch RNGs and asks cuDNN for
    deterministic kernels. It does not make results bit-identical across
    different GPU models or driver versions -- nothing reasonably can. What it
    buys is that re-running here gives the same answer, so a change in output
    means a change in code or config.
    """
    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
