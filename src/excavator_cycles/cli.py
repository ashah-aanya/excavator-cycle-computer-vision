"""Command line interface.

Grows one subcommand per pipeline stage. Today it can inspect a video; the
stages are added as they are built, so there is always something runnable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .logging_setup import configure_logging, get_logger
from .video import probe

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="excavator-cycles",
        description="Measure excavator work-cycle phase durations from video.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    # Global flags live on the top-level parser so every stage inherits them.
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log detail (repeatable)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="also write a full DEBUG trail here; useful for cluster runs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser(
        "probe",
        help="Report a video's resolution, frame rate, duration, and what the "
        "configured sampling rate implies for it.",
    )
    probe_parser.add_argument("video", type=Path)
    probe_parser.add_argument(
        "--config", type=Path, default=None, help="YAML config overriding defaults"
    )
    probe_parser.add_argument("--json", action="store_true", help="machine-readable output")
    probe_parser.set_defaults(func=_cmd_probe)

    spike_parser = subparsers.add_parser(
        "spike",
        help="Stage 1: check whether the detector finds the excavator in this "
        "footage, before anything is built on top of it.",
    )
    spike_parser.add_argument(
        "video",
        type=Path,
        help="video file, a single image, or a folder of images. Images let you "
        "check the real detector before the target video is available.",
    )
    spike_parser.add_argument(
        "--config", type=Path, default=None, help="YAML config overriding defaults"
    )
    spike_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: outputs/spike/<video stem>)",
    )
    spike_parser.add_argument(
        "--detector",
        action="append",
        choices=["grounding_dino", "owlv2"],
        help="repeatable; two detectors enables the cross-model agreement check",
    )
    spike_parser.add_argument(
        "--prompt",
        action="append",
        help="repeatable; defaults to the configured excavator prompts",
    )
    spike_parser.add_argument("--frames", type=int, default=None, help="how many to sample")
    spike_parser.add_argument(
        "--device", default=None, help="cuda / mps / cpu (default: best available)"
    )
    spike_parser.set_defaults(func=_cmd_spike)

    args = parser.parse_args(argv)
    configure_logging(args.verbose, args.log_file)
    return args.func(args)


def _cmd_probe(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    log.debug("config digest inputs: %s", sorted(config.to_dict()))
    info = probe(args.video)
    log.info("probed %s (%.2f s, %.3f fps)", info.path.name, info.duration_seconds, info.fps)

    stride = info.stride_for(config.sampling.rate_hz)
    anchor_stride = info.stride_for(config.sampling.anchor_rate_hz)
    n_samples = info.frame_count // stride if stride else 0
    n_anchors = info.frame_count // anchor_stride if anchor_stride else 0

    if args.json:
        print(
            json.dumps(
                {
                    "path": str(info.path),
                    "width": info.width,
                    "height": info.height,
                    "fps": info.fps,
                    "frame_count": info.frame_count,
                    "duration_seconds": info.duration_seconds,
                    "sampling": {
                        "rate_hz": config.sampling.rate_hz,
                        "stride_frames": stride,
                        "n_samples": n_samples,
                        "anchor_rate_hz": config.sampling.anchor_rate_hz,
                        "n_anchor_detections": n_anchors,
                    },
                },
                indent=2,
            )
        )
        return 0

    print(info.summary())
    print()
    print("  with the current config:")
    print(
        f"    features/FSM : {config.sampling.rate_hz:g} Hz "
        f"-> every {stride} frame(s), ~{n_samples} samples"
    )
    print(
        f"    detection    : {config.sampling.anchor_rate_hz:g} Hz "
        f"-> ~{n_anchors} detector calls"
    )
    print(
        f"    sample gap   : {1 / config.sampling.rate_hz:.3f} s "
        f"(tolerance is 0.6 s, so quantisation is not the limit)"
    )
    return 0


def _cmd_spike(args: argparse.Namespace) -> int:
    """Run the detector spike and print its verdict.

    Imported lazily: the spike is the only command that needs the model extras,
    and `probe` must keep working on a machine without them.
    """
    from .detect import build_detector
    from .provenance import set_seeds
    from .spike import format_report, run_spike

    set_seeds()
    config = Config.load(args.config)
    names = args.detector or ["grounding_dino"]
    out_dir = args.out or Path("outputs/spike") / Path(args.video).stem

    log.info("building detector(s): %s", ", ".join(names))
    detectors = {
        name: build_detector(name, config.detection, device=args.device) for name in names
    }

    report = run_spike(
        video_path=args.video,
        detectors=detectors,
        config=config,
        output_dir=out_dir,
        prompts=args.prompt,
        n_frames=args.frames,
    )

    print()
    print(format_report(report))
    print()
    print(f"  artifacts: {out_dir}")

    # Non-zero exit on a failed gate, so this is usable in a script -- but the
    # artifacts are written either way, because a failure is what you most need
    # to look at.
    return 0 if report.gate["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
