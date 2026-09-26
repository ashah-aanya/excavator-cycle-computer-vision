"""Command line interface.

Grows one subcommand per pipeline stage. Today it can inspect a video; the
stages are added as they are built, so there is always something runnable.
"""

from __future__ import annotations

import argparse
import json
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

    track_parser = subparsers.add_parser(
        "track",
        help="Stage 2: run detection + SAM 2 over a video and cache masks. "
        "This is the only stage that needs a GPU.",
    )
    track_parser.add_argument("video", type=Path)
    track_parser.add_argument("--config", type=Path, default=None)
    track_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="cache directory (default: outputs/track/<video stem>)",
    )
    track_parser.add_argument("--device", default=None, help="cuda / mps / cpu")
    track_parser.add_argument(
        "--detector", default="grounding_dino", choices=["grounding_dino", "owlv2"]
    )
    track_parser.add_argument(
        "--rate", type=float, default=None, help="samples per second (overrides config)"
    )
    track_parser.set_defaults(func=_cmd_track)

    render_parser = subparsers.add_parser(
        "render",
        help="Rebuild the video with the cached masks and detections drawn on it. "
        "Reads the cache only -- no model, no GPU.",
    )
    render_parser.add_argument("track_dir", type=Path, help="a directory produced by `track`")
    render_parser.add_argument("--out", type=Path, default=None, help="output .mp4")
    render_parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="resize factor; >1 makes overlays legible on small sources",
    )
    render_parser.add_argument("--no-boxes", action="store_true", help="mask only")
    render_parser.set_defaults(func=_cmd_render)

    features_parser = subparsers.add_parser(
        "features",
        help="Stage 3: derive the scene (slew centre, reach L, truck box) and every "
        "kinematic feature from the cached masks, then plot them. No models, no GPU.",
    )
    features_parser.add_argument("track_dir", type=Path, help="a directory from `track`")
    features_parser.add_argument("--config", type=Path, default=None)
    features_parser.add_argument(
        "--no-plots", action="store_true", help="skip the diagnostic figures"
    )
    features_parser.set_defaults(func=_cmd_features)

    args = parser.parse_args(argv)
    configure_logging(args.verbose, args.log_file)
    return args.func(args)


def _cmd_probe(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    log.debug("config digest inputs: %s", sorted(config.to_dict()))
    info = probe(args.video, verify=True)
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



def _cmd_track(args: argparse.Namespace) -> int:
    """Run perception over a whole video and cache the result."""
    from .track import track

    overrides = {"sampling": {"rate_hz": args.rate}} if args.rate else None
    config = Config.load(args.config, overrides)
    out_dir = args.out or Path("outputs/track") / Path(args.video).stem

    result = track(
        video_path=args.video,
        config=config,
        output_dir=out_dir,
        device=args.device,
        detector_name=args.detector,
    )

    qa = result.qa
    print()
    print(
        f"  video      : {Path(result.video).name}  "
        f"{result.width}x{result.height}  {result.duration_seconds:.1f}s"
    )
    print(f"  samples    : {len(result.frames)} at {result.rate_hz:g} Hz")
    print(
        f"  seed       : t={result.seed['time_seconds']:.2f}s  "
        f"({len(result.seed['negative_points'])} negative point(s))"
    )
    print(f"  coverage   : {qa['coverage']:.1%}")
    print(
        f"  mask area  : median {qa['mask_area']['median'] * 100:.2f}%  "
        f"(min {qa['mask_area']['min'] * 100:.2f}%, max {qa['mask_area']['max'] * 100:.2f}%)"
    )
    print(
        f"  agreement  : {qa['detection_agreement_median_iou']}  "
        f"(mask vs independent detections, unmerged frames)"
    )
    print(f"  QA         : {qa['status'].upper()}")
    for failure in qa["failures"]:
        print(f"     - {failure}")
    print(f"\n  cache: {out_dir}")
    print(f"  next : uv run run.py render {out_dir} --scale 2")
    return 0 if qa["status"] == "pass" else 1


def _cmd_render(args: argparse.Namespace) -> int:
    """Draw the cached masks back onto the video."""
    from .render import render

    stats = render(
        output_dir=args.track_dir,
        out_path=args.out,
        scale=args.scale,
        draw_boxes=not args.no_boxes,
    )
    print(f"\n  wrote {stats.output_path}")
    print(f"  {stats.frames_written} frames, {stats.frames_with_mask} carrying a mask")
    return 0



def _cmd_features(args: argparse.Namespace) -> int:
    """Stage 3: derive the scene and every kinematic feature, and plot them."""
    from .features import build_features, save
    from .masks import load_objects
    from .track import load_result

    config = Config.load(args.config)
    result, _excavator = load_result(args.track_dir)
    objects, _shape = load_objects(args.track_dir / "masks.npz")
    table, scene = build_features(result, objects, config)
    save(table, scene, args.track_dir)

    if not args.no_plots:
        from .plots import plot_features

        plot_features(table, scene, args.track_dir / "features.png")

    print()
    print(f"  samples        : {len(table)}  ({int(table.found.sum())} with a bucket)")
    print(f"  slew centre    : ({scene.pivot[0]:.0f}, {scene.pivot[1]:.0f}) px")
    print(f"  arm reach L    : {scene.scale:.0f} px")
    if scene.truck_box:
        box = ", ".join(f"{v:.0f}" for v in scene.truck_box)
        print(f"  truck box      : ({box}) px")
    else:
        print("  truck box      : none detected (overlap feature is nan)")
    print(f"  wrote          : {args.track_dir}/features.npz, scene.json")
    return 0

