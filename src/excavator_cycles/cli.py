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
        help="Stage 3: derive the scene (pivot, scale, zones, surface) and the "
        "per-sample signals from cached masks. No models, no GPU.",
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
    """Derive the scene and the signals, and plot them for inspection."""
    import cv2

    from .features import build_features, save
    from .track import load_result

    config = Config.load(args.config)
    result, masks = load_result(args.track_dir)
    table, scene = build_features(result, masks, config)
    save(table, scene, args.track_dir)

    if not args.no_plots:
        from .plots import plot_scene, plot_signals
        from .video import read_frames_at

        plot_signals(table, scene, args.track_dir / "signals.png")
        mid = read_frames_at(result.video, [result.duration_seconds / 2])
        if mid:
            plot_scene(mid[0].image, table, scene, args.track_dir / "scene.png")
        else:
            log.warning("could not read a frame for the scene plot")
        del cv2  # imported only to fail early if OpenCV is missing

    print()
    print(f"  samples        : {len(table)}  ({int(table.valid.sum())} valid)")
    print(f"  rotation centre: ({scene.centre[0]:.0f}, {scene.centre[1]:.0f}) px")
    print(f"  arm reach L    : {scene.scale:.0f} px")
    if scene.dig_zone and scene.dump_zone:
        print(f"  dig zone       : ({scene.dig_zone[0]:.0f}, {scene.dig_zone[1]:.0f})")
        print(f"  dump zone      : ({scene.dump_zone[0]:.0f}, {scene.dump_zone[1]:.0f})")
        print(f"  separation     : {scene.zone_separation:.2f} L")
    else:
        print("  zones          : NOT RESOLVED")
    surface = scene.surface_height
    print(
        f"  surface height : {surface:.3f} L"
        if surface is not None
        else "  surface height : NOT RESOLVED"
    )
    print(f"  return swing   : {'left' if scene.return_sign > 0 else 'right'}")
    print()
    print(f"  wrote features.npz, features.csv, scene.json to {args.track_dir}")
    if not args.no_plots:
        print(f"  LOOK AT: {args.track_dir}/scene.png and {args.track_dir}/signals.png")
        print("  The gate for this stage is visual: the landmarks must land where")
        print("  you would put them, and the four boundaries must be visible.")
    return 0
