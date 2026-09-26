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

    cycles_parser = subparsers.add_parser(
        "cycles",
        help="Stage 4: find the work cycles and write answer.json. --test also "
        "renders a diagnostic video and scores the result. No models, no GPU.",
    )
    cycles_parser.add_argument("track_dir", type=Path, help="a directory from `features`")
    cycles_parser.add_argument("--config", type=Path, default=None)
    cycles_parser.add_argument(
        "--out", type=Path, default=None, help="answer.json (default: <track_dir>/answer.json)"
    )
    cycles_parser.add_argument(
        "--test",
        action="store_true",
        help="also render the windows and predicted onsets over the video, and "
        "print the per-cycle breakdown",
    )
    cycles_parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="EVALUATION ONLY: hand-made ground truth to draw and score against. "
        "Passed in, never discovered -- the pipeline cannot reach the answer on "
        "its own.",
    )
    cycles_parser.add_argument("--scale", type=float, default=2.0, help="video resize factor")
    cycles_parser.set_defaults(func=_cmd_cycles)

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


def _cmd_cycles(args: argparse.Namespace) -> int:
    """Stage 4: features -> onsets -> cycles -> answer.json.

    `--test` does not change what the pipeline computes. It only adds evidence:
    the same run, plus a video showing where each window was and where each cue
    fired, plus the per-cycle breakdown. What is tested is what ships.
    """

    from .cycles import assemble, summarise, write_answer
    from .features import load as load_features
    from .fsm import calibrate, evidence_within, locate, walk

    config = Config.load(args.config)
    table, _scene = load_features(args.track_dir)

    levels = calibrate(table, config)
    print()
    print(levels.report())

    detections = walk(table, levels, config=config)
    onsets = locate(detections, table, config)
    # Evidence is asked about each cycle's OWN span. A single set built from the
    # whole video and copied into every cycle marks them all complete as soon as
    # any one of them was.
    cycles = assemble(
        onsets,
        evidence=lambda start, end: evidence_within(table, levels, start, end),
    )
    answer = summarise(cycles)

    out = args.out or args.track_dir / "answer.json"
    write_answer(answer, out)

    print()
    print(f"  cycles occurred  : {answer.cycle_count}")
    print(f"  cycles measured  : {sum(1 for c in cycles if c.measurable)}")
    print(f"  average cycle    : {answer.average_cycle_duration_seconds:.3f} s")
    for phase, seconds in answer.average_phase_duration_seconds.items():
        print(f"    {phase:9s}      : {seconds:.3f} s")
    print(f"  wrote            : {out}")

    if not args.test:
        return 0

    truth = _read_labels(args.labels) if args.labels else {}
    _print_breakdown(cycles, onsets, truth)
    _render_diagnostic(args, table, detections, onsets, truth)
    return 0


def _read_labels(path: Path) -> dict[str, float]:
    """EVALUATION ONLY. The pipeline never calls this; only --test does."""
    import json

    labels = json.loads(path.read_text())
    fps = float(labels["video"]["fps"])
    keys = {
        "digging": "digging_begins",
        "hauling": "hauling_begins",
        "dumping": "dumping_begins",
        "swinging": "swinging_begins",
    }
    return {name: labels["boundaries"][key] / fps for name, key in keys.items()}


def _print_breakdown(cycles, onsets, truth: dict[str, float]) -> None:
    print()
    print("  --- onsets " + "-" * 52)
    header = f"  {'phase':10}{'predicted':>11}"
    if truth:
        header += f"{'truth':>9}{'err':>8}"
    print(header)
    seen: set[str] = set()
    for phase, when in onsets:
        line = f"  {phase:10}{when:>11.2f}"
        if truth and phase not in seen:
            line += f"{truth[phase]:>9.2f}{when - truth[phase]:>+8.2f}"
        seen.add(phase)
        print(line)

    print()
    print("  --- cycles " + "-" * 52)
    if not cycles:
        print("  none: fewer than two digging onsets, so no span is bounded")
    for index, cycle in enumerate(cycles, 1):
        verdict = "measured" if cycle.measurable else f"EXCLUDED -- {cycle.reason}"
        print(f"  cycle {index}: {verdict}")
        for phase, seconds in cycle.durations().items():
            print(f"      {phase:9s} {seconds:6.2f} s")


def _render_diagnostic(args, table, detections, onsets, truth) -> None:
    from .render import render

    times = table.time_seconds
    windows = [
        (d.phase, float(times[d.window.lo]), float(times[min(d.window.hi, len(times) - 1)]))
        for d in detections
    ]
    predicted: dict[str, float] = {}
    for phase, when in onsets:
        predicted.setdefault(phase, when)

    out = args.track_dir / "cycles.mp4"
    stats = render(
        args.track_dir,
        out_path=out,
        scale=args.scale,
        windows=windows,
        onsets=predicted,
        reference=truth or None,
    )
    print()
    print(f"  wrote {stats.output_path}  ({stats.frames_written} frames)")
    print("    shaded span   the window pass 1 searched")
    print("    solid line    the refined onset pass 2 returned")
    if truth:
        print("    dashed white  the hand-labelled truth")
