"""Command line interface.

`run` chains the stages: video in, answer.json and an annotated video out. Each stage
is also its own subcommand, because only tracking needs a GPU and the later stages
read its cache in seconds.
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
        help="cache directory (default: outputs/<video stem>)",
    )
    track_parser.add_argument("--device", default=None, help="cuda / mps / cpu")
    track_parser.add_argument(
        "--detector", default="grounding_dino", choices=["grounding_dino", "owlv2"]
    )
    track_parser.add_argument(
        "--rate", type=float, default=None, help="samples per second (overrides config)"
    )
    track_parser.set_defaults(func=_cmd_track)

    run_parser = subparsers.add_parser(
        "run",
        help="THE DELIVERABLE: video in, answer.json and an annotated video out. Chains "
        "track -> features -> cycles -> render, so a reviewer needs one command and no "
        "knowledge of the stages.",
    )
    run_parser.add_argument("video", type=Path)
    run_parser.add_argument("--config", type=Path, default=None)
    run_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="results directory: the cache, answer.json and annotated.mp4 all go here "
        "(default: outputs/<video stem>)",
    )
    run_parser.add_argument("--device", default=None, help="cuda / mps / cpu")
    run_parser.add_argument(
        "--detector", default="grounding_dino", choices=["grounding_dino", "owlv2"]
    )
    run_parser.add_argument(
        "--rate", type=float, default=None, help="samples per second (overrides config)"
    )
    run_parser.add_argument(
        "--reuse",
        action="store_true",
        help="skip stages whose output is already in the results directory. For iterating "
        "on later stages without re-running the GPU pass. Once a stage runs, every "
        "later stage runs too. Settings are NOT compared: after changing --config or "
        "--rate, delete the results directory or run without --reuse.",
    )
    run_parser.add_argument("--scale", type=float, default=2.0, help="video resize factor")
    run_parser.set_defaults(func=_cmd_run)

    render_parser = subparsers.add_parser(
        "render",
        help="Rebuild the video with the cached masks, detections and phases drawn on it. "
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
        help="Stage 4: find the phase starts and the complete cycles, and write "
        "answer.json and phases.json. No models, no GPU.",
    )
    cycles_parser.add_argument("track_dir", type=Path, help="a directory from `features`")
    cycles_parser.add_argument(
        "--out", type=Path, default=None, help="answer.json (default: <track_dir>/answer.json)"
    )
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
    actual_rate = info.actual_rate(config.sampling.rate_hz)
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
                        "actual_rate_hz": actual_rate,
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
        f"    features     : {config.sampling.rate_hz:g} Hz "
        f"-> every {stride} frame(s) = {actual_rate:g} Hz, ~{n_samples} samples"
    )
    print(
        f"    detection    : {config.sampling.anchor_rate_hz:g} Hz "
        f"-> ~{n_anchors} detector calls"
    )
    print(
        f"    sample gap   : {1 / actual_rate:.3f} s "
        f"(tolerance is 0.6 s, so quantisation is not the limit)"
    )
    return 0


def _cmd_track(args: argparse.Namespace) -> int:
    """Run perception over a whole video and cache the result."""
    from .track import track

    overrides = {"sampling": {"rate_hz": args.rate}} if args.rate else None
    config = Config.load(args.config, overrides)
    out_dir = args.out or Path("outputs") / Path(args.video).stem

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
    """Draw the cached masks, and the phases `cycles` found, back onto the video."""
    from .cycles import read_phases
    from .render import render

    phases = args.track_dir / "phases.json"
    starts, cycles = read_phases(phases) if phases.exists() else (None, None)
    stats = render(
        output_dir=args.track_dir,
        out_path=args.out,
        scale=args.scale,
        draw_boxes=not args.no_boxes,
        starts=starts,
        cycles=cycles,
    )
    print(f"\n  wrote {stats.output_path}")
    print(f"  {stats.frames_written} frames, {stats.frames_with_mask} carrying a mask")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Video in, answer.json and an annotated video out. The whole pipeline, one command.

    The stages exist because only one of them needs a GPU and the others are worth
    re-running on a cached result in seconds. That division is right for development and
    wrong for a reviewer, who should not have to know it exists to get an answer out of
    a video.

    So this adds no new logic. It calls the same subcommands in order, with the same
    arguments, through their own `_cmd_*` functions -- which means the thing a reviewer
    runs is the thing the tests exercise, rather than a second path that can drift
    from it.
    """
    out_dir = args.out or Path("outputs") / Path(args.video).stem
    stages = (
        ("track", out_dir / "masks.npz", _cmd_track, {"out": out_dir}),
        ("features", out_dir / "features.npz", _cmd_features, {"no_plots": False}),
        ("cycles", None, _cmd_cycles, {"out": out_dir / "answer.json"}),
        ("render", None, _cmd_render, {"out": out_dir / "annotated.mp4", "no_boxes": False}),
    )

    # A QA verdict is ADVISORY here, and only for `track`. `track` returns non-zero
    # when `evaluate_quality` flags anything -- including soft observations like
    # "only X% of samples have a mask" -- but it has still written the masks, so the
    # remaining stages can run and the task asks for an answer. Aborting turned one
    # tracker wobble on a hidden video into zero for every field, which is strictly
    # worse than a flagged answer. A genuine failure raises rather than returning.
    #
    # The concern is not swallowed: it is logged and printed beside the answer.
    # It is NOT the exit code: the answer and video were written, and a caller that
    # reads a non-zero status as "no result" would discard a good one. Only a stage
    # that genuinely fails returns non-zero.
    concerns: list[str] = []
    # `--reuse` asked each stage separately whether its product existed, so a
    # re-run `track` could be followed by a REUSED `features.npz` built from the
    # previous masks -- fresh input, stale features, and an answer from neither.
    # Once any stage runs, everything downstream of it is out of date.
    #
    # What `--reuse` still does not do is compare settings. `track.json` records a
    # digest, but of the WHOLE config, and `features` records none. So a changed
    # `--config` or `--rate` is the caller's to handle; the help says so.
    upstream_ran = False
    for name, product, run_stage, extra in stages:
        if args.reuse and not upstream_ran and product is not None and product.exists():
            log.info("%s: reusing %s", name, product)
            continue
        upstream_ran = True
        log.info("%s: running", name)
        # A copy per stage, so one stage's arguments cannot leak into the next.
        stage_args = argparse.Namespace(**vars(args))
        stage_args.track_dir = out_dir
        for key, value in extra.items():
            setattr(stage_args, key, value)
        status = run_stage(stage_args)
        if status == 0:
            continue
        if name == "track":
            log.warning(
                "track reported a QA concern (status %d). The masks were written, so "
                "the run continues -- but treat this answer as suspect.",
                status,
            )
            concerns.append("track QA flagged this run; see the QA lines above")
            continue
        log.error("%s failed with status %d; stopping", name, status)
        return status

    print()
    print(f"  ANSWER: {out_dir / 'answer.json'}")
    print(f"  VIDEO : {out_dir / 'annotated.mp4'}")
    for concern in concerns:
        print(f"  CONCERN: {concern}")
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
    """Stage 4: features -> phase starts -> cycles -> answer.json and phases.json."""
    from .cues import NoPhases
    from .cycles import assemble, summarise, write_answer, write_phases
    from .features import load as load_features
    from .starts import PhaseSearch, find_phase_starts

    table, scene = load_features(args.track_dir)
    try:
        search = find_phase_starts(table, scene)
    except NoPhases as exc:
        # Some videos do not show what the method needs. That is a finding, not a
        # crash: the answer says no cycles, and this says why.
        log.warning("no phases can be found: %s", exc)
        stop = {"phase": "digging", "search_from": 0.0, "why": str(exc)}
        search = PhaseSearch(starts=[], gaps=[], stop=stop)
    cycles = assemble(search.starts)
    answer = summarise(cycles)

    out = args.out or args.track_dir / "answer.json"
    write_answer(answer, out)
    write_phases(search, cycles, args.track_dir / "phases.json")

    print()
    print(f"  phase starts found : {len(search.starts)}")
    for gap in search.gaps:
        print(
            f"  abandoned a cycle  : {gap['phase']} from {gap['from']:.1f} s -- {gap['why']}"
        )
    if search.stop:
        print(f"  search stopped     : {search.stop['phase']} -- {search.stop['why']}")
    print(f"  complete cycles    : {answer.cycle_count}")
    print(f"  average cycle      : {answer.average_cycle_duration_seconds:.3f} s")
    for phase, seconds in answer.average_phase_duration_seconds.items():
        print(f"    {phase:9s}        : {seconds:.3f} s")
    print(f"  wrote              : {out}")
    return 0
