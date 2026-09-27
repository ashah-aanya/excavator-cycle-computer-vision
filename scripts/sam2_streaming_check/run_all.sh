#!/bin/sh
# TEMPORARY (fix/sam2-streaming) -- delete before merging.
#
# Every model run stages 3, 4 and 6 need, on the cluster GPU, then the verdicts.
#   scripts/sam2_streaming_check/run_all.sh [dev clip] [long clip]
# Defaults: data/dev_clip.mp4 (the hand-labelled 29.6 s clip) and
# data/construction_excavator_cycle_duration_1.mp4 (the 83 s clip).
# Writes everything to outputs/sam2check/; the verdicts go to summary.txt.
set -u
cd "$(dirname "$0")/../.."

DEV=${1:-data/dev_clip.mp4}
LONG=${2:-data/construction_excavator_cycle_duration_1.mp4}
OUT=outputs/sam2check
PY=.venv/bin/python
RUNNER=scripts/sam2_streaming_check/runner.py

for f in "$DEV" "$LONG" "$PY"; do
  [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
mkdir -p "$OUT"

one() {
  label=$1
  mode=$2
  shift 2
  echo "== $label ($mode) started $(date +%H:%M:%S)"
  rm -rf "$OUT/$label"
  $PY -u $RUNNER "$label" "$mode" "$OUT" -- \
    run "$@" --work-dir "$OUT/$label" --device cuda \
    > "$OUT/$label.out" 2>&1
  echo "   exit $? -- log: $OUT/$label.out"
}

# Stage 3: same results? (dev clip, 10 Hz)
one dev_offline_10hz  offline "$DEV"
one dev_stream_A_10hz stream  "$DEV"
one dev_stream_B_10hz stream  "$DEV"
one dev_keepall_10hz  keepall "$DEV"

# Stage 4: flat memory? (dev clip at other rates; the offline 30 Hz run is left
# out on purpose -- it is the one expected to need the most memory)
one dev_offline_5hz   offline "$DEV" --rate 5
one dev_stream_5hz    stream  "$DEV" --rate 5
one dev_stream_30hz   stream  "$DEV" --rate 30

# Stage 6: the clip that crashed at 10 Hz
one long_stream_10hz  stream  "$LONG"

$PY scripts/sam2_streaming_check/compare.py "$OUT" | tee "$OUT/summary.txt"
