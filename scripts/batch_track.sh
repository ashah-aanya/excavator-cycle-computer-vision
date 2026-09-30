#!/usr/bin/env bash
# Track a list of videos (the `track` stage only: masks.npz + track.json for the excavator AND
# the bucket) with the default config, and copy each video's results to Drive THE MOMENT THAT
# VIDEO FINISHES, so a crash or a Colab disconnect loses at most the video in progress.
#
# The default config uses SAM 2.1 small (src/excavator_cycles/config.py).
#
#   1. Edit VIDEOS below (the first is excavator_cycle; the rest are placeholders), or pass
#      video paths as arguments and the list below is ignored.
#   2. On Colab, from the repo folder:
#        PY=/content/excavator/.venv/bin/python bash scripts/batch_track.sh
#
#   DRIVE_OUT  where finished runs are copied      (default /content/drive/MyDrive/bucket_seed_runs)
#   WORK       fast local scratch, not Drive       (default /content/work)
#   PY         interpreter with torch              (default: uv run --no-sync python)
#
# Why work locally and then copy: Drive is a network folder, slow for the video and flaky for
# many small writes during tracking. The results are ~1 MB, so one copy at the end of each
# video costs nothing and is easy to verify.
#
# Resumable: a video whose Drive folder already holds track.json is skipped. The copy goes to a
# hidden ".partial" folder and is renamed only once complete, so a half-copied video is never
# mistaken for a finished one. A video that is missing (a placeholder) or fails is reported
# and the batch carries on.

set -u
cd "$(dirname "$0")/.."

VIDEOS=(
  "/content/drive/MyDrive/construction_excavator_cycle_duration_1.mp4"   # excavator_cycle
  # "/content/drive/MyDrive/PLACEHOLDER_video_2.mp4"
  # "/content/drive/MyDrive/PLACEHOLDER_video_3.mp4"
  # "/content/drive/MyDrive/PLACEHOLDER_video_4.mp4"
)
if [ "$#" -gt 0 ]; then VIDEOS=("$@"); fi

DRIVE_OUT="${DRIVE_OUT:-/content/drive/MyDrive/bucket_seed_runs}"
WORK="${WORK:-/content/work}"
PY="${PY:-uv run --no-sync python}"

mkdir -p "$DRIVE_OUT" "$WORK" || { echo "cannot create $DRIVE_OUT or $WORK (is Drive mounted?)" >&2; exit 2; }

echo "torch / GPU check:"
$PY -c "import torch; print('  torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
model="$($PY -c "from excavator_cycles.config import Config; print(Config.load('configs/default.yaml').track.model_id)" 2>/dev/null || echo unknown)"
commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "SAM model: $model   commit: $commit"
echo "videos: ${#VIDEOS[@]}   copying each to: $DRIVE_OUT"
echo

done_count=0; skipped=0; missing=0; failed=0
for video in "${VIDEOS[@]}"; do
  stem="$(basename "${video%.*}")"
  if [ ! -f "$video" ]; then
    echo "MISSING  $stem  ($video)"; missing=$((missing + 1)); continue
  fi
  if [ -f "$DRIVE_OUT/$stem/track.json" ]; then
    echo "skip     $stem  (already in Drive)"; skipped=$((skipped + 1)); continue
  fi

  echo "run      $stem  $(date '+%H:%M:%S')"
  local_video="$WORK/$stem.${video##*.}"
  local_out="$WORK/$stem"
  rm -rf "$local_out"; mkdir -p "$local_out"
  cp "$video" "$local_video"

  $PY run.py --log-file "$WORK/$stem.log" \
    track "$local_video" --config configs/default.yaml --out "$local_out" \
    > "$WORK/$stem.stdout" 2>&1
  code=$?
  rm -f "$local_video"

  # `track` exits non-zero when its QA flags a concern but still writes the masks, so what
  # counts as success is that both outputs exist, not the exit code.
  if [ ! -f "$local_out/track.json" ] || [ ! -f "$local_out/masks.npz" ]; then
    echo "FAILED   $stem  (exit $code, no track.json/masks.npz; see $WORK/$stem.stdout)"
    tail -n 5 "$WORK/$stem.stdout" | sed 's/^/           /'
    failed=$((failed + 1)); continue
  fi

  cp "$WORK/$stem.log" "$WORK/$stem.stdout" "$local_out/" 2>/dev/null
  printf 'video: %s\nmodel: %s\ncommit: %s\nexit: %s\nfinished: %s\n' \
    "$video" "$model" "$commit" "$code" "$(date '+%Y-%m-%d %H:%M:%S')" > "$local_out/RUN_INFO.txt"

  partial="$DRIVE_OUT/.$stem.partial"
  rm -rf "$partial"
  if cp -r "$local_out" "$partial" && sync && mv "$partial" "$DRIVE_OUT/$stem"; then
    echo "saved    $stem  exit $code  ->  $DRIVE_OUT/$stem  $(date '+%H:%M:%S')"
    done_count=$((done_count + 1))
  else
    echo "FAILED   $stem  (tracked fine but the copy to Drive failed; results are in $local_out)"
    failed=$((failed + 1))
  fi
done

echo
echo "=== done: $done_count saved, $skipped skipped, $missing missing, $failed failed ==="
ls -1 "$DRIVE_OUT"
echo "ALL DONE $(date '+%H:%M:%S')"
[ "$failed" -eq 0 ]
