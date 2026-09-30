#!/usr/bin/env bash
# Track a list of videos (the `track` stage only: masks.npz + track.json for the excavator AND
# the bucket) with the default config, and copy each video's results to Drive THE MOMENT THAT
# VIDEO FINISHES, so a crash or a Colab disconnect loses at most the video in progress.
#
# Two things differ from the older pipeline, and both are in the default config:
#   * SAM 2.1 small instead of tiny;
#   * the bucket's seed frame is chosen by how far the arm is stretched ("reach"), not by the
#     older blend ("score"). Each video is run with BOTH rules so they can be compared on the
#     same video, same model, same environment (RULES below).
#
#   1. Edit VIDEOS below (the first is excavator_cycle; the rest are placeholders), or pass
#      video paths as arguments and the list below is ignored.
#   2. On Colab, from the repo folder:
#        PY=/content/excavator/.venv/bin/python bash scripts/batch_track.sh
#
#   RULES      frame rules to run, in order        (default "reach score"; "reach" = new only)
#   DRIVE_OUT  where finished runs are copied      (default /content/drive/MyDrive/bucket_seed_runs)
#   WORK       fast local scratch, not Drive       (default /content/work)
#   PY         interpreter with torch              (default: uv run --no-sync python)
#
# Why work locally and then copy: Drive is a network folder, slow for the video and flaky for
# many small writes during tracking. The results are ~1 MB, so one copy at the end of each
# video costs nothing and is easy to verify.
#
# Layout in Drive: $DRIVE_OUT/<rule>/<video name>/{track.json,masks.npz,run.json,RUN_INFO.txt,...}
#
# Resumable: a rule+video whose Drive folder already holds track.json is skipped. The copy goes to a
# hidden ".partial" folder and is renamed only once complete, so a half-copied video is never
# mistaken for a finished one. A video that is missing (a placeholder) or fails is reported
# and the batch carries on.

set -u
cd "$(dirname "$0")/.."

# In the order they were uploaded, so the ones that finish uploading first run first. A video
# that is not in Drive yet, or is still uploading, is skipped and the batch moves on; rerun the
# same command later and only the skipped ones run.
VIDEOS=(
  "/content/drive/MyDrive/construction_excavator_cycle_duration_1.mp4"   # excavator_cycle
  "/content/drive/MyDrive/clip3_15.3s-73.6s.mp4"
  "/content/drive/MyDrive/Untitled4.mov"
  "/content/drive/MyDrive/vid1.mov"
)
if [ "$#" -gt 0 ]; then VIDEOS=("$@"); fi

DRIVE_OUT="${DRIVE_OUT:-/content/drive/MyDrive/bucket_seed_runs}"
WORK="${WORK:-/content/work}"
PY="${PY:-uv run --no-sync python}"
RULES="${RULES:-reach score}"
# A function rather than an associative array, so this also runs on bash 3 (macOS).
config_for() {
  case "$1" in
    reach) echo configs/default.yaml ;;
    score) echo configs/variants/bucket-frame-score.yaml ;;
    *) return 1 ;;
  esac
}
for rule in $RULES; do
  config_for "$rule" > /dev/null || { echo "unknown rule '$rule' (use: reach score)" >&2; exit 2; }
done

mkdir -p "$DRIVE_OUT" "$WORK" || { echo "cannot create $DRIVE_OUT or $WORK (is Drive mounted?)" >&2; exit 2; }

echo "torch / GPU check:"
$PY -c "import torch; print('  torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
model="$($PY -c "from excavator_cycles.config import Config; print(Config.load('configs/default.yaml').track.model_id)" 2>/dev/null || echo unknown)"
commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "SAM model: $model   commit: $commit   frame rules: $RULES"
echo "videos: ${#VIDEOS[@]}   copying each to: $DRIVE_OUT"
echo

# Is this video there, finished uploading, and readable? A file still being written grows,
# and one cut short will not open, so both are checked before any GPU time is spent on it.
ready() {
  [ -f "$1" ] || return 1
  local before after
  before="$(wc -c < "$1" 2>/dev/null | tr -d ' ')"
  sleep 3
  after="$(wc -c < "$1" 2>/dev/null | tr -d ' ')"
  [ -n "$before" ] && [ "$before" -gt 0 ] && [ "$before" = "$after" ] || return 1
  $PY -c "import av, sys; av.open(sys.argv[1]).close()" "$1" > /dev/null 2>&1
}

done_count=0; skipped=0; missing=0; failed=0; pending=""
for video in "${VIDEOS[@]}"; do
  stem="$(basename "${video%.*}")"
  if [ ! -f "$video" ]; then
    echo "MISSING  $stem  (not in Drive yet: $video)"
    missing=$((missing + 1)); pending="$pending $stem"; continue
  fi
  if ! ready "$video"; then
    echo "NOT READY  $stem  (still uploading, or the file does not open); skipping it for now"
    missing=$((missing + 1)); pending="$pending $stem"; continue
  fi
  # The rules run back to back on one video, so an interrupted batch leaves comparable pairs.
  for rule in $RULES; do
    dest="$DRIVE_OUT/$rule/$stem"
    if [ -f "$dest/track.json" ]; then
      echo "skip     $rule  $stem  (already in Drive)"; skipped=$((skipped + 1)); continue
    fi

    echo "run      $rule  $stem  $(date '+%H:%M:%S')"
    local_video="$WORK/$stem.${video##*.}"
    local_out="$WORK/$rule/$stem"
    rm -rf "$local_out"; mkdir -p "$local_out"
    cp "$video" "$local_video"

    $PY run.py --log-file "$WORK/$rule/$stem.log" \
      track "$local_video" --config "$(config_for "$rule")" --out "$local_out" \
      > "$WORK/$rule/$stem.stdout" 2>&1
    code=$?
    rm -f "$local_video"

    # `track` exits non-zero when its QA flags a concern but still writes the masks, so what
    # counts as success is that both outputs exist, not the exit code.
    if [ ! -f "$local_out/track.json" ] || [ ! -f "$local_out/masks.npz" ]; then
      echo "FAILED   $rule  $stem  (exit $code, no track.json/masks.npz; see $WORK/$rule/$stem.stdout)"
      tail -n 5 "$WORK/$rule/$stem.stdout" | sed 's/^/           /'
      failed=$((failed + 1)); continue
    fi

    cp "$WORK/$rule/$stem.log" "$WORK/$rule/$stem.stdout" "$local_out/" 2>/dev/null
    printf 'video: %s\nrule: %s\nmodel: %s\ncommit: %s\nexit: %s\nfinished: %s\n' \
      "$video" "$rule" "$model" "$commit" "$code" "$(date '+%Y-%m-%d %H:%M:%S')" \
      > "$local_out/RUN_INFO.txt"

    mkdir -p "$DRIVE_OUT/$rule"
    partial="$DRIVE_OUT/$rule/.$stem.partial"
    rm -rf "$partial"
    if cp -r "$local_out" "$partial" && sync && mv "$partial" "$dest"; then
      echo "saved    $rule  $stem  exit $code  ->  $dest  $(date '+%H:%M:%S')"
      done_count=$((done_count + 1))
    else
      echo "FAILED   $rule  $stem  (tracked fine but the copy to Drive failed; results are in $local_out)"
      failed=$((failed + 1))
    fi
  done
done

echo
echo "=== done: $done_count saved, $skipped skipped, $missing missing or not ready, $failed failed ==="
if [ -n "$pending" ]; then
  echo "NOT RUN (upload them, then run this same cell again; finished ones are skipped):$pending"
fi
for rule in $RULES; do echo "$rule: $(ls -1 "$DRIVE_OUT/$rule" 2>/dev/null | tr '\n' ' ')"; done
echo "ALL DONE $(date '+%H:%M:%S')"
[ "$failed" -eq 0 ]
