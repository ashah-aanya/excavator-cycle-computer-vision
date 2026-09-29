#!/usr/bin/env bash
# Track the videos with several versions of the code, one shared venv, so only the code differs.
#
#   scripts/cluster/run_ab.sh A-pre-reseed B-reseed ...     names of folders under $WT
#   ONLY='random|vid2' scripts/cluster/run_ab.sh A-pre-reseed F-main   only matching videos
#
# Each version is a git worktree (`git worktree add --detach ~/wt/<name> <commit>`), so a
# version is just a commit hash. Only the GPU stage (`track`) runs; it writes masks.npz,
# track.json and run.json, which is all a bucket comparison needs. No video is rendered.
#
# What you see: a plan, then for every run a banner (progress, elapsed time, a rough ETA,
# the version's commit, GPU memory), the run's own log streamed live, and a one-line result.
# A failed run prints the tail of its log and the batch carries on -- a failure is a result
# too. The full log of each run is kept at $OUT/<version>/<video>.log.
#
# ONLY is a regular expression matched against the file name WITHOUT its extension:
# ONLY='Untitled4' works, ONLY='Untitled4.mov' matches nothing.
#
# Resumable: a run whose track.json exists is skipped. Never delete finished output.
#
# Do not use `uv run` or `uv sync` for this: they can rebuild the venv and change torch.
# Run it under nohup or tmux; closing the terminal kills a plain foreground job.
#
# Paths (override with environment variables):
#   VENV=~/venv-t210   WT=~/wt   VIDEOS=~/videos   OUT=~/ab-results

set -u
VENV="${VENV:-$HOME/venv-t210}"
WT="${WT:-$HOME/wt}"
VIDEOS="${VIDEOS:-$HOME/videos}"
OUT="${OUT:-$HOME/ab-results}"
ONLY="${ONLY:-.}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ "$#" -gt 0 ] || { echo "give at least one folder name from $WT" >&2; exit 2; }
VERSIONS=("$@")
for label in "${VERSIONS[@]}"; do
  [ -d "$WT/$label" ] || { echo "no such version: $WT/$label" >&2; exit 2; }
done

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1   # otherwise a piped python holds its output back and nothing streams

# Which videos the filter picks (matched on the name without its extension).
MATCHED=()
ALL=()
for video in "$VIDEOS"/*.mp4 "$VIDEOS"/*.mov; do
  [ -f "$video" ] || continue
  stem="$(basename "${video%.*}")"
  ALL+=("$stem")
  echo "$stem" | grep -Eq "$ONLY" && MATCHED+=("$video")
done
if [ "${#MATCHED[@]}" -eq 0 ]; then
  echo "ONLY='$ONLY' matches none of the videos in $VIDEOS." >&2
  echo "It is matched against the name WITHOUT the extension. Available:" >&2
  printf '  %s\n' "${ALL[@]}" >&2
  exit 2
fi

TOTAL=$(( ${#MATCHED[@]} * ${#VERSIONS[@]} ))
echo "=== plan ==="
echo "python : $(command -v python)"
python -c "import torch; print('torch  :', torch.__version__, '| cuda:', torch.cuda.is_available())"
echo "videos : ${#MATCHED[@]}   versions : ${#VERSIONS[@]}   runs : $TOTAL   output : $OUT"
for video in "${MATCHED[@]}"; do echo "  video   $(basename "$video")"; done
for label in "${VERSIONS[@]}"; do
  echo "  version $label  $(git -C "$WT/$label" log -1 --format='%h %s' 2>/dev/null)"
done
echo

fmt() { printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 )); }

gpu_memory() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo "n/a"; return; }
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1
}

# One line about a finished run, read from its track.json.
result_line() {
  python - "$1" <<'PY'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
data = json.loads((run / "track.json").read_text())
qa = data["qa"]
bucket = qa.get("bucket") or {}
reseeds = data.get("bucket_reseeds", [])
found = sum(1 for r in reseeds if r.get("seed_sample") is not None)
notes = "; ".join(qa.get("failures", []))
print(
    f"samples {len(data['frames'])} | bucket coverage {bucket.get('coverage', float('nan')):.2f}"
    f" | area p90/p10 {bucket.get('area_p90_over_p10', float('nan')):.1f}x"
    f" | reseeds {found}/{len(reseeds)} | QA {qa['status']}" + (f" ({notes})" if notes else "")
)
PY
}

done_runs=0
ran_seconds=0
ran_count=0
START=$SECONDS
for video in "${MATCHED[@]}"; do
  stem="$(basename "${video%.*}")"
  # Every version for one video, then the next video, so a job that dies half way still
  # leaves comparable rows.
  for label in "${VERSIONS[@]}"; do
    done_runs=$(( done_runs + 1 ))
    dest="$OUT/$label/$stem"
    if [ -f "$dest/track.json" ]; then
      echo "[$done_runs/$TOTAL] skip  $label  $stem  (already done)"
      continue
    fi
    eta="?"
    if [ "$ran_count" -gt 0 ]; then
      eta="$(fmt $(( ran_seconds / ran_count * (TOTAL - done_runs + 1) )))"
    fi
    echo "------------------------------------------------------------------------"
    echo "[$done_runs/$TOTAL] $label  on  $stem   $(date +%H:%M:%S)"
    echo "    commit  $(git -C "$WT/$label" log -1 --format='%h %s' 2>/dev/null)"
    echo "    GPU mem $(gpu_memory) MiB used   elapsed $(fmt $(( SECONDS - START )))   ETA ~$eta"
    echo "------------------------------------------------------------------------"
    mkdir -p "$dest"
    began=$SECONDS
    # `tee` shows the run live and keeps it; PIPESTATUS[0] is python's own exit code.
    ( cd "$WT/$label" && python -u run.py --log-file "$OUT/$label/$stem.debug.log" \
        track "$video" --out "$dest" ) 2>&1 | tee "$OUT/$label/$stem.log"
    status=${PIPESTATUS[0]}
    took=$(( SECONDS - began ))
    ran_seconds=$(( ran_seconds + took ))
    ran_count=$(( ran_count + 1 ))
    echo
    if [ -f "$dest/track.json" ]; then
      # `track` exits non-zero when its QA flags a concern, but it still writes the masks.
      echo ">>> RESULT $label $stem  exit $status in $(fmt "$took")"
      echo ">>> $(result_line "$dest")"
      [ "$status" -ne 0 ] && echo ">>> (exit $status with a track.json = tracking finished and QA flagged a concern)"
    else
      echo ">>> FAILED $label $stem  exit $status after $(fmt "$took")  -- no track.json"
      echo ">>> last lines of $OUT/$label/$stem.log:"
      tail -15 "$OUT/$label/$stem.log" | sed 's/^/>>>   /'
      if grep -qi "out of memory" "$OUT/$label/$stem.log"; then
        echo ">>> This is a GPU out-of-memory error, not the code: re-run this one alone."
      fi
    fi
    echo
  done
done

echo "=== all runs finished $(date +%H:%M:%S), total $(fmt $(( SECONDS - START ))) ==="
python "$HERE/summarize_ab.py" "$OUT"
echo "ALL DONE $(date +%H:%M:%S)"
