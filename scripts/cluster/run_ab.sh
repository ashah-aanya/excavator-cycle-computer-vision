#!/usr/bin/env bash
# Track the videos with several versions of the code, one shared venv, so only the code differs.
#
#   scripts/cluster/run_ab.sh A-pre-reseed B-reseed ...     names of folders under $WT
#   ONLY='random|vid2' scripts/cluster/run_ab.sh A-pre-reseed F-main   only matching videos
#
# Each version is a git worktree (`git worktree add --detach ~/wt/<name> <commit>`), so a
# version is just a commit hash. Only the GPU stage (`track`) runs; it writes masks.npz and
# track.json, which is all a bucket comparison needs.
#
# Resumable: a run whose track.json exists is skipped. A run that fails is logged and the
# batch carries on -- a failure is a result too.
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

[ "$#" -gt 0 ] || { echo "give at least one folder name from $WT" >&2; exit 2; }
for label in "$@"; do
  [ -d "$WT/$label" ] || { echo "no such version: $WT/$label" >&2; exit 2; }
done

# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "python: $(command -v python)"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

for video in "$VIDEOS"/*.mp4 "$VIDEOS"/*.mov; do
  [ -f "$video" ] || continue
  stem="$(basename "${video%.*}")"
  echo "$stem" | grep -Eq "$ONLY" || continue
  # Every version for one video, then the next video, so a job that dies half way still
  # leaves comparable rows.
  for label in "$@"; do
    dest="$OUT/$label/$stem"
    if [ -f "$dest/track.json" ]; then
      echo "skip  $label  $stem"
      continue
    fi
    mkdir -p "$dest"
    echo "run   $label  $stem  $(date +%H:%M:%S)"
    ( cd "$WT/$label" && python run.py --log-file "$OUT/$label/$stem.debug.log" \
        track "$video" --out "$dest" ) > "$OUT/$label/$stem.log" 2>&1
    echo "      exit $?  $(date +%H:%M:%S)"
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
