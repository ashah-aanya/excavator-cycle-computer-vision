#!/usr/bin/env bash
# Pack what is needed to render one video's results on another machine into ONE file:
# each version's saved run (masks, track.json, features, answer -- NOT the big annotated
# videos) plus the source video. Then download the .tgz through JupyterLab's file browser.
#
#   make_bundle.sh Untitled4          the video's name WITHOUT its extension
#
# A version's saved run is looked for in $OUT/<version>/<video>/ first, then in that
# version's own folder ($WT/<version>/outputs/track/<video>/), because the older versions
# keep their cache there.
#
# Paths (override with environment variables):
#   WT=~/wt   OUT=~/ab-results   VIDEOS=~/videos   BUNDLE=~/bundle

set -u
STEM="${1:?give the video name without its extension, e.g. Untitled4}"
WT="${WT:-$HOME/wt}"
OUT="${OUT:-$HOME/ab-results}"
VIDEOS="${VIDEOS:-$HOME/videos}"
BUNDLE="${BUNDLE:-$HOME/bundle}"

rm -rf "$BUNDLE" "$BUNDLE.tgz"
mkdir -p "$BUNDLE"

packed=0
for dir in "$WT"/*/; do
  label="$(basename "$dir")"
  [ "$label" = "tools" ] && continue
  src="$OUT/$label/$STEM"
  [ -f "$src/track.json" ] || src="$dir/outputs/track/$STEM"
  if [ ! -f "$src/track.json" ]; then
    echo "skip   $label  (no saved run for $STEM)"
    continue
  fi
  mkdir -p "$BUNDLE/$label"
  cp -r "$src"/. "$BUNDLE/$label/"
  rm -f "$BUNDLE/$label"/annotated*.mp4
  [ -f "$OUT/$label/$STEM.log" ] && cp "$OUT/$label/$STEM.log" "$BUNDLE/$label/"
  echo "packed $label  ($(du -sh "$BUNDLE/$label" | cut -f1))"
  packed=$(( packed + 1 ))
done

video="$(ls "$VIDEOS/$STEM".* 2>/dev/null | head -1)"
if [ -n "$video" ]; then
  cp "$video" "$BUNDLE/"
  echo "packed source video $(basename "$video")  ($(du -h "$video" | cut -f1))"
else
  echo "WARNING: no source video named $STEM.* in $VIDEOS" >&2
fi
cp "$OUT"/batch*.log "$BUNDLE/" 2>/dev/null

tar czf "$BUNDLE.tgz" -C "$(dirname "$BUNDLE")" "$(basename "$BUNDLE")"
echo
echo "$packed version(s) packed."
ls -lh "$BUNDLE.tgz"
echo "Download it from JupyterLab's file browser (right-click > Download)."
