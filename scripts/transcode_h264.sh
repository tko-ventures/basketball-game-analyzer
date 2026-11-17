#!/usr/bin/env bash
# Re-encode AV1 / VP9 videos in data/raw to H.264 in place (OpenCV builds often can't decode AV1).
# Usage: scripts/transcode_h264.sh [files...]   (default: all of data/raw/*.mp4)
set -euo pipefail
cd "$(dirname "$0")/.."
files=("$@"); [ ${#files[@]} -eq 0 ] && files=(data/raw/*.mp4)
for f in "${files[@]}"; do
  case "$f" in *.tmp.mp4) continue;; esac
  codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$f" 2>/dev/null || echo error)
  [ "$codec" = "h264" ] && continue
  [ "$codec" = "error" ] && { echo "skip unreadable $f"; continue; }
  tmp="${f%.mp4}.h264.tmp.mp4"
  trap 'rm -f "$tmp"' EXIT
  trap 'exit 130' INT TERM
  echo "transcode $codec → h264: $f"
  ffmpeg -hide_banner -loglevel error -y -i "$f" -c:v libx264 -preset superfast -crf 20 \
    -pix_fmt yuv420p -c:a aac -b:a 96k -movflags +faststart "$tmp"
  mv -f "$tmp" "$f"
done
