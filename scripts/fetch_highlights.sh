#!/usr/bin/env bash
# Download Dziki Warszawa game highlights ("skrót") from YouTube into data/raw/.
# Needs yt-dlp + ffmpeg (`brew install yt-dlp ffmpeg`).
#
#   scripts/fetch_highlights.sh                      # the committed clip list, data/clip_ids.txt
#   IDS=other_ids.txt scripts/fetch_highlights.sh    # another id list (first column = YouTube id)
#   SEARCH=1 N=80 SINCE=20250901 scripts/fetch_highlights.sh      # discover new clips by search
#   SEARCH=1 QUERY="dziki warszawa mecz" scripts/fetch_highlights.sh
#   YTDLP="uvx --from yt-dlp[default] yt-dlp" scripts/fetch_highlights.sh   # newest yt-dlp
#
# Chunked requests (10 MiB) avoid YouTube's per-stream throttling: one unchunked stream ran at
# 40-150 KiB/s on 2026-10-29, the same video chunked at several MiB/s.
#
# Re-runs are incremental (data/raw/.archive.txt). A manifest with
# upload date / season / title / resolution is appended to data/raw/manifest.tsv.
# Only download footage you are allowed to use for analysis.
set -euo pipefail
cd "$(dirname "$0")/.."

IDS="${IDS:-data/clip_ids.txt}"
read -r -a ytdlp <<< "${YTDLP:-yt-dlp}"
QUERY="${QUERY:-dziki warszawa skrót}"
N="${N:-40}"
SINCE="${SINCE:-20240901}"          # start of 2024/25 season
MAX_MIN="${MAX_MIN:-20}"            # skip full-game uploads / long shows
OUT=data/raw
mkdir -p "$OUT"
[ -f "$OUT/manifest.tsv" ] || printf "upload_date\tid\twidth\theight\tfps\tduration_s\ttitle\tfile\n" > "$OUT/manifest.tsv"

if [ -n "${SEARCH:-}" ]; then
  targets=("ytsearch${N}:${QUERY}")
  filters=(--dateafter "$SINCE" --match-filter "duration < $((MAX_MIN*60)) & title ~= '(?i)dzik'")
else
  targets=()
  while IFS= read -r line; do
    id="${line%%[[:space:]]*}"
    [ -n "$id" ] && targets+=("https://www.youtube.com/watch?v=${id}")
  done < <(grep -v '^#' "$IDS")
  filters=()
fi

"${ytdlp[@]}" "${targets[@]}" \
  ${filters[@]+"${filters[@]}"} \
  -f "bv*[height<=1080]+ba/b[height<=1080]" -S "vcodec:h264,res:1080,fps" \
  --merge-output-format mp4 \
  --http-chunk-size 10M --concurrent-fragments 4 \
  -o "$OUT/%(upload_date)s_%(id)s_%(title).60B.%(ext)s" \
  --restrict-filenames \
  --download-archive "$OUT/.archive.txt" \
  --write-info-json --no-write-playlist-metafiles \
  --print-to-file "%(upload_date)s	%(id)s	%(width)s	%(height)s	%(fps)s	%(duration)s	%(title)s	%(filepath)s" "$OUT/manifest.tsv" \
  --ignore-errors --no-overwrites

echo "Done. $(($(wc -l < "$OUT/manifest.tsv")-1)) videos in $OUT/manifest.tsv"
echo "Next: scripts/transcode_h264.sh (if any are AV1/VP9), then uv run tools/labeler/prepare_clips.py"
