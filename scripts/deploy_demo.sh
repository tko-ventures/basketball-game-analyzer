#!/usr/bin/env bash
# Publish the web demo (runs/demo/web, built by scripts/demo_web.py) to Vercel production.
#
#   scripts/deploy_demo.sh            # -> https://basketball-analyzer-demo.vercel.app
#
# Uploads only the page, its videos, data and court image, with noindex headers (match footage, client demo).
# Needs the Vercel CLI logged in (`vercel login`). The production URL is public.
set -euo pipefail
cd "$(dirname "$0")/.."
SCOPE="${VERCEL_SCOPE:-tomasz-kopackis-projects}"
D=runs/demo/deploy/basketball-analyzer-demo
mkdir -p "$D"
rsync -a --delete --exclude .vercel --exclude vercel.json --exclude summary.json runs/demo/web/ "$D/"
cat > "$D/vercel.json" <<'JSON'
{
  "headers": [
    { "source": "/(.*)", "headers": [{ "key": "X-Robots-Tag", "value": "noindex, nofollow" }] },
    { "source": "/(.*)\\.mp4", "headers": [{ "key": "Cache-Control", "value": "public, max-age=3600" }] }
  ]
}
JSON
(cd "$D" && vercel deploy --prod --yes --scope "$SCOPE")
