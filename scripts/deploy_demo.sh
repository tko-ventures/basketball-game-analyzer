#!/usr/bin/env bash
# Publish the web demo (runs/demo/web, built by scripts/demo_web.py) to Vercel production.
#
#   scripts/deploy_demo.sh            # -> https://basketball-analyzer.demo.tko.ventures (vercel.app redirects there)
#
# Editions: Polish at / and /analysis/, English at /en/ and /en/analysis/ (both built by demo_web.py).
# The coach page (/coach/, /en/coach/, simulated data) is rebuilt here by coach_web.py, so it is never stale.
# Also publishes docs/roadmap.html as /roadmap/ (the plan at a glance; not linked from the demo pages).
#
# Uploads only the page, its videos, data and court image, with noindex headers (match footage, client demo).
# Needs the Vercel CLI logged in (`vercel login`). The production URL is public.
# Deploys only from main as it is on GitHub, with a clean tree, so the live site always matches main;
# each deployment carries the commit as Vercel metadata. BGA_DEPLOY_ANY_BRANCH=1 skips the check.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${BGA_DEPLOY_ANY_BRANCH:-}" != 1 ]; then
  git fetch -q origin main
  branch=$(git branch --show-current)
  if [ "$branch" != main ] || [ -n "$(git status --porcelain)" ] || [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
    echo "deploy_demo.sh: deploy from a clean main that matches origin/main (now: ${branch:-detached}," \
      "$(git rev-parse --short HEAD), origin/main $(git rev-parse --short origin/main))" >&2
    exit 1
  fi
fi
COMMIT=$(git rev-parse --short HEAD)
SCOPE="${VERCEL_SCOPE:-tomasz-kopackis-projects}"
D=runs/demo/deploy/basketball-analyzer-demo
mkdir -p "$D"
uv run scripts/coach_web.py --out runs/demo/web
rsync -a --delete --exclude .vercel --exclude vercel.json --exclude summary.json runs/demo/web/ "$D/"
mkdir -p "$D/roadmap" && cp docs/roadmap.html "$D/roadmap/index.html"
cat > "$D/vercel.json" <<'JSON'
{
  "redirects": [
    { "source": "/analiza", "destination": "/analysis/", "permanent": true },
    { "source": "/analiza/(.*)", "destination": "/analysis/$1", "permanent": true }
  ],
  "headers": [
    { "source": "/(.*)", "headers": [{ "key": "X-Robots-Tag", "value": "noindex, nofollow" }] },
    { "source": "/(.*)\\.mp4", "headers": [{ "key": "Cache-Control", "value": "public, max-age=3600" }] }
  ]
}
JSON
(cd "$D" && vercel deploy --prod --yes --scope "$SCOPE" --meta "commit=$COMMIT")
