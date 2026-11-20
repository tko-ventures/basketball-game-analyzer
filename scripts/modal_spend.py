#!/usr/bin/env python3
"""Modal spend from Modal's own billing: the month's total and the cost of each app, never an estimate.

    uv run --extra remote scripts/modal_spend.py                      # this month: total, then each app, newest first
    uv run --extra remote scripts/modal_spend.py --app ap-8Qb7SB...   # one or more apps (prefixes work)
    uv run --extra remote scripts/modal_spend.py --for 2026-10        # another month

Wraps ``modal billing summary`` and ``modal billing report --show-resources --json``. Use these figures in
docs/costs.md: the time a local process spends waiting on a remote call says nothing about how long the remote ran
(on 2026-11-18 such an estimate overstated three training runs about 25-fold).

"""

import argparse
import collections
import json
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--for", dest="period", default="this month")
ap.add_argument("--app", action="append", default=[], help="app id or prefix; repeatable")
a = ap.parse_args()

summary = subprocess.run(["modal", "billing", "summary", "--for", a.period], capture_output=True, text=True).stdout
print("\n".join(line.strip() for line in summary.splitlines() if "Cost" in line or "Credits" in line))
rows = json.loads(
    subprocess.run(
        ["modal", "billing", "report", "--for", a.period, "--show-resources", "--resolution", "d", "--json"],
        capture_output=True,
        text=True,
    ).stdout
)
apps: dict[str, dict] = collections.defaultdict(lambda: {"usd": 0.0, "by": collections.Counter()})
for r in rows:
    if a.app and not any(r["object_id"].startswith(x) for x in a.app):
        continue
    e = apps[r["object_id"]]
    e.update(name=r["description"], day=r["interval_start"][:10])
    e["usd"] += float(r["cost"])
    e["by"][r["resource"]] += float(r["cost"])
print(f"{'day':10}  {'app':26} {'name':18} {'USD':>8}  resources")
for app_id, e in sorted(apps.items(), key=lambda kv: kv[1]["day"], reverse=True):
    res = ", ".join(f"{k} {v:.3f}" for k, v in e["by"].most_common() if v >= 0.0005)
    print(f"{e['day']:10}  {app_id:26} {e['name'][:18]:18} {e['usd']:8.3f}  {res}")
