#!/usr/bin/env python3
"""Check that a running labeller opens every clip with shots ready to tag (M0 done-when).

    uv run tools/labeler/server.py &          # http://127.0.0.1:8765
    uv run scripts/check_labeller.py          # markdown table, exit 1 on any failure

Per clip: /api/clip returns version-2 labels with home and away teams and more than one shot, and the
middle frame of the first and of the last shot decodes through /api/frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request


def get(base: str, route: str, **q) -> bytes:
    url = f"{base}{route}?{urllib.parse.urlencode(q)}" if q else f"{base}{route}"
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    a = ap.parse_args(argv)
    clips = json.loads(get(a.url, "/api/clips"))["clips"]
    print("| clip | fps | frames | shots | home | away | season | frames decode | open s |")
    print("|---|---|---|---|---|---|---|---|---|")
    failures = 0
    for c in clips:
        t0 = time.time()
        L = json.loads(get(a.url, "/api/clip", path=c["path"]))["labels"]
        shots = L["shots"]
        probes = [(s["start"] + s["end"]) // 2 for s in (shots[0], shots[-1])]
        decoded = sum(len(get(a.url, "/api/frame", path=c["path"], idx=i)) > 1000 for i in probes)
        m = L["meta"]
        ok = L["version"] == 2 and len(shots) > 1 and decoded == len(probes) and m["home"]["team"]
        failures += not ok
        print(
            f"| {c['path'].split('/')[-1][:44]} | {L['fps']:g} | {L['frame_count']} | {len(shots)} | "
            f"{m['home']['team']} | {m['away']['team']} | {m['season']} | {decoded}/{len(probes)} | "
            f"{time.time() - t0:.1f} |{'' if ok else ' **FAIL**'}"
        )
    print(f"\n{len(clips) - failures}/{len(clips)} clips open with shots ready to tag")
    return 1 if failures or not clips else 0


if __name__ == "__main__":
    sys.exit(main())
