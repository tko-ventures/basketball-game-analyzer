#!/usr/bin/env python3
"""Compare the shot cuts of two sets of label files and print a markdown table.

    uv run scripts/compare_shots.py OLD_DIR NEW_DIR [--tol 2]
    uv run scripts/compare_shots.py --git-rev c8ef83f data/labels   # OLD = files at a commit

A cut in NEW matches a cut in OLD if they are at most ``--tol`` frames apart (one-to-one, greedy by
distance). Frame 0 is not counted as a cut.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def cuts(data: dict) -> list[int]:
    return [s["start"] for s in data.get("shots", []) if s["start"] > 0]


def match(a: list[int], b: list[int], tol: int) -> int:
    pairs = sorted((abs(x - y), i, j) for i, x in enumerate(a) for j, y in enumerate(b) if abs(x - y) <= tol)
    used_a, used_b, n = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            n += 1
    return n


def load_old(name: str, old_dir: Path | None, rev: str | None, new_dir: Path) -> dict | None:
    if rev:
        try:
            out = subprocess.run(
                ["git", "show", f"{rev}:{(new_dir / name).as_posix()}"], capture_output=True, check=True
            )
        except subprocess.CalledProcessError:
            return None
        return json.loads(out.stdout)
    p = old_dir / name
    return json.loads(p.read_text()) if p.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("old", nargs="?", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--git-rev")
    ap.add_argument("--tol", type=int, default=2)
    a = ap.parse_args(argv)
    print(
        f"| clip | frames old/new | shots old | shots new | cuts matched (±{a.tol}) | only old | only new |"
    )
    print("|---|---|---|---|---|---|---|")
    tot = [0, 0, 0]
    for p in sorted(a.new.glob("*.json")):
        new = json.loads(p.read_text())
        old = load_old(p.name, a.old, a.git_rev, a.new)
        if old is None:
            print(f"| {p.stem[:40]} | – | – | {len(new['shots'])} | – | – | – |")
            continue
        co, cn = cuts(old), cuts(new)
        m = match(co, cn, a.tol)
        tot[0] += m
        tot[1] += len(co) - m
        tot[2] += len(cn) - m
        print(
            f"| {p.stem[:40]} | {old['frame_count']}/{new['frame_count']} | {len(old['shots'])} | "
            f"{len(new['shots'])} | {m} | {len(co) - m} | {len(cn) - m} |"
        )
    print(f"| **total** | | | | **{tot[0]}** | **{tot[1]}** | **{tot[2]}** |")


if __name__ == "__main__":
    main()
