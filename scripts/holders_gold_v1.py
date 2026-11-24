#!/usr/bin/env python3
"""Holder gold v1: write data/gold/v1/holders.json from box gold and data/labels/review/holders_v1.ops.txt.

    uv run scripts/holders_gold_v1.py

The logic lives in bga.eval.holders (shared with the labeller's /holders page, where Tomasz decides the flagged
frames); see that module and docs/gold.md, "Holder gold v1".

"""

from __future__ import annotations

from bga.eval.holders import HOLDERS_PATH, build_holders


def main() -> None:
    counts = build_holders()
    print(
        f"{HOLDERS_PATH}: {counts['frames']} frames, {counts['changed']} changed from box gold "
        f"({counts['tomasz']} by Tomasz), {counts['flagged']} flagged for Tomasz"
    )


if __name__ == "__main__":
    main()
