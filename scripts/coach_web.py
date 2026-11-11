#!/usr/bin/env python3
"""Build the coach page of the web demo from the synthetic tracking fixtures.

    uv run scripts/synth_plays.py                 # only when the plays change
    uv run scripts/coach_web.py [--out runs/demo/web]
    uv run scripts/serve_demo.py runs/demo/web    # then open http://127.0.0.1:8770/en/coach/

Writes <out>/coach/index.html (Polish) and <out>/en/coach/index.html (English), each a single file
with the fixtures and their insights inside. The page has three steps in the order a coach works:
watch the possessions, read the insights (scripts/coach_insights.py), give feedback (drawings and
notes addressed to players, kept in the browser, exported as text).

The data is simulated (scripts/synth_plays.py) and the page says so. A showcase, not a pipeline layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.coach_insights import OPEN_M, analyse, summary  # noqa: E402
from scripts.coach_strings import T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data/fixtures/synthetic"
TEMPLATE = ROOT / "scripts/coach_page.html"


def load_plays() -> list[dict]:
    plays = [json.loads(p.read_text()) for p in FIXTURES.glob("*.json")]
    plays.sort(key=lambda d: (d["game"]["quarter"], -d["game"]["clock_start_s"]))
    for d in plays:
        d["insights"] = analyse(d)
    return plays


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs/demo/web")
    a = ap.parse_args(argv)
    plays = load_plays()
    data = {"plays": plays, "summary": summary(plays), "open_m": OPEN_M}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = TEMPLATE.read_text()
    for lang in T:
        base = a.out if lang == "pl" else a.out / lang
        other = "../../en/coach/" if lang == "pl" else "../../coach/"
        html = (
            template.replace("__DATA__", blob)
            .replace("__T__", json.dumps(T[lang], ensure_ascii=False).replace("</", "<\\/"))
            .replace("{{lang}}", lang)
            .replace("{{title}}", T[lang]["title"])
            .replace("{{lang_href}}", other)
        )
        dst = base / "coach" / "index.html"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(html)
        print(f"{dst}  {len(html) // 1024} kB, {len(plays)} possessions")


if __name__ == "__main__":
    main()
