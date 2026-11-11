"""The synthetic coach-demo fixtures (scripts/synth_plays.py) stay well-formed, physically plausible, and
keep showing what each play is built to show."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.coach_insights import analyse, summary
from scripts.synth_plays import build, plays

FIX = Path(__file__).resolve().parents[1] / "data/fixtures/synthetic"
BUILT = {p.id: build(p) for p in plays()}


def test_committed_fixtures_match_the_generator():
    on_disk = {p.stem: json.loads(p.read_text()) for p in FIX.glob("*.json")}
    assert set(on_disk) == set(BUILT)
    for pid, doc in BUILT.items():
        assert json.loads(json.dumps(doc, ensure_ascii=False)) == on_disk[pid], (
            f"rerun scripts/synth_plays.py ({pid})"
        )


@pytest.mark.parametrize("pid", sorted(BUILT))
def test_schema_and_physics(pid):
    d = BUILT[pid]
    n = len(d["ball"])
    assert d["schema"] == "bga.tracking2d/v0" and d["synthetic"] is True
    assert len(d["holder"]) == n
    teams = [p["team"] for p in d["players"]]
    assert teams.count("home") == 5 and teams.count("away") == 5
    xy = np.stack([np.array(d["tracks"][p["id"]], float) / 100 for p in d["players"]], 1)  # (n, 10, 2)
    assert xy.shape == (n, 10, 2)
    assert (
        (xy[..., 0] >= 0).all()
        and (xy[..., 0] <= 28).all()
        and (xy[..., 1] >= 0).all()
        and (xy[..., 1] <= 15).all()
    )
    # sustained speed (0.4 s window) within what elite players reach, about 9.5 m/s
    step = np.linalg.norm(np.diff(xy, axis=0), axis=2) * d["fps"]
    k = np.ones(10) / 10
    sustained = np.stack([np.convolve(step[:, j], k, mode="valid") for j in range(10)], 1)
    assert sustained.max() < 9.5
    # two players never share a spot for long (a screen is contact, not 0.4 s of overlap)
    dist = np.linalg.norm(xy[:, :, None] - xy[:, None], axis=3) + np.eye(10) * 99
    assert (dist < 0.35).sum(axis=0).max() <= 0.4 * d["fps"]
    assert all(h is None or h in d["tracks"] for h in d["holder"])


EXPECT = {  # play -> (offence, shot zone, openness at release, an open-player moment)
    "q1-0842-high-pnr": ("home", "corner3", "wide_open", ("open_found", "h13")),
    "q1-0615-horns": ("home", "paint", "contested", ("open_missed", "h5")),
    "q3-0548-zone-attack": ("home", "corner3", "wide_open", None),
    "q4-0420-switch-double": ("away", "three", "wide_open", None),
}


@pytest.mark.parametrize("pid", sorted(EXPECT))
def test_insights_show_the_point_of_the_play(pid):
    side, zone, openness, moment = EXPECT[pid]
    i = analyse(BUILT[pid])
    assert BUILT[pid]["offence"] == side
    assert (i["shot"]["zone"], i["shot"]["open"]) == (zone, openness)
    if moment:
        assert any((m["kind"], m.get("by")) == moment for m in i["moments"])


def test_horns_shot_is_the_poor_one():
    q = {pid: analyse(d)["quality"] for pid, d in BUILT.items()}
    assert q["q1-0615-horns"] == "poor"
    assert sum(v == "poor" for v in q.values()) == 1


def test_summary_counts_possessions():
    ps = [dict(d, insights=analyse(d)) for d in BUILT.values()]
    s = summary(ps)
    assert s["teams"]["home"]["possessions"] + s["teams"]["away"]["possessions"] == len(ps)
    assert all(r["number"] for r in s["players"])
