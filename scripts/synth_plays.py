#!/usr/bin/env python3
# ruff: noqa: E501  (play tables: one waypoint list per player per line)
"""Write synthetic 2D tracking fixtures for the coach demo: data/fixtures/synthetic/*.json.

    uv run scripts/synth_plays.py            # regenerate every fixture (deterministic)

Each fixture is one possession: ten players and the ball at 25 fps on a FIBA 28 x 15 m court, in the
shape the pipeline is meant to deliver one day (docs/fixtures.md). Nothing here is measured. The
offence is choreographed by hand from set plays every team runs (high pick-and-roll, Horns, Spain
pick-and-roll, a 3-on-2 break, a zone attack, staggered screens); the defence is simulated: every
defender follows a man-to-man or zone rule with a reaction delay and a speed and acceleration limit,
plus the scripted decisions that make the play what it is (drop, switch, tag, double team).

Plays are written in half-court coordinates (u, v): u metres from the baseline behind the attacked
basket, v metres across the court. The basket is at (1.575, 7.5). They are then placed on the full
court by the direction the offence attacks in that quarter.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FPS = 25
L, W = 28.0, 15.0
BASKET = np.array([1.575, 7.5])
OUT = Path(__file__).resolve().parents[1] / "data/fixtures/synthetic"
SCHEMA = "bga.tracking2d/v0"

HOME, AWAY = "Dziki Warszawa", "Opponent"
# Dziki's 2026/27 roster (data/rosters/dziki_warszawa.json). The opponent is deliberately anonymous.
HOME_PLAYERS = {
    "1": ("Austin Luke", "PG"),
    "2": ("Makai Ashton", "PG"),
    "5": ("Michał Aleksandrowicz", "SG"),
    "8": ("Krzysztof Kempa", "PF"),
    "11": ("Grzegorz Kamiński", "SF"),
    "13": ("Quanterrius Jackson", "SG"),
    "22": ("Timothy Adetukasi", "SF"),
    "23": ("OsunBowale Osunniyi", "C"),
    "24": ("Zane Waterman", "PF"),
}
AWAY_PLAYERS = {"4": "PG", "7": "SG", "10": "SF", "21": "PF", "15": "C", "9": "SG", "33": "PF"}


@dataclass
class Play:
    id: str
    quarter: int
    clock: str  # game clock at the first frame, m:ss
    score: tuple[int, int]  # home, away before the possession
    offence: str  # "home" or "away"
    labels: dict  # what a play-recognition layer would say: play, defence, coverage
    dur: float
    off: dict  # jersey number -> [(t, u, v), ...]
    deff: dict  # jersey number -> {"guard": number} | {"zone": (u, v)} | {"path": [...]}, optional "ops"
    ball: list  # ("hold", n, t) ("pass", n, t_release) ("shot", t_release, made) ("reb", pid, t_catch)
    marks: list = field(default_factory=list)  # (t, type, {...}) tactical events
    shot_clock: float = 24.0


# ---------------------------------------------------------------- interpolation


def pchip(ts, ys, t):
    """Monotone cubic interpolation (Fritsch-Carlson): smooth, no overshoot, flat where a value repeats."""
    ts, ys = np.asarray(ts, float), np.asarray(ys, float)
    h = np.diff(ts)
    d = np.diff(ys) / h
    m = np.zeros_like(ys)
    for i in range(1, len(ys) - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2)
    x = np.clip((t - ts[i]) / h[i], 0, 1)
    h00, h10 = 2 * x**3 - 3 * x**2 + 1, x**3 - 2 * x**2 + x
    h01, h11 = -2 * x**3 + 3 * x**2, x**3 - x**2
    return h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1]


def path(wps, t, dur):
    wps = sorted(wps)
    if wps[0][0] > 0:
        wps = [(0.0, *wps[0][1:])] + wps
    if wps[-1][0] < dur:
        wps = wps + [(dur, *wps[-1][1:])]
    ts = [w[0] for w in wps]
    return np.stack([pchip(ts, [w[1] for w in wps], t), pchip(ts, [w[2] for w in wps], t)], axis=1)


def sway(key, t, amp=0.06):
    """Slow, small drift so nobody stands as a frozen dot."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    out = np.zeros((len(t), 2))
    for _ in range(3):
        f, ph = rng.uniform(0.15, 0.6), rng.uniform(0, 2 * math.pi, 2)
        out += amp / 2 * np.stack([np.sin(2 * math.pi * f * t + ph[0]), np.sin(2 * math.pi * f * t + ph[1])], 1)
    return out


# ---------------------------------------------------------------- ball


def ball_plan(play, pos, t):
    """Per frame: holder id or None, and ball (u, v, z). pos maps player id -> (n, 2) array."""
    n = len(t)
    holder = [None] * n
    ball = np.zeros((n, 3))
    events = []
    fr = lambda s: int(round(s * FPS))  # noqa: E731
    cur, t_from = None, 0.0
    segs = []  # (kind, f0, f1, a, b) with a, b player ids or None
    for item in play.ball:
        kind = item[0]
        if kind == "hold":
            cur, t_from = item[1], item[2]
        elif kind == "pass":
            to, t_rel = item[1], item[2]
            dist = float(np.linalg.norm(pos[to][fr(t_rel)] - pos[cur][fr(t_rel)]))
            t_catch = item[3] if len(item) > 3 else t_rel + max(0.28, dist / 13.0)
            segs.append(("hold", fr(t_from), fr(t_rel), cur, None))
            segs.append(("pass", fr(t_rel), fr(t_catch), cur, to))
            events.append({"t": t_rel, "t1": round(t_catch, 2), "type": "pass", "from": cur, "to": to})
            cur, t_from = to, t_catch
        elif kind == "shot":
            t_rel, made = item[1], item[2]
            p = pos[cur][fr(t_rel)]
            d = float(np.linalg.norm(p - BASKET))
            three = d > 6.75 or (p[0] <= 2.99 and abs(p[1] - 7.5) >= 6.6)
            flight = 0.45 + 0.08 * d
            segs.append(("hold", fr(t_from), fr(t_rel), cur, None))
            segs.append(("shot", fr(t_rel), fr(t_rel + flight), cur, None))
            events.append(
                {"t": t_rel, "t1": round(t_rel + flight, 2), "type": "shot", "by": cur, "made": made,
                 "points": 3 if three else 2}
            )  # fmt: skip
            cur, t_from = None, t_rel + flight
            segs.append(("made" if made else "rim", fr(t_from), n, None, None))
        elif kind == "reb":
            who, t_catch = item[1], item[2]
            segs = [s for s in segs if s[0] != "rim"]
            segs.append(("reb", fr(t_from), fr(t_catch), None, who))
            segs.append(("hold", fr(t_catch), n, who, None))
            events.append({"t": t_catch, "type": "rebound", "by": who})
            cur = who
    if cur is not None and not any(s[0] in ("made", "rim", "reb") for s in segs):
        segs.append(("hold", fr(t_from), n, cur, None))
    vel = {p: np.gradient(xy, 1 / FPS, axis=0) for p, xy in pos.items()}
    for kind, f0, f1, a, b in segs:
        f1 = min(f1, n)
        for f in range(f0, f1):
            x = (f - f0) / max(1, f1 - f0)
            if kind == "hold":
                holder[f] = a
                v = vel[a][max(0, f - 3) : f + 4].mean(axis=0)
                sp = np.linalg.norm(v)
                d = v / sp if sp > 0.8 else (BASKET - pos[a][f]) / (np.linalg.norm(BASKET - pos[a][f]) + 1e-6)
                ball[f, :2] = pos[a][f] + 0.35 * d
                ball[f, 2] = 1.0
            elif kind == "pass":
                ball[f, :2] = (1 - x) * (pos[a][f0] + 0) + x * pos[b][min(f1, n - 1)]
                ball[f, 2] = 1.3 + 0.8 * 4 * x * (1 - x)
            elif kind == "shot":
                ball[f, :2] = (1 - x) * pos[a][f0] + x * BASKET
                ball[f, 2] = 2.2 + (3.05 - 2.2) * x + 2.0 * 4 * x * (1 - x)
            elif kind == "made":
                ball[f, :2] = BASKET + np.array([0.25, 0.0]) * min(1, x * 6)
                ball[f, 2] = max(0.2, 3.05 - 9 * (f - f0) / FPS)
            elif kind == "rim":
                ball[f, :2] = BASKET
                ball[f, 2] = 3.05
            elif kind == "reb":
                ball[f, :2] = (1 - x) * BASKET + x * pos[b][min(f1, n - 1)]
                ball[f, 2] = 3.05 - 0.8 * x + 1.2 * 4 * x * (1 - x)
    # smooth the hand-to-flight joins a little
    k = np.ones(3) / 3
    for c in range(2):
        ball[1:-1, c] = np.convolve(ball[:, c], k, mode="valid")
    return holder, ball, events


def ball_proxy(play, pos_off, t):
    """Where the defence sees the ball before defenders exist: attackers' hands, flights, then the rim."""
    stub = dict(pos_off)
    for item in play.ball:
        if item[0] == "reb" and item[1] not in stub:
            stub[item[1]] = np.tile(BASKET + np.array([1.0, 0.0]), (len(t), 1))
    return ball_plan(play, stub, t)


# ---------------------------------------------------------------- defence


def man_target(a, ball, on_ball):
    to_b = BASKET - a
    d = np.linalg.norm(to_b) + 1e-6
    if on_ball:
        gap = float(np.clip(0.95 + 0.09 * (d - 5.0), 0.9, 2.2))
        return a + to_b / d * min(gap, d * 0.6)
    w = float(np.clip(np.linalg.norm(a - ball) / 9.0, 0, 1))
    off = (0.14 + 0.20 * w) * to_b + (0.05 + 0.15 * w) * (ball - a)
    n = np.linalg.norm(off)
    return a + off * min(1.0, 3.2 / (n + 1e-6))


def zone_target(home, ball, holder_pos, others_home):
    shift = np.array([0.22, 0.42]) * (ball - np.array([5.0, 7.5]))
    tgt = home + shift
    if holder_pos is not None:
        d = np.linalg.norm(holder_pos - home)
        if d < 5.2 and all(d <= np.linalg.norm(holder_pos - o) + 1e-9 for o in others_home):
            to_b = BASKET - holder_pos
            tgt = holder_pos + to_b / (np.linalg.norm(to_b) + 1e-6) * 1.15
    return tgt


def simulate_defence(play, off, t, atk, dfn):
    """off: attacker id -> track. Returns defender id -> track."""
    n = len(t)
    holder, ball, _ = ball_proxy(play, off, t)
    delay = int(0.2 * FPS)
    vel = {p: np.gradient(xy, 1 / FPS, axis=0) for p, xy in off.items()}
    out, state = {}, {}
    scripted = {}
    for num, spec in play.deff.items():
        if "path" in spec:
            scripted[dfn + num] = path(spec["path"], t, play.dur)
    homes = {dfn + num: np.array(spec["zone"], float) for num, spec in play.deff.items() if "zone" in spec}

    def target(num, spec, f):
        g = max(0, f - delay)
        tt = t[f]
        guard = spec.get("guard")
        cap, tau = 6.6, 0.32
        fixed = None
        for op in spec.get("ops", []):
            if op[0] == "guard" and tt >= op[1]:
                guard = op[2]
            elif op[0] == "slow" and op[1] <= tt < op[2]:
                cap = op[3]
            elif op[0] == "goto" and op[1] <= tt < op[2]:
                fixed = np.array(op[3], float)
            elif op[0] == "ball" and op[1] <= tt < op[2]:
                h = holder[g]
                src = off[h][g] if h in off else ball[g, :2]
                fixed = man_target(src, ball[g, :2], True) if h in off else src
            elif op[0] == "lazy" and op[1] <= tt < op[2]:
                tau = op[3]
        if fixed is not None:
            return fixed, cap, tau
        if "zone" in spec:
            me = dfn + num
            hp = off[holder[g]][g] if holder[g] in off else None
            return zone_target(homes[me], ball[g, :2], hp, [h for k, h in homes.items() if k != me]), cap, tau
        a = off[atk + guard][g]
        on_ball = holder[g] == atk + guard
        if not on_ball:  # a defender reads where his man is running and goes with him
            a = a + 0.3 * vel[atk + guard][g]
        return man_target(a, ball[g, :2], on_ball), cap, tau

    for num, spec in play.deff.items():
        pid = dfn + num
        if pid in scripted:
            out[pid] = scripted[pid]
            continue
        p, _, _ = target(num, spec, 0)
        state[pid] = (p.copy(), np.zeros(2))
        out[pid] = np.zeros((n, 2))
    dt, amax = 1 / FPS, 9.0
    for f in range(n):
        for num, spec in play.deff.items():
            pid = dfn + num
            if pid in scripted:
                continue
            p, v = state[pid]
            tgt, cap, tau = target(num, spec, f)
            want = (tgt - p) / tau
            sp = np.linalg.norm(want)
            if sp > cap:
                want *= cap / sp
            dv = want - v
            na = np.linalg.norm(dv)
            if na > amax * dt:
                dv *= amax * dt / na
            v = v + dv
            p = p + v * dt
            state[pid] = (p, v)
            out[pid][f] = p
    # bodies do not overlap: push each simulated defender out to 0.75 m from attackers and 0.7 m from
    # other defenders (both give way), a few relaxation rounds per frame, then smooth the pushes
    moving = [pid for pid in out if pid not in scripted]
    for f in range(n):
        for _ in range(3):
            for pid in moving:
                p = out[pid][f]
                for q, xy in [*off.items(), *out.items()]:
                    if q == pid:
                        continue
                    d = p - xy[f]
                    dist = float(np.hypot(d[0], d[1]))
                    lim = 0.75 if q in off else 0.7
                    if dist < lim:
                        p += d / (dist + 1e-6) * (lim - dist) * (1.0 if q in off or q not in moving else 0.5)
    for pid in moving:
        xy = out[pid]
        pad = np.pad(xy, ((3, 3), (0, 0)), mode="edge")
        xy[:] = np.stack([np.convolve(pad[:, c], np.ones(7) / 7, mode="valid") for c in range(2)], 1)
    return out


# ---------------------------------------------------------------- build


def clock_seconds(s):
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def build(play: Play) -> dict:
    n = int(round(play.dur * FPS)) + 1
    t = np.arange(n) / FPS
    atk, dfn = ("h", "a") if play.offence == "home" else ("a", "h")
    off = {atk + num: path(w, t, play.dur) + sway(play.id + atk + num, t) for num, w in play.off.items()}
    de = simulate_defence(play, off, t, atk, dfn)
    de = {pid: xy + sway(play.id + pid, t, 0.04) for pid, xy in de.items()}
    pos = {**off, **de}
    for xy in pos.values():
        xy[:, 0] = np.clip(xy[:, 0], 0.3, L - 0.3)
        xy[:, 1] = np.clip(xy[:, 1], 0.3, W - 0.3)
    holder, ball, events = ball_plan(play, pos, t)
    for mt, typ, extra in play.marks:
        ev = {"t": mt, "type": typ}
        for k, v in extra.items():
            ev[k] = (atk if k in ("by", "for") else dfn) + v if k in ("by", "for", "on", "from_def", "to_def") else v
        events.append(ev)
    events.sort(key=lambda e: e["t"])
    # Dziki attack the right basket in the first half, the left one in the second.
    attacks = ("right" if play.quarter <= 2 else "left") if play.offence == "home" else (
        "left" if play.quarter <= 2 else "right")  # fmt: skip

    def place(uv):
        return np.stack([L - uv[:, 0], W - uv[:, 1]], 1) if attacks == "right" else uv

    cm = lambda a: np.round(a * 100).astype(int).tolist()  # noqa: E731
    players = []
    for pid in sorted(pos, key=lambda p: (p[0] != "h", int(p[1:]))):
        num = pid[1:]
        if pid[0] == "h":
            name, role = HOME_PLAYERS[num]
        else:
            name, role = "", AWAY_PLAYERS[num]
        players.append({"id": pid, "team": "home" if pid[0] == "h" else "away", "number": num, "name": name,
                        "pos": role})  # fmt: skip
    shot = next((e for e in events if e["type"] == "shot"), None)
    pts = shot["points"] if shot and shot["made"] else 0
    after = list(play.score)
    after[0 if play.offence == "home" else 1] += pts
    b = place(ball[:, :2])
    return {
        "schema": SCHEMA,
        "id": play.id,
        "synthetic": True,
        "fps": FPS,
        "court": {"length_cm": int(L * 100), "width_cm": int(W * 100)},
        "game": {
            "home": HOME, "away": AWAY, "quarter": play.quarter, "clock_start_s": clock_seconds(play.clock),
            "shot_clock_start_s": play.shot_clock, "score_before": list(play.score), "score_after": after,
        },  # fmt: skip
        "offence": play.offence,
        "attacks": attacks,
        "labels": play.labels,
        "players": players,
        "tracks": {p["id"]: cm(place(pos[p["id"]])) for p in players},
        "ball": [[x, y, z] for (x, y), z in zip(cm(b), cm(ball[:, 2]), strict=True)],
        "holder": holder,
        "events": events,
    }


# ---------------------------------------------------------------- the plays

C = (1.575, 7.5)  # the basket, for reference when reading waypoints


def plays() -> list[Play]:
    out = []

    # 1. High pick-and-roll against drop coverage. The roll pulls the weak-side low man into the paint
    #    (the tag); the ball goes to the corner he left. Open corner three, made.
    out.append(Play(
        id="q1-0842-high-pnr", quarter=1, clock="8:42", score=(2, 5), offence="home", dur=12.4,
        labels={"play": "high_pnr", "defence": "man", "coverage": "drop"},
        off={
            "1": [(0, 15.5, 7.5), (2.6, 10.2, 8.6), (5.2, 9.8, 8.4), (6.8, 7.6, 5.6), (8.0, 5.6, 5.3), (8.6, 5.2, 5.5),
                  (10.5, 6.4, 5.0)],
            "23": [(0, 10.5, 10.5), (3.0, 6.5, 9.2), (5.0, 9.2, 7.0), (6.5, 9.0, 6.9), (8.2, 3.6, 6.9), (9.2, 2.4, 6.6),
                   (11.5, 2.6, 6.4)],
            "13": [(0, 11.5, 13.0), (3.2, 1.3, 14.4), (12.4, 1.3, 14.4)],
            "11": [(0, 12.0, 2.2), (3.6, 1.3, 0.6), (12.4, 1.3, 0.6)],
            "24": [(0, 12.5, 11.0), (3.0, 7.2, 13.0), (7.0, 7.2, 13.0), (9.0, 8.8, 11.6)],
        },
        deff={
            "4": {"guard": "1", "ops": [("slow", 6.3, 7.4, 2.2)]},
            "15": {"guard": "23", "ops": [("goto", 5.0, 7.0, (6.0, 6.4)), ("ball", 7.0, 8.7)]},
            "7": {"guard": "13", "ops": [("goto", 7.3, 9.15, (3.2, 7.4))]},
            "10": {"guard": "11"},
            "21": {"guard": "24"},
        },
        ball=[("hold", "h1", 0), ("pass", "h13", 8.5), ("shot", 9.95, True)],
        marks=[(5.4, "screen", {"by": "23", "for": "1", "on": "4"}), (7.4, "tag", {"on": "7", "for": "23"})],
    ))

    # 2. Horns: elbow entry, the guard cuts into the corner that is already taken, hand-off, drive
    #    into a crowded lane, contested pull-up. The lifted weak-side wing stands open and is not found.
    out.append(Play(
        id="q1-0615-horns", quarter=1, clock="6:15", score=(9, 10), offence="home", dur=12.6,
        labels={"play": "horns", "defence": "man", "coverage": "show"},
        off={
            "2": [(0, 14.5, 7.5), (3.0, 9.6, 7.5), (3.6, 9.4, 7.3), (5.2, 5.6, 3.4), (6.6, 2.8, 2.4), (12.6, 2.9, 2.5)],
            "8": [(0, 11.0, 4.5), (2.6, 6.2, 5.0), (5.6, 6.3, 4.8), (7.1, 6.9, 3.9), (9.2, 3.8, 5.4), (12.0, 3.2, 5.8)],
            "13": [(0, 12.0, 1.6), (2.6, 1.3, 0.6), (5.8, 1.5, 0.7), (7.2, 7.2, 3.0), (8.6, 5.7, 5.9), (9.4, 4.9, 6.6),
                   (11.0, 5.2, 6.4)],
            "23": [(0, 11.0, 10.5), (2.6, 5.9, 10.0), (7.0, 5.8, 10.0), (8.6, 4.4, 9.4), (11.0, 3.0, 8.6)],
            "5": [(0, 12.5, 13.2), (2.8, 1.3, 14.4), (5.6, 1.4, 14.3), (7.5, 6.3, 13.3), (12.6, 6.4, 13.3)],
        },
        deff={
            "4": {"guard": "2"},
            "21": {"guard": "8", "ops": [("ball", 7.2, 8.3)]},
            "7": {"guard": "13", "ops": [("slow", 6.9, 7.7, 2.5)]},
            "15": {"guard": "23", "ops": [("goto", 8.2, 10.0, (4.6, 7.4))]},
            "10": {"guard": "5", "ops": [("goto", 7.0, 10.2, (3.6, 9.8))]},
        },
        ball=[("hold", "h2", 0), ("pass", "h8", 3.4), ("pass", "h13", 7.0, 7.25), ("shot", 9.6, False),
              ("reb", "a15", 11.5)],
        marks=[(3.9, "cut", {"by": "2"}), (7.1, "handoff", {"by": "8", "for": "13"}), (8.0, "drive", {"by": "13"})],
    ))

    # 3. Dziki defend. Spain pick-and-roll: a guard back-screens the dropping centre, the roller is free.
    out.append(Play(
        id="q2-0730-spain-pnr", quarter=2, clock="7:30", score=(24, 26), offence="away", dur=10.6,
        labels={"play": "spain_pnr", "defence": "man", "coverage": "drop"},
        off={
            "4": [(0, 14.5, 7.5), (3.0, 10.2, 7.0), (5.0, 10.0, 7.2), (6.8, 7.4, 9.6), (7.6, 6.3, 9.9), (9.5, 6.8, 10.4)],
            "15": [(0, 8.5, 4.8), (3.0, 7.0, 5.5), (4.6, 9.4, 8.0), (5.7, 9.3, 8.0), (7.5, 4.4, 7.7), (8.2, 3.1, 7.5),
                   (8.7, 2.3, 7.3), (10.0, 2.2, 6.6)],
            "7": [(0, 9.5, 11.5), (3.0, 3.4, 7.8), (5.0, 3.6, 7.6), (5.9, 5.8, 8.2), (6.7, 5.9, 8.2), (8.2, 9.2, 7.0)],
            "10": [(0, 11.5, 2.0), (3.2, 1.3, 0.6), (10.6, 1.3, 0.6)],
            "21": [(0, 12.0, 13.2), (3.2, 1.3, 14.4), (10.6, 1.3, 14.4)],
        },
        deff={
            "1": {"guard": "4", "ops": [("slow", 5.3, 6.5, 2.0)]},
            "23": {"guard": "15", "ops": [("goto", 4.6, 6.1, (6.5, 8.9)), ("ball", 6.1, 7.7), ("slow", 6.0, 8.8, 1.3)]},
            "13": {"guard": "7"},
            "11": {"guard": "10"},
            "24": {"guard": "21"},
        },
        ball=[("hold", "a4", 0), ("pass", "a15", 7.65, 8.15), ("shot", 8.7, True)],
        marks=[(5.0, "screen", {"by": "15", "for": "4", "on": "1"}),
               (6.1, "backscreen", {"by": "7", "for": "15", "on": "23"})],
    ))

    # 4. Dziki break after a defensive rebound: outlet, three lanes filled wide, 3-on-2, lay-up.
    out.append(Play(
        id="q2-0305-fast-break", quarter=2, clock="3:05", score=(33, 34), offence="home", dur=7.6,
        labels={"play": "fast_break", "defence": "transition", "coverage": ""},
        off={
            "24": [(0, 24.6, 8.6), (1.2, 23.8, 8.8), (4.0, 17.0, 8.4), (7.0, 9.5, 8.2)],
            "1": [(0, 21.5, 11.0), (1.3, 19.8, 11.4), (2.8, 13.6, 9.2), (4.3, 8.4, 8.0), (5.1, 6.6, 8.3), (7.0, 6.4, 9.6)],
            "13": [(0, 21.0, 4.2), (2.6, 12.0, 2.1), (4.4, 6.0, 2.6), (5.5, 3.1, 5.5), (6.0, 2.3, 6.5), (7.4, 1.6, 9.0)],
            "11": [(0, 20.8, 13.5), (3.2, 11.5, 13.6), (5.2, 4.8, 13.3), (6.4, 1.6, 14.2)],
            "23": [(0, 25.4, 6.4), (1.5, 24.0, 6.4), (7.6, 13.5, 6.6)],
        },
        deff={
            "4": {"path": [(0, 17.5, 7.6), (1.0, 16.0, 7.6), (3.6, 6.2, 7.6), (4.6, 5.0, 7.8), (5.2, 5.2, 8.0),
                           (6.2, 3.4, 7.2)]},
            "10": {"path": [(0, 21.8, 12.6), (1.2, 20.4, 12.4), (4.4, 7.5, 10.8), (5.6, 4.6, 10.6), (6.8, 3.0, 10.6)]},
            "7": {"path": [(0, 21.6, 2.8), (1.4, 20.6, 3.2), (5.0, 9.0, 4.4), (6.4, 5.0, 5.2), (7.6, 3.4, 5.6)]},
            "15": {"path": [(0, 25.0, 7.4), (1.6, 23.6, 7.6), (7.6, 12.0, 7.8)]},
            "21": {"path": [(0, 24.2, 9.8), (1.5, 23.0, 9.8), (7.6, 10.5, 9.4)]},
        },
        ball=[("hold", "h24", 0), ("pass", "h1", 0.8), ("pass", "h13", 5.0, 5.42), ("shot", 5.95, True)],
        marks=[(0.8, "outlet", {"by": "24", "for": "1"})],
    ))

    # 5. Dziki defend well: wing drive stopped by help, two kick-outs, every rotation on time,
    #    contested corner three, miss, defensive rebound.
    out.append(Play(
        id="q3-0910-rotations", quarter=3, clock="9:10", score=(44, 45), offence="away", dur=13.6,
        labels={"play": "drive_and_kick", "defence": "man", "coverage": "help_and_recover"},
        off={
            "4": [(0, 14.5, 7.8), (3.0, 9.6, 8.0), (4.4, 9.6, 8.6), (13.6, 9.4, 8.4)],
            "7": [(0, 11.5, 2.4), (2.8, 7.0, 2.2), (4.6, 6.9, 2.3), (5.6, 4.6, 2.6), (6.3, 3.2, 3.6), (7.4, 3.0, 2.4),
                  (9.0, 1.5, 0.7)],
            "10": [(0, 12.0, 12.8), (2.8, 6.8, 13.0), (8.2, 6.8, 13.0), (9.1, 5.6, 11.9), (9.6, 5.4, 11.6),
                   (12.0, 6.6, 12.4)],
            "21": [(0, 12.5, 11.0), (3.4, 1.3, 14.4), (13.6, 1.3, 14.4)],
            "15": [(0, 9.0, 5.5), (3.0, 3.0, 9.8), (6.0, 2.8, 9.9), (9.0, 3.0, 9.6), (12.0, 2.6, 8.8)],
        },
        deff={
            "2": {"guard": "4"},
            "5": {"guard": "7"},
            "22": {"guard": "10"},
            "8": {"guard": "21", "ops": [("goto", 8.9, 9.45, (3.4, 11.8))]},
            "23": {"guard": "15", "ops": [("goto", 5.4, 6.7, (3.2, 5.2))]},
        },
        ball=[("hold", "a4", 0), ("pass", "a7", 3.6), ("pass", "a4", 6.45), ("pass", "a10", 7.5),
              ("pass", "a21", 9.6), ("shot", 10.75, False), ("reb", "h23", 12.7)],
        marks=[(5.0, "drive", {"by": "7"}), (5.9, "help", {"on": "23"}), (8.6, "drive", {"by": "10"}),
               (9.2, "help", {"on": "8"})],
    ))

    # 6. Dziki against a 2-3 zone: wing entry, high-post flash, kick to the far wing, extra pass to
    #    the corner while two defenders close on the wing. Open corner three, made.
    out.append(Play(
        id="q3-0548-zone-attack", quarter=3, clock="5:48", score=(50, 54), offence="home", dur=10.6,
        labels={"play": "zone_offence", "defence": "zone_2_3", "coverage": ""},
        off={
            "1": [(0, 14.5, 7.5), (2.8, 9.6, 8.0), (4.0, 9.5, 8.4), (6.0, 9.4, 7.0), (10.6, 9.4, 7.2)],
            "13": [(0, 12.0, 12.6), (2.6, 6.9, 12.9), (10.6, 7.0, 12.9)],
            "8": [(0, 10.5, 9.5), (2.6, 2.6, 9.6), (4.0, 2.8, 9.4), (4.9, 5.7, 7.6), (6.4, 5.6, 7.6), (8.4, 3.4, 8.0)],
            "24": [(0, 12.0, 3.0), (2.8, 3.0, 3.6), (5.0, 3.2, 3.4), (6.3, 6.9, 2.0), (10.6, 7.0, 2.1)],
            "11": [(0, 12.5, 1.6), (3.0, 1.3, 0.6), (10.6, 1.3, 0.6)],
        },
        deff={
            "4": {"zone": (7.3, 5.3)},
            "7": {"zone": (7.3, 9.7)},
            "10": {"zone": (3.0, 2.9), "ops": [("ball", 6.25, 7.35)]},
            "15": {"zone": (2.9, 7.5), "ops": [("ball", 5.1, 6.2)]},
            "21": {"zone": (3.0, 12.1)},
        },
        ball=[("hold", "h1", 0), ("pass", "h13", 3.4), ("pass", "h8", 4.9), ("pass", "h24", 6.0),
              ("pass", "h11", 7.05), ("shot", 8.15, True)],
        marks=[(4.6, "flash", {"by": "8"})],
    ))

    # 7. Dziki defend. They switch the ball screen, the centre posts up the guard, the double team
    #    comes, the ball is swung twice and the last rotation is late. Open wing three, made.
    out.append(Play(
        id="q4-0420-switch-double", quarter=4, clock="4:20", score=(68, 66), offence="away", dur=13.4,
        labels={"play": "pnr_post_up", "defence": "man", "coverage": "switch"},
        off={
            "4": [(0, 14.5, 7.2), (2.8, 10.0, 7.0), (4.4, 9.6, 6.4), (5.4, 8.8, 4.6), (6.6, 7.6, 2.8), (13.4, 7.6, 2.6)],
            "15": [(0, 9.0, 9.5), (2.6, 7.5, 8.4), (3.6, 9.2, 5.6), (4.6, 9.1, 5.6), (6.0, 5.4, 5.2), (7.2, 3.3, 4.7),
                   (9.0, 3.1, 4.9), (12.0, 2.6, 5.6)],
            "21": [(0, 13.0, 9.5), (3.0, 9.4, 9.6), (6.0, 9.2, 8.6), (13.4, 9.2, 8.4)],
            "7": [(0, 12.0, 12.8), (3.0, 6.9, 13.0), (13.4, 6.9, 13.0)],
            "10": [(0, 12.5, 11.5), (3.4, 1.3, 14.4), (13.4, 1.3, 14.4)],
        },
        deff={
            "2": {"guard": "4", "ops": [("guard", 4.5, "15")]},
            "23": {"guard": "15", "ops": [("guard", 4.5, "4")]},
            "22": {"guard": "21", "ops": [("ball", 7.7, 9.3), ("guard", 9.3, "10"), ("slow", 9.3, 13.4, 5.5)]},
            "13": {"guard": "7", "ops": [("guard", 9.15, "21"), ("guard", 10.15, "7"), ("slow", 10.15, 10.6, 2.5)]},
            "24": {"guard": "10"},
        },
        ball=[("hold", "a4", 0), ("pass", "a15", 7.2), ("pass", "a21", 9.0), ("pass", "a7", 9.9),
              ("shot", 11.0, True)],
        marks=[(4.2, "screen", {"by": "15", "for": "4", "on": "2"}), (4.6, "switch", {"on": "2", "to_def": "23"}),
               (7.9, "double", {"on": "22"}), (9.3, "rotation", {"on": "13"})],
    ))

    # 8. Last minute, Dziki down two: staggered screens for the shooter coming out of the paint
    #    to the wing. Catch and shoot, made.
    out.append(Play(
        id="q4-0038-stagger", quarter=4, clock="0:38", score=(77, 79), offence="home", dur=9.6,
        labels={"play": "stagger_screens", "defence": "man", "coverage": "chase"},
        off={
            "1": [(0, 14.5, 8.5), (2.6, 9.8, 8.6), (4.4, 9.6, 7.4), (5.8, 9.4, 5.6), (9.6, 9.6, 6.2)],
            "13": [(0, 10.5, 12.5), (2.4, 2.3, 8.6), (3.8, 2.2, 7.6), (4.7, 3.0, 4.6), (5.5, 5.2, 2.9), (6.2, 7.0, 2.5),
                   (9.6, 7.1, 2.5)],
            "8": [(0, 9.5, 4.0), (2.6, 3.3, 5.2), (4.9, 3.4, 5.2), (6.4, 2.6, 6.0), (9.6, 2.6, 6.4)],
            "23": [(0, 10.5, 5.5), (2.8, 5.4, 3.7), (5.7, 5.5, 3.7), (7.0, 3.4, 6.6), (9.0, 2.4, 8.4)],
            "11": [(0, 12.0, 13.0), (3.0, 6.6, 13.2), (5.0, 1.4, 14.3), (9.6, 1.3, 14.4)],
        },
        deff={
            "4": {"guard": "1"},
            "7": {"guard": "13", "ops": [("slow", 4.5, 5.0, 2.2), ("slow", 5.3, 6.6, 2.6)]},
            "21": {"guard": "8"},
            "15": {"guard": "23"},
            "10": {"guard": "11"},
        },
        ball=[("hold", "h1", 0), ("pass", "h13", 5.75), ("shot", 6.95, True)],
        marks=[(4.7, "screen", {"by": "8", "for": "13", "on": "7"}),
               (5.5, "screen", {"by": "23", "for": "13", "on": "7"})],
    ))
    out[3] = faster(out[3], 0.86)  # a break is run at a sprint
    return out


def faster(p: Play, k: float) -> Play:
    """The same play with every time multiplied by k."""
    sc = lambda w: [(round(t * k, 2), u, v) for t, u, v in w]  # noqa: E731
    p.off = {n: sc(w) for n, w in p.off.items()}
    p.deff = {n: {**spec, **({"path": sc(spec["path"])} if "path" in spec else {})} for n, spec in p.deff.items()}
    p.ball = [
        tuple(round(x * k, 2) if isinstance(x, float | int) and not isinstance(x, bool) else x for x in item)
        for item in p.ball
    ]
    p.marks = [(round(t * k, 2), typ, extra) for t, typ, extra in p.marks]
    p.dur = round(p.dur * k, 1)
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    for p in plays():
        doc = build(p)
        (OUT / f"{p.id}.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
        sh = next((e for e in doc["events"] if e["type"] == "shot"), {})
        print(f"{p.id:<26} {len(doc['ball']):>4} frames  shot {sh.get('t')} made={sh.get('made')} "
              f"pts={sh.get('points')}")  # fmt: skip


if __name__ == "__main__":
    main()
