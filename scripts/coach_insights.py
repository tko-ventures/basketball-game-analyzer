"""Turn one 2D tracking fixture (docs/fixtures.md) into what the coach page shows as insights.

Everything is computed from positions, ball possession and the event list; nothing is written by hand
per play. Used by scripts/coach_web.py. The expected-points table is illustrative, not fitted to PLK.
"""

from __future__ import annotations

import numpy as np

L, W = 28.0, 15.0
BASKET = np.array([1.575, 7.5])
OPEN_M = 3.5  # an off-ball attacker this far from every defender counts as open
# Illustrative shooting percentages by zone and by distance of the nearest defender at release.
FG = {
    "rim": (0.55, 0.65, 0.72),
    "paint": (0.38, 0.44, 0.48),
    "mid": (0.36, 0.41, 0.45),
    "three": (0.30, 0.35, 0.39),
    "corner3": (0.33, 0.38, 0.42),
}
OPENNESS = ("contested", "open", "wide_open")  # < 1.2 m, 1.2-1.8 m, > 1.8 m


def to_uv(xy_cm, attacks):
    a = np.asarray(xy_cm, float)[:, :2] / 100
    return np.stack([L - a[:, 0], W - a[:, 1]], 1) if attacks == "right" else a


def in_paint(p):
    return p[0] <= 5.8 and abs(p[1] - 7.5) <= 2.45


def zone_of(p):
    d = float(np.linalg.norm(p - BASKET))
    if p[0] <= 2.99 and abs(p[1] - 7.5) >= 6.6:
        return "corner3"
    if d > 6.75:
        return "three"
    if d <= 2.0:
        return "rim"
    return "paint" if in_paint(p) else "mid"


def seg_dist(p, a, b):
    """Distance of p from the passing lane a -> b. A defender within 1.2 m of the passer (the pass goes
    over or around him) or 1.0 m of the receiver (he is guarding the catch, not the lane) does not count."""
    ab = b - a
    length = float(np.linalg.norm(ab)) + 1e-9
    along = float(np.dot(p - a, ab)) / length
    if along <= 1.2 or along >= length - 1.0:
        return 99.0
    return float(np.linalg.norm(p - (a + along / length * ab)))


def hull_area(pts):
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return 0.0

    def half(seq):
        h = []
        for p in seq:
            while (
                len(h) >= 2
                and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0
            ):
                h.pop()
            h.append(p)
        return h

    hull = half(pts)[:-1] + half(reversed(pts))[:-1]
    return 0.5 * abs(
        sum(
            hull[i][0] * hull[(i + 1) % len(hull)][1] - hull[(i + 1) % len(hull)][0] * hull[i][1]
            for i in range(len(hull))
        )
    )


def runs(mask, fps, min_s):
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if (i - start) / fps >= min_s:
                out.append((start, i - 1))
            start = None
    return out


def analyse(d: dict) -> dict:
    fps = d["fps"]
    side = d["offence"]
    atk = [p["id"] for p in d["players"] if p["team"] == side]
    dfn = [p["id"] for p in d["players"] if p["team"] != side]
    pos = {pid: to_uv(xy, d["attacks"]) for pid, xy in d["tracks"].items()}
    ball = to_uv(d["ball"], d["attacks"])
    holder = d["holder"]
    n = len(ball)
    A = np.stack([pos[p] for p in atk], 1)  # (n, 5, 2)
    D = np.stack([pos[p] for p in dfn], 1)
    dist = np.linalg.norm(A[:, :, None, :] - D[:, None, :, :], axis=3)  # attacker x defender
    nearest = dist.min(axis=2)
    shot = next((e for e in d["events"] if e["type"] == "shot"), None)
    f_shot = int(round(shot["t"] * fps)) if shot else n - 1
    passes = [e for e in d["events"] if e["type"] == "pass"]

    # spacing: mean distance between attackers, while all five are in the front court, before the shot
    half = [f for f in range(f_shot + 1) if (A[f, :, 0] < 14).all()] or list(range(f_shot + 1))
    iu = np.triu_indices(len(atk), 1)
    pair = np.linalg.norm(A[:, :, None, :] - A[:, None, :, :], axis=3)[:, iu[0], iu[1]].mean(axis=1)
    step = fps // 5
    spacing = {
        "mean": round(float(pair[half].mean()), 1),
        "area": round(float(np.mean([hull_area(A[f]) for f in half[::step]])), 0),
        "series": [round(float(x), 2) for x in pair[::step]],
        "series_hz": 5,
    }

    # ball movement
    sides = [1 if pos[e["to"]][int(e["t1"] * fps)][1] > 7.5 else -1 for e in passes]
    start_side = [1 if pos[e["from"]][int(e["t"] * fps)][1] > 7.5 else -1 for e in passes]
    reversals = sum(
        1
        for e, a, b in zip(passes, start_side, sides, strict=True)
        if a != b and abs(pos[e["to"]][int(e["t1"] * fps)][1] - pos[e["from"]][int(e["t"] * fps)][1]) > 4.0
    )
    touch = [holder[f] in atk and in_paint(ball[f]) for f in range(n)]
    paint_touches = len(runs(touch, fps, 0.2))

    # open team-mates: far from every defender, in range, with a clear passing lane from the ball
    open_windows = []
    speed = np.linalg.norm(np.gradient(A, 1 / fps, axis=0), axis=2)
    front = next((f for f in range(n) if ball[f][0] < 14), 0)  # the offence needs a moment to set up
    near = next((f for f in range(n) if np.linalg.norm(ball[f] - BASKET) < 9), n)
    f_set = min(front + int(2.5 * fps), near)
    for i, pid in enumerate(atk):
        mask = []
        for f in range(f_shot + 1):
            h = holder[f]
            ok = (
                f >= f_set
                and h in atk
                and h != pid
                and speed[f, i] < 3.0
                and pos[h][f][0] < 12.0
                and np.linalg.norm(A[f, i] - BASKET) <= 8.3
                and nearest[f, i] >= OPEN_M
                and min(seg_dist(q, pos[h][f], A[f, i]) for q in D[f]) >= 0.8
            )
            mask.append(bool(ok))
        for f0, f1 in runs(mask, fps, 0.5):
            t0, t1 = f0 / fps, f1 / fps
            found = any(e["to"] == pid and t0 - 0.2 <= e["t"] <= t1 + 0.6 for e in passes)
            open_windows.append(
                {
                    "t0": round(t0, 2),
                    "t1": round(t1, 2),
                    "pid": pid,
                    "found": found,
                    "dist": round(float(nearest[f0 : f1 + 1, i].max()), 1),
                }
            )
    open_windows.sort(key=lambda w: -(w["t1"] - w["t0"]) * w["dist"])
    open_windows = sorted(open_windows[:3], key=lambda w: w["t0"])

    out_shot = None
    if shot:
        i = atk.index(shot["by"])
        p = A[f_shot, i]
        j = int(dist[f_shot, i].argmin())
        dd = float(dist[f_shot, i, j])
        z = zone_of(p)
        k = 0 if dd < 1.2 else (1 if dd <= 1.8 else 2)
        catch = max((e for e in passes if e["to"] == shot["by"]), key=lambda e: e["t"], default=None)
        f_catch = int(catch["t1"] * fps) if catch else f_shot
        out_shot = {
            "t": shot["t"],
            "by": shot["by"],
            "made": shot["made"],
            "points": shot["points"],
            "zone": z,
            "dist": round(float(np.linalg.norm(p - BASKET)), 1),
            "def_dist": round(dd, 1),
            "def_by": dfn[j],
            "open": OPENNESS[k],
            "xpts": round(FG[z][k] * shot["points"], 2),
            "catch_def_dist": round(float(dist[f_catch, i].min()), 1),
            "catch_to_release": round(shot["t"] - f_catch / fps, 1),
            "defenders_near": int((dist[f_shot, i] <= 3.0).sum()),
            "defenders_in_paint": int(sum(in_paint(q) for q in D[f_shot])),
        }

    onball = [float(dist[f, atk.index(holder[f])].min()) for f in half if holder[f] in atk]

    def load(xy, pid):
        k = np.ones(5) / 5
        sm = np.stack([np.convolve(xy[:, c], k, mode="valid") for c in range(2)], 1)
        v = np.linalg.norm(np.diff(sm, axis=0), axis=1) * fps
        v10 = np.convolve(v, np.ones(10) / 10, mode="valid")
        return {
            "dist_m": round(float(v.sum() / fps), 1),
            "top_kmh": round(float(v10.max() * 3.6), 1),
            "avg_kmh": round(float(v.mean() * 3.6), 1),
            "ball_s": round(sum(1 for h in holder if h == pid) / fps, 1),
        }

    moments = []
    for e in d["events"]:
        if e["type"] in ("pass", "shot", "rebound"):
            continue
        moments.append({**e, "kind": e["type"]})
    for w in open_windows:
        moments.append(
            {
                "t": w["t0"],
                "kind": "open_found" if w["found"] else "open_missed",
                "by": w["pid"],
                "dist": w["dist"],
                "dur": round(w["t1"] - w["t0"], 1),
            }
        )
    if out_shot:
        moments.append(
            {"t": out_shot["t"], "kind": "shot", **{k: v for k, v in out_shot.items() if k != "t"}}
        )
    for e in d["events"]:
        if e["type"] == "rebound":
            moments.append({"t": e["t"], "kind": "rebound", "by": e["by"]})
    moments.sort(key=lambda m: m["t"])

    x = out_shot["xpts"] if out_shot else 0
    return {
        "duration": round((n - 1) / fps, 1),
        "time_to_shot": shot["t"] if shot else None,
        "passes": len(passes),
        "reversals": reversals,
        "paint_touches": paint_touches,
        "spacing": spacing,
        "shot": out_shot,
        "quality": "good" if x >= 1.1 else ("average" if x >= 0.9 else "poor"),
        "onball_def_dist": round(float(np.mean(onball)), 1) if onball else None,
        "open": open_windows,
        "moments": moments,
        "players": {pid: load(xy, pid) for pid, xy in pos.items()},
    }


def summary(plays: list[dict]) -> dict:
    """Totals over the possessions: one row per team on offence, and Dziki's players."""
    teams = {}
    for side in ("home", "away"):
        ps = [p for p in plays if p["offence"] == side]
        shots = [p["insights"]["shot"] for p in ps if p["insights"]["shot"]]
        pts = sum(s["points"] for s in shots if s["made"])
        teams[side] = {
            "possessions": len(ps),
            "points": pts,
            "ppp": round(pts / max(1, len(ps)), 2),
            "xpts": round(sum(s["xpts"] for s in shots) / max(1, len(shots)), 2),
            "open_shots": sum(1 for s in shots if s["open"] != "contested"),
            "shots": len(shots),
            "spacing": round(float(np.mean([p["insights"]["spacing"]["mean"] for p in ps])), 1),
            "passes": round(float(np.mean([p["insights"]["passes"] for p in ps])), 1),
        }
    players = {}
    for p in plays:
        for pl in p["players"]:
            if pl["team"] != "home":
                continue
            s = p["insights"]["players"][pl["id"]]
            row = players.setdefault(
                pl["id"],
                {
                    "number": pl["number"],
                    "name": pl["name"],
                    "pos": pl["pos"],
                    "plays": 0,
                    "dist_m": 0.0,
                    "top_kmh": 0.0,
                    "time_s": 0.0,
                },
            )
            row["plays"] += 1
            row["dist_m"] = round(row["dist_m"] + s["dist_m"], 1)
            row["top_kmh"] = max(row["top_kmh"], s["top_kmh"])
            row["time_s"] = round(row["time_s"] + p["insights"]["duration"], 1)
    return {"teams": teams, "players": sorted(players.values(), key=lambda r: -r["dist_m"])}
