#!/usr/bin/env python3
"""Local ground-truth labeller for basketball tracking.

Run from the repo root:

    python tools/labeler/server.py            # http://127.0.0.1:8765
    python tools/labeler/server.py --port 9000 --clips-dir data/clips --clips-dir data/raw

Stdlib HTTP server + OpenCV. Labels are written to ``data/labels/<clip-stem>.json`` (source of
truth) plus ``<clip-stem>.mot.txt`` (MOTChallenge gt) and ``<clip-stem>.identities.csv`` on every save.

Teams are ``home`` / ``away`` / ``ref`` / ``other``; ``meta.home`` and ``meta.away`` map each side to a
team name, a roster file in ``data/rosters/`` and a kit description, per clip.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from bga.ingest.hist_diff import DEFAULT_MIN_LEN_S, DEFAULT_THRESHOLD, detect_shots

REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
LABEL_VERSION = 2
TEAMS = ("home", "away", "ref", "other")
MOT_CLASS = {"home": 1, "away": 2, "ref": 3}  # anything else is 4 (other)


# --------------------------------------------------------------------------- #
# Video access
# --------------------------------------------------------------------------- #
class Clip:
    """One VideoCapture per clip, guarded by a lock. Sequential reads are used when
    the requested frame is shortly after the last decoded one (much faster than seek)."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.next_idx = 0  # index of the frame the next cap.read() returns

    def read(self, idx: int) -> np.ndarray | None:
        with self.lock:
            gap = idx - self.next_idx
            if gap < 0 or gap > 60:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self.next_idx = idx
                gap = 0
            for _ in range(gap):
                if not self.cap.grab():
                    return None
                self.next_idx += 1
            ok, frame = self.cap.read()
            if not ok:
                return None
            self.next_idx += 1
            return frame


class ClipRegistry:
    def __init__(self):
        self._clips: dict[str, Clip] = {}
        self._lock = threading.Lock()

    def get(self, rel: str) -> Clip:
        with self._lock:
            if rel not in self._clips:
                self._clips[rel] = Clip(safe_path(rel))
            return self._clips[rel]


def safe_path(rel: str) -> Path:
    p = (REPO / rel).resolve()
    if REPO not in p.parents and p != REPO:
        raise PermissionError(rel)
    return p


def stem_for(rel: str) -> str:
    return Path(rel).stem


# --------------------------------------------------------------------------- #
# Shot (camera cut) detection — background job
# --------------------------------------------------------------------------- #
class ShotJobs:
    """Background shot detection per clip, using the layer-0 ``hist_diff`` detector."""

    def __init__(self, labels_dir: Path):
        self.labels_dir = labels_dir
        self.progress: dict[str, float] = {}
        self.results: dict[str, list] = {}
        self._lock = threading.Lock()

    def start(self, rel: str, threshold: float = DEFAULT_THRESHOLD, min_len_s: float = DEFAULT_MIN_LEN_S):
        with self._lock:
            if rel in self.progress and self.progress[rel] < 1.0:
                return
            self.progress[rel] = 0.0
        threading.Thread(target=self._run, args=(rel, threshold, min_len_s), daemon=True).start()

    def _run(self, rel: str, threshold: float, min_len_s: float):
        def progress(p: float) -> None:
            self.progress[rel] = p

        shots = detect_shots(safe_path(rel), threshold, min_len_s, progress)
        self.results[rel] = shots
        self.progress[rel] = 1.0


# --------------------------------------------------------------------------- #
# Label storage
# --------------------------------------------------------------------------- #
def default_labels(rel: str, clip: Clip) -> dict:
    return {
        "version": LABEL_VERSION,
        "clip": rel,
        "fps": clip.fps,
        "frame_count": clip.frame_count,
        "width": clip.width,
        "height": clip.height,
        "stride": max(1, round(clip.fps / 10)),  # ~10 labelled frames per second
        "meta": default_meta(),
        "shots": [{"start": 0, "end": max(0, clip.frame_count - 1), "camera": "unset"}],
        "identities": {},
        "frames": {},
    }


def default_meta() -> dict:
    side = {"team": "", "roster": "", "kit": ""}
    return {"season": "", "home": dict(side), "away": dict(side), "notes": ""}


DZIKI = {"team": "Dziki Warszawa", "roster": "dziki_warszawa"}
_TITLE_NOISE = r"(?i)skr[oó]t( meczu| spotkania)?|\d{1,2}\.\d{1,2}\.\d{4}|\d+:\d+|🎞️"


def teams_from_title(title: str) -> tuple[str, str]:
    """Home and away team names from a highlight title ("SKRÓT | Dziki - Legia | 24. kolejka").
    PLK titles name the home team first. Dziki is returned as "Dziki Warszawa"; unparseable titles
    give ("", "")."""
    t = re.sub(r"\s+", " ", re.sub(_TITLE_NOISE, " ", title))  # also folds non-breaking spaces
    t = re.sub(r" I ", "|", t)  # a lone capital I stands in for "|" in some titles
    segs = [x for x in t.split("|") if re.search(r"(?i)dzik", x)] or [t]
    parts = [p.strip(" .") for p in re.split(r"\s*[-–]\s*|\s+vs\.?\s+|\s*🆚\s*", segs[0], flags=re.I)]
    parts = [p for p in parts if p]
    if len(parts) != 2:
        return "", ""
    return tuple(DZIKI["team"] if re.search(r"(?i)dzik", p) else p[:40] for p in parts)  # type: ignore[return-value]


def side_meta(team: str, kit: str = "") -> dict:
    roster = DZIKI["roster"] if team == DZIKI["team"] else ""
    return {"team": team, "roster": roster, "kit": kit}


def migrate_labels(data: dict) -> dict:
    """v1 (team = dziki / opp, meta.opponent, meta.dziki_kit) to v2 (home / away per clip)."""
    if data.get("version", 1) >= 2:
        return data
    meta = data.setdefault("meta", {})
    opponent = meta.pop("opponent", "")
    kit = meta.pop("dziki_kit", "")
    home, away = teams_from_title(meta.get("title", ""))
    if DZIKI["team"] not in (home, away):
        home, away = DZIKI["team"], opponent
    meta["home"] = side_meta(home, kit if home == DZIKI["team"] else "")
    meta["away"] = side_meta(away, kit if away == DZIKI["team"] else "")
    meta.setdefault("season", "")
    meta.setdefault("notes", "")
    dziki_side = "home" if home == DZIKI["team"] else "away"
    side = {"dziki": dziki_side, "opp": "away" if dziki_side == "home" else "home"}
    for ident in data.get("identities", {}).values():
        ident["team"] = side.get(ident.get("team", ""), ident.get("team", ""))
    data["version"] = 2
    return data


def load_labels(labels_dir: Path, rel: str, clip: Clip) -> dict:
    p = labels_dir / f"{stem_for(rel)}.json"
    if p.exists():
        return migrate_labels(json.loads(p.read_text(encoding="utf-8")))
    return default_labels(rel, clip)


def write_exports(labels_dir: Path, data: dict) -> None:
    stem = stem_for(data["clip"])
    ids = data.get("identities", {})
    meta = data.get("meta", {})
    # MOTChallenge gt: frame(1-based),id,x,y,w,h,conf,class,visibility
    # class: 1 = home player, 2 = away player, 3 = referee, 4 = other
    rows = []
    for fidx, fr in data.get("frames", {}).items():
        if not (fr.get("done") or fr.get("interp")):
            continue
        for b in fr.get("boxes", []):
            ident = ids.get(str(b["tid"]), {})
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            vis = 0.5 if b.get("occluded") else 1.0
            rows.append(
                (
                    int(fidx) + 1,
                    int(b["tid"]),
                    round(x1, 1),
                    round(y1, 1),
                    round(x2 - x1, 1),
                    round(y2 - y1, 1),
                    1,
                    MOT_CLASS.get(ident.get("team", ""), 4),
                    vis,
                )
            )
    rows.sort()
    with open(labels_dir / f"{stem}.mot.txt", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    with open(labels_dir / f"{stem}.identities.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tid", "team", "number", "name", "team_name", "note"])
        for tid, ident in sorted(ids.items(), key=lambda kv: int(kv[0])):
            team = ident.get("team", "")
            team_name = (meta.get(team) or {}).get("team", "") if team in ("home", "away") else ""
            w.writerow(
                [tid, team, ident.get("number", ""), ident.get("name", ""), team_name, ident.get("note", "")]
            )


def save_labels(labels_dir: Path, data: dict) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = stem_for(data["clip"])
    p = labels_dir / f"{stem}.json"
    tmp = p.with_suffix(".json.tmp")
    data["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    write_exports(labels_dir, data)


def label_progress(labels_dir: Path, rel: str) -> dict:
    p = labels_dir / f"{stem_for(rel)}.json"
    if not p.exists():
        return {"done": 0, "boxes": 0}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"done": 0, "boxes": 0}
    frames = d.get("frames", {}).values()
    return {
        "done": sum(1 for f in frames if f.get("done")),
        "boxes": sum(len(f.get("boxes", [])) for f in frames if f.get("done")),
    }


# --------------------------------------------------------------------------- #
# Pipeline prefill (outputs/**/tracks.json from run_pipeline / demo overlay)
# --------------------------------------------------------------------------- #
def pipeline_boxes(tracks_json: Path) -> dict:
    d = json.loads(tracks_json.read_text(encoding="utf-8"))
    out: dict[str, list] = {}
    for fr in d.get("frames", []):
        boxes = []
        for t in fr.get("tracks", []):
            xyxy = t.get("xyxy") or t.get("bbox")
            if not xyxy:
                continue
            boxes.append(
                {
                    "track_id": t.get("track_id"),
                    "xyxy": xyxy,
                    "team": t.get("team") if t.get("team") in TEAMS else "",
                    "number": t.get("jersey_number"),
                }
            )
        out[str(fr["frame_idx"])] = boxes
    return {"clip_path": d.get("clip_path"), "frames": out}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, clips_dirs: list[Path], labels_dir: Path, rosters_dir: Path):
        self.clips_dirs = clips_dirs
        self.labels_dir = labels_dir
        self.rosters_dir = rosters_dir
        self.cache_dir = labels_dir / ".cache"
        self.clips = ClipRegistry()
        self.shots = ShotJobs(labels_dir)

    def list_clips(self) -> list[dict]:
        out = []
        for d in self.clips_dirs:
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in VIDEO_EXT and not p.name.endswith(".part"):
                    rel = str(p.relative_to(REPO))
                    out.append(
                        {
                            "path": rel,
                            "size_mb": round(p.stat().st_size / 1e6, 1),
                            **label_progress(self.labels_dir, rel),
                        }
                    )
        return out

    def rosters(self) -> dict:
        out = {}
        if self.rosters_dir.exists():
            for p in sorted(self.rosters_dir.glob("*.json")):
                out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        return out

    def pipeline_outputs(self) -> list[str]:
        res = []
        for root in ("outputs", "runs"):
            r = REPO / root
            if r.exists():
                res += [str(p.relative_to(REPO)) for p in r.rglob("tracks.json")]
        return sorted(res)

    def frame_jpeg(self, rel: str, idx: int) -> bytes | None:
        cdir = self.cache_dir / stem_for(rel)
        cpath = cdir / f"{idx:07d}.jpg"
        if cpath.exists():
            return cpath.read_bytes()
        frame = self.clips.get(rel).read(idx)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return None
        cdir.mkdir(parents=True, exist_ok=True)
        cpath.write_bytes(buf.tobytes())
        return buf.tobytes()


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "bga-labeler/2"

        def log_message(self, fmt, *args):  # quieter
            if "/api/frame" not in (args[0] if args else ""):
                sys.stderr.write(f"{self.log_date_time_string()} - {fmt % args}\n")

        # helpers
        def _json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _bytes(self, data: bytes, ctype: str, cache=False):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _err(self, status, msg):
            self._json({"error": msg}, status)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self._bytes((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if u.path.startswith("/static/"):
                    p = (STATIC / u.path[len("/static/") :]).resolve()
                    if STATIC not in p.parents or not p.exists():
                        return self._err(404, "not found")
                    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                    return self._bytes(p.read_bytes(), ctype)
                if u.path == "/api/clips":
                    return self._json(
                        {
                            "clips": app.list_clips(),
                            "rosters": app.rosters(),
                            "pipeline_outputs": app.pipeline_outputs(),
                        }
                    )
                if u.path == "/api/clip":
                    rel = q["path"]
                    clip = app.clips.get(rel)
                    return self._json({"labels": load_labels(app.labels_dir, rel, clip)})
                if u.path == "/api/frame":
                    data = app.frame_jpeg(q["path"], int(q["idx"]))
                    if data is None:
                        return self._err(404, "frame not decodable")
                    return self._bytes(data, "image/jpeg", cache=True)
                if u.path == "/api/shots":
                    rel = q["path"]
                    return self._json(
                        {"progress": app.shots.progress.get(rel), "shots": app.shots.results.get(rel)}
                    )
                if u.path == "/api/pipeline":
                    return self._json(pipeline_boxes(safe_path(q["tracks"])))
                return self._err(404, "not found")
            except (KeyError, ValueError) as e:
                return self._err(400, f"bad request: {e}")
            except PermissionError:
                return self._err(403, "path outside repo")
            except Exception as e:  # noqa: BLE001
                return self._err(500, repr(e))

        def do_POST(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if u.path == "/api/labels":
                    if not isinstance(body, dict) or "clip" not in body:
                        return self._err(400, "labels must include clip")
                    safe_path(body["clip"])
                    save_labels(app.labels_dir, body)
                    return self._json({"ok": True, "saved_at": body.get("saved_at")})
                if u.path == "/api/shots":
                    app.shots.start(q["path"], float(body.get("threshold", DEFAULT_THRESHOLD)))
                    return self._json({"started": True})
                return self._err(404, "not found")
            except PermissionError:
                return self._err(403, "path outside repo")
            except Exception as e:  # noqa: BLE001
                return self._err(500, repr(e))

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--clips-dir",
        action="append",
        default=None,
        help="Folder(s) with videos, relative to repo (default: data/clips, data/raw)",
    )
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--rosters-dir", default="data/rosters")
    a = ap.parse_args(argv)
    clips_dirs = [REPO / d for d in (a.clips_dir or ["data/clips", "data/raw"])]
    app = App(clips_dirs, REPO / a.labels_dir, REPO / a.rosters_dir)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(app))
    print(f"Labeller on http://{a.host}:{a.port}  (repo {REPO})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
