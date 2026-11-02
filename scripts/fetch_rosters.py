#!/usr/bin/env python3
"""Fetch every PLK team's squad from plk.pl into data/rosters/<slug>.json (plan question 14).

    uv run scripts/fetch_rosters.py                    # seasons 2024/25, 2025/26, 2026/27
    uv run scripts/fetch_rosters.py --season 2026/27

Squad pages are Next.js pages whose RSC payload carries ``"players":[...]`` and ``"coaches":[...]``
as JSON; nothing is executed. One file per club (keyed by the PLK team id, named by its latest slug),
one entry per season, same schema as the original ``dziki_warszawa.json``. Squad pages list the
squad at the end of a season (archive) or today (current season): players who left mid-season may be
missing, so gold-set numbers outside the roster are flagged as conflicts, not trusted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import urllib.request
from pathlib import Path

BASE = "https://plk.pl"
# plk.pl archive ids; the current season has no archive prefix.
SEASONS = {"2024/25": "/archiwum/27", "2025/26": "/archiwum/28", "2026/27": ""}
POS = {
    "rozgrywający": "PG",
    "rzucający": "SG",
    "rzucający obrońca": "SG",
    "obrońca": "G",
    "niski skrzydłowy": "SF",
    "silny skrzydłowy": "PF",
    "skrzydłowy": "F",
    "środkowy": "C",
}
NAT = {
    "Polska": "POL",
    "Stany Zjednoczone": "USA",
    "Kanada": "CAN",
    "Szwecja": "SWE",
    "Litwa": "LTU",
    "Łotwa": "LAT",
    "Estonia": "EST",
    "Ukraina": "UKR",
    "Chorwacja": "CRO",
    "Serbia": "SRB",
    "Słowenia": "SLO",
    "Czarnogóra": "MNE",
    "Bośnia i Hercegowina": "BIH",
    "Macedonia Północna": "MKD",
    "Grecja": "GRE",
    "Włochy": "ITA",
    "Hiszpania": "ESP",
    "Francja": "FRA",
    "Niemcy": "GER",
    "Wielka Brytania": "GBR",
    "Irlandia": "IRL",
    "Holandia": "NED",
    "Belgia": "BEL",
    "Czechy": "CZE",
    "Słowacja": "SVK",
    "Węgry": "HUN",
    "Finlandia": "FIN",
    "Dania": "DEN",
    "Norwegia": "NOR",
    "Australia": "AUS",
    "Nigeria": "NGR",
    "Senegal": "SEN",
    "Jamajka": "JAM",
    "Bahamy": "BAH",
    "Gruzja": "GEO",
    "Izrael": "ISR",
    "Turcja": "TUR",
    "Austria": "AUT",
    "Szwajcaria": "SUI",
    "Kamerun": "CMR",
    "Mali": "MLI",
    "Portoryko": "PUR",
    "Dominikana": "DOM",
    "Brazylia": "BRA",
    "Argentyna": "ARG",
    "Białoruś": "BLR",
    "Rumunia": "ROU",
    "Bułgaria": "BUL",
    "Kosowo": "KOS",
}
# Title spellings that word matching cannot recover.
EXTRA_ALIASES = {1490: ["MKS DG"], 769: ["ORLEN Zastal"]}


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (bga roster fetch)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def rsc_payload(html: str) -> str:
    """Concatenate the string chunks of ``self.__next_f.push([1, "..."])``."""
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', html)
    return "".join(json.loads(f'"{c}"') for c in chunks)


def json_array_after(text: str, key: str) -> list:
    i = text.find(f'"{key}":[')
    if i < 0:
        return []
    return json.JSONDecoder().raw_decode(text, i + len(key) + 3)[0]


def team_links(season_prefix: str) -> dict[int, str]:
    html = get(f"{BASE}{season_prefix}/druzyny")
    pat = re.escape(season_prefix) + r"/druzyny/(\d+)/([a-z0-9-]+)"
    return {int(i): slug for i, slug in re.findall(pat, html)}


def team_name(payload: str, team_id: int) -> str:
    m = re.search(rf'"id":{team_id},"name":"([^"]+)"', payload)
    return m.group(1) if m else ""


def squad(season_prefix: str, team_id: int, slug: str) -> dict:
    url = f"{BASE}{season_prefix}/druzyny/{team_id}/{slug}/sklad"
    payload = rsc_payload(get(url))
    players = []
    for p in json_array_after(payload, "players"):
        num = (p.get("shirtNumber") or "").strip() or None
        players.append(
            {
                "number": num,
                "name": f"{p.get('firstName', '').strip()} {p.get('lastName', '').strip()}".strip(),
                "pos": "/".join(POS.get(x, x) for x in p.get("positions") or []),
                "height_cm": p.get("height"),
                "nat": NAT.get(p.get("passport") or "", p.get("passport") or ""),
                "id_plk": p.get("id"),
            }
        )
    players.sort(
        key=lambda p: (p["number"] is None, int(p["number"]) if (p["number"] or "").isdigit() else 999)
    )
    coaches = json_array_after(payload, "coaches")
    head = next((c for c in coaches if c.get("function") == "Trener"), None)
    return {
        "team_name": team_name(payload, team_id),
        "source": url,
        "coach": f"{head['firstName']} {head['lastName']}" if head else "",
        "players": players,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", action="append", choices=list(SEASONS))
    ap.add_argument("--out", type=Path, default=Path("data/rosters"))
    a = ap.parse_args(argv)
    today = dt.date.today().isoformat()
    clubs: dict[int, dict] = {}
    for season in a.season or list(SEASONS):
        prefix = SEASONS[season]
        for team_id, slug in sorted(team_links(prefix).items()):
            s = squad(prefix, team_id, slug)
            note = (
                f"Current PLK squad page as of {today}."
                if not prefix
                else "End-of-season PLK squad page; players who left mid-season may be missing."
            )
            club = clubs.setdefault(team_id, {"slug": slug, "seasons": {}})
            club["slug"] = slug if not prefix else club["slug"]
            club["seasons"][season] = {**s, "note": note}
            print(f"{season} {team_id} {s['team_name'] or slug}: {len(s['players'])} players", flush=True)
            time.sleep(0.5)
    a.out.mkdir(parents=True, exist_ok=True)
    for team_id, club in clubs.items():
        names = [v["team_name"] for v in club["seasons"].values() if v["team_name"]]
        path = a.out / f"{club['slug'].replace('-', '_')}.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        seasons = {**old.get("seasons", {}), **club["seasons"]}
        doc = {
            "team": names[-1] if names else club["slug"],
            "team_id_plk": team_id,
            "aliases": sorted(set(names[:-1] + EXTRA_ALIASES.get(team_id, []))),
            "retrieved": today,
            "seasons": dict(sorted(seasons.items())),
        }
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(clubs)} clubs -> {a.out}")


if __name__ == "__main__":
    main()
