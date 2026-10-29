# Court Labeller

Local web app for building the gold set: player boxes, persistent identities, team (home / away /
referee / other) and jersey number, on central-camera shots only. Imported from the prior prototype
(`tko-ventures/basketball-tracking@feat/labeller-rosters`) without its SAM 3.1 assist.

```bash
uv sync
scripts/fetch_highlights.sh                  # the clips in data/clip_ids.txt into data/raw/
uv run tools/labeler/prepare_clips.py --jobs 6   # shot cuts, season, home/away from the title
uv run tools/labeler/server.py               # http://127.0.0.1:8765
```

Videos are listed from `data/clips/` and `data/raw/`. Rosters come from `data/rosters/*.json`.

## Teams

Each clip's `meta.home` and `meta.away` hold `{team, roster, kit}`: the team name, the stem of its
roster file in `data/rosters/` (empty if there is none yet), and a free-text kit description.
`prepare_clips.py` fills the names from the YouTube title (PLK titles list the home team first).
Identities carry `team = home | away | ref | other`; the roster of that side drives number
autocomplete and names.

## Workflow per video

1. Pick the clip, check the **season** and the **home / away** teams, and note each side's kit.
2. Walk shots with Shift+↓ and tag each one **Central (G)** or **Other (X)**. With "skip non-central
   shots" on, navigation jumps over replays and close-ups. **Detect camera cuts** reruns the layer-0
   detector if the shots look wrong.
3. Optionally **prefill** boxes from a pipeline run (`runs/**/tracks.json`). They arrive as IDs 1000+,
   not marked done. Use **Merge…** to fold duplicate pipeline IDs into one identity.
4. Label a keyframe: draw boxes, `1`–`4` for team, `N` to type the number (roster numbers
   autocomplete, names fill in). Press **D** when the frame is complete: every player and referee on
   screen.
5. Jump ahead (→ steps by `stride`, default 10 labelled frames/s), **C** to copy the previous boxes,
   adjust, **D**. Then **I** fills the frames in between by interpolation (yellow on the timeline).

Identity = one real person for the whole clip. If someone leaves and re-enters the frame, arm their
identity in the list (click it) before drawing the box, so the ID stays the same.

## Outputs (`data/labels/`, autosaved)

| File | Content |
|------|---------|
| `<clip>.json` | source of truth (version 2: meta, shots and camera tags, identities, per-frame boxes, done/interp flags) |
| `<clip>.mot.txt` | MOTChallenge gt: `frame(1-based),id,x,y,w,h,1,class,visibility`; class 1 home, 2 away, 3 referee, 4 other; done and interpolated frames only |
| `<clip>.identities.csv` | `tid,team,number,name,team_name,note`: gt ID to `(team, number)` for identity accuracy |

Version 1 files (team `dziki` / `opp`) are migrated to version 2 on load. Frame JPEGs are cached
under `data/labels/.cache/` (gitignored, safe to delete).
