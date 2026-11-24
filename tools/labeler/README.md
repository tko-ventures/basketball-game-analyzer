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

## Box gold v1 (`/boxes`)

http://127.0.0.1:8765/boxes edits `data/labels/box_gold_v1.json`: tight boxes in the 10-class schema on the 100
box-gold frames, prefilled by `scripts/box_gold_v1.py`. Flagged boxes come first (Tab), a loupe on the right shows the
selected box at up to 8×, and Enter marks the frame done. Keys are listed under the loupe; conventions are in
`docs/gold.md`, "Box gold v1".

## Gold number crops (`/numbers`)

http://127.0.0.1:8765/numbers checks the labels of the gold number crops (`data/labels/numbers/gold_crops.json`,
written by `scripts/number_gold.py`). It opens on the 30-crop spot check; `?set=illegible` shows the crops labelled illegible and `?set=all` every crop. Each crop is shown
close up and in context. `Enter` means the label is right; type the digits and press `Enter` for another number, or press
`I` for illegible and `X` for not a number. Each answer saves at once; the keys are listed on the page.

## Holder gold (`/holders`)

http://127.0.0.1:8765/holders answers the holder-gold frames flagged for Tomasz (`data/labels/review/holders_v1.ops.txt`,
see `docs/gold.md`, "Holder gold v1"). Each question shows the gold frame zoomed around the ball, with the frames from 0.6 s
before to 0.6 s after it below (`Space` plays them, `↑` `↓` step). `A` accepts the proposed holder, `N` means nobody holds
the ball, or click a player box and press `Enter`; `←` `→` move without answering and `?` lists the keys. Each answer saves at
once as a `tomasz` line under the flagged one and rebuilds `data/gold/v1/holders.json`. The second tab shows the implementer's
applied changes. `--holders-ops` and `--holders-out` point the page at copies for testing.
