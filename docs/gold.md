# Gold set

Plan section 8, as amended by the 2026-11-02 decision (`docs/decisions.md`).

## Versions

| Version | Who reviewed | What it supports | Status |
|---|---|---|---|
| v0 | Implementer, full pass on all 12 shots; Tomasz's dry-run labels on Legia shot 10 kept as they are | Comparing backends; HOTA, IDF1, MOTA, team and number accuracy, AP50 | Committed 2026-11-02 |
| v1 | Tomasz reviews at least the held-out game; boxes tightened by hand | Absolute numbers for the client; AP50:95 | Not started |

v0 was prefilled by the public models the baseline uses (Skalski's player detector, SAM 2.1), so v0 absolute scores are biased upwards. Every row in `docs/metrics.md` names its gold version.

## Contents

- **Games:** 4 games, 3 central shots each, listed in `data/labels/gold.json`. The held-out game is `Tnuscrm9RyU` (Dziki – Zastal, 2026/27). It is scored once per milestone and never trained on.
- **Track gold:** `data/gold/v0/<youtube_id>_s<shot>/` holds one MOTChallenge sequence per shot, every 5th frame at 50 fps:
  - `gt/gt.txt`: `frame,id,x,y,w,h,1,class,visibility`, where class is 1 home, 2 away, 3 referee, 4 other;
  - `frames.txt`: the source frame index of each sequence frame;
  - `seqinfo.ini`;
  - `identities.csv`: `tid,team,number,name,team_name,note`.

  Gaps of up to 3 sampled frames inside an identity are interpolated and written with visibility 0.5. `manifest.json` lists the sequences, their held-out flags and the hash of the source label file.
- **Box gold:** `data/gold/v0/boxes.json` covers 100 frames chosen evenly over the 12 shots (`data/labels/box_gold.json`). It takes the track-gold boxes, adds people the tracks miss (detector candidates the reviewer accepted) and removes the boxes the reviewer dropped, as listed in `data/labels/review/box_gold.ops.txt`. Geometry is the prefill's, so v0 supports AP50 only.
- **Not in v0:** number crops move to M7 and court landmarks to M3, both labelled just in time.

## How v0 was made

1. **Prefill.** Skalski's detector boxes, with identities from SAM 2.1 on Modal (`docs/costs.md`).
2. **Kit-colour proposals.** For each shot, `scripts/review_sheets.py --propose` fits Skalski's TeamClassifier on that shot's box crops and gives every box a light or dark kit label. It then proposes ops:
   - the team of each track, by majority;
   - a cut where a box briefly sits on an opponent during contact;
   - a split or move where a track changes team for 8 or more sampled frames.

   The reviewer checked every proposal on the sheets and rejected the wrong ones, each with its reason. Files: `data/labels/review/<id>.s<shot>.ops.txt`.
3. **Visual fixes on larger sheets.** These cover off-court people (coaches, bench, fans), same-team identity switches, duplicates, and jersey numbers where legible, checked against `data/rosters/`. Files: `data/labels/review/<id>.s<shot>.fix.ops.txt`.
4. **Duplicate check.** Any pair of tracks with box IoU above 0.6 on at least 4 frames was flagged. One duplicate was removed (`1jDaJBkUGmc.dupcheck.ops.txt`).
5. **Export.** `scripts/export_gold.py --version v0`, then `scripts/box_gold.py export --version v0`.

Every sheet that was read has a row in `data/labels/sheet_ledger.tsv`. The ops files are applied in order with `scripts/review_ops.py`, so they are the audit trail.

## Known weaknesses of v0

- **Missed people.** Some people the tracks miss stay missing in track gold: players lying on the floor, near-side referees cut off by the frame edge or the score overlay, and players hidden behind a teammate. Box gold adds them on its 100 frames.
- **Short contact moments.** Boxes that briefly sit on the wrong player were cut, which leaves short gaps rather than errors.
- **Loose boxes.** Some boxes in scrums cover two people; they carry notes in `identities.csv`.
- **Numbers.** Numbers are set only where legible on 1080p highlights; many tracks have none.
