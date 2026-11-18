# Gold set

Plan section 8, as amended by the 2026-11-02 decision (`docs/decisions.md`).

## Versions

| Version | Who reviewed | What it supports | Status |
|---|---|---|---|
| v0 | Implementer, full pass on all 12 shots; Tomasz's dry-run labels on Legia shot 10 kept as they are | Comparing backends; HOTA, IDF1, MOTA, team and number accuracy, AP50 | Committed 2026-11-02 |
| v1 | Tomasz reviews at least the held-out game; boxes tightened by hand | Absolute numbers for the client; AP50:95 | Box gold queue prefilled and given the implementer's first pass on 2026-11-18; Tomasz's review not started |

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
- **Court keyframes** (labelled for M3, outside the v0 export): one keyframe per shot in `data/labels/court/`, with the queue in `queue.json`.
  - **What:** 24 gold keyframes, 6 per gold game, for evaluation only; and 18 training keyframes from 12 other halls.
  - **Who:** Tomasz labelled all 42 on 2026-11-02 and 2026-11-17, in two passes totalling 84.5 min.
  - **Result:** 41 are usable; one Anwil frame is marked unusable.
  - **Each record:** the image→court homography in cm, the landmark dots Tomasz placed, any pins on painted lines, and the status.
  - **Agreement:** the saved courts fit his dots to a median of 5.6 px rms, with 10–14 px on the worst frames (Lublin and the held-out game).
- **Not in v0:** number crops, which move to M7 and are labelled just in time.

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

## Box gold v1

The 100 box-gold frames of `data/labels/box_gold.json`, with every object of the 10-class schema (`bga.types.DETECTION_CLASSES`) boxed tight by hand, so detection can be scored per class at AP50:95.

### Conventions

These follow Roboflow's training set `basketball-player-detection-3-ycjdo` (version 18, CC BY 4.0), checked on crops of its annotations on 2026-11-18, so that gold and training data draw boxes the same way.

- **Visible extent.** A box runs to the outermost visible pixel of the object: hair, fingers, shoes, an outstretched arm. Occluded parts are not guessed, and a box cut by the frame edge ends at the edge.
- **People:**
  - `player`, or one of the four action classes:
    - `player-in-possession`: holding or dribbling the ball. One ball, so at most one per frame.
    - `player-jump-shot`: in the shooting motion, including just after release.
    - `player-layup-dunk`: finishing at the rim.
    - `player-shot-block`: contesting a shot with a raised arm.
  - The box includes the ball when it is in the hands.
  - `referee` for the referees.
  - Coaches, bench and fans get no box.
- **`ball`:** the game ball only, tight, also when held. A spare ball off court gets no box.
- **`ball-in-basket`:** one box over the rim and the net around the ball while it is in the basket; the ball also gets its own `ball` box.
- **`rim`:** the metal ring only, not the net, bracket or backboard. Every visible rim, including the far one.
- **`number`:** each jersey number, front or back, tight around all its digits, also when partly hidden. Sponsor logos, chest badges, the stanchion logo and score graphics are not numbers.

### Prefill and first pass (2026-11-18)

`scripts/box_gold_v1.py` builds the queue `data/labels/box_gold_v1.json`, which the labeller's `/boxes` page edits:
1. **People:** the v0 person boxes, with track id and team. The class is an action class where the best-matching prefill detection says so at score ≥ 0.3, else `player`, or `referee` for referee tracks.
2. **Objects:** the prefill detector's ball, ball-in-basket, rim and number boxes at score ≥ 0.3, after per-class NMS at IoU 0.5 (the prefill ran with IoU 0.9 and put up to three boxes on one ball). Scores from 0.1 to 0.3 are kept as candidates the page can show and adopt.
3. **SAM 2.1 geometry** (`src/bga/remote/sam2_boxmask.py`, one box prompt per box on Modal, 0.03 USD):
   - where SAM's mask box agrees with the prefill box (IoU ≥ 0.85), the mask box is used;
   - elsewhere the prefill box stays and the mask box is offered as the alternative.

   On crops of 24 disagreements, SAM mostly cut off arms and feet or took in rim brackets, so a disagreement is a hint, not a flag.
4. **Implementer's first pass** on crop sheets (ledger rows of 2026-11-18), recorded in `data/labels/review/box_gold_v1.ops.txt`:
   - 91 low-score objects: 33 dropped, 53 confirmed, 5 left flagged as unclear;
   - 89 candidates: 40 adopted (33 numbers, 5 rims, 2 balls);
   - 24 flagged people confirmed as they are;
   - 15 proposed possessors that do not hold the frame's ball box set back to `player`.

   Most rejected balls were beige or white shorts, the crowd or seats. Most rejected numbers were the stanchion's sponsor logo or chest sponsors. Both are PLK confusions for M4 to learn.

Result: 1,475 boxes:

| Class | Boxes |
|---|---|
| player | 834 |
| number | 242 |
| referee | 213 |
| rim | 88 |
| ball | 50 |
| player-in-possession | 44 |
| player-jump-shot | 1 |
| player-layup-dunk | 1 |
| player-shot-block | 1 |
| ball-in-basket | 1 |

147 boxes in 61 frames are flagged for Tomasz:
- 110 people that overlap another person, whose geometry needs a human eye;
- 43 action classes;
- a few low-score objects and class conflicts.

8 frames carry a note with more than one proposed possessor.

### Tomasz's review

Tomasz reviews in the labeller at `/boxes`. Each frame shows flagged boxes first, and a loupe shows the selected box at up to 8× for pixel-level edges. Enter marks the frame done. `scripts/box_gold_v1.py export` writes `data/gold/v1/boxes.json` from the frames marked done.
