# Gold set

Plan section 8, as amended by the 2026-11-02 decision (`docs/decisions.md`).

## Versions

| Version | Who reviewed | What it supports | Status |
|---|---|---|---|
| v0 | Implementer, full pass on all 12 shots; Tomasz's dry-run labels on Legia shot 10 kept as they are | Comparing backends; HOTA, IDF1, MOTA, team and number accuracy, AP50 | Committed 2026-11-02 |
| v1 | Tomasz reviews at least the held-out game; boxes tightened by hand | Absolute numbers for the client; AP50:95; number readers | Box gold done 2026-11-18: `data/gold/v1/boxes.json`, 100 frames reviewed by Tomasz (151 min). Number crops labelled 2026-11-22: `data/gold/v1/numbers.json`, implementer's read, checked by Tomasz (110 crops, 8 min). Track gold v1 not started |

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
- **Number crops:** not in v0. Labelled for M7 on the box gold v1 frames, in `data/gold/v1/numbers.json` (section below).

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

## Team audit (2026-11-20)

The v0 track teams were proposed by Skalski's TeamClassifier (step 2 above), the same model the `siglip_kmeans` team backend runs, so before M5 scores that backend they were checked by eye with no classifier involved:

- **Sheets:** `scripts/team_gold_sheets.py` draws every v0 track of the 12 shots as one row of 7 crops from the gold boxes with full visibility, rows grouped home, away, referee, so a kit in the wrong group stands out (24 sheets, ledger row of 2026-11-20).
- **Result:** all 164 track teams match the kit (129 players, 35 referees). No change to `identities.csv`.
- **Single crops of another player inside a track:** six tracks show one such crop each (`OSp445e0mXU_s10` tracks 9, 1018 and 1026, `1jDaJBkUGmc_s16` 1040, `4XMD9FWmQws_s42` 1028, `Tnuscrm9RyU_s2` 1012), a box or identity slip within the track. Team accuracy takes the majority of a track's matched boxes, so these do not change it; they matter to track gold v1.
- **Box gold v1 against the tracks:** 1,061 v1 person boxes carry a v0 track id; 1,059 have the track's team. The other two (`4XMD9FWmQws_s78` track 1035 at frame 25800, `OSp445e0mXU_s10` track 1019 at frame 3180) were moved in Tomasz's review onto another player and kept the old track id. Their v1 team is right for the person boxed, so the stale id is a track gold v1 matter, not a team error.

Track teams in gold v0 are therefore usable as team gold. Tomasz's time: none.

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
2. **Objects:** the prefill detector's ball, ball-in-basket, rim and number boxes at score ≥ 0.3, after per-class NMS at IoU 0.5 (the prefill ran with IoU 0.9 and put up to three boxes on one ball). Scores from 0.1 to 0.3 are kept as candidates the page can show and adopt. So are the detector's player and referee boxes that overlap no person box (IoU < 0.3), for people the tracks missed; `box_gold_v1.py candidates` recomputes them on the frames not done without touching boxes (added 2026-11-18 after Tomasz's first frame, 199 candidates).
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

### Tomasz's review (2026-11-18)

Tomasz reviewed all 100 frames at `/boxes` in 151 minutes of active time.

| Against the prefill | Boxes |
|---|---|
| geometry kept | 1,174 |
| geometry changed (median 13.5 px on the largest edge, p90 50 px) | 281 |
| added: 84 players, 60 balls, 56 numbers, 19 referees, 8 rims, 7 others | 234 |
| deleted | 20 |
| class changed (mostly possession and the other action classes) | about 90 |

The prefill missed many people and nearly all balls: track gold v0 had left out players and referees at the frame edges and behind others, and the detector misses small, blurred balls.

### QA after the review

A sanity check looked for:
- boxes moved more than 120 px or changed between person and object classes: all were boxes Tomasz had reused;
- tiny boxes, persons without a team, and duplicates: none;
- frames with more than one ball: 14.

In every multi-ball frame the real ball was the one Tomasz drew. The other was a detector box scored at 0.6 or more, which no first pass had checked: the implementer's pass covered only scores below 0.6, and the page did not flag them.

All 270 such object boxes were then checked on crop sheets (ledger rows of 2026-11-18):

| Class | Checked | False |
|---|---|---|
| ball | 32 | 20 (crowd, fan items, the Eurobasket graphic by the shot clock) |
| number | 166 | 1 (the stanchion logo) |
| rim | 72 | 0 |

The 21 false boxes were dropped (`data/labels/review/box_gold_v1.qa.ops.txt`, applied with `apply --include-done`). Every frame now has one ball, or none where it is out of view (20 frames).

### Export

`scripts/box_gold_v1.py export` wrote `data/gold/v1/boxes.json`: 100 frames, 24 of them from the held-out game, and 1,669 boxes.

| Class | Boxes |
|---|---|
| player | 900 |
| number | 287 |
| referee | 233 |
| rim | 96 |
| ball | 80 |
| player-in-possession | 50 |
| player-shot-block | 10 |
| player-jump-shot | 7 |
| ball-in-basket | 3 |
| player-layup-dunk | 3 |

Classes with fewer than about 20 boxes are too small for a stable AP of their own. They are reported with their counts.

### How to review again

Tomasz reviews in the labeller at `/boxes`. Each frame shows flagged boxes first, and a loupe shows the selected box at up to 8× for pixel-level edges. Enter marks the frame done. `scripts/box_gold_v1.py export` writes `data/gold/v1/boxes.json` from the frames marked done.

## Holder gold v1 (2026-11-24)

Box gold v1 was labelled on 2026-11-18, two days before the possession schema of 2026-11-20 ([ADR](adr/2026-11-20-detector-classes-and-possession-schema.md)) defined the holder as whoever holds or dribbles the ball, a shooter still holding it included. A shooter boxed as `player-jump-shot` or `player-layup-dunk` was therefore never a holder in box gold, and some clear holds were left as `player`. `data/gold/v1/holders.json` gives each of the 100 frames its holder under the 2026-11-20 definition, as the index of a person box in the frame's `boxes.json` entry (or none). `bga eval-ball` scores possession against it; box gold is unchanged, so detection rows still score the boxes Tomasz drew.

`scripts/holders_gold_v1.py` builds it from the `player-in-possession` boxes and the ops file `data/labels/review/holders_v1.ops.txt`.

The implementer's first pass (ledger row of 2026-11-24) checked every frame on crops around the gold ball, and the wide frames without a gold ball again at full resolution:

| Against box gold | Frames |
|---|---|
| unchanged | 79 |
| changed (5 shooters still holding the ball, 5 clear holds or dribbles, 1 possession box on a shooter whose ball is already in flight) | 11 |
| flagged for Tomasz, box gold kept until he decides (release or hold, dribble or loose ball, a dead-ball hold) | 10 |

Box gold also has no ball box in two wide frames where the ball is visible at full resolution (`1jDaJBkUGmc` 1405, dribbled; 6510, in a shooter's raised hands). That is box geometry and waits for Tomasz.

### How to decide the flagged frames

Each flagged line in the ops file proposes an answer. Change its status to `tomasz` (keeping or changing the box index, or `none`) and rerun `uv run scripts/holders_gold_v1.py`.

## Number crops (v1, 2026-11-22)

`data/gold/v1/numbers.json` scores number readers crop by crop, independent of detector and tracker. Written by `scripts/number_gold.py export` from `data/labels/numbers/gold_crops.json`. The implementer read every crop; Tomasz then checked a sample of 30 and every crop the implementer had left `illegible` (110 crops, 7 min 53 s), and his answer is the label wherever he gave one.

### Crops

| Source | Crops | Digits | Illegible | Not a number |
|---|---|---|---|---|
| Tomasz's `number` boxes in box gold v1 (`n001`–`n287`) | 287 | 255 | 25 | 7 |
| The PLK detector's `number` boxes on the same frames that overlap no gold box (`d001`–`d089`) | 89 | 27 | 23 | 39 |
| Total (292 dev, 84 held-out) | 376 | 282 | 48 | 46 |

- **Gold boxes:** the box geometry comes from Tomasz's hand-drawn boxes, so it does not depend on any model.
- **Detector boxes:** `plk-r3-poss` boxes at a score of 0.3 or more, the threshold the numbers layer reads from. Most are sponsor logos, chest badges and the back patch of the Trefl kit, or parts of numbers. These are the crops a reader has to refuse.
- **Confusions present:** of the 282 legible crops, 33 contain a 6 or a 9 and 104 a 1 or a 7 (a 7 in 10 of them). Back sponsor patches and chest logos are among the `not_number` crops.

### Labels

- **Digits** (`"7"`, `"23"`): the number can be named from the crop's pixels alone, without knowing who the player is.
- **`illegible`:** a jersey number that cannot be named from the pixels: blurred, folded, turned away, torn between two numbers (3 or 8), or partly hidden (only the 6 of a 16 showing).
- **`not_number`:** the box holds no jersey number: a logo, a badge, a patch, or a crease in the shirt. When it is unclear whether the box holds a logo or an unreadable number, the label is `illegible`.
- **`team`:** the side of the gold track that holds the box, taken from the smallest person box containing at least 90% of it. When the box sits on two overlapping players and only the other side's roster has the read number, that side is used instead (8 crops).

### How they were labelled

1. **Blind read.** `number_gold.py sheets` shows each crop twice: tight and upscaled, and in context. Each crop is captioned with its id only, so the gold track's number does not anchor the read. The implementer read all 376 crops.
2. **Compare.** `compare` lists every read that differs from the holder's gold number, or whose holder has none. Each was checked against the clip season's rosters:
   - A doubtful read was changed only to `illegible`, never to the track's number, so no label comes from gold v0.
   - 14 reads became `illegible`: 11 against 31, 0 against 8, 10 against 40, and partial numbers.
3. **Audit trail.**
   - Ops: `data/labels/review/number_gold.ops.txt`.
   - Sheet reads: `data/labels/sheet_ledger.tsv`, rows of 2026-11-22.
4. **Spot check.** `number_gold.py spotcheck` draws a seeded sample of 30 and stores it in the working file. Tomasz checks it in the labeller at `/numbers`, where each answer is saved to the crop's `review`. `export` takes his label wherever he changes one.
   - **Result (2026-11-22, Tomasz, 117 s):** all 20 digit labels in the sample are right, as are the 1 `not_number` and 3 of the 9 `illegible` labels. He read the other 6 `illegible` crops as numbers: n051 #4, n076 #10, n145 #45, n198 #0, d030 #32 and d087 #3.
   - **Reading:** the implementer's `illegible` is too strict, so the 80 `illegible` crops not yet reviewed go to Tomasz too (`/numbers?set=illegible`).
   - **Caveat:** during the spot check the page showed each holder's gold track number beside the crop. 5 of the 6 numbers Tomasz read match it (d030 shows #32 among holders #3, #32 and #1; n198's holder has none), so these reads may be anchored. The number was removed from the page, and these 6 answers were cleared so that they come up again, blind, in the `illegible` pass.
   - **Blind re-check (2026-11-22, Tomasz):** of the 6, n051 #4, n145 #45, n198 #0 and d030 #32 stand. n076 and d087 are now illegible, so those two earlier reads were the number on screen. He also read d012 as #4, which he had confirmed as illegible before. On the 9 `illegible` crops of the sample, blind, he reads 5.
5. **Illegible pass (2026-11-22, Tomasz, 222 s).** Every crop the implementer labelled `illegible` (89, the sample's 9 included), shown blind at `/numbers?set=illegible`:
   - 30 are numbers he can read, 13 are not numbers (chest logos and creases), and 46 stay illegible.
   - Two of his reads were checked against the rosters and changed to `illegible` with his agreement: n179 shows only the 6 of Chávez's 16, and n104 reads 6 on a Dziki kit whose roster has no 6.
   - 5 of his own box gold v1 `number` boxes are among the `not_number` crops (n035, n220, n224, n236, n256: chest logos on the light kits). That is a box gold v1 matter for the `number` class AP, left as it is.
6. **Agreement.** The implementer's digit reads were never contradicted (20 of 20 in the sample). The implementer's `illegible` was right on 48 of 89 crops (final labels: 28 numbers, 13 not numbers, 48 illegible).

### Gold v0 gaps found on the way

- **Tracks without a number:** 14 legible crops sit on gold v0 tracks that have no number, and 3 on no gold track. For example, `4XMD9FWmQws_s4` track 1001 shows #4, and `OSp445e0mXU_s19` track 1040 shows #0.
- **Stale holders:** two holder boxes are another player (n079, n282), as in the team audit.

Number accuracy per track still reads `identities.csv`, so these gaps belong to track gold v1.
