# Gold set

Plan section 8, as amended by the 2026-11-02 decision (`docs/decisions.md`).

## Versions

| Version | Who reviewed | What it supports | Status |
|---|---|---|---|
| v0 | Implementer, full pass on all 12 shots; Tomasz's dry-run labels on Legia shot 10 kept as they are | Comparing backends; HOTA, IDF1, MOTA, team and number accuracy, AP50 | Committed 2026-11-02 |
| v1 | Tomasz reviews at least the held-out game; boxes tightened by hand | Absolute numbers for the client; AP50:95; number readers | Box gold done 2026-11-18: `data/gold/v1/boxes.json`, 100 frames reviewed by Tomasz (151 min). Number crops labelled 2026-11-22: `data/gold/v1/numbers.json`, implementer's read, checked by Tomasz (110 crops, 8 min). Track gold v1 not started |

v0 was prefilled by the public models the baseline uses (Skalski's player detector, SAM 2.1), so v0 absolute scores are biased upwards. Every row in `docs/metrics.md` names its gold version.

## Contents

- **Games:** 4 highlight games, 3 central shots each, listed in `data/labels/gold.json`. The held-out game is `Tnuscrm9RyU` (Dziki – Zastal, 2026/27). It is scored once per milestone and never trained on. Two full-length games (`full_game: true`, one dev, one held-out) hold shot gold only (section "Shot gold").
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

Box gold v1 was labelled on 2026-11-18, two days before the possession schema of 2026-11-20 ([ADR](adr/2026-11-20-detector-classes-and-possession-schema.md)) defined the holder as whoever holds or dribbles the ball, a shooter still holding it included. A shooter boxed as `player-jump-shot` or `player-layup-dunk` was therefore never a holder in box gold. `data/gold/v1/holders.json` gives each of the 100 frames its holder under the 2026-11-20 definition, as the index of a person box in the frame's `boxes.json` entry (or none). `bga eval-ball` scores possession against it; box gold is unchanged, so detection rows still score the boxes Tomasz drew.

`scripts/holders_gold_v1.py` builds it from the `player-in-possession` boxes and the ops file `data/labels/review/holders_v1.ops.txt`.

The implementer's first pass (ledger row of 2026-11-24) checked every frame on crops around the gold ball, and the wide frames without a gold ball again at full resolution. It changed 11 frames and flagged 10 for Tomasz. Tomasz then answered all 21 on the `/holders` page, which shows the frames from 0.6 s before to 0.6 s after each gold frame:

| First pass | Frames | Tomasz's answers |
|---|---|---|
| shooter still holding the ball, no holder in box gold | 5 | all 5 kept |
| possession box on a shooter whose ball is already in flight | 1 | kept (no holder) |
| hold or dribble judged from the single frame, no holder in box gold | 5 | all 5 overruled (no holder) |
| flagged: release or hold, dribble or loose ball, a dead-ball hold | 10 | 3 holders (1 a pass, not a shot, in `4XMD9FWmQws` 14585; 1 another player than proposed), 7 no holder |

A single frame overstates holds: all five holds called from one frame were passes or catches in the neighbouring frames.

### Box gold fixes (2026-11-24)

The holder review turned up three box-gold errors, which Tomasz fixed at `/boxes` before `scripts/box_gold_v1.py export`. No box index moved, so the holders still point at the right boxes:
- `4XMD9FWmQws` 14585: the passer (box 10) was `player-jump-shot`; now `player-in-possession`.
- `1jDaJBkUGmc` 1405 and 6510: the game ball was boxed as `rim` and as `ball-in-basket`; both are now `ball`.
- `4XMD9FWmQws` 1935: one player box 5 px narrower.

Dev now has 60 ball, 71 rim, 38 in-possession, 5 jump-shot and 2 ball-in-basket boxes (58, 72, 37, 6 and 3 before). Detection and ball rows were scored again on the fixed gold (`docs/metrics.md`, 2026-11-24). With the fixes, 8 of the 100 holders differ from box gold.

### How to answer holder questions

Open the labeller's `/holders` page (`uv run python tools/labeler/server.py`, then http://127.0.0.1:8765/holders). Each question shows the gold frame zoomed around the ball, the frames from 0.6 s before to 0.6 s after it (Space plays them), and the proposed holder. Press A to accept the proposal or N for no holder, or click a player box and press Enter. Each answer is saved straight away: a `tomasz` line goes under the flagged line in the ops file, and `data/gold/v1/holders.json` is rebuilt. The second tab shows the implementer's applied changes for a second look. `bga.eval.holders` holds the logic, which `scripts/holders_gold_v1.py` also runs.

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

## Shot gold (2026-11-26)

Gold v1 scores the ball layer on 16 shot attempts and 100 frames, too few to tell one more right shooter from noise ([ADR](adr/2026-11-26-shot-gold-more-layer7-gold.md)). Shot gold adds 26 shots of the same four games, labelled only for what that layer needs. It lives in one working file, `data/labels/shot_gold.json` (`bga.eval.shot_gold`); a shot or a frame counts as gold once Tomasz has confirmed it (`status: done`).

### Contents

| | Dev | Held-out |
|---|---|---|
| Shots | 18 (6 per game) | 8 |
| Frames, one a second | 133 | 55 |

- **Shots:** central shots of at least 6 s that are not in gold v0, spread evenly over each clip by shot index (`pick_shots`). The choice reads the shot list only, never a model's output.
- **Per shot, its shot attempts:** release frame, rim frame, made or missed, 2 or 3 points, and the shooter's box and team at the release frame.
- **Per frame:** the ball box, or none when the ball is not in view, and the holder's box and team, or nobody.
- **No training use:** every frame of the gold games is refused by the leakage guard, as before.

### Conventions

- **Shot attempt:** every try at the basket in live play: jump shots, layups, dunks and tip-ins, and also blocked shots and airballs, which are missed. Free throws are left out.
- **Rim frame:** the ball touches the rim or the backboard, or passes through or past the ring. For a blocked shot it is the frame of the block.
- **Release frame:** the first frame with the ball out of the shooter's hands. For a dunk it equals the rim frame.
- **Shooter:** the player who sends the ball at the basket; on an alley-oop or a tip-in, the finisher.
- **Holder:** the 2026-11-20 definition, as in holder gold v1. A dribbler holds the ball also between hand and floor; a ball in flight, loose, fought for or in a referee's hands has no holder; a receiver holds it from the frame it is under control.
- **Frames are judged with their neighbours.** The page shows 0.6 s either side of each frame, and the first pass read 0.4 s either side, because single-frame hold calls were all overruled in holder gold v1.

### Boxes offered for clicking

`configs/shot-gold-prefill.yaml` runs the active detector (`plk-r3-poss`) on the 26 shots, and `scripts/shot_gold.py prefill` caches its person, ball and rim boxes under `data/labels/.cache/shot_gold/`. A holder or shooter is a click on one of those boxes, or a box Tomasz drags when the player has none. A ball is a click: it takes the detector's ball box when the click lands within 30 px of one, else a 22 px box at the click. So box geometry leans towards the detector; scoring matches boxes at IoU 0.4 to 0.5 and the ball within 25 px, where that matters little. The held-out shots went through the detector for these boxes only; no held-out score was read.

### First pass (2026-11-26)

`scripts/shot_gold.py sheets` draws the implementer's sheets (ledger rows of 2026-11-26):
- **Holders:** each of the 188 frames at 1.2 times its size around the detector's ball, with numbered person boxes and the ball candidates ringed, beside the same region 0.4 and 0.2 s before and after (47 sheets).
- **Shot attempts:** each moment where the detector's ball comes within 2 rim widths of a rim, at most one per second (31 moments in 24 of the 26 shots): the rim over 0.9 s, the proposed release over 0.4 s, and the whole release frame with numbered boxes.

Result (ledgers `data/labels/review/shot_gold_holders.tsv` and `data/labels/review/shot_gold_events.tsv`, applied via `scripts/shot_gold.py apply`):
- **Holders:** 95 of 188 frames settled by the first pass (67 player, 28 nobody; 63 dev, 32 held-out) and 93 flagged for Tomasz (70 dev, 23 held-out).
- **Shot attempts:** 25 proposed attempts across 23 of the 26 shots (18 dev, 7 held-out; 3 shots with 0 attempts; 6 candidate sheets rejected as non-shots). Outcome: 6 made, 5 missed, 14 unclear; points: 18 two-pointers, 3 threes, 4 unclear; 12 carry a proposed shooter box and 13 need a click or release-frame adjustment. Every shot stays unconfirmed until Tomasz saves it on Tab 1.

### How to label

Open the labeller's `/shotgold` page (`uv run python tools/labeler/server.py`, then http://127.0.0.1:8765/shotgold). **How to** (`?`) holds the instructions and every key; `tools/labeler/README.md` lists them too. Tab 1 is the shot attempts of each shot, tab 2 the frames the first pass could not settle, tab 3 the frames it settled, for a second look. Each answer saves at once.

### Confirmed export (2026-11-26)

Tomasz confirmed all 26 shots and all 188 frames on `/shotgold`. `uv run scripts/shot_gold.py export` writes the confirmed answers to `data/gold/v1/shot_gold.json`:
- **Dev:** 18 shots, 20 shot attempts (17 made, 3 missed; 11 two-pointers, 9 threes; every attempt carries a shooter box and team), 133 frames (119 with ball, 72 with holder).
- **Held-out:** 8 shots, 8 shot attempts (8 made, 0 missed; 5 two-pointers, 3 threes; every attempt carries a shooter box and team), 55 frames (53 with ball, 32 with holder).
- **Combined Layer 7 gold (`bga eval-ball`):** 27 dev shots / 209 frames / 31 attempts and 11 held-out shots / 79 frames / 13 attempts (use `--no-shot-gold` to score gold `v1` alone).

### Full-game shots (2026-11-26)

The 26 shots above hold 37 made and 7 missed attempts (gold v1 included), because highlights show mostly baskets. Outcome accuracy on them mostly measures how often a backend says "made" (#48). Two full-length YouTube games add live play in which about half the attempts miss. Both games are gold games in `data/labels/gold.json` (`full_game: true`), so no training set may use them.

| Game | Split | Shots picked | Seconds | Frames (every 3 s) |
|---|---|---|---|---|
| `hSXPCy5NedU` Dziki – Tauron GTK Gliwice, 2026-02-15 | Dev | 40 | 792 | 268 |
| `1A79NXz9CFM` Dziki – PGE Start Lublin, 2026-10-11 | Held-out | 20 | 455 | 154 |

- **Shots:** `tools/labeler/prepare_clips.py` cuts each game with `hist_diff`. The implementer tags every shot of at least 6 s from its middle frame (`sheet_ledger.tsv`): `central` is the main camera with the scorebug on screen. Timeouts and breaks without the scorebug, replays, close-ups and the half-time highlights of other games are `other`, and shorter shots stay `unset`.
- **Pick:** `uv run scripts/shot_gold.py add --clip <stem> --count N --step-s 3 --max-s 40` picks `N` central shots evenly over the game (`pick_shots`, reading the shot list only), keeps the middle 40 s of each (`trim_window`), and labels a frame every 3 s, a third as often as the highlight shots, so the batch stays reviewable.
- **Same game, other uploads:** `same_game_ids` in a game's entry names other uploads of that game, and the leakage guard refuses them too. The PGE Start highlights `Fulq9md9DZc` are cut from the same broadcast as `1A79NXz9CFM`.
- **First pass:** as for the highlight shots, plus rim close-ups every 0.25 s placed by the detector's rim box (never its ball), so attempts the ball detector misses are still read. Such attempts enter the events ledger as `<sequence>#x<n>` rows with explicit frames.
