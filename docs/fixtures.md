# Synthetic 2D tracking fixtures

`data/fixtures/synthetic/*.json` holds eight possessions of a Dziki Warszawa game, written by
`scripts/synth_plays.py`. They are **not measured**: the offence is choreographed by hand from set plays
every team runs, and the defence is simulated. They exist so the coach page (`scripts/coach_web.py`,
`/coach/` on the demo site) can show what a coach gets once the pipeline delivers accurate 2D positions,
and so the analysis code has a fixed input to be tested against. They are never gold and never
training data.

The format is the one the pipeline's 2D output should take (schema `bga.tracking2d/v0`), so the coach
page will read real possessions without changes once M3 and M4 deliver them.

## Format

| Field | Meaning |
| --- | --- |
| `schema`, `id`, `synthetic` | `bga.tracking2d/v0`, the possession id, `true` for these files |
| `fps` | 25 |
| `court` | `length_cm` 2800, `width_cm` 1500 (FIBA) |
| `game` | `home`, `away`, `quarter`, `clock_start_s` (game clock at frame 0), `shot_clock_start_s`, `score_before`, `score_after` |
| `offence`, `attacks` | the side with the ball (`home` or `away`) and the basket it attacks (`left` at x = 157.5 cm, `right`) |
| `labels` | what a play-recognition layer would report: `play`, `defence`, `coverage` |
| `players` | ten entries: `id` (`h<number>` or `a<number>`), `team`, `number`, `name` (Dziki only), `pos` |
| `tracks` | per player id, one `[x, y]` per frame in cm; x along the court from the left baseline, y across from the top sideline |
| `ball` | one `[x, y, z]` per frame in cm; z is the height |
| `holder` | per frame, the player id holding the ball, or `null` in flight |
| `events` | `pass` (`from`, `to`, `t`, `t1`), `shot` (`by`, `made`, `points`), `rebound`, and tactical marks: `screen`, `backscreen`, `handoff`, `cut`, `drive`, `tag`, `help`, `switch`, `double`, `rotation`, `flash`, `outlet` |

## How the plays are made

- **Offence:** waypoints `(t, u, v)` per player in half-court metres (u from the attacked baseline), interpolated with monotone cubic splines, plus a few centimetres of slow drift.
- **Defence:** each defender follows a rule:
  - man-to-man: between his man and the basket, sagging towards the ball the further his man is from it;
  - zone: a home spot that shifts with the ball, stepping out to a nearby ball handler.

  Each defender has a 0.2 s reaction delay and limits on speed (6.6 m/s) and acceleration (9 m/s²). The play's own decisions are scripted on top: drop, tag, switch, double, a late rotation. Bodies are pushed apart: 0.75 m from attackers, 0.7 m between defenders.
- **Ball:** held 0.35 m in front of the holder, passes at about 13 m/s, shots on an arc.

`tests/test_synthetic_plays.py` checks the following:
- the committed files match the generator;
- every player stays on the court;
- sustained speed stays under 9.5 m/s;
- no two players overlap for more than 0.4 s;
- each play still shows its point (for example, the Horns play ends in the one poor shot, with an open wing left unused).

## The eight possessions

| Id | Dziki | What it shows |
| --- | --- | --- |
| q1-0842-high-pnr | attack | High pick-and-roll against drop coverage. The roll draws the corner defender (the tag) and the corner three is open. |
| q1-0615-horns | attack | Horns into a hand-off and a drive into a crowded lane. A contested shot while the weak-side wing stands open. |
| q2-0730-spain-pnr | defend | A back-screen on the dropping centre frees the roller for a lay-up. |
| q2-0305-fast-break | attack | Outlet, wide lanes, 3-on-2, lay-up. |
| q3-0910-rotations | defend | Two drives, help and recover on time, a contested corner three, a defensive rebound. |
| q3-0548-zone-attack | attack | Against a 2-3 zone: high-post flash, an extra pass, an open corner three. |
| q4-0420-switch-double | defend | A switch, a post-up, a double team, and a late last rotation that leaves an open three. |
| q4-0038-stagger | attack | Staggered screens free the shooter for a catch-and-shoot three in the last minute. |

The insights the page shows (shot quality, open team-mates, spacing, passes, distance and speed) are
computed by `scripts/coach_insights.py` from positions and events only. Its expected-points table is
illustrative, not fitted to PLK.
