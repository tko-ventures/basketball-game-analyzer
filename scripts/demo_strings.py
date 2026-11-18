"""On-screen text of the web demo (scripts/demo_web.py).

Keys are used in the page templates as {{key}}; the ``JS`` table goes to the page script as T, with {name}
placeholders filled by tr(). The site is English only.

"""

from __future__ import annotations

# Burnt into the rendered videos.
VIDEO = {
    "referee": "REFEREE",
    "automatic": "automatic, uncorrected",
}

PAGE = {
    "lang": "en",
    "title": "Video analysis – demo",
    "h1": "Game video analysis – demo",
    "sub_analysis": "Player detection and tracking, teams, jersey numbers, positions on a 2D court, "
    "coverage and spacing. Fully automatic, no manual correction.",
    "to_coach": "Coach view (simulated data) →",
    "camera": "Broadcast camera",
    "court_analysis": "2D court · analysis",
    "fullscreen": "Full screen",
    "click_hint": "Click a player to highlight him",
    "play": "Play / pause (space)",
    "back": "Previous frame (←)",
    "fwd": "Next frame (→)",
    "speed": "Playback speed",
    "layers": "Layers:",
    "cover": "Coverage",
    "cover_tip": "A line from each attacker to the nearest defender; yellow ring: the most open player",
    "shape": "Spacing",
    "shape_tip": "Each team's shape, its area and the mean distance between attackers",
    "trails": "Trails",
    "trails_tip": "Movement in the last second",
    "footer_analysis": "Click a player in the video or on the court to highlight him in both views (Esc: "
    "clear). Space: play/pause · ←/→: one frame (with Shift: 1 s) · 1–__N__: clip (each keeps its own "
    "position, speed and selected player) · ⛶: full screen for one view.<br>Coverage: a line from each "
    "attacking player to the nearest defender, with the distance in metres; the yellow ring marks the most "
    "open player. The attacking team is inferred from positions (the defence stands closer to its own "
    "basket). Only players visible in the video are counted.",
}

JS = {
    "no_calibration": "no reliable camera calibration in this segment",
    "no_calibration_info": "2D positions return once the camera shows enough court lines.",
    "no_number": "{team} (no number)",
    "attack": "Attacking: {team} → {side} basket",
    "left": "left",
    "right": "right",
    "open": "Most open: {who}, {m} m from the nearest defender",
    "shape_off": "offence: {m} m between players on average, area {a} m²",
    "shape_def": "defence: area {a} m²",
    "shape_line": "Spacing – {parts}",
    "transition": "Transition or mid-court play – coverage is computed in the half-court offence",
    "selected": "Selected: {who} · speed {v} km/h · distance in this clip {d} m",
    "selected_out": "The selected player is out of frame",
    "visible": "Visible in the video: {home} {nh}/5 · {away} {na}/5",
}


def surname(name: str) -> str:
    """Display name on labels: everything after the first given name ("Bennett Vander Plas" -> "Vander Plas")."""
    parts = name.split(maxsplit=1)
    return parts[-1] if parts else ""
