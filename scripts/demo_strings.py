# ruff: noqa: E501  (sentences are kept whole for translation)
"""On-screen text of the web demo (scripts/demo_web.py), one table per language.

Polish is the default edition at /; English is built into /en/. Keys used in the page templates as
{{key}}; the "js" table goes to the page script as T, with {name} placeholders filled by tr().
Polish is client-facing text: plain, correct Polish, no calques.
"""

from __future__ import annotations

LANGS = ("pl", "en")

# Burnt into the rendered videos.
VIDEO = {
    "pl": {
        "referee": "SĘDZIA",
        "automatic": "automatycznie, bez korekty",
        "no_calibration": "brak wiarygodnej kalibracji kamery w tym fragmencie",
        "court_caption": "pozycje na boisku FIBA 28 x 15 m, automatycznie",
    },
    "en": {
        "referee": "REFEREE",
        "automatic": "automatic, uncorrected",
        "no_calibration": "no reliable camera calibration in this segment",
        "court_caption": "positions on a FIBA 28 x 15 m court, automatic",
    },
}

PAGE = {
    "pl": {
        "lang": "pl",
        "other_lang": "English",
        "title": "Analiza wideo – demo",
        "h1": "Analiza wideo meczów – demo",
        "sub_video": "Wykrywanie i śledzenie zawodników, drużyny, numery, pozycje na boisku 2D. "
        "Wynik automatyczny, bez ręcznej korekty.",
        "sub_analysis": "Wykrywanie i śledzenie zawodników, drużyny, numery, pozycje na boisku 2D, "
        "krycie i ustawienie. Wynik automatyczny, bez ręcznej korekty.",
        "to_analysis": "Wersja z analizą (krycie, ustawienie) →",
        "to_video": "← Wersja: dwa zsynchronizowane filmy",
        "camera": "Obraz z kamery",
        "court": "Rzut 2D boiska",
        "court_analysis": "Rzut 2D boiska · analiza",
        "fullscreen": "Pełny ekran",
        "click_hint": "Kliknij zawodnika, aby go wyróżnić",
        "play": "Start / pauza (spacja)",
        "back": "Klatka wstecz (←)",
        "fwd": "Klatka naprzód (→)",
        "speed": "Prędkość",
        "layers": "Warstwy:",
        "cover": "Krycie",
        "cover_tip": "Linia od każdego atakującego do najbliższego obrońcy; żółty pierścień: najbardziej wolny",
        "shape": "Ustawienie",
        "shape_tip": "Obrys ustawienia drużyn, pole i średnia odległość w ataku",
        "trails": "Ślady",
        "trails_tip": "Ruch w ostatniej sekundzie",
        "footer_video": "Spacja: start/pauza · ←/→: klatka (z Shift: 1 s) · 1–__N__: akcja (każda pamięta "
        "swoją pozycję i prędkość) · ⛶: pełny ekran wybranego widoku, sterowanie działa dla obu.",
        "footer_analysis": "Kliknij zawodnika na obrazie albo na boisku, aby go wyróżnić w obu widokach (Esc: "
        "odznacz). Spacja: start/pauza · ←/→: klatka (z Shift: 1 s) · 1–__N__: akcja (każda pamięta swoją "
        "pozycję, prędkość i wybranego zawodnika) · ⛶: pełny ekran wybranego widoku.<br>Krycie: linia od "
        "każdego zawodnika ataku do najbliższego obrońcy z odległością w metrach; żółty pierścień to zawodnik "
        "najbardziej wolny. Kto atakuje, wynika z ustawienia (obrona stoi bliżej swojego kosza). Liczone tylko "
        "z zawodników widocznych na obrazie.",
    },
    "en": {
        "lang": "en",
        "other_lang": "Polski",
        "title": "Video analysis – demo",
        "h1": "Game video analysis – demo",
        "sub_video": "Player detection and tracking, teams, jersey numbers, positions on a 2D court. "
        "Fully automatic, no manual correction.",
        "sub_analysis": "Player detection and tracking, teams, jersey numbers, positions on a 2D court, "
        "coverage and spacing. Fully automatic, no manual correction.",
        "to_analysis": "Analysis view (coverage, spacing) →",
        "to_video": "← Two synchronised videos",
        "camera": "Broadcast camera",
        "court": "2D court",
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
        "footer_video": "Space: play/pause · ←/→: one frame (with Shift: 1 s) · 1–__N__: clip (each keeps its "
        "own position and speed) · ⛶: full screen for one view; the controls drive both.",
        "footer_analysis": "Click a player in the video or on the court to highlight him in both views (Esc: "
        "clear). Space: play/pause · ←/→: one frame (with Shift: 1 s) · 1–__N__: clip (each keeps its own "
        "position, speed and selected player) · ⛶: full screen for one view.<br>Coverage: a line from each "
        "attacking player to the nearest defender, with the distance in metres; the yellow ring marks the most "
        "open player. The attacking team is inferred from positions (the defence stands closer to its own "
        "basket). Only players visible in the video are counted.",
    },
}

JS = {
    "pl": {
        "no_calibration": VIDEO["pl"]["no_calibration"],
        "no_calibration_info": "Pozycje 2D pojawią się, gdy kamera pokaże wystarczająco dużo linii boiska.",
        "no_number": "{team} (bez numeru)",
        "attack": "Atak: {team} → kosz {side}",
        "left": "lewy",
        "right": "prawy",
        "open": "Najbardziej wolny: {who}, {m} m od najbliższego obrońcy",
        "shape_off": "atak: średnio {m} m między zawodnikami, pole {a} m²",
        "shape_def": "obrona: pole {a} m²",
        "shape_line": "Ustawienie – {parts}",
        "transition": "Przejście albo akcja w środkowej strefie – krycie liczone w ataku pozycyjnym",
        "selected": "Wybrany: {who} · prędkość {v} km/h · dystans w akcji {d} m",
        "selected_out": "Wybrany zawodnik jest teraz poza kadrem",
        "visible": "Widoczni na obrazie: {home} {nh}/5 · {away} {na}/5",
    },
    "en": {
        "no_calibration": VIDEO["en"]["no_calibration"],
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
    },
}


def surname(name: str) -> str:
    """Display name on labels: everything after the first given name ("Bennett Vander Plas" -> "Vander Plas")."""
    parts = name.split(maxsplit=1)
    return parts[-1] if parts else ""
