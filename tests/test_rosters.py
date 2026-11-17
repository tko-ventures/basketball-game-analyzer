import json

from scripts.fetch_rosters import json_array_after, rsc_payload, team_name
from tools.labeler.server import roster_key_for


def _roster(tmp_path, stem, team, aliases=(), season_names=()):
    seasons = {f"202{i}/2{i + 1}": {"team_name": n, "players": []} for i, n in enumerate(season_names)}
    (tmp_path / f"{stem}.json").write_text(
        json.dumps({"team": team, "aliases": list(aliases), "seasons": seasons})
    )


def test_roster_key_for_exact_alias_words_and_ambiguity(tmp_path):
    _roster(tmp_path, "legia_warszawa", "Legia Warszawa")
    _roster(tmp_path, "zastal", "Zastal Zielona Góra", aliases=["ORLEN Zastal"])
    _roster(tmp_path, "gtk_a", "Euvic GTK Gliwice")
    _roster(tmp_path, "gtk_b", "Tauron GTK Gliwice")
    assert roster_key_for("Legia", tmp_path) == "legia_warszawa"
    assert roster_key_for("legia warszawa", tmp_path) == "legia_warszawa"
    assert roster_key_for("ORLEN Zastal", tmp_path) == "zastal"
    assert roster_key_for("GTK", tmp_path) == ""  # two clubs match: refuse to guess
    assert roster_key_for("Euvic GTK", tmp_path) == "gtk_a"
    assert roster_key_for("Nobody", tmp_path) == ""
    assert roster_key_for("", tmp_path) == ""


def test_rsc_payload_players_and_team_name():
    inner = json.dumps(
        '{"team":{"id":11257,"name":"Dziki Warszawa"},"players":[{"id":1,"firstName":"A","lastName":"B",'
        '"shirtNumber":"7"}],"coaches":[]}'
    )[1:-1]
    html = f'<script>self.__next_f.push([1,"{inner}"])</script>'
    payload = rsc_payload(html)
    assert json_array_after(payload, "players")[0]["shirtNumber"] == "7"
    assert json_array_after(payload, "missing") == []
    assert team_name(payload, 11257) == "Dziki Warszawa"
