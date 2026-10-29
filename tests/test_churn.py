from bga.eval.churn import compute_churn, format_churn, id_set_change_events, track_lifetimes


def test_empty():
    c = compute_churn([])
    assert c["unique_track_ids"] == 0 and c["frame_count"] == 0 and c["id_set_delta_rate"] == 0.0


def test_set_delta_ignores_sentinel():
    per = [[1, 2], [1, 2], [1, 3], [1, 3, -1]]
    assert id_set_change_events(per) == 1
    c = compute_churn(per)
    assert c["id_set_change_events"] == 1
    assert c["id_set_delta_rate"] == 1 / 3
    assert c["unique_track_ids"] == 3


def test_short_lived_and_survival():
    per = [[1, 3], [1, 3], [1, 2], [1], [1]]
    assert track_lifetimes(per) == {1: 5, 3: 2, 2: 1}
    c = compute_churn(per, short_lived_frames=3)
    assert c["short_lived_id_count"] == 2
    assert abs(c["short_lived_id_fraction"] - 2 / 3) < 1e-9
    assert c["mean_track_survival_frames"] == 8 / 3
    assert c["median_track_survival_frames"] == 2.0
    assert format_churn(c).startswith("churn unique=3 ")
