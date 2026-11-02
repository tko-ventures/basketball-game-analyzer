from bga.eval.mot import evaluate


def _seq(n=10):
    return [[(1, 0, 0, 10, 20), (2, 50, 0, 60, 20)] for _ in range(n)]


def test_perfect_tracking_scores_one():
    gt = _seq()
    r = evaluate({"a": (gt, gt)})
    assert abs(r["HOTA"] - 1) < 1e-6 and abs(r["IDF1"] - 1) < 1e-6 and r["MOTA"] == 1 and r["IDSW"] == 0


def test_identity_swap_halfway_costs_association_not_detection():
    gt = _seq()
    pred = [
        [(1, 0, 0, 10, 20), (2, 50, 0, 60, 20)] if t < 5 else [(2, 0, 0, 10, 20), (1, 50, 0, 60, 20)]
        for t in range(10)
    ]
    r = evaluate({"a": (gt, pred)})
    assert abs(r["DetA"] - 1) < 1e-6 and r["AssA"] < 0.7 and r["IDSW"] == 2 and abs(r["IDF1"] - 0.5) < 1e-6
