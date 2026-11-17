import re

import numpy as np

from mtah import drift


def _ref(rng, names):
    X = rng.normal(0, 1, (4000, len(names))).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan
    return drift.stats_from_matrix(X, names)


def test_same_distribution_not_flagged_and_shift_detected():
    rng = np.random.default_rng(0)
    names = [f"f{i}" for i in range(20)]
    ref = {"profiles": {"nominal": _ref(rng, names), "stoppage": _ref(rng, names)}}
    same = _ref(rng, names)
    assert drift.compare(same, ref["profiles"]["nominal"])["n_outside"] == 0
    X = rng.normal(0, 1, (4000, len(names))).astype(np.float32)
    X[:, 3] += 3.0                      # one feature shifted by ~2.2 IQR
    X[:, 7] = np.nan                    # one feature always missing
    c = drift.compare(drift.stats_from_matrix(X, names), ref["profiles"]["nominal"])
    assert c["rows"][0]["feature"] == "f7" and c["rows"][1]["feature"] == "f3"
    assert c["rows"][1]["shift"] > 2 and c["n_outside"] == 2


def test_report_lines_pass_the_safe_filter():
    rng = np.random.default_rng(1)
    names = [f"h_feat_{i}" for i in range(30)]
    ref = {"profiles": {"nominal": _ref(rng, names), "stoppage": _ref(rng, names)}}
    lines = drift.report_lines(_ref(rng, names), ref)
    pat = re.compile(r"^SAFE\| [A-Za-z0-9_.=:/+, -]{0,240}$")
    assert all(pat.match("SAFE| " + l) for l in lines)
    assert len(lines) <= 80 and lines[-1].startswith("drift closer_profile=")


def test_four_profiles_fit_the_safe_line_budget():
    rng = np.random.default_rng(2)
    names = [f"n_feature_{i:03d}" for i in range(300)]
    ref = {"profiles": {p: _ref(rng, names) for p in ("nominal", "stoppage", "nominal_lj", "stoppage_lj")}}
    lines = drift.report_lines(_ref(rng, names), ref)
    pat = re.compile(r"^SAFE\| [A-Za-z0-9_.=:/+, -]{0,240}$")
    assert len(lines) == 4 * 16 + 1 <= 80 and all(pat.match("SAFE| " + l) for l in lines)


def test_five_profiles_stay_within_80_lines():
    rng = np.random.default_rng(3)
    names = [f"n_feature_{i:03d}" for i in range(300)]
    ref = {"profiles": {p: _ref(rng, names) for p in ("nominal", "stoppage", "nominal_lj", "stoppage_lj", "stoppage_sm")}}
    lines = drift.report_lines(_ref(rng, names), ref)
    assert len(lines) == 5 * 15 + 1 <= 80
