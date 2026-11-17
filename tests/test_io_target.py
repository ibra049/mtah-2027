import hashlib
import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from fixtures import meta_xml, raw_line, random_game, tags_xml, write_game  # noqa: E402
from mtah import baseline, io_target, submit  # noqa: E402
from mtah.canonical import Sequence  # noqa: E402

REPO = io_target.REPO
STARTER = os.path.join(REPO, "vendor", "challenge", "starter", "baseline.py")
L, W = 105.3, 68.0


def small_cfg(T, split, blocks=None):
    cfg = io_target.load_config()
    cfg.update(n_frames=T, period_split_frame=split,
               graded_blocks=blocks or [[2, split - 2], [split, T - 3]])
    return cfg


def official():
    spec = importlib.util.spec_from_file_location("official_baseline", STARTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ parser: edge cases
def test_edge_cases(tmp_path):
    sides = [0, 1, 0, 1]  # interleaved teams in <Player> order
    lines = [
        raw_line(1, [(0.1, 0.2), (0.9, 0.8), None, (0.5, 0.5)], (0.25, 0.75)),
        raw_line(2, [(0.1, 0.2), None, None, None], None),                      # ball missing
        "3:0.1,0.2,1;;NaN,NaN,NaN;0.3,0.4,1:0.5,0.5,0",                           # empty slot
        "4:0.1,abc,1;0.2,0.3;0.4,0.4,1,9;0.6,0.6,1:0.5,0.5,0",                   # odd tokens
        "5:garbage",                                                            # malformed
        raw_line(6, [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.5, 0.5)], "nan,nan,nan"),
        raw_line(99, [None] * 4, None),                                         # out of range
    ]
    (tmp_path / "BAR-ATH_tactical_atd.xml").write_text(meta_xml(4, sides, 1, 7))
    (tmp_path / "BAR-ATH_tactical_atd.txt").write_text("\r\n".join(lines) + "\r\n\n")
    cfg = small_cfg(7, 3)
    seqs, info = io_target.load(str(tmp_path), cfg, dtype=np.float64, return_meta=True)
    st = info["stats"]
    assert [s.period for s in seqs] == [1, 2]
    assert np.array_equal(np.concatenate([s.frame for s in seqs]), np.arange(7))
    h = np.concatenate([s.home for s in seqs])
    a = np.concatenate([s.away for s in seqs])
    b = np.concatenate([s.ball for s in seqs])
    assert h.shape == (7, 2, 2) and a.shape == (7, 2, 2) and b.shape == (7, 3)
    assert list(seqs[0].home_slot) == [0, 2] and list(seqs[0].away_slot) == [1, 3]
    # frame 0 (id 1): metres about the centre, y up
    np.testing.assert_allclose(h[0, 0], [(0.1 - .5) * L, (.5 - .2) * W])
    np.testing.assert_allclose(a[0, 0], [(0.9 - .5) * L, (.5 - .8) * W])
    assert np.isnan(h[0, 1]).all()
    np.testing.assert_allclose(b[0, :2], [(0.25 - .5) * L, (.5 - .75) * W])
    assert np.isnan(b[:, 2]).all()
    assert np.isnan(b[1]).all()                                # ball missing
    assert np.isnan(a[2, 0]).all() and np.isfinite(a[2, 1]).all()   # empty slot -> NaN
    assert np.isnan(h[3, 0, 1]) and np.isfinite(h[3, 0, 0])   # 'abc' -> NaN, x kept
    np.testing.assert_allclose(a[3, 0], [(0.2 - .5) * L, (.5 - .3) * W])  # 2-field group: x,y kept
    assert np.isfinite(h[3, 1]).all()                          # 4-field group: x,y still read
    assert np.isnan(h[4]).all() and np.isnan(b[4]).all()       # malformed line -> NaN
    assert not info["seen"][4] and info["seen"][:4].all()
    assert np.isnan(b[5]).all()                                # lowercase nan
    assert np.isnan(h[6]).all()                                # frame 7 never written
    assert st["malformed"] == 1 and st["out_of_range"] + st["no_layout"] == 1 and st["odd_tokens"] >= 3
    assert seqs[0].home.dtype == np.float64
    assert io_target.load(str(tmp_path), cfg)[0].home.dtype == np.float32


def test_spec_group_order_differs_from_player_order(tmp_path):
    """Raw text columns follow DataFormatSpecification, not <Player> order."""
    sides = [0, 0, 1, 1]
    (tmp_path / "m.xml").write_text(meta_xml(4, sides, 1, 1, group_order=[3, 2, 1, 0]))
    (tmp_path / "r.txt").write_text(raw_line(1, [(0.4, .5), (0.3, .5), (0.2, .5), (0.1, .5)],
                                             None) + "\n")
    seqs = io_target.load_epts(str(tmp_path / "m.xml"), str(tmp_path / "r.txt"), "t",
                               pitch=(100.0, 50.0), split_frames=[], dtype=np.float64)
    np.testing.assert_allclose(seqs[0].home[0, :, 0], [(0.1 - .5) * 100, (0.2 - .5) * 100])
    np.testing.assert_allclose(seqs[0].away[0, :, 0], [(0.3 - .5) * 100, (0.4 - .5) * 100])


def test_two_field_groups_and_xy_ball(tmp_path):
    """Metrica sample layout: x,y only per player and ball."""
    (tmp_path / "m.xml").write_text(meta_xml(2, [0, 1], 1, 2, channel_ids=("x", "y"),
                                             ball_channels=("x", "y")))
    (tmp_path / "r.txt").write_text("1:0.1,0.2;0.3,0.4:0.5,0.6\n2:NaN,NaN;0.3,0.4:NaN,NaN\n")
    s = io_target.load_epts(str(tmp_path / "m.xml"), str(tmp_path / "r.txt"), "t",
                            pitch=(L, W), split_frames=[], dtype=np.float64)[0]
    np.testing.assert_allclose(s.ball[0, :2], [0, (.5 - .6) * W], atol=1e-12)
    assert np.isnan(s.home[1]).all() and np.isnan(s.ball[1]).all()


def test_save_load_roundtrip(tmp_path):
    P, B = random_game(np.random.default_rng(0), 50, [0, 1] * 3)
    tx = tags_xml([(0.0, 1.0, "BUILD UP", {"Team": "Barcelona", "Half": "1st Half"})])
    write_game(str(tmp_path), P, B, [0, 1] * 3, tags_xml=tx)
    s = io_target.load(str(tmp_path), small_cfg(50, 20))[0]
    s.save(tmp_path / "s.npz")
    r = Sequence.load(tmp_path / "s.npz")
    for k in ("frame", "ball", "home", "away", "pitch", "y", "home_slot", "away_slot"):
        np.testing.assert_array_equal(getattr(r, k), getattr(s, k))
    assert r.tags.tolist() == s.tags.tolist() and r.period == 1 and r.match_id == "BAR-ATH"


# ------------------------------------------------------------------ smart tagging
def test_tags(tmp_path):
    inst = [(0.0, 2.02, "BUILD UP", {"Team": "Barcelona", "Half": "1st Half"}),
            (1.5, 3.0, "PROGRESSION", {"Team": "Atletic Club"}),
            (2.0, 4.5, "SET PIECES", {"Team": "N/A", "Type": "Throw-in"}),
            (3.0, 3.5, "LONG BALL", {"Team": " barcelona "}),
            (5.0, 6.0, "GOALS", {"Half": "2nd Half"}),
            (6.06, 6.1, "CREATING CHANCES", {"team": "ATLETIC CLUB"})]
    p = tmp_path / "t.xml"
    p.write_text(tags_xml(inst))
    t = io_target.load_tags(str(p))
    assert len(t) == 6
    assert list(t["team"]) == ["H", "A", "N", "H", "?", "A"]
    assert list(t["start"]) == [0, 38, 50, 75, 125, int(round(6.06 * 25))]
    assert list(t["end"]) == [int(round(2.02 * 25)), 75, 112, 88, 150, int(round(6.1 * 25))]
    assert t["type"][2] == "Throw-in" and t["code"][0] == "BUILD UP" and t["half"][1] == ""


def test_tags_match_starter(tmp_path):
    """Frames identical to starter/load_smart_tagging.py on the same file."""
    rng = np.random.default_rng(3)
    inst = [(float(a), float(a + d), "PROGRESSION", {"Team": "Barcelona"})
            for a, d in zip(np.round(rng.uniform(0, 5000, 300), 3), rng.uniform(1, 30, 300))]
    p = tmp_path / "t.xml"
    p.write_text(tags_xml(inst))
    spec = importlib.util.spec_from_file_location(
        "st", os.path.join(REPO, "vendor", "challenge", "starter", "load_smart_tagging.py"))
    if not os.path.isfile(spec.origin):
        pytest.skip("vendor missing")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.pattern_path = lambda *a, **k: str(p)
    ref = mod.load()
    t = io_target.load_tags(str(p))
    assert [e["start_frame"] for e in ref] == list(t["start"])
    assert [e["end_frame"] for e in ref] == list(t["end"])


# ------------------------------------------------------------------ baseline parity
def _official_codes(mod, folder, T):
    meta = os.path.join(folder, "BAR-ATH_tactical_atd.xml")
    raw = os.path.join(folder, "BAR-ATH_tactical_atd.txt")
    return mod.nearest_player_labels(raw, mod.team_of_each_track(meta), T, (L, W))


@pytest.mark.skipif(not os.path.isfile(STARTER), reason="vendor missing")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_baseline_parity_random(tmp_path, seed):
    rng = np.random.default_rng(seed)
    T = 3000
    sides = list(rng.integers(0, 2, 46))
    P, B = random_game(rng, T, sides)
    # engineered exact ties: two players mirrored about the ball, home slot first / away first
    for t in range(0, T, 37):
        B[t] = (0.5, 0.5)
        P[t, :, :] = np.nan
        P[t, 3] = (0.49, 0.5)
        P[t, 7] = (0.51, 0.5)
        P[t, 9] = (0.5, 0.52)
    write_game(str(tmp_path), P, B, sides)
    mod = official()
    ref = _official_codes(mod, str(tmp_path), T)
    seqs = io_target.load(str(tmp_path), small_cfg(T, 1500), dtype=np.float64, with_tags=False)
    mine = baseline.full_codes(seqs, T)
    assert np.array_equal(ref, mine)
    for w in (0, 25, 50, 125):
        assert baseline.official_csv_bytes(baseline.smooth(mine, w)) == \
            _official_csv(mod, ref, w, tmp_path)


def _official_csv(mod, codes, w, tmp_path):
    p = mod.write_submission(str(tmp_path / f"o{w}.csv"), mod.smooth(codes, w))
    return open(p, "rb").read()


# ------------------------------------------------------------------ submit
def test_submit(tmp_path):
    cfg = io_target.load_config()
    n = cfg["n_frames"]
    codes = np.zeros(n, dtype=np.int8)
    sha = submit.write(codes, str(tmp_path / "s.csv"), cfg)
    data = (tmp_path / "s.csv").read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha
    lines = data.decode().splitlines()
    assert lines[0] == "frame,label" and len(lines) == n + 1
    assert lines[1] == "0,X" and lines[11185 + 1] == "11185,H" and lines[11184 + 1] == "11184,X"
    assert lines[90652 + 1] == "90652,X" and lines[190197 + 1] == "190197,H"
    assert lines[-1] == "199318,X"
    if os.path.isdir(os.path.join(REPO, "vendor", "challenge", "evaluation")):
        assert submit.validate(str(tmp_path / "s.csv"), n) is None
    assert submit.render(np.array(["h"] * n), cfg) == submit.render(codes, cfg)
