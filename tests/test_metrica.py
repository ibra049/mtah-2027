"""Parser + baseline checks on Metrica sample game 3 (EPTS, same family as the target ATD)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from mtah import baseline, io_target  # noqa: E402

G3 = os.path.expanduser("~/mtah-external/metrica-sample/data/Sample_Game_3")
META = os.path.join(G3, "Sample_Game_3_metadata.xml")
RAW = os.path.join(G3, "Sample_Game_3_tracking.txt")
pytestmark = pytest.mark.skipif(not os.path.isfile(RAW), reason="Metrica sample not present")


@pytest.fixture(scope="module")
def g3():
    return io_target.load_epts(META, RAW, "metrica3", pitch=(105.0, 68.0), dtype=np.float64,
                               return_meta=True)


def test_g3_structure(g3):
    seqs, meta, seen, stats = g3
    assert len(meta.layouts) == 11
    assert [s.period for s in seqs] == [1, 2]
    assert seqs[0].frame[0] == 0 and seqs[1].frame[0] == 69661       # second_half_start 69662 - 1
    assert sum(len(s) for s in seqs) == 143761 and seen.all()
    assert stats["malformed"] == stats["odd_groups"] == stats["odd_tokens"] == 0
    # both teams have ~11 visible players during play
    n_home = np.isfinite(seqs[0].home[..., 0]).sum(1)
    n_away = np.isfinite(seqs[0].away[..., 0]).sum(1)
    assert np.median(n_home) == 11 and np.median(n_away) == 11
    # inside the pitch (metres about the centre), allowing a small run-off
    for s in seqs:
        for arr in (s.home, s.away, s.ball[:, :2]):
            v = arr[np.isfinite(arr[..., 0])]
            assert (np.abs(v[:, 0]) <= 52.5 + 5).mean() > 0.99
            assert (np.abs(v[:, 1]) <= 34 + 5).mean() > 0.99


def test_g3_vs_kloppy(g3):
    """Every player/ball coordinate equals kloppy's on a 1-in-50 sample across all 11 specs."""
    kloppy = pytest.importorskip("kloppy")
    from kloppy import metrica
    seqs, meta, _, _ = g3
    ds = metrica.load_tracking_epts(meta_data=META, raw_data=RAW, sample_rate=1 / 50)
    df = ds.to_df()
    ball = np.concatenate([s.ball for s in seqs])
    players = np.full((len(ball), len(meta.slot_player), 2), np.nan)
    for s in seqs:
        players[s.frame[:, None], s.home_slot[None]] = s.home
        players[s.frame[:, None], s.away_slot[None]] = s.away
    frames = df["frame_id"].to_numpy().astype(int) - 1
    assert len(frames) > 2500 and frames.max() > 130000
    bx = (df["ball_x"].to_numpy(float) - 0.5) * 105.0
    by = (0.5 - df["ball_y"].to_numpy(float)) * 68.0
    np.testing.assert_allclose(ball[frames, 0], bx, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(ball[frames, 1], by, atol=1e-6, equal_nan=True)
    n_checked = 0
    for k, pid in enumerate(meta.slot_player):
        if f"{pid}_x" not in df:
            continue
        px = (df[f"{pid}_x"].to_numpy(float) - 0.5) * 105.0
        py = (0.5 - df[f"{pid}_y"].to_numpy(float)) * 68.0
        np.testing.assert_allclose(players[frames, k, 0], px, atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(players[frames, k, 1], py, atol=1e-6, equal_nan=True)
        n_checked += 1
    assert n_checked >= 25


def test_g3_baseline_parity(tmp_path):
    """Official starter baseline vs our port on real Metrica coordinates (first half, spec 1),
    converted to the target layout (x,y,speed groups; slot order == <Player> order)."""
    import importlib.util
    starter = os.path.join(io_target.REPO, "vendor", "challenge", "starter", "baseline.py")
    if not os.path.isfile(starter):
        pytest.skip("vendor missing")
    from fixtures import meta_xml
    meta = io_target.parse_meta(META)
    lay = meta.layouts[0]
    sides = [int(meta.slot_side[s]) for s in lay.group_slot]
    T = lay.end
    folder = tmp_path / "g3"
    folder.mkdir()
    (folder / "BAR-ATH_tactical_atd.xml").write_text(meta_xml(len(sides), sides, 1, T))
    with open(RAW) as src, open(folder / "BAR-ATH_tactical_atd.txt", "w") as dst:
        for i, line in enumerate(src):
            if i >= T:
                break
            fid, players, ball = line.rstrip("\n").split(":")
            players = ";".join(g + ",1.0" for g in players.split(";"))
            dst.write(f"{fid}:{players}:{ball},0.0\n")
    spec = importlib.util.spec_from_file_location("official_baseline", starter)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ref = mod.nearest_player_labels(str(folder / "BAR-ATH_tactical_atd.txt"),
                                    mod.team_of_each_track(str(folder / "BAR-ATH_tactical_atd.xml")),
                                    T, (105.3, 68.0))
    cfg = io_target.load_config()
    cfg.update(n_frames=T, period_split_frame=T // 2)
    seqs = io_target.load(str(folder), cfg, dtype=np.float64, with_tags=False)
    mine = baseline.full_codes(seqs, T)
    assert (ref != -1).mean() > 0.5
    assert int((ref != mine).sum()) == 0
    for w in (0, 50):
        assert np.array_equal(mod.smooth(ref, w), baseline.smooth(mine, w))
