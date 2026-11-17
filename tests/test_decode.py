"""Decoder tests: exactness of the segment search, the football scenarios of the work order
(same-team passes, deflections, short turnovers, occlusion gaps, stoppages, restarts, half
reset), the forced mask, determinism, post-processing, tag priors and the X argument."""
import itertools

import numpy as np
import pytest

from mtah import decode as dec
from mtah.canonical import A, D, H
from mtah.evaluate import pm, to_codes

CFG = {"method": "segment", "cost_cap": 4605, "dead_bias": 0, "switch_cost": 20000,
       "min_poss": 50, "min_dead": 25}

SURE_H, SURE_A, SURE_D = (0.90, 0.05, 0.05), (0.05, 0.90, 0.05), (0.05, 0.05, 0.90)
FLAT = (1 / 3, 1 / 3, 1 / 3)
NAN = (np.nan, np.nan, np.nan)


def proba(*parts):
    """parts: (n_frames, (pH, pA, pD)) -> float32 [T,3]."""
    return np.concatenate([np.tile(np.asarray(p, np.float32), (n, 1)) for n, p in parts])


def runs(labels):
    return [(lab, e - s) for s, e, lab in dec.segments(labels)]


# ------------------------------------------------------------------ exactness of the search
def _feasible(labels, mins):
    segs = dec.segments(np.asarray(labels))
    for i, (s, e, lab) in enumerate(segs):
        if i == 0 or i == len(segs) - 1:
            continue                       # first and last segment are exempt
        prev = segs[i - 1][2]
        need = mins["dead"] if lab == D else mins["restart"] if prev == D else mins["turn"]
        if e - s < need:
            return False
    return True


@pytest.mark.parametrize("seed", range(40))
def test_segment_viterbi_is_exact(seed):
    rng = np.random.default_rng(seed)
    T = int(rng.integers(1, 9))
    cost = rng.integers(0, 12, size=(T, 3))
    pens = [int(v) for v in rng.integers(0, 8, size=3)]
    mins = {"turn": int(rng.integers(1, 5)), "restart": int(rng.integers(1, 4)),
            "dead": int(rng.integers(1, 5))}
    labels, total = dec.segment_viterbi(cost, *pens, mins["turn"], mins["restart"], mins["dead"])
    assert _feasible(labels, mins)
    assert dec.path_cost(labels, cost, *pens) == total
    best = min(dec.path_cost(np.array(c), cost, *pens)
               for c in itertools.product((H, A, D), repeat=T) if _feasible(c, mins))
    assert total == best


def test_plain_viterbi_when_minima_are_one():
    rng = np.random.default_rng(7)
    cost = rng.integers(0, 3000, size=(400, 3))
    labels, total = dec.segment_viterbi(cost, 2500, 2000, 1500)
    # reference: textbook Viterbi
    pen = np.array([[0, 2500, 2000], [2500, 0, 2000], [1500, 1500, 0]])
    best = cost[0].astype(np.int64)
    for t in range(1, len(cost)):
        best = (best[:, None] + pen).min(axis=0) + cost[t]
    assert total == best.min() == dec.path_cost(labels, cost, 2500, 2000, 1500)


# ------------------------------------------------------------------------- football scenarios
def test_same_team_passes_stay_one_possession():
    # carrier, ball in flight (model unsure), receiver, flight, receiver: one H possession
    p = proba((80, SURE_H), (20, (0.40, 0.35, 0.25)), (60, SURE_H), (15, (0.36, 0.40, 0.24)),
              (90, SURE_H))
    assert runs(dec.decode(p, None, CFG)) == [(H, 265)]


def test_deflection_does_not_change_possession():
    # an opponent touches the ball for a third of a second, the pass still arrives
    p = proba((120, SURE_H), (8, SURE_A), (120, SURE_H))
    assert runs(dec.decode(p, None, CFG)) == [(H, 248)]


def test_short_turnover_absorbed_long_turnover_kept_at_onset():
    short = proba((150, SURE_H), (30, SURE_A), (150, SURE_H))        # 1.2 s: no controlled change
    assert runs(dec.decode(short, None, CFG)) == [(H, 330)]
    real = proba((150, SURE_H), (120, SURE_A), (150, SURE_H))        # 4.8 s: a real possession
    assert runs(dec.decode(real, None, CFG)) == [(H, 150), (A, 120), (H, 150)]   # exact onset


def test_occlusion_gap_inside_a_possession_is_bridged():
    for gap in (FLAT, NAN):          # ball lost: flat probabilities, or no model output at all
        p = proba((100, SURE_H), (150, gap), (100, SURE_H))
        assert runs(dec.decode(p, None, CFG)) == [(H, 350)]


def test_occlusion_gap_across_a_turnover_gives_exactly_two_possessions():
    p = proba((100, SURE_H), (60, NAN), (100, SURE_A))
    out = dec.decode(p, None, CFG)
    assert [lab for lab, _ in runs(out)] == [H, A]
    boundary = int(np.flatnonzero(out == A)[0])
    assert 100 <= boundary <= 160    # somewhere in the blind stretch, never a third label


def test_stoppage_splits_possessions_and_short_dead_blip_does_not():
    p = proba((200, SURE_H), (200, SURE_D), (200, SURE_H))
    assert runs(dec.decode(p, None, CFG)) == [(H, 200), (D, 200), (H, 200)]
    blip = proba((200, SURE_H), (6, SURE_D), (200, SURE_H))            # too weak for the search
    assert runs(dec.decode(blip, None, CFG)) == [(H, 406)]
    blip = proba((200, SURE_H), (20, SURE_D), (200, SURE_H))           # found, then under min_dead
    assert runs(dec.decode(blip, None, CFG)) == [(H, 420)]
    assert runs(dec.decode(blip, None, dict(CFG, min_dead=1))) == [(H, 200), (D, 20), (H, 200)]


def test_short_dead_between_teams_goes_to_the_team_that_had_the_ball():
    p = proba((150, SURE_H), (15, SURE_D), (150, SURE_A))
    assert runs(dec.decode(p, None, CFG)) == [(H, 165), (A, 150)]


def test_min_poss_is_min_control_on_the_decoded_path():
    # two short opponent runs in a row are both absorbed; the third is long enough
    p = proba((100, SURE_H), (30, SURE_A), (20, SURE_H), (30, SURE_A), (40, SURE_H), (100, SURE_A))
    assert runs(dec.decode(p, None, CFG)) == [(H, 220), (A, 100)]
    assert [n for _, n in runs(dec.decode(p, None, dict(CFG, min_poss=1)))] == [100, 30, 20, 30, 40, 100]


def test_hard_minimum_inside_the_search_is_still_available():
    p = proba((150, SURE_H), (30, SURE_A), (150, SURE_H))
    out = dec.decode(p, None, dict(CFG, switch_cost=5000, min_poss=1, hard_min_turn=50,
                                   hard_min_restart=50))
    assert runs(out) == [(H, 130), (A, 50), (H, 150)]      # stretched to the minimum, not absorbed


def test_restart_names_the_team_the_whole_spell_supports():
    # after the dead ball the first frames lean H weakly, then A clearly holds it
    p = proba((150, SURE_D), (10, (0.45, 0.35, 0.20)), (140, SURE_A))
    assert runs(dec.decode(p, None, CFG)) == [(D, 150), (A, 150)]


def test_restart_spell_may_be_short_but_turnover_spell_may_not():
    # 1 s of A between two dead balls is a possession; 1 s of A inside H is not
    p = proba((100, SURE_D), (25, SURE_A), (100, SURE_D))
    assert runs(dec.decode(p, None, CFG)) == [(D, 100), (A, 25), (D, 100)]
    p = proba((100, SURE_H), (25, SURE_A), (100, SURE_H))
    assert runs(dec.decode(p, None, CFG)) == [(H, 225)]


def test_half_reset_no_state_crosses_halftime():
    first = proba((300, SURE_H))
    second = proba((30, (0.40, 0.35, 0.25)), (300, SURE_A))
    both = dec.decode_match([first, second], [None, None], CFG)
    assert runs(both[0]) == [(H, 300)]
    assert runs(both[1]) == [(A, 330)]       # the ambiguous opening takes the half's own evidence
    np.testing.assert_array_equal(both[1], dec.decode(second, None, CFG))
    joined = dec.decode(np.concatenate([first, second]), None, CFG)
    assert (joined[300:330] == H).all()      # what would happen without the reset


# ------------------------------------------------------------------ interface and determinism
def test_forced_mask_is_honoured_and_shapes_the_neighbourhood():
    p = proba((400, SURE_H))
    forced = np.full(400, -1, np.int8)
    forced[:120] = D                         # e.g. the pre-kickoff stretch of a half
    forced[300] = A                          # a single contradictory frame is still emitted
    out = dec.decode(p, None, CFG, forced)
    assert (out[:120] == D).all() and out[300] == A
    assert (out[120:250] == H).all() and set(out.tolist()) <= {H, A, D}
    for method in ("argmax", "smooth"):
        out = dec.decode(p, None, dict(CFG, method=method), forced)
        assert (out[:120] == D).all() and out[300] == A


def test_output_is_always_h_a_d_and_int8():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], size=3000).astype(np.float32)
    p[100:200] = np.nan
    for method in ("argmax", "smooth", "segment"):
        out = dec.decode(p, None, dict(CFG, method=method))
        assert out.dtype == np.int8 and out.shape == (3000,)
        assert set(out.tolist()) <= {H, A, D}


def test_integer_costs_and_rounding():
    cost = dec.frame_costs(np.array([[1.0, 0.0, 0.5], [0.12344, 0.12346, 1e-9]]), dict(CFG, cost_cap=0))
    assert cost.dtype == np.int64
    assert cost[0].tolist() == [0, 9210, 693]            # p=0 clipped to 1e-4
    assert cost[1, 0] == round(-1000 * np.log(0.1234)) and cost[1, 1] == round(-1000 * np.log(0.1235))
    assert dec.frame_costs(np.array([[0.0, 1.0, 0.0]]), CFG)[0].tolist() == [4605, 0, 4605]


def test_deterministic_and_insensitive_to_sub_quantum_noise():
    rng = np.random.default_rng(3)
    y = np.repeat(rng.integers(0, 3, size=60), rng.integers(20, 200, size=60))
    p = np.full((len(y), 3), 0.15)
    p[np.arange(len(y)), y] = 0.7
    p = np.clip(p + rng.normal(0, 0.25, p.shape), 0.01, None)
    p = np.round(p / p.sum(1, keepdims=True), 4)
    a = dec.decode(p.astype(np.float64), None, CFG)
    b = dec.decode((p + rng.uniform(-2e-5, 2e-5, p.shape)).astype(np.float32), None, CFG)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, dec.decode(p.astype(np.float64), None, CFG))


def test_unknown_config_key_is_an_error():
    with pytest.raises(KeyError):
        dec.resolve_cfg({"pen_trun": 1})


# ---------------------------------------------------------------------------- post-processing
def test_prune_merges_short_low_margin_spell_only():
    cfg = dict(CFG, switch_cost=0, min_poss=1, min_dead=1, prune_len_poss=75, prune_margin_poss=400)
    weak = proba((150, SURE_H), (40, (0.40, 0.50, 0.10)), (150, SURE_H))     # margin 223/frame
    assert runs(dec.decode(weak, None, cfg)) == [(H, 340)]
    strong = proba((150, SURE_H), (40, SURE_A), (150, SURE_H))               # margin 2890/frame
    assert runs(dec.decode(strong, None, cfg)) == [(H, 150), (A, 40), (H, 150)]
    long_weak = proba((150, SURE_H), (90, (0.40, 0.50, 0.10)), (150, SURE_H))
    assert runs(dec.decode(long_weak, None, cfg)) == [(H, 150), (A, 90), (H, 150)]


def test_prune_dead_joins_the_possession_around_it():
    cfg = dict(CFG, switch_cost=0, min_poss=1, min_dead=1, prune_len_dead=50, prune_margin_dead=400)
    p = proba((150, SURE_A), (20, (0.10, 0.40, 0.50)), (150, SURE_A))
    assert runs(dec.decode(p, None, cfg)) == [(A, 320)]


def test_shift_boundaries_by_transition_type():
    y = np.array([H] * 50 + [A] * 50 + [D] * 50 + [H] * 50, np.int8)
    out = dec.shift_boundaries(y, off_turn=-3, off_stop=4, off_restart=-2)
    assert runs(out) == [(H, 47), (A, 57), (D, 44), (H, 52)]
    tiny = np.array([H, A, D, H], np.int8)                      # nothing may vanish
    assert runs(dec.shift_boundaries(tiny, 30, -30, 30)) == [(H, 1), (A, 1), (D, 1), (H, 1)]


def test_snap_moves_team_changes_to_the_onset_peak_only():
    y = np.array([H] * 100 + [A] * 100 + [D] * 50 + [H] * 50, np.int8)
    onset = np.zeros(300)
    onset[92], onset[205], onset[250] = 0.8, 0.9, 0.9
    out = dec.snap_boundaries(y, onset, 12, 0.3)
    assert runs(out) == [(H, 92), (A, 108), (D, 50), (H, 50)]          # D boundaries untouched
    assert runs(dec.snap_boundaries(y, onset * 0.3, 12, 0.3)) == runs(y)        # peak too low
    assert runs(dec.snap_boundaries(y, onset, 5, 0.3)) == runs(y)               # peak too far
    p = proba((100, SURE_H), (100, SURE_A), (50, SURE_D), (50, SURE_H))
    cfg = dict(CFG, snap_radius=12)
    assert runs(dec.decode(p, None, cfg, aux={"turn_onset": onset})) == runs(out)
    assert runs(dec.decode(p, None, cfg)) == runs(y)                   # works without aux
    assert runs(dec.decode(p, None, cfg, aux={"other": onset})) == runs(y)


# -------------------------------------------------------------------------------- tag priors
TAG_DTYPE = np.dtype([("start", "i8"), ("end", "i8"), ("code", "U64"), ("team", "U1"),
                      ("type", "U64"), ("half", "U16")])


class Seq:
    def __init__(self, T, tags, f0=1000):
        self.frame = np.arange(f0, f0 + T)
        self.tags = np.array(tags, dtype=TAG_DTYPE)


def test_tags_are_off_by_default_and_bounded():
    seq = Seq(600, [(1100, 1400, "BUILD UP", "A", "", "1st Half")])
    p = proba((600, (0.38, 0.36, 0.26)))
    low = dict(CFG, switch_cost=3000)
    assert runs(dec.decode(p, seq, low)) == [(H, 600)]                       # switch off
    on = dict(low, tags={"enabled": True})
    out = dec.decode(p, seq, on)
    assert [lab for lab, _ in runs(out)] == [H, A, H] and (out[100:400] == A).all()
    sure = proba((600, SURE_H))                                              # tag vs clear model
    assert runs(dec.decode(sure, seq, on)) == [(H, 600)]
    tc = dec.tag_costs(Seq(600, [(1100, 1400, "BUILD UP", "A", "", "")] * 20), dec.DEFAULTS["tags"], 600)
    assert tc.max() == dec.DEFAULTS["tags"]["cap"]


def test_defending_phase_is_evidence_for_the_other_team_and_na_is_neutral():
    tcfg = dec.DEFAULTS["tags"]
    att = dec.tag_costs(Seq(300, [(1050, 1150, "Progression", "H", "", "")]), tcfg, 300)
    assert att[60, A] > 0 and att[60, H] == 0 and att[60, D] == 0
    dfn = dec.tag_costs(Seq(300, [(1050, 1150, "DEFENDING IN MIDDLE THIRD", "H", "", "")]), tcfg, 300)
    assert dfn[60, H] > 0 and dfn[60, A] == 0
    for team in ("N", "?"):
        assert not dec.tag_costs(Seq(300, [(1050, 1150, "PROGRESSION", team, "", "")]), tcfg, 300).any()
    assert not dec.tag_costs(Seq(300, [(1050, 1150, "GOALS", "H", "", "")]), tcfg, 300).any()


def test_set_piece_window_favours_dead_before_and_team_after():
    tcfg = dec.DEFAULTS["tags"]
    tc = dec.tag_costs(Seq(600, [(1200, 1325, "SET PIECES", "A", "Corner Kick", "")]), tcfg, 600)
    assert tc[150, H] > 0 and tc[150, A] > 0 and tc[150, D] == 0             # before: dead
    assert not tc[200:325].any()                                             # inside: unknown
    assert tc[330, D] > 0 and tc[330, H] > 0 and tc[330, A] == 0             # after: alive, away
    throw = dec.tag_costs(Seq(600, [(1200, 1325, "SET PIECES", "N", "Throw-in", "")]), tcfg, 600)
    assert throw[330, D] > 0 and throw[330, H] == 0 and throw[330, A] == 0   # no team for N/A
    off = dict(tcfg, restart=False)
    assert not dec.tag_costs(Seq(600, [(1200, 1325, "SET PIECES", "A", "Corner Kick", "")]), off, 600).any()
    # end to end: an unsure model around a tagged corner
    seq = Seq(600, [(1200, 1325, "SET PIECES", "A", "Corner Kick", "")])
    p = proba((200, (0.30, 0.30, 0.40)), (125, FLAT), (275, (0.37, 0.38, 0.25)))
    out = dec.decode(p, seq, dict(CFG, tags={"enabled": True}))
    assert out[100] == D and out[400] == A and H not in set(out.tolist())


def test_forced_mask_helper():
    seq = Seq(500, [(1300, 1420, "SET PIECES", "H", "Kick Off (Start)", "2nd Half")])
    f = dec.forced_mask(seq, kickoff_tag=True)
    assert (f[:300] == D).all() and (f[300:] == -1).all()
    f = dec.forced_mask(seq, graded_blocks=[[1100, 1399]], dead_intervals=[[1100, 1149]])
    assert (f[:150] == D).all() and (f[150:400] == -1).all() and (f[400:] == D).all()


# ----------------------------------------------------------------------- abstention argument
def test_x_never_beats_the_label_it_repeats():
    """Official scorer: on spellF1 an X is identical to repeating the previous label, and on
    macroF1 it is never better -- so the decoder holds its label instead of abstaining."""
    rng = np.random.default_rng(11)
    gt = np.repeat(rng.integers(0, 3, size=80), rng.integers(10, 300, size=80)).astype(np.int8)
    pred = gt.copy()
    for s in rng.integers(0, len(gt) - 60, size=40):                 # damage the prediction
        pred[s:s + rng.integers(5, 60)] = rng.integers(0, 3)
    abst = pred.copy()
    for s in rng.integers(0, len(gt) - 80, size=60):                 # then abstain in places
        abst[s:s + rng.integers(1, 80)] = pm.X
    g = to_codes(gt)
    filled = pm.fill_abstentions(to_codes(abst), pm.valid_runs(g))
    with_x, held = pm.evaluate(g, to_codes(abst)), pm.evaluate(g, filled)
    assert with_x["spell_f1_mean"] == held["spell_f1_mean"]
    assert with_x["macro_f1"] <= held["macro_f1"]
    assert with_x["n_abstain_on_graded"] > 0 and held["n_abstain_on_graded"] == 0
