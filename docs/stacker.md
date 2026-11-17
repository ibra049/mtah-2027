# Temporal stacker (lane DECODER)

`mtah.stack` refines the frame model's per-frame P(H, A, D) with a second model that looks at the
probabilities themselves over about ±2 s. `mtah.model_stacked` wraps frame model + stacker behind
the ordinary model interface, so the pipeline (`mtah.predict`) and the decoder are unchanged.

## Why

The decoder is within about 0.02 spellF1 of what the frame model's probabilities allow
(`docs/decoder.md`): the remaining loss is possessions of 1–5 s that the frame model does not
commit to, because its probabilities lag behind a touch and stay sticky through a short spell. A
model that sees the *shape* of the probability curves around a frame (a bump of the other team
framed by two onset peaks, a slow drift, a stretch without the ball) can re-sharpen them. The
frame model cannot do this itself: it never sees its own output.

## What it is

- **Input, per frame**: `pH − pA`, `pH + pA`, the frame model's team-change onset output, the
  nearest-player team margin to the ball (metres, 0 without a ball observation) and the
  ball-observed flag, each sampled at 21 offsets from −50 to +50 frames; means of those channels
  over six windows; the onset maximum before and after the frame; frames since and until the ball
  is observed. 135 columns, rounded to 1e-4.
- **Team symmetry by construction**: every column is either unchanged or sign-flipped when home
  and away are exchanged, so a swap is one multiplication. Training swaps a random half of the
  rows.
- **Model**: LightGBM multiclass (H, A, D), 200 rounds, 63 leaves, deterministic settings.
- **Labels**: `relabel.min_control(y, 0.5 s)`, the same view the frame model is trained on.

## How it is fitted without leakage

The stacker is fitted on the frame model's **out-of-fold** probabilities (leave-one-match-out over
the 5 non-holdout DFL matches), i.e. on probabilities of the same quality the frame model
produces on a match it has never seen, which is the situation on the target. For scoring, the
stacker itself is fitted leave-one-match-out again: the stacked prediction of match *m* comes from
a stacker fitted on the other four matches only. The artefact used on the target is fitted on all
five. Holdout matches, the Metrica games and the target are never used for fitting.

One command per base model (every degradation profile the base model exported is used):

```bash
PYTHONPATH=src python scripts/stack_oof.py --base lgb_v2 --name lgb_v2s \
    --metrica --holdout --save models/stack_lgb_v2
PYTHONPATH=src python scripts/stack_check.py --artefact models/stack_lgb_v2 --name lgb_v2s
```

`stack_check.py` proves the saved artefact reproduces the stored stacked predictions bit for bit,
both from the stored frame-model outputs and end to end from the degraded sequences.

## Use in the pipeline

```json
"model": {"module": "mtah.model_stacked", "artefact": "models/stack_lgb_v2"}
```

`meta.json` in the artefact names the base model module and artefact; the booster file is checked
against its recorded SHA-256 on load, and the feature list is checked on every call.

## Three feature sets

- **standard** (135 columns): everything listed above.
- **outputs only** (79 columns, `use_cues: false`): the frame model's probabilities and onset
  only, with no ball-observed flag, no nearest-player margin and no time since/until the ball.
  This variant cannot react to a tracking property the frame model has not already digested, so
  it transfers exactly as far as the frame model does.
- **probabilities only** (54 columns, `use_cues: false, use_onset: false`): `pH − pA` and
  `pH + pA` only. For a frame model whose onset output is out of range on the target.

Why the third set exists: the label-free input check (below) showed that `lgb_v1`'s
probabilities are in range on the target, but its onset output is almost silent there (about a
tenth of the external peak rate), because that head keys on ball-velocity jumps that the target's
low-pass-filtered ball does not produce. Forcing the onset input of the outputs-only stacker to
zero on the Metrica games drops it 0.03–0.05 *below* the plain frame model (it has learned "no
onset nearby means no change"), so a stacker may only use the onset channel of a frame model
whose onset output is in range on the target.

## Results (frozen decoder on both sides; paired rule: ≥ +0.01 mean, non-negative on ≥ 4 of 5)

Family = mean official spellF1 over the label views 0 / 0.5 / 1.0 s.

### On `lgb_v1` (the frame model that transfers cleanly to the target)

| data | plain v1 | + stacker, standard | + outputs only | + probabilities only |
|---|---|---|---|---|
| DFL nominal (5 matches) | 0.6469 | 0.6729 (+0.026, 5/5) | 0.6697 (+0.023, 5/5) | 0.6570 (+0.010, 5/5) |
| DFL harsh | 0.6161 | 0.6348 (+0.019, 5/5) | 0.6300 (+0.014, 5/5) | 0.6247 (+0.009, 4/5) |
| Metrica nominal (3 games) | 0.6575 | 0.6677 (+0.010, 2/3) | 0.6910 (+0.034, 3/3) | 0.6835 (+0.026, 3/3) |
| Metrica harsh | 0.6088 | 0.6175 (+0.009, 3/3) | 0.6436 (+0.035, 3/3) | 0.6342 (+0.025, 3/3) |
| Metrica nominal, low jitter | 0.6740 | 0.6982 (+0.024, 2/3) | 0.7137 (+0.040, 3/3) | 0.6914 (+0.017, 3/3) |
| Metrica stoppage, low jitter | 0.7115 | 0.7456 (+0.034, 3/3) | 0.7487 (+0.037, 3/3) | 0.7384 (+0.027, 3/3) |
| outputs only with the onset input forced to 0, Metrica nominal | | | 0.6199 (−0.038 vs v1) | n/a |

The probabilities-only stacker (`models/stack_lgb_v1q`, `configs/v1q.json`) keeps about half of
the outputs-only gain, and it is the half that depends only on inputs that are in range on the
target. With the onset input silenced, the outputs-only stacker falls below the plain frame
model on every Metrica draw (−0.03 to −0.05), so it is not used with `lgb_v1`.

The standard set is 0.003 better on DFL nominal but loses most of its gain on the Metrica games,
through the alive/dead decision (alive-dead F1 0.953 → 0.943): the ball cues it learned on the
DFL degradation draws misread Metrica's stoppage gaps. The outputs-only set keeps alive-dead F1
at 0.952 and gains on every Metrica game in every draw (`models/stack_lgb_v1p`,
`configs/v1p.json`), but it depends on the onset channel (see above), which rules it out for
`lgb_v1` on the target.

What changes in the output (view 0.5 s, outputs-only): recall at IoU 0.5 of true possessions of
1–2 s rises from 0.24 to 0.31 on DFL and from 0.25 to 0.36 on Metrica, of 2–5 s from 0.54 to 0.58
and from 0.51 to 0.58; team changes matched within one second from 0.42 to 0.47 (DFL) and from
0.51 to 0.58 (Metrica), with the rate of spurious changes unchanged. The dead-ball share and the
number of dead-ball runs do not move.

### On `lgb_v2` (three degradation profiles, so the selection value exists)

| data | plain v2 | + stacker, standard |
|---|---|---|
| DFL nominal | 0.6730 | 0.6845 (+0.012, 5/5) |
| DFL stoppage | 0.6842 | 0.6962 (+0.012, 5/5) |
| DFL select (mean of the two) | 0.6786 | 0.6903 (+0.012, 5/5) |
| DFL harsh | 0.6270 | 0.6438 (+0.017, 5/5) |
| Metrica nominal / harsh | 0.7110 / 0.6592 | 0.7371 / 0.6878 (+0.026 / +0.029, 3/3) |

### Variants tried and rejected

- **Possession-length sample weights** (short possessions weighted up to 4×): recall of 1–2 s
  possessions rises to 0.51, but precision falls from 0.78 to 0.67; with the frozen decoder the
  selection value drops by 0.015 and the Metrica games by 0.06.
- **Re-tuning the decoder on stacked probabilities**: the switch-cost plateau is 3000–6000 as
  before (0.6893 / 0.6903 / 0.6915 / 0.6893 at 3000 / 4500 / 6000 / 9000); the frozen decoder is
  kept.

### Related experiment: hybrid of two frame models (`scripts/hybrid_oof.py`)

Dead-ball mass from `lgb_v1`, team split from `lgb_v2` or `lgb_v3`
(`mtah.model_hybrid.combine`): +0.023 and +0.020 over v1 on DFL nominal. Gating shows where that
comes from: using the newer team split only on frames where both models name the same team gains
+0.002; using it only where they disagree gains +0.022. The benefit is the newer model overruling
v1's team call, i.e. it rests on the touch features that are out of range on the target. The
hybrid was therefore not selected; the stacker, which adds no new input, was.

## Label-free transfer check

`mtah.stack.input_summary(probas, onsets)` summarises the stacker's inputs (quantiles of
`pH − pA`, `|pH − pA|`, `pD`, onset; shares of confident frames; onset peaks per minute).
`configs/stack_input_reference_lgb_v1.json` holds it for the external predictions of `lgb_v1`;
running the same function on the frame model's target output shows whether the stacker sees
inputs in the range it was fitted on. No labels are involved.
