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

## Results

Filled from `scripts/decoder_bench.py compare --model <base> --model-b <base>s` (frozen decoder on
both sides); see the tables below.
