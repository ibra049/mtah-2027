"""Canonical per-match-half container (CONTRACT.md v0). Shared by every lane.

Label codes: 0=H, 1=A, 2=D, -1 unknown (masked). 25 fps, metres, origin at pitch centre.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

H, A, D, UNK = 0, 1, 2, -1
FPS = 25


@dataclass
class Sequence:
    match_id: str
    period: int
    frame: np.ndarray            # int64 [T]
    ball: np.ndarray             # float32 [T,3], NaN where unobserved
    home: np.ndarray             # float32 [T,K,2], NaN where slot empty
    away: np.ndarray             # float32 [T,K,2]
    pitch: np.ndarray            # float32 [2] length, width
    y: np.ndarray                # int8 [T]
    tags: np.ndarray | None = None  # optional structured array (start, end, code, team)
    extra: dict = field(default_factory=dict)  # optional named per-frame arrays (not relied on by other lanes)

    def __post_init__(self):
        T = len(self.frame)
        assert self.ball.shape == (T, 3), self.ball.shape
        assert self.home.ndim == 3 and self.home.shape[0] == T and self.home.shape[2] == 2
        assert self.away.ndim == 3 and self.away.shape[0] == T and self.away.shape[2] == 2
        assert self.y.shape == (T,)

    def __len__(self):
        return len(self.frame)

    def copy(self) -> "Sequence":
        return Sequence(self.match_id, self.period, self.frame.copy(), self.ball.copy(),
                        self.home.copy(), self.away.copy(), self.pitch.copy(), self.y.copy(),
                        None if self.tags is None else self.tags.copy(),
                        {k: v.copy() for k, v in self.extra.items()})

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(match_id=np.array(self.match_id), period=np.array(self.period),
                      frame=self.frame.astype(np.int64), ball=self.ball.astype(np.float32),
                      home=self.home.astype(np.float32), away=self.away.astype(np.float32),
                      pitch=self.pitch.astype(np.float32), y=self.y.astype(np.int8))
        if self.tags is not None:
            arrays["tags"] = self.tags
        for k, v in self.extra.items():
            arrays["extra__" + k] = v
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> "Sequence":
        with np.load(path, allow_pickle=False) as z:
            extra = {k[len("extra__"):]: z[k] for k in z.files if k.startswith("extra__")}
            return cls(str(z["match_id"]), int(z["period"]), z["frame"], z["ball"], z["home"],
                       z["away"], z["pitch"], z["y"], z["tags"] if "tags" in z.files else None,
                       extra)


# Smart Tagging intervals (lane target-io). Global video frames, end exclusive
# (start = round(start_s * 25), end = round(end_s * 25), as starter/load_smart_tagging.py).
# team: H (Barcelona), A (Athletic), N (N/A), ? (missing/unknown). type/half: raw tag text.
TAG_DTYPE = np.dtype([("start", "i8"), ("end", "i8"), ("code", "U64"), ("team", "U1"),
                      ("type", "U64"), ("half", "U16")])
