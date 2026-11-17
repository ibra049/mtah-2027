"""CONTRACT Sequence: one match-half of tracking, in metres, origin at the pitch centre.

Minimal implementation of the CONTRACT.md table (owned by the captain). Extra optional fields
used by lane target-io:
  home_slot / away_slot : int16 [K]  provider slot index (global, file order) of each column, so
                          code that needs the provider's slot order (e.g. the official baseline's
                          tie-break) can recover it.
  tags                  : structured array, see TAG_DTYPE (frames are global video frames,
                          end exclusive).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

TAG_DTYPE = np.dtype([("start", "i8"), ("end", "i8"), ("code", "U64"), ("team", "U1"),
                      ("type", "U64"), ("half", "U16")])


@dataclass
class Sequence:
    match_id: str
    period: int
    frame: np.ndarray            # int64 [T]
    ball: np.ndarray             # float [T,3] m, NaN where unobserved
    home: np.ndarray             # float [T,K,2] m, NaN where slot empty
    away: np.ndarray             # float [T,K,2]
    pitch: np.ndarray            # float32 [2] length, width
    y: np.ndarray                # int8 [T], -1 unknown
    tags: Optional[np.ndarray] = None
    home_slot: Optional[np.ndarray] = None
    away_slot: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.frame.shape[0])

    def save(self, path) -> None:
        arrays = dict(match_id=np.array(self.match_id), period=np.array(self.period),
                      frame=self.frame, ball=self.ball, home=self.home, away=self.away,
                      pitch=self.pitch, y=self.y)
        for name in ("tags", "home_slot", "away_slot"):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "Sequence":
        with np.load(path, allow_pickle=False) as z:
            get = lambda k: z[k] if k in z.files else None  # noqa: E731
            return cls(match_id=str(z["match_id"]), period=int(z["period"]), frame=z["frame"],
                       ball=z["ball"], home=z["home"], away=z["away"], pitch=z["pitch"],
                       y=z["y"], tags=get("tags"), home_slot=get("home_slot"),
                       away_slot=get("away_slot"))
