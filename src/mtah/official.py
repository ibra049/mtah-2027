"""The committed official challenge code (third_party/mtah_challenge, MIT, unmodified).

    from mtah.official import games, pm      # games.py constants, possession_metrics
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL_DIR = os.path.join(REPO, "third_party", "mtah_challenge")
if OFFICIAL_DIR not in sys.path:
    sys.path.insert(0, OFFICIAL_DIR)

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402

__all__ = ["games", "pm", "OFFICIAL_DIR"]
