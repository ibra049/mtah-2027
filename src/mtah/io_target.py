"""Target recording (Metrica FIFA/EPTS ATD txt+xml, SportsCode Smart Tagging) -> CONTRACT Sequences.

Written from vendor/challenge/DATA.md and starter/*.py only. Tested on synthetic fixtures and on
Metrica sample game 3 (same EPTS family). It never prints data.

Conventions (CONTRACT.md):
  * frame = ATD frame id - 1 (submission row / video frame).
  * metres, origin at the pitch centre: x = (x_n - 0.5) * L,  y = (0.5 - y_n) * W
    (x along the length, y across, right-handed; Metrica's normalised y grows downwards).
  * NaN for every missing observation (ball and players).
  * slots: every <Player> of the metadata in file order is one slot; home = first <Team>.
    home[:, k] holds the k-th home slot; extra['home_slot'][k] is its global slot index.
  * the raw text layout is read from <DataFormatSpecification> (per frame range), so
    group order, field count (x,y[,speed]) and ball channels come from the metadata.
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from mtah.canonical import TAG_DTYPE, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(REPO, "configs", "target.json")
HOME, AWAY = 0, 1


def load_config(path: str | None = None) -> dict:
    """configs/target.json + the published constants of third_party/mtah_challenge/games.py:
    n_frames, fps, pitch (105.3 x 68.0) and graded_blocks (games.half_video_blocks)."""
    from mtah.official import games
    with open(path or CONFIG_PATH) as fh:
        cfg = json.load(fh)
    g = games.GAMES[cfg["game"]]
    cfg.setdefault("n_frames", int(games.source_frames("tactical", cfg["game"])))
    cfg.setdefault("fps", int(g["fps"]))
    cfg.setdefault("pitch", [float(v) for v in g["pitch"]])
    cfg.setdefault("graded_blocks", [[int(a), int(b)] for _, a, b in
                                     games.half_video_blocks(cfg["game"])])
    return cfg


# ----------------------------------------------------------------------------------------------
# metadata
# ----------------------------------------------------------------------------------------------
@dataclass
class Layout:
    start: int                 # first frame id covered (inclusive)
    end: int                   # last frame id covered (inclusive)
    sep: str                   # top-level separator (':')
    n_parts: int               # number of top-level parts
    frame_part: int            # index of the frame-count part
    player_part: int           # index of the players part (-1 if none)
    ball_part: int             # index of the ball part (-1 if none)
    group_sep: str             # separator between player groups (';')
    field_sep: str             # separator inside a group (',')
    group_slot: np.ndarray     # int [G] global slot index of each group (-1 unknown player)
    group_nf: np.ndarray       # int [G] number of fields in each group
    group_x: np.ndarray        # int [G] field index of x in the group (-1 none)
    group_y: np.ndarray        # int [G]
    ball_sep: str
    ball_nf: int
    ball_x: int
    ball_y: int
    ball_z: int


@dataclass
class Meta:
    team_ids: list
    team_names: list
    home_team_id: str
    slot_player: list          # player id per global slot (file order)
    slot_side: np.ndarray      # int8 [S] 0 home 1 away
    field_size: tuple          # (length, width) from <FieldSize>, may be (nan, nan)
    layouts: list              # list[Layout], sorted by start
    provider_params: dict      # ProviderGlobalParameters name -> value

    @property
    def home_slots(self) -> np.ndarray:
        return np.flatnonzero(self.slot_side == HOME).astype(np.int16)

    @property
    def away_slots(self) -> np.ndarray:
        return np.flatnonzero(self.slot_side == AWAY).astype(np.int16)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _semantic(channel_id: str, channel_name: str | None) -> str:
    """'x' | 'y' | 'z' | 'other' for a channel, from its id and its Device channel name."""
    cid = (channel_id or "").strip().lower()
    name = (channel_name or "").strip().lower()
    for axis in ("x", "y", "z"):
        if cid == axis or name in (axis, "position_" + axis) or cid.endswith("_" + axis):
            return axis
    return "other"


def parse_meta(xml_path: str) -> Meta:
    root = ET.parse(xml_path).getroot()
    find_all = lambda tag: [e for e in root.iter() if _local(e.tag) == tag]  # noqa: E731
    text = lambda e, tag: next((c.text for c in e if _local(c.tag) == tag), None)  # noqa: E731

    teams = find_all("Team")
    if len(teams) != 2:
        raise ValueError(f"expected 2 teams in metadata, found {len(teams)}")
    team_ids = [t.get("id") for t in teams]
    team_names = [(text(t, "Name") or "").strip() for t in teams]
    home_id = team_ids[0]

    players = find_all("Player")
    if not players:
        raise ValueError("no Player entries in metadata")
    slot_player = [p.get("id") for p in players]
    if len(set(slot_player)) != len(slot_player):
        raise ValueError("duplicate player ids in metadata")
    unknown = [p.get("teamId") for p in players if p.get("teamId") not in team_ids]
    if unknown:
        raise ValueError(f"{len(unknown)} players with an unknown teamId")
    slot_side = np.array([HOME if p.get("teamId") == home_id else AWAY for p in players],
                         dtype=np.int8)
    slot_of_player = {pid: i for i, pid in enumerate(slot_player)}

    channel_name = {c.get("id"): text(c, "Name") for c in find_all("Channel")}
    player_channel = {pc.get("id"): (pc.get("playerId"), pc.get("channelId"))
                      for pc in find_all("PlayerChannel")}

    def num(e, tag):
        try:
            return float(text(e, tag))
        except (TypeError, ValueError):
            return float("nan")
    fs = find_all("FieldSize")
    field_size = (num(fs[0], "Width"), num(fs[0], "Height")) if fs else (np.nan, np.nan)

    provider = {}
    for gc in find_all("ProviderGlobalParameters"):
        for pp in gc:
            provider[(text(pp, "Name") or "").strip()] = (text(pp, "Value") or "").strip()

    layouts = [_parse_spec(spec, player_channel, channel_name, slot_of_player)
               for spec in find_all("DataFormatSpecification")]
    if not layouts:
        raise ValueError("no DataFormatSpecification in metadata")
    layouts.sort(key=lambda l: l.start)
    for a, b in zip(layouts, layouts[1:]):
        if b.start <= a.end:
            raise ValueError("overlapping DataFormatSpecification frame ranges")
    return Meta(team_ids, team_names, home_id, slot_player, slot_side, field_size, layouts,
                provider)


def _parse_spec(spec, player_channel, channel_name, slot_of_player) -> Layout:
    sep = spec.get("separator", ":")
    children = [c for c in spec if _local(c.tag) in ("StringRegister", "SplitRegister")]
    frame_part = player_part = ball_part = -1
    group_sep = field_sep = ball_sep = ","
    groups, ball = [], None
    for i, child in enumerate(children):
        kind = _local(child.tag)
        if kind == "StringRegister":
            if child.get("name", "").lower() in ("framecount", "frame", "frameid") or frame_part < 0:
                frame_part = i
            continue
        sub = [c for c in child if _local(c.tag) == "SplitRegister"]
        refs = [c for c in child if _local(c.tag) in ("PlayerChannelRef", "BallChannelRef")]
        if sub:  # players: one nested register per player
            player_part, group_sep = i, child.get("separator", ";")
            for g in sub:
                field_sep = g.get("separator", ",")
                fields, pid = [], None
                for ref in g:
                    if _local(ref.tag) != "PlayerChannelRef":
                        fields.append("other")
                        continue
                    p, ch = player_channel.get(ref.get("playerChannelId"), (None, None))
                    pid = pid or p
                    fields.append(_semantic(ch, channel_name.get(ch)))
                groups.append((slot_of_player.get(pid, -1), fields))
        elif refs and all(_local(r.tag) == "BallChannelRef" for r in refs):
            ball_part, ball_sep = i, child.get("separator", ",")
            ball = [_semantic(r.get("channelId"), channel_name.get(r.get("channelId")))
                    for r in refs]
    if frame_part < 0:
        raise ValueError("DataFormatSpecification without a frame register")
    idx = lambda fields, axis: fields.index(axis) if axis in fields else -1  # noqa: E731
    ball = ball or []
    return Layout(
        start=int(spec.get("startFrame", 0)), end=int(spec.get("endFrame", 2**62)), sep=sep,
        n_parts=len(children), frame_part=frame_part, player_part=player_part,
        ball_part=ball_part, group_sep=group_sep, field_sep=field_sep,
        group_slot=np.array([g[0] for g in groups], dtype=np.int64),
        group_nf=np.array([len(g[1]) for g in groups], dtype=np.int64),
        group_x=np.array([idx(g[1], "x") for g in groups], dtype=np.int64),
        group_y=np.array([idx(g[1], "y") for g in groups], dtype=np.int64),
        ball_sep=ball_sep, ball_nf=len(ball), ball_x=idx(ball, "x"), ball_y=idx(ball, "y"),
        ball_z=idx(ball, "z"))


# ----------------------------------------------------------------------------------------------
# raw tracking
# ----------------------------------------------------------------------------------------------
def _float(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        return float("nan")


def read_raw(txt_path: str, meta: Meta, n_frames: int | None = None, first_frame_id: int = 1,
             dtype=np.float32):
    """Stream the raw file into normalised arrays indexed by frame_id - first_frame_id.

    Returns (players [T,S,2], ball [T,3] (z NaN), seen bool[T], stats dict). Values stay 0-1.
    Malformed lines are skipped (left NaN, not marked seen); odd tokens become NaN. Both are
    counted in stats. Memory: T*S*2*itemsize (float32: ~73 MB for the target).
    """
    if n_frames is None:
        n_frames = max(l.end for l in meta.layouts) - first_frame_id + 1
    S = len(meta.slot_player)
    players = np.full((n_frames, S, 2), np.nan, dtype=dtype)
    ball = np.full((n_frames, 3), np.nan, dtype=dtype)
    seen = np.zeros(n_frames, dtype=bool)
    stats = dict(lines=0, malformed=0, out_of_range=0, duplicate=0, no_layout=0,
                 odd_groups=0, odd_tokens=0, odd_ball=0)

    layouts = meta.layouts
    li = 0
    cache = {}

    def prep(lay):
        uniform = (len(lay.group_nf) > 0 and np.all(lay.group_nf == lay.group_nf[0])
                   and np.all(lay.group_x == lay.group_x[0]) and np.all(lay.group_y == lay.group_y[0]))
        ok = lay.group_slot >= 0
        return uniform, ok, lay.group_slot[ok]

    with open(txt_path, "r", newline=None) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            head = line.split(layouts[0].sep, 1)[0]
            try:
                fid = int(head)
            except ValueError:
                stats["malformed"] += 1
                continue
            if not (layouts[li].start <= fid <= layouts[li].end):
                li = next((i for i, l in enumerate(layouts) if l.start <= fid <= l.end), -1)
                if li < 0:
                    stats["no_layout"] += 1
                    li = 0
                    continue
            lay = layouts[li]
            parts = line.split(lay.sep)
            if len(parts) != lay.n_parts:
                stats["malformed"] += 1
                continue
            t = fid - first_frame_id
            if not 0 <= t < n_frames:
                stats["out_of_range"] += 1
                continue
            if seen[t]:
                stats["duplicate"] += 1
            seen[t] = True

            if lay.player_part >= 0:
                if li not in cache:
                    cache[li] = prep(lay)
                uniform, ok, slots = cache[li]
                G = len(lay.group_slot)
                xy = None
                if uniform:
                    flat = parts[lay.player_part].replace(lay.group_sep, lay.field_sep).split(lay.field_sep)
                    if len(flat) == G * lay.group_nf[0]:
                        try:
                            arr = np.array(flat, dtype=np.float64).reshape(G, lay.group_nf[0])
                            xy = np.stack([arr[:, lay.group_x[0]], arr[:, lay.group_y[0]]], axis=1)
                        except ValueError:
                            xy = None
                if xy is None:
                    xy = _slow_groups(parts[lay.player_part], lay, stats)
                players[t, slots] = xy[ok]

            if lay.ball_part >= 0 and lay.ball_x >= 0 and lay.ball_y >= 0:
                toks = parts[lay.ball_part].split(lay.ball_sep)
                if len(toks) != lay.ball_nf:
                    stats["odd_ball"] += 1
                if len(toks) > max(lay.ball_x, lay.ball_y):
                    ball[t, 0] = _float(toks[lay.ball_x])
                    ball[t, 1] = _float(toks[lay.ball_y])
    return players, ball, seen, stats


def _slow_groups(text: str, lay: Layout, stats: dict) -> np.ndarray:
    G = len(lay.group_slot)
    out = np.full((G, 2), np.nan)
    groups = text.split(lay.group_sep)
    if len(groups) != G:
        stats["odd_groups"] += 1
    for g, grp in enumerate(groups[:G]):
        toks = grp.split(lay.field_sep)
        if len(toks) != lay.group_nf[g]:
            stats["odd_tokens"] += 1
            if not grp.strip():
                continue
        xi, yi = lay.group_x[g], lay.group_y[g]
        if 0 <= xi < len(toks) and 0 <= yi < len(toks):
            out[g] = (_float(toks[xi]), _float(toks[yi]))   # same as the fast path: NaN stays NaN
            for tok in (toks[xi], toks[yi]):
                if tok.strip().lower() != "nan" and not np.isfinite(_float(tok)):
                    stats["odd_tokens"] += 1
    return out


def to_metres(arr: np.ndarray, length: float, width: float) -> np.ndarray:
    """In place: normalised (..., 2+) -> metres about the centre (x right, y up)."""
    arr[..., 0] -= 0.5
    arr[..., 0] *= length
    arr[..., 1] *= -1.0
    arr[..., 1] += 0.5
    arr[..., 1] *= width
    return arr


def ranges_from_splits(n_frames, split_frames):
    """[(period, first, last)] covering 0..n_frames-1, cut at each split frame."""
    cuts = [0] + [f for f in split_frames if 0 < f < n_frames] + [n_frames]
    return [(i + 1, a, b - 1) for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]))]


def sequences_from_arrays(match_id, players, ball, meta, pitch, ranges, tags=None):
    """Global arrays (index == video frame) -> one Sequence per (period, first, last) range,
    inclusive. Provider slot indices go to extra['home_slot'] / extra['away_slot']."""
    hs, aw = meta.home_slots, meta.away_slots
    seqs = []
    for period, a, b in ranges:
        frames = np.arange(a, b + 1, dtype=np.int64)
        seq_tags = None
        if tags is not None:
            m = (tags["end"] > a) & (tags["start"] <= b)
            seq_tags = tags[m].copy()
        seqs.append(Sequence(
            match_id=match_id, period=period, frame=frames, ball=ball[a:b + 1].copy(),
            home=players[a:b + 1][:, hs].copy(), away=players[a:b + 1][:, aw].copy(),
            pitch=np.asarray(pitch, dtype=np.float32), y=np.full(b + 1 - a, -1, dtype=np.int8),
            tags=seq_tags, extra={"home_slot": hs.copy(), "away_slot": aw.copy()}))
    return seqs


# ----------------------------------------------------------------------------------------------
# Smart Tagging
# ----------------------------------------------------------------------------------------------
def map_team(value, team_map: dict) -> str:
    """Barcelona->H, Atletic Club->A, N/A->N (case-insensitive). Unknown/missing -> '?'."""
    v = (value or "").strip().lower()
    return team_map.get(v, "?")


def load_tags(xml_path: str, fps: int = 25, team_map: dict | None = None) -> np.ndarray:
    """SportsCode instances -> TAG_DTYPE array in file order. Frames: int(round(s*fps)),
    end exclusive, exactly as starter/load_smart_tagging.py."""
    team_map = {k.lower(): v for k, v in (team_map or load_config()["tag_team_map"]).items()}
    rows = []
    for inst in ET.parse(xml_path).iter("instance"):
        start_s = float(inst.findtext("start"))
        end_s = float(inst.findtext("end"))
        labels = {}
        for lab in inst.findall("label"):
            group = (lab.findtext("group") or "").strip()
            if group:
                labels[group.lower()] = (lab.findtext("text") or "").strip()
        rows.append((int(round(start_s * fps)), int(round(end_s * fps)),
                     (inst.findtext("code") or "").strip(), map_team(labels.get("team"), team_map),
                     labels.get("type", ""), labels.get("half", "")))
    return np.array(rows, dtype=TAG_DTYPE)


# ----------------------------------------------------------------------------------------------
# public entry points
# ----------------------------------------------------------------------------------------------
def target_paths(data_dir: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    folder = os.path.join(data_dir, "tactical")
    if not os.path.isdir(folder):
        folder = data_dir
    stem = os.path.join(folder, cfg["stem"])
    return {"atd_meta": stem + "_atd.xml", "atd_raw": stem + "_atd.txt",
            "smart_tagging": stem + "_smart_tagging.xml"}


def load_epts(meta_path, raw_path, match_id, pitch=None, split_frames=None, n_frames=None,
              dtype=np.float32, tags=None, return_meta=False, ranges=None):
    """Generic EPTS pair -> list[Sequence]. frame = frame_id - 1. `ranges` [(period, first,
    last)] selects frames; default: split at `split_frames` (or the provider's
    second_half_start)."""
    meta = parse_meta(meta_path)
    players, ball, seen, stats = read_raw(raw_path, meta, n_frames=n_frames, first_frame_id=1,
                                          dtype=dtype)
    L, W = pitch if pitch is not None else meta.field_size
    to_metres(players, L, W)
    to_metres(ball, L, W)
    if ranges is None:
        if split_frames is None:
            s = meta.provider_params.get("second_half_start", "")
            split_frames = [int(s) - 1] if s.isdigit() else []
        ranges = ranges_from_splits(players.shape[0], split_frames)
    seqs = sequences_from_arrays(match_id, players, ball, meta, (L, W), ranges, tags)
    del players, ball
    if return_meta:
        return seqs, meta, seen, stats
    return seqs


def graded_ranges(cfg: dict) -> list:
    """[(1, 11185, 90651), (2, 110808, 190197)] -- games.half_video_blocks, from the config."""
    return [(i + 1, int(a), int(b)) for i, (a, b) in enumerate(cfg["graded_blocks"])]


def load(data_dir: str, cfg: dict | None = None, dtype=np.float32, with_tags: bool = True,
         return_meta: bool = False, coverage: str = "graded"):
    """CONTRACT: tactical ATD (+ Smart Tagging) -> Sequences.

    coverage="graded" (default): exactly two Sequences, the graded blocks, periods 1 and 2.
    coverage="full": one Sequence, period 0, all n_frames video frames -- only for the official
    baseline reproduction (its smoothing window reaches past the block edges) and parse checks.
    """
    cfg = cfg or load_config()
    paths = target_paths(data_dir, cfg)
    tags = None
    if with_tags and os.path.isfile(paths["smart_tagging"]):
        tags = load_tags(paths["smart_tagging"], cfg["fps"], cfg["tag_team_map"])
    if coverage == "graded":
        ranges = graded_ranges(cfg)
    elif coverage == "full":
        ranges = [(0, 0, cfg["n_frames"] - 1)]
    else:
        raise ValueError(f"coverage must be 'graded' or 'full', not {coverage!r}")
    seqs, meta, seen, stats = load_epts(
        paths["atd_meta"], paths["atd_raw"], cfg["game"], pitch=tuple(cfg["pitch"]),
        n_frames=cfg["n_frames"], dtype=dtype, tags=tags, return_meta=True, ranges=ranges)
    if return_meta:
        return seqs, dict(meta=meta, seen=seen, stats=stats, tags=tags)
    return seqs


def kickoff_frames(tags, cfg: dict | None = None) -> dict:
    """{period: start frame of the 'Kick Off (Start)' Smart Tag window, or None}.

    Matched on the Type tag, case-insensitive. Period from the Half tag ('1st Half'/'2nd Half');
    if that is missing, from the graded block the window overlaps. If a half has several, the
    earliest start wins. No kickoff frame is hard-coded here: this reads the tags only.
    """
    cfg = cfg or load_config()
    out = {p: None for p, _, _ in graded_ranges(cfg)}
    if tags is None:
        return out
    half_names = {"1st half": 1, "2nd half": 2}
    for t in tags:
        if str(t["type"]).strip().lower() != "kick off (start)":
            continue
        period = half_names.get(str(t["half"]).strip().lower())
        if period is None:
            period = next((p for p, a, b in graded_ranges(cfg)
                           if t["end"] > a and t["start"] <= b), None)
        if period in out and (out[period] is None or int(t["start"]) < out[period]):
            out[period] = int(t["start"])
    return out


def kickoff_windows(tags, cfg: dict | None = None) -> dict:
    """{period: (start, end) of the window kickoff_frames picked, or None} (for checks)."""
    starts = kickoff_frames(tags, cfg)
    out = {}
    for p, st in starts.items():
        hit = None
        if st is not None:
            for t in tags:
                if (str(t["type"]).strip().lower() == "kick off (start)" and int(t["start"]) == st):
                    hit = (int(t["start"]), int(t["end"]))
                    break
        out[p] = hit
    return out
