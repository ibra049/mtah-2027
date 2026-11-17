"""Synthetic EPTS / SportsCode fixtures following vendor/challenge/DATA.md + starter formats."""
from __future__ import annotations

import os

import numpy as np

HOME_ID, AWAY_ID = "T_BAR", "T_ATH"


def meta_xml(n_slots, sides, start=1, end=100, channel_ids=("x", "y", "speed"),
             ball_channels=("x", "y", "z"), field=(105, 68), home_name="FC Barcelona",
             away_name="Athletic Club", group_order=None):
    """sides: list of 0/1 per slot (file order of <Player>). group_order: slot order in the raw
    text (default = file order)."""
    group_order = list(range(n_slots)) if group_order is None else group_order
    players = "\n".join(
        f'<Player id="P{i+1}" teamId="{HOME_ID if s == 0 else AWAY_ID}"><Name>Track {i+1}</Name>'
        f"<ShirtNumber>{i+1}</ShirtNumber></Player>" for i, s in enumerate(sides))
    chans = "\n".join(f'<Channel id="{c}"><Name>{"position_" + c if c in "xy" else c}</Name>'
                      f"<Unit>normalized</Unit></Channel>" for c in channel_ids)
    pch = "\n".join(f'<PlayerChannel channelId="{c}" id="player{i+1}_{c}" playerId="P{i+1}"/>'
                    for i in range(n_slots) for c in channel_ids)
    groups = "\n".join(
        '<SplitRegister separator=",">' + "".join(
            f'<PlayerChannelRef playerChannelId="player{i+1}_{c}"/>' for c in channel_ids)
        + "</SplitRegister>" for i in group_order)
    ball = "".join(f'<BallChannelRef channelId="{c}"/>' for c in ball_channels)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<main><Metadata><GlobalConfig><FrameRate>25</FrameRate></GlobalConfig>
<Sessions><Session id="s"><FieldSize><Width>{field[0]}</Width><Height>{field[1]}</Height></FieldSize></Session></Sessions>
<Teams><Team id="{HOME_ID}"><Name>{home_name}</Name></Team><Team id="{AWAY_ID}"><Name>{away_name}</Name></Team></Teams>
<Players>{players}</Players>
<Devices><Device id="d1"><Sensors><Sensor id="position"><Channels>{chans}</Channels></Sensor></Sensors></Device></Devices>
<PlayerChannels>{pch}</PlayerChannels></Metadata>
<DataFormatSpecifications><DataFormatSpecification endFrame="{end}" separator=":" startFrame="{start}">
<StringRegister name="frameCount"/><SplitRegister separator=";">{groups}</SplitRegister>
<SplitRegister separator=",">{ball}</SplitRegister></DataFormatSpecification></DataFormatSpecifications></main>
"""


def fmt(v):
    return "NaN" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.5f}"


def raw_line(fid, players, ball, nf=3):
    """players: list of (x,y) / None / raw str; ball: (x,y) / None / raw str."""
    groups = []
    for p in players:
        if isinstance(p, str):
            groups.append(p)
        elif p is None:
            groups.append(",".join(["NaN"] * nf))
        else:
            groups.append(",".join([fmt(p[0]), fmt(p[1])] + ["1.23"] * (nf - 2)))
    if isinstance(ball, str):
        b = ball
    elif ball is None:
        b = "NaN,NaN,NaN"
    else:
        b = f"{fmt(ball[0])},{fmt(ball[1])},0.00000"
    return f"{fid}:{';'.join(groups)}:{b}"


def random_game(rng, n_frames, sides, p_ball_missing=0.4, p_player_missing=0.3, quant=5):
    """Random normalised positions, quantised like the EPTS text (5 decimals)."""
    S = len(sides)
    P = np.round(rng.uniform(0, 1, (n_frames, S, 2)), quant)
    P[rng.uniform(size=(n_frames, S)) < p_player_missing] = np.nan
    B = np.round(rng.uniform(0, 1, (n_frames, 2)), quant)
    B[rng.uniform(size=n_frames) < p_ball_missing] = np.nan
    return P, B


def write_game(folder, P, B, sides, stem="BAR-ATH_tactical", tags_xml=None, extra_lines=None):
    os.makedirs(folder, exist_ok=True)
    T, S = P.shape[:2]
    with open(os.path.join(folder, stem + "_atd.xml"), "w") as fh:
        fh.write(meta_xml(S, sides, 1, T))
    with open(os.path.join(folder, stem + "_atd.txt"), "w") as fh:
        for t in range(T):
            players = [None if np.isnan(P[t, s, 0]) else (P[t, s, 0], P[t, s, 1]) for s in range(S)]
            ball = None if np.isnan(B[t, 0]) else (B[t, 0], B[t, 1])
            fh.write(raw_line(t + 1, players, ball) + "\n")
        for line in extra_lines or []:
            fh.write(line + "\n")
    if tags_xml is not None:
        with open(os.path.join(folder, stem + "_smart_tagging.xml"), "w") as fh:
            fh.write(tags_xml)


def tags_xml(instances):
    """instances: list of (start_s, end_s, code, {group: text})."""
    out = ['<?xml version="1.0" encoding="utf-8"?>\n<file><ALL_INSTANCES>']
    for i, (a, b, code, labels) in enumerate(instances):
        labs = "".join(f"<label><group>{g}</group><text>{t}</text></label>"
                       for g, t in labels.items())
        out.append(f"<instance><ID>{i+1}</ID><start>{a}</start><end>{b}</end>"
                   f"<code>{code}</code>{labs}</instance>")
    out.append("</ALL_INSTANCES><ROWS></ROWS></file>")
    return "\n".join(out)
