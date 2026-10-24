"""Builds cow variants from the supplied low-poly cow (Z-up, head +X) -> Y-up, head +Z, with a zebu hump.
usage: python3 tools/make_cows.py path/to/low_poly_cow.glb [preview.png]  -> assets/models/cow_0..3.glb"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ped_lib import read_parts
from glbkit import hex2lin, Mesh
from glbparts import flat, write_glb

SRC = sys.argv[1]; PREV = sys.argv[2] if len(sys.argv) > 2 else None
OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'models'); os.makedirs(OUT, exist_ok=True)
parts = read_parts(SRC)
BODYCOL = ['Body', 'Chest', 'Neck', 'Head', 'BackLeg_L', 'BackLeg_R', 'FrontLeg_L', 'FrontLeg_R', 'Tail1', 'Tail2']
DARK = ['TailTuft', 'Ear_-1', 'Ear_1', 'EyePatch_-1', 'EyePatch_1']
S = 0.78
VARIANTS = [  # body colour, spot colour (None = hide spots), tuft/ears colour
    (None, 'orig', None),
    (0xc99a62, None, 0x6a4a2c),
    (0x8a5a3a, 0xe6d9bd, 0x3a2a1c),
    (0xddd8cc, None, 0xb9b2a2)]


def xf(P):   # model (x fwd, y side, z up) -> game (X side, Y up, Z fwd); proper rotation, floor at 0, scaled
    Q = np.stack([P[:, 1], P[:, 2] - 0.19, P[:, 0]], 1)
    return Q * S


def build(v):
    body, spot, dark = v
    P_all, C_all, I_all, n = [], [], [], 0
    def add(P, C, I):
        nonlocal n
        P_all.append(P); C_all.append(C); I_all.append(I + n); n += len(P)
    for name, (P, C, I) in parts.items():
        C = C.copy()
        if name.startswith('Spot'):
            if spot is None: continue
            if spot != 'orig': C[:] = hex2lin(spot)
        elif name in BODYCOL and body is not None: C[:] = hex2lin(body)
        elif name in DARK and dark is not None: C[:] = hex2lin(dark)
        elif name.startswith('Horn') and body is not None: pass
        add(xf(P), C, I)
    # zebu hump on the shoulders
    m = Mesh(); m.rbox((0.38, 0, 1.98), (0.30, 0.20, 0.22), 0x888888, e=1.0, nu=8, nv=12)
    Ph, Nh, Ch, Ih = m.arrays()
    colb = hex2lin(body) if body is not None else np.array([0.92, 0.9, 0.83])
    add(xf(Ph.astype(float)), np.tile(colb, (len(Ph), 1)), Ih.reshape(-1, 3).astype(np.int64))
    P = np.concatenate(P_all); C = np.concatenate(C_all); I = np.concatenate(I_all)
    return [('cow', (0, 0, 0), flat(P, C, I))]


for k, v in enumerate(VARIANTS):
    g = build(v); p = os.path.join(OUT, 'cow_%d.glb' % k); write_glb(p, g)
    print(p, os.path.getsize(p) // 1024, 'KB', len(g[0][2][3]) // 3, 'tris')
    if PREV and k == 2:
        m = Mesh(); pos, nrm, col, idx = g[0][2]
        m.P.append(pos.astype(float)); m.N.append(nrm.astype(float)); m.C.append(col.astype(float)); m.I.append(idx.astype(np.int64)); m.n = len(pos)
        m.preview(PREV, views=((15, 150), (10, 60), (0, 90)), size=4.5)
