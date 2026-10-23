"""Builds the auto-rickshaw models: python3 tools/make_auto.py  (writes assets/models/auto_*.glb)"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from glbkit import Mesh, rotx
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'models')
PREV = sys.argv[1] if len(sys.argv) > 1 else None

DARK, RUBBER, CHROME, GLASS = 0x232323, 0x151515, 0xb4b4b4, 0x35505e
SEAT, SHIRT, SKIN, LAMP, RED, AMBER = 0x7a2a1a, 0x8a7a55, 0xb98760, 0xfff3c4, 0xd02424, 0xf2a020


def build(lower, roof):
    m = Mesh()
    # chassis and lower body
    m.box((0, 0.36, -0.10), (0.56, 0.05, 1.22), DARK)
    m.rbox((0, 0.68, -0.62), (0.66, 0.30, 0.78), lower, e=0.35)
    m.box((0, 0.52, -0.62), (0.672, 0.022, 0.79), roof)                 # trim stripe
    # front cowl and nose
    m.rbox((0, 0.78, 0.72), (0.46, 0.40, 0.50), lower, e=0.45)
    m.rbox((0, 0.72, 1.12), (0.30, 0.32, 0.36), lower, e=0.65)
    m.rbox((0, 0.58, 1.15), (0.19, 0.04, 0.31), lower, e=0.5)           # front mudguard
    # windshield (slanted)
    m.box((0, 1.31, 0.40), (0.50, 0.28, 0.012), GLASS, R=rotx(np.radians(-20)))
    m.seg((-0.52, 1.03, 0.45), (-0.52, 1.58, 0.33), 0.016, DARK)
    m.seg((0.52, 1.03, 0.45), (0.52, 1.58, 0.33), 0.016, DARK)
    # roof canopy
    m.rbox((0, 1.70, -0.30), (0.74, 0.045, 1.08), roof, e=0.3)
    m.rbox((0, 1.665, 0.74), (0.62, 0.03, 0.14), roof, e=0.4)           # front visor
    for x, z in [(-0.66, -1.30), (0.66, -1.30)]:
        m.seg((x, 0.98, z), (x, 1.66, z), 0.02, DARK)
    # rear canvas panel, small window, quarter panels
    m.box((0, 1.34, -1.39), (0.64, 0.33, 0.02), roof)
    m.box((0, 1.36, -1.405), (0.30, 0.15, 0.008), GLASS)
    for sx in (-1, 1):
        m.box((sx * 0.685, 1.34, -1.00), (0.015, 0.32, 0.36), roof)
    # seats
    m.rbox((0, 0.93, -0.84), (0.55, 0.08, 0.34), SEAT, e=0.4)
    m.rbox((0, 1.18, -1.17), (0.55, 0.28, 0.07), SEAT, e=0.4)
    m.rbox((0, 0.87, 0.10), (0.28, 0.06, 0.20), DARK, e=0.4)
    # driver
    m.rbox((0, 1.10, 0.04), (0.19, 0.26, 0.13), SHIRT, e=0.5)
    m.rbox((0, 1.49, 0.06), (0.115, 0.125, 0.115), SKIN, e=0.9)
    m.rbox((0, 1.57, 0.05), (0.12, 0.05, 0.12), DARK, e=0.8)
    for sx in (-1, 1):
        m.seg((sx * 0.20, 1.18, 0.10), (sx * 0.30, 1.04, 0.50), 0.035, SHIRT)   # arms
    # handlebar, mirrors
    m.cyl((0, 1.02, 0.52), 0.78, 0.016, CHROME, axis="x", segs=8)
    for sx in (-1, 1):
        m.seg((sx * 0.38, 1.03, 0.52), (sx * 0.55, 1.18, 0.50), 0.01, CHROME)
        m.box((sx * 0.56, 1.21, 0.50), (0.035, 0.05, 0.01), CHROME)
    # lights
    m.rbox((0, 0.98, 1.46), (0.12, 0.12, 0.07), LAMP, e=1.0)
    for sx in (-1, 1):
        m.box((sx * 0.22, 0.88, 1.43), (0.035, 0.035, 0.03), AMBER)
        m.box((sx * 0.50, 0.74, -1.41), (0.07, 0.05, 0.015), RED)
    m.cyl((0, 0.46, 1.47), 0.5, 0.02, CHROME, axis="x", segs=8)             # front bumper bar
    # wheels (tyre + hub) and rear mudguards
    for x, z, w in [(0, 1.15, 0.13), (-0.64, -0.55, 0.15), (0.64, -0.55, 0.15)]:
        m.cyl((x, 0.26, z), w, 0.26, RUBBER, axis="x", segs=16)
        m.cyl((x, 0.26, z), w + 0.02, 0.12, CHROME, axis="x", segs=10)
    for sx in (-1, 1):
        m.rbox((sx * 0.64, 0.52, -0.55), (0.10, 0.03, 0.30), lower, e=0.5)
    return m


VARIANTS = {'auto_green': (0x2f7a3a, 0xf2c230), 'auto_blue': (0x3a6ab0, 0xf2c230), 'auto_black': (0x262626, 0xf2c230)}
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, (lo, ro) in VARIANTS.items():
        m = build(lo, ro)
        p = m.save(os.path.join(OUT, name + '.glb'), name)
        print(name, m.tri_count(), 'tris', os.path.getsize(p) // 1024, 'KB')
        if PREV and name == 'auto_green':
            m.preview(PREV)
