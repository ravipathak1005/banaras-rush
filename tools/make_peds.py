"""Builds animated-by-code pedestrians from the static human model.
usage: python3 tools/make_peds.py path/to/low_poly_human_character.glb [preview.png]
Writes assets/models/people/ped_0..5.glb  (nodes: body, legL, legR, armL, armR; face +Z, 1.72 m tall)"""
import os, sys, json, struct
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ped_lib import read_parts
from glbkit import hex2lin, Mesh

SRC = sys.argv[1]
PREV = sys.argv[2] if len(sys.argv) > 2 else None
OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'models', 'people')
os.makedirs(OUT, exist_ok=True)

parts = read_parts(SRC)
SHIRT = ['shirt_torso', 'left_shoulder', 'right_shoulder', 'left_sleeve', 'right_sleeve', 'left_cuff', 'right_cuff']
SKIN = ['neck', 'head', 'left_hand', 'right_hand']
SKIN2 = ['left_ear', 'right_ear', 'nose']
PANTS = ['left_leg', 'right_leg']
GROUPS = {
    'legL': ['left_leg', 'left_shoe'], 'legR': ['right_leg', 'right_shoe'],
    'armL': ['left_shoulder', 'left_sleeve', 'left_hand', 'left_cuff'],
    'armR': ['right_shoulder', 'right_sleeve', 'right_hand', 'right_cuff'],
}
PIVOT = {'legL': (-0.145, 0.90, 0.0), 'legR': (0.145, 0.90, 0.0), 'armL': (-0.40, 1.58, 0.0), 'armR': (0.40, 1.58, 0.0)}
used = {n for v in GROUPS.values() for n in v}
DROP = lambda n: n.startswith(('shirt_motif', 'button', 'left_eye', 'right_eye', 'left_brow', 'right_brow', 'mouth', 'shirt_placket', 'hair_fringe'))   # too small to see at game scale
parts = {k: v for k, v in parts.items() if not DROP(k)}
VARIANTS = [  # shirt, trousers, skin multiplier
    (None, None, 1.0), (0x3a6ab0, 0xb0a070, 0.9), (0xe08a2a, 0x1e1e1e, 1.1),
    (0x3a8a4a, 0x233044, 0.95), (0xb03030, 0xe8e0c8, 1.05), (0xf0f0f0, 0x1e1e1e, 0.85)]
H = 1.72 / (2.47 - 0.04)


def part_arrays(name, v):
    P, C, I = parts[name]
    C = C.copy()
    shirt, pants, sk = v
    if shirt is not None and name in SHIRT: C[:] = hex2lin(shirt)
    if shirt is not None and name == 'shirt_placket': C[:] = np.clip(hex2lin(shirt) * 1.25, 0, 1)
    if pants is not None and name in PANTS: C[:] = hex2lin(pants)
    if name in SKIN or name in SKIN2: C = np.clip(C * sk, 0, 1)
    # model faces -Z, we want +Z: rotate 180 about Y, drop to the floor and scale to 1.72 m
    P = P.copy(); P[:, 0] *= -1; P[:, 2] *= -1
    P[:, 1] -= 0.04
    P *= H
    # triangle winding is preserved by a proper rotation (x,z negated) so no flip is needed
    return P, C, I


def flat(P, C, I, pivot=(0, 0, 0)):
    tri = P[I]
    pos = (tri - np.array(pivot)).reshape(-1, 3)
    col = C[I].reshape(-1, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1
    nrm = np.repeat(n / ln, 3, axis=0)
    idx = np.arange(len(pos), dtype=np.uint16 if len(pos) < 65535 else np.uint32)
    return pos.astype(np.float32), nrm.astype(np.float32), col.astype(np.float32), idx


def write_glb(path, groups):
    """groups: list of (name, translation, (pos, nrm, col, idx))"""
    blobs, views, accs, meshes, nodes = [], [], [], [], []
    off = 0
    for gi, (name, tr, (pos, nrm, col, idx)) in enumerate(groups):
        base = len(accs)
        for arr, tgt, typ, ct in [(pos, 34962, 'VEC3', 5126), (nrm, 34962, 'VEC3', 5126), (col, 34962, 'VEC3', 5126),
                                  (idx, 34963, 'SCALAR', 5123 if idx.dtype == np.uint16 else 5125)]:
            b = arr.tobytes(); b += b'\0' * ((4 - len(b) % 4) % 4)
            views.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(arr.tobytes()), 'target': tgt})
            a = {'bufferView': len(views) - 1, 'componentType': ct, 'count': len(arr) if typ == 'SCALAR' else len(arr), 'type': typ}
            if arr is pos: a['min'] = pos.min(0).tolist(); a['max'] = pos.max(0).tolist()
            accs.append(a); blobs.append(b); off += len(b)
        meshes.append({'name': name, 'primitives': [{'attributes': {'POSITION': base, 'NORMAL': base + 1, 'COLOR_0': base + 2}, 'indices': base + 3, 'material': 0}]})
        nodes.append({'name': name, 'mesh': gi, 'translation': [float(x) for x in tr]})
    nodes.insert(0, {'name': 'ped', 'children': list(range(1, len(nodes) + 1))})
    binb = b''.join(blobs)
    g = {'asset': {'version': '2.0', 'generator': 'make_peds.py'}, 'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': nodes, 'meshes': meshes,
         'materials': [{'name': 'vcol', 'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1], 'metallicFactor': 0.0, 'roughnessFactor': 0.9}}],
         'buffers': [{'byteLength': len(binb)}], 'bufferViews': views, 'accessors': accs}
    js = json.dumps(g, separators=(',', ':')).encode(); js += b' ' * ((4 - len(js) % 4) % 4)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binb)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js); f.write(struct.pack('<II', len(binb), 0x004E4942) + binb)


for vi, v in enumerate(VARIANTS):
    groups = []
    body = [n for n in parts if n not in used]
    arrs = [part_arrays(n, v) for n in body]
    P = np.concatenate([a[0] for a in arrs]); C = np.concatenate([a[1] for a in arrs])
    off = np.cumsum([0] + [len(a[0]) for a in arrs[:-1]])
    I = np.concatenate([a[2] + o for a, o in zip(arrs, off)])
    groups.append(('body', (0, 0, 0), flat(P, C, I)))
    for gname, names in GROUPS.items():
        arrs = [part_arrays(n, v) for n in names]
        P = np.concatenate([a[0] for a in arrs]); C = np.concatenate([a[1] for a in arrs])
        off = np.cumsum([0] + [len(a[0]) for a in arrs[:-1]])
        I = np.concatenate([a[2] + o for a, o in zip(arrs, off)])
        pv = (PIVOT[gname][0] * -1 * H, (PIVOT[gname][1] - 0.04) * H, 0.0)   # same x/z flip + floor + scale as the mesh
        groups.append((gname, pv, flat(P, C, I, pv)))
    p = os.path.join(OUT, 'ped_%d.glb' % vi)
    write_glb(p, groups)
    print(p, os.path.getsize(p) // 1024, 'KB', sum(len(g[2][3]) // 3 for g in groups), 'tris')
    if PREV and vi == 1:
        m = Mesh()
        for name, tr, (pos, nrm, col, idx) in groups:
            m.P.append(pos.astype(float) + np.array(tr)); m.N.append(nrm.astype(float)); m.C.append(col.astype(float)); m.I.append(idx.astype(np.int64).reshape(-1, 3) + m.n if False else (idx.astype(np.int64) + m.n)); m.n += len(pos)
        m.preview(PREV, views=((15, 150), (10, 60), (0, 90)), size=4)
