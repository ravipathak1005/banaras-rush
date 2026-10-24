"""Shared helpers: flat-shaded vertex-colour meshes -> .glb with several named nodes."""
import json, struct
import numpy as np


def flat(P, C, I, pivot=(0, 0, 0)):
    tri = P[I]
    pos = (tri - np.array(pivot)).reshape(-1, 3)
    col = C[I].reshape(-1, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1
    nrm = np.repeat(n / ln, 3, axis=0)
    idx = np.arange(len(pos), dtype=np.uint16 if len(pos) < 65535 else np.uint32)
    return pos.astype(np.float32), nrm.astype(np.float32), col.astype(np.float32), idx


def welded(P, C, I, pivot=(0, 0, 0)):
    """Indexed mesh (shared vertices, no normals): use flatShading in the game. About 4x smaller than flat()."""
    P = np.asarray(P) - np.array(pivot)
    key = np.concatenate([np.round(P, 5), np.round(C, 5)], 1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    pos = P[first].astype(np.float32); col = C[first].astype(np.float32)
    idx = inv[I].reshape(-1).astype(np.uint16 if len(pos) < 65535 else np.uint32)
    return pos, None, col, idx


def write_glb(path, groups):
    """groups: list of (name, translation, (pos, nrm, col, idx)) -> one node per group under a root node."""
    blobs, views, accs, meshes, nodes = [], [], [], [], []
    off = 0
    for gi, (name, tr, (pos, nrm, col, idx)) in enumerate(groups):
        base = len(accs)
        items = [(pos, 34962, 'VEC3', 5126)] + ([(nrm, 34962, 'VEC3', 5126)] if nrm is not None else []) + [(col, 34962, 'VEC3', 5126), (idx, 34963, 'SCALAR', 5123 if idx.dtype == np.uint16 else 5125)]
        for arr, tgt, typ, ct in items:
            b = arr.tobytes(); raw = len(b); b += b'\0' * ((4 - len(b) % 4) % 4)
            views.append({'buffer': 0, 'byteOffset': off, 'byteLength': raw, 'target': tgt})
            a = {'bufferView': len(views) - 1, 'componentType': ct, 'count': len(arr), 'type': typ}
            if arr is pos: a['min'] = pos.min(0).tolist(); a['max'] = pos.max(0).tolist()
            accs.append(a); blobs.append(b); off += len(b)
        attrs = {'POSITION': base, 'NORMAL': base + 1, 'COLOR_0': base + 2} if nrm is not None else {'POSITION': base, 'COLOR_0': base + 1}
        meshes.append({'name': name, 'primitives': [{'attributes': attrs, 'indices': base + len(items) - 1, 'material': 0}]})
        nodes.append({'name': name, 'mesh': gi, 'translation': [float(x) for x in tr]})
    nodes.insert(0, {'name': 'root', 'children': list(range(1, len(nodes) + 1))})
    binb = b''.join(blobs)
    g = {'asset': {'version': '2.0', 'generator': 'glbparts'}, 'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': nodes, 'meshes': meshes,
         'materials': [{'name': 'vcol', 'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1], 'metallicFactor': 0.0, 'roughnessFactor': 0.9}}],
         'buffers': [{'byteLength': len(binb)}], 'bufferViews': views, 'accessors': accs}
    js = json.dumps(g, separators=(',', ':')).encode(); js += b' ' * ((4 - len(js) % 4) % 4)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binb)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js); f.write(struct.pack('<II', len(binb), 0x004E4942) + binb)
