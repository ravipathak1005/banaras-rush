"""Reads the static low-poly human .glb (trimesh export): returns {part_name: (P, C, I)}."""
import json, struct
import numpy as np

def read_parts(path):
    d = open(path, 'rb').read()
    jl = struct.unpack('<I', d[12:16])[0]
    j = json.loads(d[20:20 + jl]); b = d[20 + jl + 8:]
    dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
    nn = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    def acc(i):
        a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]
        off = bv.get('byteOffset', 0) + a.get('byteOffset', 0); n = nn[a['type']]; t = dt[a['componentType']]
        arr = np.frombuffer(b[off:off + a['count'] * n * np.dtype(t).itemsize], dtype=t).reshape(a['count'], n)
        if a.get('normalized') or (a['componentType'] == 5121): arr = arr.astype(np.float64) / 255.0
        return arr
    parts = {}
    for nd in j['nodes']:
        if 'mesh' not in nd: continue
        m = j['meshes'][nd['mesh']]; p = m['primitives'][0]
        P = acc(p['attributes']['POSITION']).astype(np.float64)
        C = acc(p['attributes']['COLOR_0']).astype(np.float64)[:, :3]
        I = acc(p['indices']).reshape(-1, 3).astype(np.int64)
        parts[nd['name']] = (P, C, I)
    return parts
