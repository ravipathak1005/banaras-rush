"""glbkit: tiny procedural 3D toolkit (numpy only) that writes .glb files.

Conventions: metres, Y up, model front = +Z. Colours are sRGB hex and are stored as
linear vertex colours (what Three.js expects). One mesh = one primitive = one draw call.
"""
import json
import struct
import numpy as np


def hex2lin(h):
    c = np.array([(h >> 16) & 255, (h >> 8) & 255, h & 255], dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def smooth_normals(P, I):
    """Area-weighted vertex normals, welded by position so seams and poles stay smooth."""
    tri = P[I]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    key = np.round(P, 4)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    acc = np.zeros((inv.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, inv[I[:, k]], fn)
    n = acc[inv]
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return n / ln


class Mesh:
    def __init__(self):
        self.P, self.N, self.C, self.I = [], [], [], []
        self.n = 0

    def _add(self, P, N, I, color, R=None, T=(0, 0, 0)):
        P = np.asarray(P, dtype=np.float64)
        N = np.asarray(N, dtype=np.float64)
        if R is not None:
            P = P @ np.asarray(R).T
            N = N @ np.asarray(R).T
        P = P + np.asarray(T, dtype=np.float64)
        self.P.append(P)
        self.N.append(N)
        self.C.append(np.tile(hex2lin(color), (len(P), 1)))
        self.I.append(np.asarray(I, dtype=np.int64) + self.n)
        self.n += len(P)

    # ---- primitives -------------------------------------------------------------
    def box(self, c, h, color, R=None):
        """Flat-shaded box. c = centre, h = half sizes."""
        hx, hy, hz = h
        faces = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
                 ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
                 ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (-1, 0, 0), (0, 1, 0))]
        P, N, I = [], [], []
        for nrm, u, v in faces:
            nrm, u, v = map(np.array, (nrm, u, v))
            ce = nrm * np.array([hx, hy, hz])
            uu, vv = u * np.array([hx, hy, hz]), v * np.array([hx, hy, hz])
            b = len(P)
            P += [ce - uu - vv, ce + uu - vv, ce + uu + vv, ce - uu + vv]
            N += [nrm] * 4
            I += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
        self._add(P, N, I, color, R, c)

    def rbox(self, c, h, color, e=0.35, R=None, nu=10, nv=16):
        """Rounded box (superellipsoid). Smaller e = squarer corners."""
        hx, hy, hz = h
        u = np.linspace(-np.pi / 2, np.pi / 2, nu + 1)
        v = np.linspace(-np.pi, np.pi, nv + 1)
        U, V = np.meshgrid(u, v, indexing="ij")

        def sp(x):
            return np.sign(x) * np.abs(x) ** e
        X = hx * sp(np.cos(U)) * sp(np.cos(V))
        Y = hy * sp(np.sin(U))
        Z = hz * sp(np.cos(U)) * sp(np.sin(V))
        P = np.stack([X, Y, Z], -1).reshape(-1, 3)
        I = []
        W = nv + 1
        for i in range(nu):
            for j in range(nv):
                a = i * W + j
                I += [(a, a + 1, a + W + 1), (a, a + W + 1, a + W)]
        I = np.array(I)
        N = smooth_normals(P, I)
        # keep winding outward
        tri = P[I]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if np.mean(np.sum(fn * tri.mean(1), axis=1)) < 0:
            I = I[:, ::-1]
            N = smooth_normals(P, I)
        self._add(P, N, I, color, R, c)

    def cyl(self, c, length, r, color, axis="y", segs=18, caps=True, r2=None):
        """Cylinder/cone along an axis, centred on c. r2 = radius at the +axis end (cone)."""
        r2 = r if r2 is None else r2
        a = np.linspace(0, 2 * np.pi, segs, endpoint=False)
        P, N, I = [], [], []
        hl = length / 2.0
        # side with per-vertex smooth normals
        for k in range(segs):
            P.append((r * np.cos(a[k]), r * np.sin(a[k]), -hl))
            N.append((np.cos(a[k]), np.sin(a[k]), 0))
        for k in range(segs):
            P.append((r2 * np.cos(a[k]), r2 * np.sin(a[k]), hl))
            N.append((np.cos(a[k]), np.sin(a[k]), 0))
        for k in range(segs):
            k2 = (k + 1) % segs
            I += [(k, k2, segs + k2), (k, segs + k2, segs + k)]
        if caps:
            for z, rad, sgn in [(-hl, r, -1), (hl, r2, 1)]:
                b = len(P)
                P.append((0, 0, z)); N.append((0, 0, sgn))
                for k in range(segs):
                    P.append((rad * np.cos(a[k]), rad * np.sin(a[k]), z)); N.append((0, 0, sgn))
                for k in range(segs):
                    k2 = (k + 1) % segs
                    I.append((b, b + 1 + k, b + 1 + k2) if sgn > 0 else (b, b + 1 + k2, b + 1 + k))
        P = np.array(P, dtype=np.float64)
        N = np.array(N, dtype=np.float64)
        # orient the local Z axis along the requested axis
        if axis == "x":
            R = roty(np.pi / 2)
        elif axis == "y":
            R = rotx(-np.pi / 2)
        else:
            R = np.eye(3)
        self._add(P, N, I, color, R, c)

    def seg(self, p0, p1, r, color, segs=10):
        """Thin cylinder between two points."""
        p0, p1 = np.array(p0, float), np.array(p1, float)
        d = p1 - p0
        L = np.linalg.norm(d)
        if L < 1e-9:
            return
        z = d / L
        up = np.array([0, 1, 0]) if abs(z[1]) < 0.9 else np.array([1, 0, 0])
        x = np.cross(up, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], 1)
        a = np.linspace(0, 2 * np.pi, segs, endpoint=False)
        P, N, I = [], [], []
        for k in range(segs):
            P.append((r * np.cos(a[k]), r * np.sin(a[k]), -L / 2)); N.append((np.cos(a[k]), np.sin(a[k]), 0))
        for k in range(segs):
            P.append((r * np.cos(a[k]), r * np.sin(a[k]), L / 2)); N.append((np.cos(a[k]), np.sin(a[k]), 0))
        for k in range(segs):
            k2 = (k + 1) % segs
            I += [(k, k2, segs + k2), (k, segs + k2, segs + k)]
        self._add(P, N, I, color, R, (p0 + p1) / 2)

    # ---- export -----------------------------------------------------------------
    def arrays(self):
        P = np.concatenate(self.P).astype(np.float32)
        N = np.concatenate(self.N).astype(np.float32)
        C = np.concatenate(self.C).astype(np.float32)
        I = np.concatenate(self.I)
        return P, N, C, I

    def tri_count(self):
        return sum(len(i) for i in self.I)

    def save(self, path, name="model"):
        P, N, C, I = self.arrays()
        idx_t = np.uint32 if len(P) > 65535 else np.uint16
        I = I.astype(idx_t).reshape(-1)
        blobs = [P.tobytes(), N.tobytes(), C.tobytes(), I.tobytes()]
        offs, off = [], 0
        for b in blobs:
            offs.append(off)
            off += (len(b) + 3) // 4 * 4
        binchunk = b"".join(b + b"\0" * ((len(b) + 3) // 4 * 4 - len(b)) for b in blobs)
        views = [{"buffer": 0, "byteOffset": offs[i], "byteLength": len(blobs[i]),
                  "target": 34962 if i < 3 else 34963} for i in range(4)]
        g = {
            "asset": {"version": "2.0", "generator": "glbkit"},
            "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": name}],
            "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                                                       "indices": 3, "material": 0}]}],
            "materials": [{"name": "vcol", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                                                                  "metallicFactor": 0.0, "roughnessFactor": 0.85}}],
            "buffers": [{"byteLength": len(binchunk)}],
            "bufferViews": views,
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": len(P), "type": "VEC3",
                 "min": P.min(0).tolist(), "max": P.max(0).tolist()},
                {"bufferView": 1, "componentType": 5126, "count": len(N), "type": "VEC3"},
                {"bufferView": 2, "componentType": 5126, "count": len(C), "type": "VEC3"},
                {"bufferView": 3, "componentType": 5125 if idx_t == np.uint32 else 5123, "count": len(I), "type": "SCALAR"},
            ],
        }
        js = json.dumps(g, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(binchunk)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
            f.write(struct.pack("<II", len(binchunk), 0x004E4942) + binchunk)
        return path

    # ---- preview (matplotlib painter's algorithm) --------------------------------
    def preview(self, path, views=((22, -35), (0, -90), (22, 145), (88, -90)), size=4.2):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        P, N, C, I = self.arrays()
        tri = P[I.reshape(-1, 3)]
        col = lin2srgb(C[I.reshape(-1, 3)].mean(1))
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(nrm, axis=1, keepdims=True); ln[ln == 0] = 1
        nrm /= ln
        light = np.array([0.5, 0.9, 0.6]); light /= np.linalg.norm(light)
        shade = 0.45 + 0.55 * np.clip(nrm @ light, 0, 1)
        rgb = np.clip(col * shade[:, None], 0, 1)
        # matplotlib's axes: x right, y into screen, z up -> map (x, y, z)_model -> (x, z, y)
        T = tri[:, :, [0, 2, 1]]
        fig = plt.figure(figsize=(size * len(views), size))
        ctr = (P.min(0) + P.max(0)) / 2
        span = (P.max(0) - P.min(0)).max() / 2 * 1.05
        for k, (el, az) in enumerate(views):
            ax = fig.add_subplot(1, len(views), k + 1, projection="3d", proj_type="ortho")
            pc = Poly3DCollection(T, facecolors=rgb, edgecolors="none")
            ax.add_collection3d(pc)
            ax.set_xlim(ctr[0] - span, ctr[0] + span)
            ax.set_ylim(ctr[2] - span, ctr[2] + span)
            ax.set_zlim(ctr[1] - span, ctr[1] + span)
            ax.view_init(elev=el, azim=az)
            ax.set_axis_off()
            ax.set_box_aspect((1, 1, 1))
        fig.patch.set_facecolor("#d9cfb8")
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        fig.savefig(path, dpi=70)
        plt.close(fig)
