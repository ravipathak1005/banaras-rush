"""Convert School_Campus GLB (Z-up, 860 meshes) -> game GLB: Y-up, origin at gate centre on the road line,
campus extends toward -Z, front faces +Z, one primitive per material. Usage: make_campus.py in.glb out.glb"""
import json, struct, sys, numpy as np
src, dst = sys.argv[1], sys.argv[2]
d = open(src, 'rb').read()
jl = struct.unpack('<I', d[12:16])[0]; j = json.loads(d[20:20+jl])
bl = struct.unpack('<I', d[20+jl:24+jl])[0]; bin0 = d[28+jl:28+jl+bl]
CT = {5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
NC = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}
def acc(i):
    a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]
    n = NC[a['type']]; dt = np.dtype(CT[a['componentType']])
    off = bv.get('byteOffset',0)+a.get('byteOffset',0); stride = bv.get('byteStride') or n*dt.itemsize
    raw = np.frombuffer(bin0, np.uint8, count=stride*(a['count']-1)+n*dt.itemsize, offset=off)
    return np.lib.stride_tricks.as_strided(raw.view(np.uint8), (a['count'], n*dt.itemsize), (stride, 1)).copy().view(dt).reshape(a['count'], n)
GX = 23.7  # gate centre x
def conv(p): return np.stack([p[:,0]-GX, p[:,2]+0.03, -p[:,1]], 1)
groups = {}
def walk(ni):
    n = j['nodes'][ni]
    if 'mesh' in n:
        for pr in j['meshes'][n['mesh']]['primitives']:
            at = pr['attributes']; P = acc(at['POSITION']).astype(np.float32)
            N = acc(at['NORMAL']).astype(np.float32) if 'NORMAL' in at else None
            U = acc(at['TEXCOORD_0']).astype(np.float32) if 'TEXCOORD_0' in at else np.zeros((len(P),2),np.float32)
            I = acc(pr['indices']).astype(np.uint32).ravel() if 'indices' in pr else np.arange(len(P), dtype=np.uint32)
            if N is None:
                N = np.zeros_like(P)
            g = groups.setdefault(pr.get('material',0), [[],[],[],[],0])
            g[0].append(conv(P)); g[1].append(conv(N)); g[2].append(U); g[3].append(I+g[4]); g[4] += len(P)
    for c in n.get('children', []): walk(c)
for r in j['scenes'][0]['nodes']: walk(r)
out = bytearray(); bvs = []; accs = []; prims = []
def add(arr, target=None):
    while len(out) % 4: out.append(0)
    bvs.append({'buffer':0,'byteOffset':len(out),'byteLength':arr.nbytes, **({'target':target} if target else {})}); out.extend(arr.tobytes()); return len(bvs)-1
def addacc(arr, ctype, typ, target, mm=False):
    bv = add(arr, target); a = {'bufferView':bv,'componentType':ctype,'count':len(arr),'type':typ}
    if mm: a['min'] = arr.min(0).tolist(); a['max'] = arr.max(0).tolist()
    accs.append(a); return len(accs)-1
tris = 0
for mi, g in sorted(groups.items()):
    P = np.concatenate(g[0]); N = np.concatenate(g[1]); U = np.concatenate(g[2]); I = np.concatenate(g[3])
    tris += len(I)//3
    ia = addacc(I.astype(np.uint16 if len(P) < 65535 else np.uint32), 5123 if len(P) < 65535 else 5125, 'SCALAR', 34963)
    pa = addacc(P, 5126, 'VEC3', 34962, True); na = addacc(N.astype(np.float32), 5126, 'VEC3', 34962); ua = addacc(U, 5126, 'VEC2', 34962)
    prims.append({'attributes':{'POSITION':pa,'NORMAL':na,'TEXCOORD_0':ua},'indices':ia,'material':mi})
# images
imgs = []
for im in j.get('images', []):
    b = j['bufferViews'][im['bufferView']]; raw = bin0[b.get('byteOffset',0):b.get('byteOffset',0)+b['byteLength']]
    arr = np.frombuffer(raw, np.uint8); imgs.append({'bufferView': add(arr), 'mimeType': im['mimeType']})
meshes = [{'name':'campus_%d'%i,'primitives':[p]} for i,p in enumerate(prims)]
nodes = [{'name':'campus','children':list(range(1,len(meshes)+1))}] + [{'mesh':i} for i in range(len(meshes))]
g = {'asset':{'version':'2.0','generator':'tools/make_campus.py'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':nodes,'meshes':meshes,
     'materials':j['materials'],'textures':j.get('textures',[]),'samplers':j.get('samplers',[]),'images':imgs,
     'bufferViews':bvs,'accessors':accs,'buffers':[{'byteLength':len(out)}]}
jb = json.dumps(g, separators=(',',':')).encode(); jb += b' '*(-len(jb)%4)
bb = bytes(out) + b'\0'*(-len(out)%4)
open(dst,'wb').write(struct.pack('<III',0x46546C67,2,12+8+len(jb)+8+len(bb)) + struct.pack('<II',len(jb),0x4E4F534A)+jb + struct.pack('<II',len(bb),0x004E4942)+bb)
print('prims',len(prims),'tris',tris,'bytes',12+8+len(jb)+8+len(bb))
