"""Compound_3D (Y-up, 1 mesh, 24 prims) -> game GLB: scaled by S, origin at gate centre, gate faces +X, small lift off the ground,
grass pad clipped to the city block. Usage: make_compound.py in.glb out.glb"""
import json, struct, sys, numpy as np
src, dst = sys.argv[1], sys.argv[2]
S = 0.65; GX, GZ = 27.4, -31.4
d = open(src,'rb').read(); jl = struct.unpack('<I', d[12:16])[0]; j = json.loads(d[20:20+jl]); b = d[28+jl:]
def get(i, n):
    a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]
    return np.frombuffer(b, np.float32, a['count']*n, bv.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1, n).copy()
out = bytearray(); bvs = []; accs = []; prims = []
def addacc(arr, mm=False):
    while len(out) % 4: out.append(0)
    bvs.append({'buffer':0,'byteOffset':len(out),'byteLength':arr.nbytes,'target':34962}); out.extend(arr.tobytes())
    a = {'bufferView':len(bvs)-1,'componentType':5126,'count':len(arr),'type':'VEC3'}
    if mm: a['min'] = arr.min(0).tolist(); a['max'] = arr.max(0).tolist()
    accs.append(a); return len(accs)-1
tris = 0
for pr in j['meshes'][0]['primitives']:
    P = get(pr['attributes']['POSITION'], 3); N = get(pr['attributes']['NORMAL'], 3)
    P = np.stack([(P[:,0]-GX)*S, P[:,1]*S, (P[:,2]-GZ)*S], 1)
    if j['materials'][pr['material']]['name'] == 'grass':
        P[:,0] = np.minimum(P[:,0], 0.5); P[:,2] = np.clip(P[:,2], -14, 22); P[:,1] = np.clip(P[:,1], -0.02, 0.0)
    P[:,1] += 0.03
    tris += len(P)//3
    prims.append({'attributes':{'POSITION':addacc(P.astype(np.float32), True),'NORMAL':addacc(N)},'material':pr['material'],'mode':4})
g = {'asset':{'version':'2.0','generator':'tools/make_compound.py'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'mesh':0,'name':'compound'}],
     'meshes':[{'name':'compound','primitives':prims}],'materials':j['materials'],'bufferViews':bvs,'accessors':accs,'buffers':[{'byteLength':len(out)}]}
jb = json.dumps(g, separators=(',',':')).encode(); jb += b' '*(-len(jb)%4); bb = bytes(out)+b'\0'*(-len(out)%4)
open(dst,'wb').write(struct.pack('<III',0x46546C67,2,12+8+len(jb)+8+len(bb))+struct.pack('<II',len(jb),0x4E4F534A)+jb+struct.pack('<II',len(bb),0x004E4942)+bb)
print('tris',tris,'bytes',12+8+len(jb)+8+len(bb))
