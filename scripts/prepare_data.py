"""Download source assets; reconstruct the exact 321-frame50Hz reference, no policy weights."""
from common import B,R,ROOT
import urllib.request,hashlib,json,subprocess
from pathlib import Path
import numpy as np
B.mkdir(parents=True,exist_ok=True)
def fetch(url,path,expected=None):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():urllib.request.urlretrieve(url,path)
 sha=hashlib.sha256(path.read_bytes()).hexdigest()
 if expected and sha!=expected:raise RuntimeError(f'Checksum mismatch: {path}; do not silently use changed data')
 return sha
manifest=json.loads((ROOT/'environment/asset-manifest.json').read_text())
for item in manifest['assets']:
 fetch(item['source'],R/'data/g1_description'/item['relative_file'],item['sha256'])
url='https://raw.githubusercontent.com/kevinzakka/mjbatch/77966f85bcd8f7ef4351cb4a1a6f42e133d19725/examples/assets/flip.npz'
src=B/'downloads/flip.npz';sha=fetch(url,src,'f9ad1275f978bf891ee141e272d8279eb56685e4209f42d5cd7dc23529bcb4d6')
x=np.load(src)
raw=np.concatenate([x['body_pos_w'][:,0],x['body_quat_w'][:,0,[1,2,3,0]],x['joint_pos']],axis=-1)[30:241]
raw=np.concatenate([np.repeat(raw[:1],10,axis=0),raw,np.repeat(raw[-1:],100,axis=0)])
assert raw.shape==(321,36)
out=R/'data/backflip/backflip.csv';out.parent.mkdir(parents=True,exist_ok=True);np.savetxt(out,raw,delimiter=',',fmt='%.10g')
expected=manifest['motion_csv_sha256'];actual=hashlib.sha256(out.read_bytes()).hexdigest()
assert actual==expected,('Motion conversion differs from recorded run',actual,expected)
(B/'data_provenance.json').write_text(json.dumps({'source':url,'source_sha256':sha,'source_frames_inclusive':[30,240],'initial_hold_frames':10,'final_hold_frames':100,'fps':50,'csv_shape':[321,36],'csv_quaternion':'xyzw','csv_sha256':actual,'external_policy_weights':False},indent=2))
for tool in ['urdf_to_mjcf.py','mjcf_to_usd.py']:
 subprocess.run([str(B/'.venv/bin/python'),str(R/'tools'/tool)],cwd=R,check=True)
print('Motion and original G1 assets ready.')
