"""Install the recorded Linux x86_64 runtime into a new isolated directory."""
from common import B,ROOT
from pathlib import Path
import subprocess,shutil,sys,urllib.request,hashlib
if sys.platform!='linux':raise SystemExit('Training requires Linux x86_64 + NVIDIA GPU; this is not a macOS install.')
uv=shutil.which('uv')
if not uv:raise SystemExit('Install uv first: https://docs.astral.sh/uv/getting-started/installation/')
B.mkdir(parents=True,exist_ok=True)
def run(*args,cwd=None):subprocess.run(list(args),cwd=cwd,check=True)
def clone(url,path,commit):
 if not path.exists():run('git','clone',url,str(path))
 actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
 if actual!=commit:
  if subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip():raise RuntimeError(f'Dirty dependency: {path}')
  run('git','-C',str(path),'checkout','--detach',commit)
clone('https://github.com/isaac-sim/IsaacLab.git',B/'IsaacLab','46d6cc3d7b115906fad12efd372fbf69f4e3d3ed')
clone('https://github.com/newton-physics/newton.git',B/'newton-71eeba16a717d4dcf0b9dd7056a7b6f863054dc3','71eeba16a717d4dcf0b9dd7056a7b6f863054dc3')
python=B/'.venv/bin/python'
if not python.exists():run(uv,'venv','--python','3.12',str(B/'.venv'))
wheel=B/'warp_lang-1.17.0+cu12-py3-none-manylinux_2_28_x86_64.whl'
if not wheel.exists():
 urllib.request.urlretrieve('https://github.com/NVIDIA/warp/releases/download/v1.17.0/warp_lang-1.17.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl',wheel)
expected=(ROOT/'environment/warp-wheel.sha256').read_text().split()[0]
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==expected,'Warp wheel checksum mismatch'
run(uv,'--no-config','pip','install','--python',str(python),'torch==2.11.0','torchvision==0.26.0','--index-url','https://download.pytorch.org/whl/cu128')
lines=[]
for line in (ROOT/'environment/observed-freeze.txt').read_text().splitlines():
 if line.startswith(('torch==','torchvision==','triton==')):continue
 if line.startswith('-e file://<WORKDIR>/'):line='-e '+str(B/line.split('<WORKDIR>/')[1])
 if line.startswith('newton @'):line='newton @ '+(B/'newton-71eeba16a717d4dcf0b9dd7056a7b6f863054dc3').as_uri()
 if line.startswith('warp-lang @'):line='warp-lang @ '+wheel.as_uri()
 lines.append(line)
req=B/'requirements-native.txt';req.write_text('\n'.join(lines)+'\n')
run(uv,'--no-config','pip','install','--python',str(python),'--no-deps','-r',str(req),'--index-strategy','unsafe-best-match','--index-url','https://pypi.org/simple','--extra-index-url','https://pypi.nvidia.com',cwd=B/'IsaacLab')
print('Runtime ready:',python)
