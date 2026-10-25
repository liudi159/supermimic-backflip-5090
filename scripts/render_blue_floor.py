"""Display-only re-render of final V14 recorded poses; no simulation stepping."""
from common import B,R,configure_gpu
configure_gpu()
from pathlib import Path
import json,hashlib,subprocess,xml.etree.ElementTree as ET
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
import sys
P=Path(sys.argv[1]);O=P/'blue_floor';O.mkdir(exist_ok=True)
source=P/'rollout.npz';before=hashlib.sha256(source.read_bytes()).hexdigest()
r=np.load(source);states=r['states'];metrics=json.loads((P/'metrics.json').read_text())
asset=R/'data/g1_description/g1_29dof.xml'
root=ET.parse(asset).getroot();root.find('compiler').set('meshdir',str(asset.parent))
a=root.find('asset')
ET.SubElement(a,'texture',name='blue_checks',type='2d',builtin='checker',rgb1='.055 .16 .28',rgb2='.14 .29 .44',mark='edge',markrgb='.25 .42 .57',width='512',height='512')
ET.SubElement(a,'material',name='blue_floor',texture='blue_checks',texrepeat='4 4',texuniform='true',reflectance='.15',shininess='.3',specular='.4')
w=root.find('worldbody')
ET.SubElement(w,'geom',name='display_floor',type='plane',size='0 0 .1',material='blue_floor',group='2',contype='0',conaffinity='0')
ET.SubElement(w,'light',pos='-2 -3 5',dir='.3 .3 -1',diffuse='.8 .8 .8',castshadow='true')
model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));assert model.nq==36
data=mujoco.MjData(model);W,H=1280,720
model.vis.global_.offwidth=W;model.vis.global_.offheight=H
model.vis.headlight.ambient[:]=[.45]*3;model.vis.headlight.diffuse[:]=[.7]*3
renderer=mujoco.Renderer(model,H,W)
opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[3]=0
heading=np.degrees(Rotation.from_quat(states[0,[4,5,6,3]]).as_euler('xyz')[2])
cam=mujoco.MjvCamera();cam.distance=3.5;cam.azimuth=heading+65;cam.elevation=-16
cam.lookat[:]=[float(states[0,0]),float(states[0,1]),.85]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
video=O/'policy_blue_checker.mp4'
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','25','-i','-','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
for i in range(0,len(states),2):
 data.qpos[:]=states[i,:36];mujoco.mj_forward(model,data)
 renderer.update_scene(data,camera=cam,scene_option=opt)
 im=Image.fromarray(renderer.render());d=ImageDraw.Draw(im)
 d.rectangle((0,0,W,34),fill=(12,18,26));d.text((12,6),f'Iteration {metrics['iteration']} | Actual simulated policy trajectory | t={i*.02:.2f}s',font=font,fill='white')
 proc.stdin.write(np.asarray(im).tobytes())
 if i in [0,88,96,104,114,120,132,474]:im.save(O/f'frame_{i:03d}.jpg')
 if i%100==0:print('RENDER',i,flush=True)
proc.stdin.close();assert proc.wait()==0;renderer.close()
assert hashlib.sha256(source.read_bytes()).hexdigest()==before
(O/'provenance.json').write_text(json.dumps({'checkpoint':metrics['checkpoint'],'iteration':metrics['iteration'],'source_rollout_sha256':before,'source_unchanged':True,'pose_mapping':'All displayed joint/root positions copied directly from saved actual states; every second50Hz state at25fps; no retiming or interpolation','render_only_changes':['blue checker material','lighting','fixed three-quarter camera'],'new_dynamics_solved':False,'training_or_evaluation_modified':False,'self_collision_during_original_training':False},indent=2))
print('DONE',flush=True)
