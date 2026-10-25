"""Actual upstream policy rollout plus five-second hold; no resets after failure.

Training is unchanged. This evaluator adds physical diagnostics to the repository
criterion and records solved contact forces at every physics substep.
"""
from pathlib import Path
import sys,os,argparse,json,hashlib
from common import B,R,ROOT
sys.path.insert(0,str(R))
from common import configure_gpu
configure_gpu()
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
add_launcher_args(p);args=p.parse_args()
import numpy as np,torch,mujoco
from scipy.spatial.transform import Rotation
from supermimic.env import TrackingSim,make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import MDP,OBS_DIM
from supermimic.networks import Policy
import supermimic.networks as networks
from supermimic.robot import G1_JOINT_NAMES,default_joint_pos_list,MJCF_PATH,motor_of
from supermimic.rotations import mat_to_quat
from contact_recorder import ContactRecorder
torch.set_num_threads(1);torch.manual_seed(137)
assert torch.cuda.device_count()==1
dev='cuda:0';kin=G1Kinematics(dev)
motion=MotionLib([str(R/'data/backflip/backflip.csv')],kin,dev,src_fps=50.)
mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev))
ck=torch.load(args.checkpoint,map_location=dev,weights_only=False);a=ck['args'];K,H=a['goal_frames'],a['hist']
networks.AMP_DTYPE=torch.bfloat16 if a.get('amp','none')=='bf16' else None
pi=Policy(OBS_DIM,73*K,hist=H).to(dev);pi.load_state_dict(ck['pi']);pi.eval()
cfg=make_sim(dev,args);cfg.physics.use_cuda_graph=False
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
n=motion.num_frames;assert n==321
def vector(s):
    p,r,v,w,q,qd=s
    return torch.cat((p,mat_to_quat(r),q,v,(r.transpose(-1,-2)@w.unsqueeze(-1)).squeeze(-1),qd),-1)[0].cpu().numpy()
with launch_simulation(cfg,args):
    env=TrackingSim(cfg,1,motion,mdp,eval_mode=True)
    ids=torch.tensor([0],device=dev);env.reset(ids,ids);env.refresh()
    # This motion contains ten initial static frames and 100 final static frames.
    assert float(motion.root_v[0].norm()+motion.root_w[0].norm()+motion.qd[0].norm())<1e-6
    assert float(motion.root_v[-1].norm()+motion.root_w[-1].norm()+motion.qd[-1].norm())<1e-6
    recorder=ContactRecorder(env)
    states=[vector(env.state())];actions=[];targets=[];rewards=[];failed=[];hist=None
    with torch.no_grad():
        for step in range(n+250):
            s=env.state();obs=mdp.obs(s,env.a_prev,env.xi,env.f)
            hist=obs.unsqueeze(1).repeat(1,H,1) if hist is None else torch.cat((hist[:,1:],obs.unsqueeze(1)),1)
            act=pi(hist,motion.goal(env.f,K))
            targets.append((env.default_q+env.alpha*act)[0].cpu().numpy());actions.append(act[0].cpu().numpy())
            tr=env.step(act);states.append(vector(env.state()));rewards.append(float(tr['r'][0]));failed.append(bool(tr['term'][0]))
            env.f.clamp_(max=n-2) # identical static final target, no restart/retiming
    recorder.save(out/'contact_forces_200hz.npz');recorder.close()
    physics_model=recorder.model
    mujoco.mj_saveModel(physics_model,str(out/'physics_model.mjb'))
    states=np.asarray(states);rewards=np.asarray(rewards)
    np.savez_compressed(out/'rollout.npz',states=states,rewards=rewards,actions=actions,targets=targets,
        reference_q=motion.q.cpu().numpy(),reference_p=motion.root_p.cpu().numpy(),reference_R=motion.root_R.cpu().numpy())
    # Geometric quality checks use saved simulated poses. Forces above remain actual GPU outputs.
    pd=mujoco.MjData(physics_model);floor=next(iter(recorder.floor));feet=recorder.feet
    ankle_ids=[int(physics_model.geom_bodyid[next(iter(f))]) for f in feet]
    air=[];bad=[];ankles=[];gaps=[]
    assert physics_model.nq==36 and physics_model.nv==35
    for s in states:
        pd.qpos[:]=s[:36];pd.qvel[:]=s[36:];mujoco.mj_forward(physics_model,pd)
        gap=[min(mujoco.mj_geomDistance(physics_model,pd,g,floor,10,None) for g in group) for group in feet]
        contact=any(c.dist<0 and ((int(c.geom[0]) in recorder.floor and int(c.geom[1]) not in feet[0]|feet[1]) or
                    (int(c.geom[1]) in recorder.floor and int(c.geom[0]) not in feet[0]|feet[1])) for c in pd.contact)
        gaps.append(gap);air.append(min(gap)>.03);bad.append(contact)
        ankles.append(float(np.linalg.norm(pd.xpos[ankle_ids[0]]-pd.xpos[ankle_ids[1]])))
    fullair=np.array(air[:n+1]) & ~np.array(bad[:n+1]);gaps=np.array(gaps)
    rot=Rotation.from_quat(states[:,[4,5,6,3]]);m=rot.as_matrix();up=m[:,2,2]
    phase=np.unwrap(np.arctan2(-(m[:,:,0]@m[0,:,2]),m[:,:,0]@m[0,:,0]))
    edge=np.diff(np.r_[False,fullair,False].astype(int));starts=np.flatnonzero(edge==1);ends=np.flatnonzero(edge==-1)
    if len(starts):i=int(np.argmax(ends-starts));aa,zz=int(starts[i]),int(ends[i]-1)
    else:aa=zz=0
    lateral=m[:n+1,:,1]@m[0,:,1];latmin=float(lateral[aa:zz+1].min())
    aerial=float(np.degrees(phase[zz]-phase[aa]));net=float(np.degrees(phase[n]-phase[0]))
    indices=[j for j,name in enumerate(G1_JOINT_NAMES) if 'hip_' in name or 'knee_' in name]
    qref=motion.q.cpu().numpy();ri=np.minimum(np.arange(aa,zz+1),n-1)
    rmse=float(np.sqrt(np.mean(np.degrees(states[aa:zz+1,7:36][:,indices]-qref[ri][:,indices])**2)))
    # Reference ankle spacing is measured with this run's own robot asset.
    refdist=[]
    for f in ri:
        ref=motion.ref_state(torch.tensor([int(f)],device=dev));pd.qpos[:]=vector(ref)[:36];mujoco.mj_kinematics(physics_model,pd)
        refdist.append(float(np.linalg.norm(pd.xpos[ankle_ids[0]]-pd.xpos[ankle_ids[1]])))
    tail=states[-100:];ls=np.linalg.norm(tail[:,36:39],axis=1);ws=np.linalg.norm(tail[:,39:42],axis=1)
    drift=float(np.linalg.norm(tail[-1,:2]-tail[0,:2]));fail=np.flatnonzero(failed[:n])
    metrics={'checkpoint':str(Path(args.checkpoint).resolve()),'iteration':int(ck['iteration']),'frames':n,
        'quality_protocol':'strict_v3 thresholds, reference implementation robot asset and native reward',
        'normalized_return':float(rewards[:n].sum()/(.1*n)),
        'repository_early_termination_return':float(rewards[:min(int(fail[0])+1 if len(fail) else n-1,n-1)].sum()/(.1*(n-1))),
        'first_failure_step':int(fail[0]+1) if len(fail) else None,
        'initial_root_speed':float(np.linalg.norm(states[0,36:39])),'initial_joint_speed_norm':float(np.linalg.norm(states[0,42:])),
        'sagittal_net_rotation_degrees':net,'aerial_rotation_degrees':aerial,
        'axis_deviation_degrees':float(np.degrees(np.arccos(np.clip(latmin,-1,1)))),
        'aerial_lateral_axis_min_alignment':latmin,'aerial_leg_joint_rmse_degrees':rmse,
        'aerial_max_ankle_distance':float(max(ankles[aa:zz+1])),'reference_max_ankle_distance_same_window':max(refdist),
        'nonfoot_floor_contact_frames':int(sum(bad[:n+1])),'hold_nonfoot_contact_frames':int(sum(bad[n+1:])),
        'last_2s_linear_speed_rms':float(np.sqrt(np.mean(ls**2))),'last_2s_linear_speed_max':float(ls.max()),
        'last_2s_angular_speed_rms':float(np.sqrt(np.mean(ws**2))),'last_2s_horizontal_drift':drift,
        'last_2s_min_up':float(up[-100:].min()),'extra_hold_seconds':5,'self_collisions_enabled':False,
        'source':'actual reference implementation policy; no state substitution or resets after failure',
        'comparability_note':'Native pelvis reward and different robot assets/collisions vs previous mjlab runs; physical thresholds unchanged, returns are not interchangeable.'}
    checks={'zero_initial_speed':metrics['initial_root_speed']<1e-6 and metrics['initial_joint_speed_norm']<1e-6,
        'full_backward_turn':-370<=net<=-350,'aerial_rotation':aerial<=-250 and latmin>.8,
        'no_nonfoot_contact':not any(bad),'stable_last_two_seconds':np.sqrt(np.mean(ls**2))<.1 and ls.max()<.2 and np.sqrt(np.mean(ws**2))<.2 and drift<.1 and up[-100:].min()>.99,
        'tracking_return':metrics['normalized_return']>=.75,
        'reference_leg_posture':len(starts)>0 and rmse<25 and max(ankles[aa:zz+1])<=max(refdist)+.15}
    pitch=float((rot[:-1].inv()*rot[1:]).as_rotvec()[:n,1].sum())
    legacy=bool(not len(fail) and up[:n+1].min()<-.6 and pitch<-5 and up[-25:].min()>.85 and
        states[-25:,2].min()>.5 and np.linalg.norm(states[-1,36:39])<.5 and fullair.sum()*.02>=.06 and
        (up[:n+1][fullair].min() if fullair.any() else 1)<-.6)
    metrics.update(legacy_gate_pass=legacy,pitch_rotation_radians=pitch,fully_airborne_seconds=float(fullair.sum()*.02))
    metrics['quality_checks']={k:bool(v) for k,v in checks.items()};metrics['success']=bool(all(checks.values()) and legacy)
    metrics['whole_motion_max_ankle_distance']=float(max(ankles[:n+1]))
    metrics['fixed_reference_flight_max_ankle_distance']=float(max(ankles[90:118]))
    metrics['fixed_reference_flight_leg_rmse_degrees']=float(np.sqrt(np.mean(np.degrees(states[90:118,7:36][:,indices]-qref[90:118][:,indices])**2)))
    metrics['has_airborne_segment']=bool(len(starts))
    if not len(starts):
        for key in ['axis_deviation_degrees','aerial_lateral_axis_min_alignment','aerial_leg_joint_rmse_degrees','aerial_max_ankle_distance','reference_max_ankle_distance_same_window']:
            metrics[key]=None # No flight is not a zero-error successful posture.
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    sign=np.array([1,-1,-1,1,1,-1]);err=(states[:n,7:13]-states[:n,13:19]*sign)-(qref[:,:6]-qref[:,6:12]*sign)
    asym={'phases':{name:{'excess_pair_asymmetry_rmse_deg':float(np.sqrt(np.mean(np.degrees(err[lo:hi])**2)))} for name,lo,hi in [('preparation',50,75),('takeoff',75,90),('flight_fixed_reference_window',90,118)]}}
    depart=[]
    for j in range(2):
        hits=np.flatnonzero(gaps[:aa+1,j]<=.03);depart.append(int(hits[-1]+1) if len(hits) else 0)
    asym.update(left_right_departure_step=depart,departure_gap_seconds=abs(depart[0]-depart[1])*.02,
        note='Same fixed windows and geometric departure threshold as V12/V13; force-based diagnostics are additional.')
    (out/'asymmetry_metrics.json').write_text(json.dumps(asym,indent=2))
    np.savez_compressed(out/'rollout.npz',states=states,rewards=rewards,actions=actions,targets=targets,full_air=fullair,
        nonfoot_contact=bad,sagittal_phase=phase,reference_q=qref,reference_p=motion.root_p.cpu().numpy(),reference_R=motion.root_R.cpu().numpy())
    print('ACTUAL_EVALUATION_COMPLETE',json.dumps(metrics),flush=True)
