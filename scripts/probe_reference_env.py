"""Pre-training FK / ordering / actual contact-force and read-only instrumentation checks."""
from pathlib import Path
import sys,os,argparse,json
from common import B,R,ROOT
sys.path.insert(0,str(R))
from common import configure_gpu
configure_gpu()
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser();add_launcher_args(p);args=p.parse_args()
import numpy as np,torch,mujoco
from supermimic.env import TrackingSim,make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import MDP
from supermimic.robot import G1_JOINT_NAMES,TRACKED_BODIES,default_joint_pos_list,motor_of
from supermimic.rotations import quat_to_mat,rot_angle_sq,so3_log
from contact_recorder import ContactRecorder
torch.set_num_threads(1);torch.manual_seed(137)
assert torch.cuda.device_count()==1
dev='cuda:0';kin=G1Kinematics(dev)
motion=MotionLib([str(R/'data/backflip/backflip.csv')],kin,dev,src_fps=50.)
mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev))
cfg=make_sim(dev,args);cfg.physics.use_cuda_graph=False
out=B/'validation';out.mkdir(exist_ok=True)
result={'source':'pristine upstream env + evaluation-only recorder','gpu':torch.cuda.get_device_name(0),'frames':motion.num_frames}
with launch_simulation(cfg,args):
    env=TrackingSim(cfg,1,motion,mdp,eval_mode=True)
    ids=torch.tensor([0],device=dev);frame=torch.tensor([0],device=dev)
    env.reset(ids,frame);env.refresh()
    state=env.state();ref=motion.ref_state(frame)
    result['roundtrip_max']={n:float((a-b).abs().max()) for n,a,b in zip(['p','R','v','w','q','qd'],state,ref)}
    body_ids=[env.robot.body_names.index(n) for n in TRACKED_BODIES]
    poses=env.robot.data.body_link_pose_w.torch[:,body_ids]
    r=quat_to_mat(torch.cat((poses[...,6:7],poses[...,3:6]),-1))
    bp,br,_,_=kin.forward(*state)
    result['fk_position_max_m']=float((bp-poses[...,:3]).norm(dim=-1).max())
    result['fk_rotation_max_rad']=float(rot_angle_sq(br,r).sqrt().max())
    result['joint_order_matches']=list(env.robot.joint_names)==G1_JOINT_NAMES
    # Static calibration uses the original robot's nominal standing pose, not a held
    # motion-frame pose which need not be open-loop balanced.
    target=env.default_q.unsqueeze(0).clone();act=torch.zeros_like(target)
    # Static calibration inverse statics: compensate gravity and nominal equal
    # sole support through joint motor setpoints, never through root vertical force.
    model=env.sim.backend_manager._solver.mj_model;cal=mujoco.MjData(model)
    cal.qpos[:]=np.r_[0.,0.,.76,1.,0.,0.,0.,target[0].cpu().numpy()]
    mujoco.mj_forward(model,cal);tau=cal.qfrc_bias.copy();weight=model.body_mass.sum()*9.81
    for side in ('left','right'):
        bid=next(i for i in range(model.nbody) if side+'_ankle_roll_link' in (model.body(i).name or ''))
        geoms=np.flatnonzero(model.geom_bodyid==bid)
        point=cal.geom_xpos[geoms].mean(0);point[2]=0
        jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jac(model,cal,jp,jr,point,bid)
        tau-=jp.T@np.array([0,0,weight/2])
    limits=np.array([motor_of(n)[0] for n in G1_JOINT_NAMES]);act=torch.tensor(4*tau[6:]/limits,device=dev,dtype=torch.float32)[None]
    result['calibration_joint_feedforward_Nm']=tau[6:].tolist()
    trajectories=[];rec=None
    for recording in (False,False,True):
        env.reset(ids,frame)
        env.robot.write_root_link_pose_to_sim_index(root_pose=torch.tensor([[0.,0.,.76,0.,0.,0.,1.]],device=dev),env_ids=ids)
        env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(1,6,device=dev),env_ids=ids)
        env.robot.write_joint_position_to_sim_index(position=target,env_ids=ids)
        env.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(target),env_ids=ids)
        env.robot.set_joint_position_target_index(target=target,env_ids=ids)
        env.sim.backend_manager._solver.mjw_data.qacc_warmstart.zero_()
        env.refresh();states=[]
        if recording: rec=ContactRecorder(env)
        for step in range(100):
            # Calibration fixture only: stabilize horizontal position and orientation.
            # Apply NO vertical support; both feet must support the real full weight.
            # This fixture is absent from all policy training and evaluation.
            ss=env.state();force=(-100*ss[0]-10*ss[2]).clamp(-50,50);force[:,2]=0
            torque=(-30*so3_log(ss[1])-3*ss[3]).clamp(-20,20)
            env.robot.set_external_force_and_torque(force[:,None],torque[:,None],body_ids=[env.robot.body_names.index('pelvis')],env_ids=ids,is_global=True)
            env.step(act)
            s=env.state();states.append(torch.cat((s[0],s[1].flatten(1),s[2],s[3],s[4],s[5]),-1).cpu().numpy()[0])
            assert np.isfinite(states[-1]).all(),('Nonfinite calibration state',recording,step)
        trajectories.append(np.array(states))
        if rec:
            rec.save(out/'static_contact_probe.npz');rec.close()
    result['unrecorded_repeat_max_abs']=float(np.max(np.abs(trajectories[0]-trajectories[1])))
    result['recorder_trajectory_max_abs']=float(np.max(np.abs(trajectories[1]-trajectories[2])))
    dif=np.abs(trajectories[1]-trajectories[2]);result['recorder_position_rotation_joint_max']=float(np.max(dif[:,np.r_[0:12,18:47]]))
    result['recorder_velocity_max']=float(np.max(dif[:,np.r_[12:18,47:76]]))
    sample=np.load(out/'static_contact_probe.npz');f=sample['forces'];tail=f[-200:].sum(1)
    result['contact_sampling_hz']=1/float(sample['dt'])
    result['total_mass_kg']=float(sample['mass'])
    result['static_tail_mean_ground_force_N']=tail.mean(0).tolist()
    ratio=float(tail[:,2].mean()/(float(sample['mass'])*9.81))
    result['static_support_weight_ratio']=ratio
    result['left_right_tail_vertical_force_N']=f[-200:,:,2].mean(0).tolist()
    result['total_ground_force_tail_N']=sample['total_ground_force'][-200:].mean(0).tolist()
    result['total_constraint_force_tail_N']=sample['qfrc_constraint'][-200:,:3].mean(0).tolist()
    result['root_height_final_m']=float(sample['qpos'][-1,2])
    result['force_sum_vs_root_constraint_max_N']=float(np.max(np.abs(sample['total_ground_force']-sample['qfrc_constraint'][:,:3])))
    result['calibration_fixture']='Only horizontal force and attitude torque; zero vertical force. Not used in training/evaluation.'
    result['force_on_robot_convention']='actual GPU-solved force on each foot, world XYZ, upward Z positive'
    result['passed']=bool(result['joint_order_matches'] and motion.num_frames==321 and
        max(result['roundtrip_max'].values())<1e-4 and result['fk_position_max_m']<1e-4 and
        result['fk_rotation_max_rad']<.002 and result['recorder_position_rotation_joint_max']<1e-5 and
        result['recorder_velocity_max']<1e-4 and result['recorder_trajectory_max_abs']<2*result['unrecorded_repeat_max_abs']+1e-5 and
        .9<ratio<1.1 and min(result['left_right_tail_vertical_force_N'])>0 and
        result['force_sum_vs_root_constraint_max_N']<.01)
(out/'reference_env_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
assert result['passed'],'Pre-training validation failed; do not start baseline'
