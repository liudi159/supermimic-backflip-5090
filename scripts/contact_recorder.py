"""Read actual MuJoCo-Warp constraint forces after each 5 ms solver substep.

Evaluation only, CUDA graph disabled for host-side copies. No forward dynamics,
inverse dynamics, state projection, or reference playback is used to create forces.
"""
import numpy as np
import json
import mujoco
import mujoco_warp as mjw


class ContactRecorder:
    def __init__(self, env):
        self.manager = env.sim.backend_manager
        assert not self.manager._cfg.use_cuda_graph, 'Host recording requires eager evaluation'
        self.solver = self.manager._solver
        self.model = self.solver.mj_model
        assert self.solver.mjw_data.nworld == 1
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.manager.get_solver_dt())
        assert abs(self.dt-.005)<1e-8,self.dt
        self.floor = set(np.flatnonzero(self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist())
        self.feet = []
        for side in ('left','right'):
            bodies = [i for i in range(self.model.nbody) if side+'_ankle_roll_link' in (self.model.body(i).name or '')]
            assert len(bodies)==1,(side,bodies)
            self.feet.append(set(np.flatnonzero(self.model.geom_bodyid == bodies[0]).tolist()))
        assert self.floor and all(self.feet)
        self.samples=[]
        self.debug_contacts=[]
        self.original=self.manager._step_solver
        def recording_step(*args,**kwargs):
            result=self.original(*args,**kwargs)
            self.record()
            return result
        self.manager._step_solver=recording_step

    def record(self):
        # GPU-solved efc_force and contact frames are copied, never recomputed by mj_forward.
        mjw.get_data_into(self.data,self.model,self.solver.mjw_data)
        d,m=self.data,self.model
        forces=np.zeros((2,3));ground_total=np.zeros(3);nonfoot_force=np.zeros(3)
        count=np.zeros(2,dtype=int);bad=0
        for cidx in range(d.ncon):
            c=d.contact[cidx];g1,g2=map(int,c.geom)
            if g1 not in self.floor and g2 not in self.floor:continue
            f=np.zeros(6);mujoco.mj_contactForce(m,d,cidx,f)
            # Contact frame rows are world axes. Force on geom2; reverse if ground is geom2.
            world=c.frame.reshape(3,3).T @ f[:3]
            if g2 in self.floor:world=-world;other=g1
            else:other=g2
            ground_total+=world
            if len(self.debug_contacts)<12 and np.linalg.norm(world)>1:
                self.debug_contacts.append(dict(step=len(self.samples),geom1=g1,geom2=g2,
                    names=[m.geom(g1).name,m.geom(g2).name],bodies=[m.body(int(m.geom_bodyid[g1])).name,m.body(int(m.geom_bodyid[g2])).name],
                    frame=c.frame.tolist(),local_force=f.tolist(),world_force=world.tolist(),root_constraint=d.qfrc_constraint[:3].tolist()))
            side=next((i for i,ids in enumerate(self.feet) if other in ids),None)
            if side is None:
                nonfoot_force+=world
                if np.linalg.norm(world)>1e-3:bad+=1
            else:
                forces[side]+=world
                if np.linalg.norm(world)>1e-3:count[side]+=1
        self.samples.append(dict(time=len(self.samples)*self.dt+self.dt,forces=forces,contact_counts=count,
             total_ground_force=ground_total,nonfoot_force=nonfoot_force,nonfoot_contacts=bad,
             qpos=d.qpos.copy(),qvel=d.qvel.copy(),qfrc_actuator=d.qfrc_actuator.copy(),qfrc_applied=d.qfrc_applied.copy(),
             actuator_force=d.actuator_force.copy(),ctrl=d.ctrl.copy(),qfrc_constraint=d.qfrc_constraint.copy()))

    def save(self,path):
        assert self.samples
        from pathlib import Path
        Path(str(path)+'.contacts.json').write_text(json.dumps({'floor':list(self.floor),'feet':[list(x) for x in self.feet],'contacts':self.debug_contacts},indent=2))
        np.savez_compressed(path,dt=self.dt,mass=float(self.model.body_mass.sum()),
            joint_names=np.array([self.model.joint(j).name for j in range(self.model.njnt)]),
            **{k:np.array([s[k] for s in self.samples]) for k in self.samples[0]})

    def close(self):
        self.manager._step_solver=self.original
