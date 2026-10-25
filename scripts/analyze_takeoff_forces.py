"""Summarize actual solver contact forces and synchronized left/right leg motion."""
from pathlib import Path
import sys,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(sys.argv[1]);f=np.load(P/'contact_forces_200hz.npz');r=np.load(P/'rollout.npz')
t=f['time'];dt=float(f['dt']);z=f['forces'][:,:,2];s=r['states'];q=s[:,7:36]
sgn=np.array([1,-1,-1,1,1,-1]);ref=r['reference_q'];N=len(ref)
err=np.degrees((q[:N,:6]-q[:N,6:12]*sgn)-(ref[:,:6]-ref[:,6:12]*sgn))
result={'source':'actual 200 Hz GPU solver contact forces; no calibration fixture','mass_kg':float(f['mass']),'sampling_hz':1/dt,'phases':{},'force_sign':'positive Z acts upward on the robot','contact_threshold_N':5.0}
phases=[('reference_preparation_window',1.,1.5),('reference_takeoff_window',1.5,1.8),('reference_airborne_window',1.8,2.36)]
air=r['full_air'];edges=np.diff(np.r_[False,air,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
if len(starts):
    k=int(np.argmax(ends-starts));leave=float(starts[k])*.02
    phases.append(('actual_pre_airborne_window',max(0,leave-.3),leave))
    result['selected_actual_airborne_window_seconds']=[leave,float(ends[k])*.02]
else:result['selected_actual_airborne_window_seconds']=None
for name,lo,hi in phases:
    mask=(t>lo)&(t<=hi);imp=z[mask].sum(0)*dt;peak=z[mask].max(0)
    result['phases'][name]={'window_seconds':[lo,hi],'left_right_vertical_impulse_Ns':imp.tolist(),
        'absolute_impulse_imbalance_fraction':float(abs(imp[0]-imp[1])/abs(imp).sum()) if abs(imp).sum()>1e-8 else None,
        'left_right_peak_vertical_force_N':peak.tolist(),'left_right_contact_fraction':(z[mask]>5).mean(0).tolist(),
        'excess_leg_asymmetry_rmse_degrees':float(np.sqrt(np.mean(err[round(lo*50):round(hi*50)]**2)))}
force_error=f['total_ground_force']-f['qfrc_constraint'][:,:3]
result['summed_contact_vs_constraint_max_error_N']=float(np.max(abs(force_error)))
# Report all sustained loss-of-contact events; do not select a flattering departure.
events=[]
for side in range(2):
    support=z[:,side]>5;idx=np.flatnonzero(support[:-1]&~support[1:])+1
    events.append([float(t[i]) for i in idx if 1.2<=t[i]<=2.8 and i+4<len(t) and not support[i:i+4].any()])
result['left_right_sustained_departure_events_seconds']=events
tau=f['qfrc_actuator'][:,6:18];limits=np.array([88,139,88,139,50,50]*2)
mask=(t>1.5)&(t<=2.36)
result['left_right_leg_max_torque_limit_ratio']=[float(np.max(abs(tau[mask,i:i+6])/limits[i:i+6])) for i in [0,6]]
(P/'takeoff_force_metrics.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
ax[0].plot(t,z[:,0],label='Left foot');ax[0].plot(t,z[:,1],label='Right foot');ax[0].set_ylabel('Vertical force (N)');ax[0].legend()
tt=np.arange(len(q))*.02
for j,label in [(0,'Hip pitch'),(3,'Knee')]:
    a=ax[1 if j==0 else 2];a.plot(tt,np.degrees(q[:,j]),label='Left');a.plot(tt,np.degrees(q[:,6+j]),label='Right')
    a.plot(np.arange(N)*.02,np.degrees(ref[:,j]),'--',alpha=.6,label='Reference left')
    a.plot(np.arange(N)*.02,np.degrees(ref[:,6+j]),'--',alpha=.6,label='Reference right')
    a.set_ylabel(label+' (deg)');a.legend(ncol=4,fontsize=8)
for a in ax:a.axvspan(1.5,1.8,alpha=.1,color='orange');a.axvspan(1.8,2.36,alpha=.07,color='blue');a.grid(alpha=.25);a.set_xlim(1.,3.)
ax[-1].set_xlabel('Actual simulation time (s)');fig.suptitle('Actual policy: takeoff forces and leg posture (no retiming)');fig.tight_layout();fig.savefig(P/'takeoff_force_and_posture.png',dpi=160)
print(json.dumps(result),flush=True)
