"""Bounded single-GPU baseline runner, gated on physics validation and capacity smoke."""
from pathlib import Path
import os,sys,json,time,subprocess,csv,math
from common import B,R,ROOT
phase=sys.argv[1];assert phase in ('smoke','main')
assert json.loads((B/'validation/reference_env_probe.json').read_text())['passed']
from common import configure_gpu
env=configure_gpu()
env.update(TORCHINDUCTOR_CACHE_DIR=str(B/'torchinductor-cache'),TORCHINDUCTOR_COMPILE_THREADS='4')
run=B/'runs'/('capacity_smoke' if phase=='smoke' else 'backflip_v14_full_reference_seed137')
assert not run.exists(),f'Never overwrite an existing run: {run}'
if phase=='main':
    prior=json.loads((B/'smoke_result.json').read_text())
    assert prior['exit_code']==0 and prior['finite_metrics']
run.mkdir(parents=True)
cmd=[str(B/'.venv/bin/python'),'-u',str(R/'supermimic/train_bounded_backflip.py'),
     '--motion_dir',str(R/'data/backflip'),'--log_dir',str(run),'--seed','137','--max_policy_minutes','90']
if phase=='smoke':cmd+=['--iterations','2','--wm_only_iters','0','--log_every','1']
else:cmd+=['--iterations','320000','--wm_only_iters','500','--log_every','50','--ckpt_every','500']
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
record={'phase':phase,'command':cmd,'run':str(run),'source_commit':commit,'start_unix':time.time(),
        'selected_gpu':env['CUDA_VISIBLE_DEVICES'],'policy_budget_minutes':90,'external_pretrained':False,'resume':False,'initialization':'fresh seed137'}
with (B/('smoke.log' if phase=='smoke' else 'train_v14.log')).open('w') as f:
    p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    record['trainer_pid']=p.pid;(B/(phase+'_launch.json')).write_text(json.dumps(record,indent=2))
    try:code=p.wait(timeout=1800 if phase=='smoke' else 10800)
    except subprocess.TimeoutExpired:
        import signal
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        code=124
record.update(exit_code=code,end_unix=time.time())
csv_path=run/'train_log.csv';rows=list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
record['finite_metrics']=bool(rows) and all(math.isfinite(float(v)) for row in rows for v in row.values())
record['last_logged_iteration']=int(rows[-1]['iteration']) if rows else None
(B/(phase+'_result.json')).write_text(json.dumps(record,indent=2))
with (B/'results.tsv').open('a') as f:
    f.write(f'{phase}\t{commit}\t{record["last_logged_iteration"]}\t'+('completed' if code==0 else 'crash')+f'\tfull reference baseline; see {phase}_result.json\n')
print(json.dumps(record),flush=True)
if phase=='main' and code==0:
    cmd=[str(B/'.venv/bin/python'),'-u',str(ROOT/'scripts/evaluate_reference_backflip.py'),'--checkpoint',str(run/'last.pt'),'--output',str(run/'final_evaluation')]
    with (B/'eval_v14_final.log').open('w') as f:
        p=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
    (B/'final_evaluation_status.json').write_text(json.dumps({'exit_code':p.returncode,'completed_unix':time.time(),'checkpoint':str(run/'last.pt')},indent=2))
    p.check_returncode()
    if p.returncode==0:
        for script,log in [('analyze_takeoff_forces.py','takeoff_v14_final.log'),('render_blue_floor.py','render_v14_final.log')]:
            with (B/log).open('w') as f:
                q=subprocess.run([str(B/'.venv/bin/python'),'-u',str(ROOT/'scripts'/script),str(run/'final_evaluation')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
            print(script,'exit_code',q.returncode,flush=True)
            q.check_returncode()

if code != 0:
    raise SystemExit(code)
