"""Build a reviewable, non-algorithmic adapter; retain pristine upstream train.py."""
from pathlib import Path
import hashlib,json,subprocess
from common import B,R,ROOT
B.mkdir(parents=True,exist_ok=True)
p=R/'supermimic/train.py'; original=p.read_text(); s=original
def replace_once(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new)
replace_once('parser.add_argument("--seed", type=int, default=0)',
    'parser.add_argument("--seed", type=int, default=0)\nparser.add_argument("--max_policy_minutes", type=float, default=90.0)')
replace_once('motion = MotionLib.from_dir(args.motion_dir, kin, dev)',
    'import glob\nmotion = MotionLib(sorted(glob.glob(os.path.join(args.motion_dir, "*.csv"))), kin, dev, src_fps=50.0)')
replace_once('    for it in range(start_it, args.iterations):',
    '    policy_budget_start = None\n    for it in range(start_it, args.iterations):\n'
    '        if it >= args.wm_only_iters and policy_budget_start is None:\n'
    '            policy_budget_start = time.monotonic()\n'
    '            print("POLICY_BUDGET_START", flush=True)')
replace_once('        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:',
    '        budget_done = policy_budget_start is not None and time.monotonic() - policy_budget_start >= 60 * args.max_policy_minutes\n'
    '        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations or budget_done:')
s += '\n        if budget_done:\n            print("BOUNDED_TRAINING_FINISHED", it + 1, flush=True)\n            break\n'
(R/'supermimic/train_bounded_backflip.py').write_text(s)
subprocess.run(['git','diff','--no-index',str(p),str(R/'supermimic/train_bounded_backflip.py')],stdout=(B/'training_adapter.diff').open('w'))
(B/'training_adapter_manifest.json').write_text(json.dumps({
    'upstream_sha256':hashlib.sha256(original.encode()).hexdigest(),
    'adapter_sha256':hashlib.sha256(s.encode()).hexdigest(),
    'changes':['source fps50 via existing constructor','wall-time cap with final checkpoint'],
    'algorithm_changes':False},indent=2))
compile(s,str(R/'supermimic/train_bounded_backflip.py'),'exec')
print('ADAPTER_READY')
