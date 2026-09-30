import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import uuid
from collections import Counter
from datetime import date
import numpy as np
import httpx
from .clients import Client
from .environment import CONFIG, ACTIONS, make_env, snapshot, encode_system
from .oracle import sweep, grade

ROOT = Path(__file__).resolve().parent.parent

def seeds_arg(value):
    if '..' in value:
        a,b = map(int,value.split('..')); return list(range(a,b+1))
    return list(map(int,value.split(',')))

def atomic_write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def persist(data, path, no_git):
    atomic_write(path,data)
    if no_git: return
    subprocess.run(['git','add','--',str(path)],cwd=ROOT,check=True)
    changed = subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode
    if changed:
        subprocess.run(['git','commit','-m','Record completed benchmark episode'],cwd=ROOT,check=True)
    if subprocess.check_output(['git','remote'],cwd=ROOT,text=True).strip():
        subprocess.run(['git','push','-u','origin','main'],cwd=ROOT,check=True)
    else:
        print('Push skipped: no git remote configured.',flush=True)

def pricing(model):
    basis = dict(date=str(date.today()),unit='USD per million tokens')
    if model == 'mock': return dict(**basis,input=0,output=0,source='local mock, no billable tokens')
    if model == 'baseline': return dict(**basis,input=1,output=5,source='https://www.anthropic.com/news/claude-haiku-4-5')
    if 'JEV_INPUT_USD_PER_MILLION' not in os.environ:
        raise ValueError('Set JEV_INPUT_USD_PER_MILLION from your dated billing agreement; no guessed price.')
    return dict(**basis,input=float(os.environ['JEV_INPUT_USD_PER_MILLION']),output=0,
                source=os.environ.get('JEV_PRICING_SOURCE','user supplied account pricing; output free per official OpenAPI'))

def episode(args,seed,number,role,price):
    env = make_env(seed)
    client = Client(args.model,seed)
    run_id = f'{args.model}-{role}-{seed}-{number}-{uuid.uuid4().hex[:8]}'
    artifact = ROOT/'artifacts'/f'{run_id}.jsonl'
    artifact.parent.mkdir(exist_ok=True)
    score, steps, lost, previous = 0.,0,0,None
    decisions, accuracies, adjusted, errors = [],[],[],Counter()
    mixed_safe, mixed_correct, unsafe_count, safe_count = 0,0,0,0
    initial_frame = env.ale.getEpisodeFrameNumber()
    try:
        with artifact.open('w') as trace:
            while not env.ale.game_over() and (not args.max_steps or steps < args.max_steps):
                state, system = snapshot(env,score,previous)
                frame = env.ale.getEpisodeFrameNumber()
                labels = sweep(env,system,args.horizon,args.max_delay)
                # Only this serialization crosses the client boundary. Never pass labels or emulator state.
                decision = client.decide(json.dumps(state,sort_keys=True,separators=(',',':')))
                delay = math.ceil(decision['latency_ms']*60/1000)
                ordinary, adjusted_ok = grade(labels,decision['action'],delay,not decision['fallback'])
                accuracies.append(ordinary); adjusted.append(adjusted_ok)
                decisions.append(decision); errors.update(decision['errors_by_status'])
                safety = [not v['outcomes'][0]['life_lost'] for v in labels.values()]
                correct = [v['correct'] for v in labels.values()]
                mixed_safe += len(set(safety))>1; mixed_correct += len(set(correct))>1
                safe_count += sum(safety); unsafe_count += 6-sum(safety)
                before = env.ale.lives()
                # Online trajectory uses ordinary Gym steps. Deadline scoring is a paired, frozen-state
                # evaluation of wall latency, not a claim that the online game ran asynchronously.
                _,reward,terminated,truncated,_ = env.step(ACTIONS.index(decision['action']))
                score += reward; steps += 1; lost += max(0,before-env.ale.lives())
                trace.write(json.dumps(dict(step=steps,frame=frame,state=state,
                    emulator_state_b64=encode_system(system),oracle=labels,decision=decision,
                    latency_frames=delay,ordinary_correct=ordinary,deadline_correct=adjusted_ok))+'\n')
                previous = state
        latencies = [d['latency_ms'] for d in decisions]
        n = max(1,len(decisions))
        tin,tout = sum(d['input_tokens'] for d in decisions),sum(d['output_tokens'] for d in decisions)
        confidence = [d['confidence'] for d in decisions]
        bins = []
        for lo in np.arange(0,1,0.1):
            indices = [i for i,c in enumerate(confidence) if lo <= c < lo+0.1 or (lo>0.89 and c==1)]
            if indices:
                bins.append(dict(lower=float(lo),count=len(indices),confidence=float(np.mean([confidence[i] for i in indices])),accuracy=float(np.mean([accuracies[i] for i in indices]))))
        result = dict(run_id=run_id,backend=args.model,role=role,seed=seed,episode=number,score=score,steps=steps,
            frames=env.ale.getEpisodeFrameNumber()-initial_frame,lives_lost=lost,
            terminated=env.ale.game_over(with_truncation=False),
            truncated=bool(env.ale.game_truncated() or (args.max_steps and steps>=args.max_steps and not env.ale.game_over())),
            truncation_reason='max_steps' if args.max_steps and steps>=args.max_steps and not env.ale.game_over() else None,
            model_calls=sum(d['attempts'] for d in decisions),input_tokens=tin,output_tokens=tout,
            latency_ms_p50=float(np.percentile(latencies,50)),latency_ms_p95=float(np.percentile(latencies,95)),latency_ms_total=sum(latencies),
            errors_by_status=dict(errors),retries=sum(d['retries'] for d in decisions),fallback_actions=sum(d['fallback'] for d in decisions),
            mean_confidence=float(np.mean(confidence)),ordinary_accuracy=sum(accuracies)/n,deadline_adjusted_accuracy=sum(adjusted)/n,
            timeout_count=sum(d['timeout'] for d in decisions),invalid_output_count=sum(d['invalid'] for d in decisions),
            beyond_sweep_count=sum(math.ceil(d['latency_ms']*60/1000)>args.max_delay for d in decisions),
            cost_usd=(tin*price['input']+tout*price['output'])/1e6,pricing_basis=price,
            brier_score=float(np.mean([(c-int(a))**2 for c,a in zip(confidence,accuracies)])),
            calibration_bins=bins,ece=sum(b['count']*abs(b['confidence']-b['accuracy']) for b in bins)/n,
            mixed_safety_snapshots=mixed_safe,mixed_correctness_snapshots=mixed_correct,
            safe_action_labels=safe_count,unsafe_action_labels=unsafe_count,
            trace=str(artifact.relative_to(ROOT)),served_models=sorted({d['served_model'] for d in decisions if d['served_model']}))
        return result
    finally:
        client.close(); env.close()

def run(args):
    if args.horizon != 48: raise ValueError('This protocol fixes horizon=48 to match the shared question.')
    if not 0 <= args.max_delay < args.horizon: raise ValueError('Require 0 <= max-delay < horizon')
    if args.episodes_per_seed < 1 or args.max_steps < 0 or not args.seeds: raise ValueError('Invalid run length')
    price = pricing(args.model)
    roles = ['decider','baseline'] if args.model=='mock' and args.mock_role=='both' else [args.mock_role if args.model=='mock' else ('decider' if args.model=='jev' else 'baseline')]
    protocol = dict(**CONFIG,oracle_horizon_frames=args.horizon,oracle_max_delay_frames=args.max_delay,
                    oracle_delay_step_frames=1,oracle_policy='NOOP delay then hold action to common horizon',
                    latency_policy='offline counterfactual; online immediate action',fps=60,
                    beyond_sweep_policy='conservatively fail deadline metric; report count')
    path = Path(args.output).resolve()
    data = json.loads(path.read_text()) if path.exists() else dict(schema_version=2,models=[],config=protocol,runs=[],baseline=dict(model='claude-haiku-4-5',runs=[]))
    if data['config'] != protocol: raise ValueError('Existing results use different protocol; choose --output')
    for role in roles:
        target = data['runs'] if role=='decider' else data['baseline']['runs']
        if any(r['backend']!=args.model for r in target): raise ValueError('Do not mix mock and real runs; use --output results-real.json')
        for seed in args.seeds:
            for number in range(args.episodes_per_seed):
                result = episode(args,seed,number,role,price)
                target.append(result)
                requested = {'mock':'mock-random-v1','jev':'jev-latest','baseline':'claude-haiku-4-5'}[args.model]
                metadata = dict(role=role,provider={'mock':'mock','jev':'typesafe','baseline':'anthropic'}[args.model],
                    requested_model=requested,served_model=result['served_models'][-1] if result['served_models'] else None,
                    sdk_package='httpx' if args.model!='mock' else 'bench',sdk_version=httpx.__version__ if args.model!='mock' else '0.1.0')
                if metadata not in data['models']: data['models'].append(metadata)
                if role=='baseline': data['baseline']['model']=requested
                persist(data,path,args.no_git)
                print(json.dumps(result),flush=True)

def analyze(args):
    data=json.loads(Path(args.output).read_text())
    for role,runs in [('decider',data['runs']),('baseline',data['baseline']['runs'])]:
        if not runs: continue
        n=sum(r['steps'] for r in runs)
        weighted=lambda key: sum(r[key]*r['steps'] for r in runs)/n
        print(f"{role} ({','.join(sorted({r['backend'] for r in runs}))} | {len(runs)} episodes, {n} decisions")
        print(f"  ordinary={weighted('ordinary_accuracy'):.4f} deadline-adjusted={weighted('deadline_adjusted_accuracy'):.4f}")
        print(f"  mean latency={sum(r['latency_ms_total'] for r in runs)/n:.3f}ms; mean episode p50={np.mean([r['latency_ms_p50'] for r in runs]):.3f}ms p95={np.mean([r['latency_ms_p95'] for r in runs]):.3f}ms")
        print(f"  cost=${sum(r['cost_usd'] for r in runs):.6f}; confidence={weighted('mean_confidence'):.4f}; Brier={weighted('brier_score'):.4f}; mean episode ECE={weighted('ece'):.4f}")
        print(f"  mixed safety snapshots={sum(r['mixed_safety_snapshots'] for r in runs)}; unsafe actions={sum(r['unsafe_action_labels'] for r in runs)}")

def main():
    parser=argparse.ArgumentParser(description='The Last Safe Millisecond')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run'); p.add_argument('--model',choices=['mock','jev','baseline'],required=True)
    p.add_argument('--seeds',type=seeds_arg,default=seeds_arg('1..5')); p.add_argument('--episodes-per-seed',type=int,default=1)
    p.add_argument('--mock-role',choices=['decider','baseline','both'],default='both')
    p.add_argument('--max-steps',type=int,default=0,help='0 runs until game over; positive is explicitly truncated smoke test')
    p.add_argument('--horizon',type=int,default=48); p.add_argument('--max-delay',type=int,default=12)
    p.add_argument('--no-git',action='store_true'); p.add_argument('--output',default=str(ROOT/'results.json'))
    p=sub.add_parser('analyze'); p.add_argument('--output',default=str(ROOT/'results.json'))
    args=parser.parse_args(); (run if args.command=='run' else analyze)(args)

if __name__=='__main__': main()
