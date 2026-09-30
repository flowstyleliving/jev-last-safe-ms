"""Key-free structural validation and replay of recorded counterfactual evidence."""
import base64
import json
from pathlib import Path
import ale_py
import jsonschema
from .environment import make_env
from .oracle import sweep

ROOT=Path(__file__).resolve().parent.parent

def verify(path, min_seeds=2, evidence_path=None):
    data=json.loads(Path(path).read_text())
    jsonschema.validate(data,json.loads((ROOT/'results.schema.json').read_text()))
    summary={}
    examples=[]
    for role,runs in [('decider',data['runs']),('baseline',data['baseline']['runs'])]:
        assert len({r['seed'] for r in runs})>=min_seeds, f'{role}: not enough seeds'
        mixed=0
        finite=0
        for run in runs:
            count=0
            for line in (ROOT/run['trace']).open():
                row=json.loads(line); count+=1
                assert len(row['state']['ram'])==128
                assert all(isinstance(v,int) and 0<=v<=255 for v in row['state']['ram'])
                labels=row['oracle']
                safety=[not v['outcomes'][0]['life_lost'] for v in labels.values()]
                mixed_here=len(set(safety))>1
                mixed+=mixed_here
                bounded=any(0<=v['deadline_frames']<data['config']['oracle_max_delay_frames'] for v in labels.values())
                finite+=bounded
                if mixed_here and not any(x['role']==role for x in examples):
                    examples.append(dict(role=role,seed=run['seed'],**row))
                if bounded and not any(x.get('example_type')=='finite_deadline' for x in examples):
                    examples.append(dict(role=role,seed=run['seed'],example_type='finite_deadline',**row))
            assert count==run['steps'], f'trace length mismatch: {run["run_id"]}'
        assert mixed>0, f'{role}: no action-dependent safety labels'
        summary[role]=dict(episodes=len(runs),seeds=sorted({r['seed'] for r in runs}),
                           mixed_safety_snapshots=mixed,finite_deadline_snapshots=finite,
                           safe_action_labels=sum(r['safe_action_labels'] for r in runs),
                           unsafe_action_labels=sum(r['unsafe_action_labels'] for r in runs))
    for row in examples:
        env=make_env(row['seed'])
        try:
            system=ale_py.ALEState(base64.b64decode(row['emulator_state_b64']))
            replay=sweep(env,system,data['config']['oracle_horizon_frames'],data['config']['oracle_max_delay_frames'])
            assert replay==row['oracle'], 'serialized counterfactual replay mismatch'
        finally: env.close()
    evidence=dict(schema_version=2,source_results=str(path),validation='schema, full trace counts, natural mixed labels, serialized ALE replay',summary=summary,examples=examples)
    if evidence_path: Path(evidence_path).write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print(f'PASS: schema, all trace lengths and {len(examples)} exact counterfactual replays')
    return evidence
