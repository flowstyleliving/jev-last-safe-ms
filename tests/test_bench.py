import json
import httpx
import pytest
from bench.environment import make_env, snapshot, ACTIONS, decode
from bench.oracle import sweep, grade
from bench.clients import Client


def test_replay_rng_and_oracle_noninterference():
    env=make_env(7)
    for i in range(100): env.step(i%6)
    _,system=snapshot(env,0)
    expected=env.step(4)
    env.ale.restoreSystemState(system)
    labels=sweep(env,system,max_delay=2)
    actual=env.step(4)
    assert actual[0].tolist()==expected[0].tolist()
    assert actual[1:4]==expected[1:4]
    env.ale.restoreSystemState(system)
    assert labels==sweep(env,system,max_delay=2)
    assert set(labels)==set(ACTIONS)
    env.close()


def test_nonmonotonic_and_out_of_range_deadlines():
    labels={'NOOP':dict(correct=True,safe_delays=[0,1,3],deadline_frames=3)}
    assert grade(labels,'NOOP',1)==(True,True)
    assert grade(labels,'NOOP',2)==(True,False)
    assert grade(labels,'NOOP',4)==(True,False)
    assert grade(labels,'NOOP',0,False)==(False,False)


def test_mock_determinism_and_no_keys(monkeypatch):
    monkeypatch.delenv('TYPESAFE_API_KEY',raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY',raising=False)
    a,b=Client('mock',1),Client('mock',1)
    assert [a.decide('{}')['action'] for _ in range(20)]==[b.decide('{}')['action'] for _ in range(20)]
    a.close(); b.close()


@pytest.mark.parametrize('backend',['jev','baseline'])
def test_api_contract_and_usage_without_network(monkeypatch,backend):
    monkeypatch.setenv('TYPESAFE_API_KEY','test-not-a-real-key')
    monkeypatch.setenv('ANTHROPIC_API_KEY','test-not-a-real-key')
    def respond(request):
        body=json.loads(request.content)
        assert 'oracle' not in request.content.decode()
        answer=dict(action='LEFT',confidence=.8)
        if backend=='jev':
            assert str(request.url)=='https://api.typesafe.ai/v1/systemone'
            assert body['model']=='jev-latest'
            assert set(body['questions']['move']['criteria'])==set(ACTIONS)
            content={'answers':{'move':dict(type='choice',choice='LEFT',confidence=.8,probabilities={a:1/6 for a in ACTIONS})}}
        else:
            assert body['output_config']['format']['type']=='json_schema'
            content={'content':[dict(type='text',text=json.dumps(answer))]}
        return httpx.Response(200,json=dict(**content,model='returned-version',usage=dict(input_tokens=100,output_tokens=10)))
    client=Client(backend,transport=httpx.MockTransport(respond))
    d=client.decide('{"ram":[]}')
    assert (d['action'],d['served_model'],d['input_tokens'],d['output_tokens'])==('LEFT','returned-version',100,10)
    assert not d['fallback']
    client.close()


def test_retries_and_invalid_response(monkeypatch):
    monkeypatch.setenv('TYPESAFE_API_KEY','test')
    responses=iter([httpx.Response(429),httpx.Response(200,json={'model':'x','usage':dict(input_tokens=3,output_tokens=4),'answers':{'move':dict(choice='BAD',confidence=.3,probabilities={})}})])
    c=Client('jev',transport=httpx.MockTransport(lambda _:next(responses)))
    d=c.decide('{}')
    assert d['fallback'] and d['invalid'] and d['action']=='NOOP'
    assert d['retries']==1 and d['attempts']==2
    assert d['errors_by_status']=={'429':1,'invalid_output':1}
    assert d['input_tokens']==3
    c.close()


def test_decoder():
    r=[0]*128; r[18:24]=[63]*6; r[28]=40; r[81]=70; r[83]=50
    s=decode(r,3,20)
    assert len(s['ram'])==128 and s['ship']['x']==39
    assert sum(map(sum,s['alien_grid']))==36
    assert s['bullets'][0]['y']==143 and s['bullets'][0]['vy'] is None
    r[81]+=4
    assert decode(r,3,20,s)['bullets'][0]['vy']==2


def test_natural_snapshot_has_action_dependent_life_loss():
    import base64
    import ale_py
    from pathlib import Path
    row=json.loads((Path(__file__).parent/'fixtures/natural_threat.json').read_text())
    env=make_env(1)
    system=ale_py.ALEState(base64.b64decode(row['emulator_state_b64']))
    labels=sweep(env,system,horizon=180,max_delay=12)
    safe=[not x['outcomes'][0]['life_lost'] for x in labels.values()]
    assert any(safe) and not all(safe)
    assert any(x['correct'] for x in labels.values()) and not all(x['correct'] for x in labels.values())
    assert any(x['deadline_frames']>=0 for x in labels.values())
    assert 0 not in labels['RIGHT']['safe_delays']
    assert 10 in labels['RIGHT']['safe_delays']
    env.close()


def test_timeout_accounting(monkeypatch):
    monkeypatch.setenv('TYPESAFE_API_KEY','test')
    def timeout(request): raise httpx.ReadTimeout('simulated',request=request)
    c=Client('jev',retries=0,transport=httpx.MockTransport(timeout))
    d=c.decide('{}')
    assert d['timeout'] and d['fallback'] and not d['invalid']
    assert d['errors_by_status']=={'timeout':1} and d['attempts']==1
    c.close()


def test_runner_never_sends_oracle_to_client(monkeypatch,tmp_path):
    from argparse import Namespace
    from bench import runner
    original=runner.Client
    class SpyClient(original):
        def decide(self,state_json):
            state=json.loads(state_json)
            assert len(state['ram'])==128
            assert not {'oracle','deadline_frames','correct','outcomes','emulator_state_b64'} & state.keys()
            return super().decide(state_json)
    monkeypatch.setattr(runner,'Client',SpyClient)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    args=Namespace(model='mock',max_steps=1,horizon=2,max_delay=0)
    result=runner.episode(args,3,0,'decider',runner.pricing('mock'))
    assert result['steps']==1 and result['truncated'] and result['cost_usd']==0
