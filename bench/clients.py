import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
import httpx
from .environment import ACTIONS

QUESTION = ('Choose one Space Invaders action. First avoid losing a life, then maximize score. '
            'The action will be held over a 48-frame lookahead; frames run at 60 Hz. '
            'Choices: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE.')

@dataclass
class Decision:
    action: str = 'NOOP'
    confidence: float = 0.0
    latency_ms: float = 0.0
    raw_response: object = None
    input_tokens: int = 0
    output_tokens: int = 0
    errors_by_status: dict = field(default_factory=dict)
    retries: int = 0
    attempts: int = 0
    fallback: bool = False
    timeout: bool = False
    invalid: bool = False
    served_model: str | None = None
    probabilities: dict | None = None

def _usage(body):
    u = body.get('usage') or {}
    inp = u.get('input_tokens', u.get('prompt_tokens', u.get('input', 0)))
    out = u.get('output_tokens', u.get('completion_tokens', u.get('output', 0)))
    return int(inp or 0), int(out or 0)

class Client:
    def __init__(self, model, seed=1, timeout=10, retries=2, transport=None, horizon=180):
        self.question = QUESTION.replace('48-frame', f'{horizon}-frame')
        self.model, self.rng, self.timeout, self.max_retries = model, random.Random(seed), timeout, retries
        self.http = httpx.Client(timeout=timeout, transport=transport)
        self.requested = {'mock': 'mock-random-v1', 'jev': 'jev-latest',
                          'baseline': os.environ.get('BASELINE_MODEL', 'anthropic/claude-haiku-4.5')}[model]
        if model != 'mock':
            self.key = os.environ.get('TYPESAFE_API_KEY' if model == 'jev' else 'OPENROUTER_API_KEY')
            if not self.key:
                raise ValueError('Missing TYPESAFE_API_KEY' if model == 'jev' else 'Missing OPENROUTER_API_KEY')

    def close(self):
        self.http.close()

    def decide(self, state_json):
        start = time.perf_counter()
        d = Decision()
        if self.model == 'mock':
            d.action = self.rng.choice(ACTIONS)
            d.confidence = 1 / 6
            d.served_model = self.requested
            d.raw_response = dict(action=d.action, confidence=d.confidence)
            d.attempts = 1
        else:
            for attempt in range(self.max_retries + 1):
                d.attempts += 1
                d.retries = attempt
                try:
                    if self.model == 'jev':
                        response = self.http.post('https://api.typesafe.ai/v1/systemone',
                            headers={'Authorization': f'Bearer {self.key}'},
                            json=dict(model=self.requested, state=state_json,
                                      questions={'move': dict(type='choice', instructions=self.question,
                                                             criteria={a: a for a in ACTIONS})}))
                    else:
                        prompt = (self.question + '\nState:\n' + state_json + '\n\n'
                                  'Reply with a JSON object only, no prose: '
                                  '{"action": "<one of ' + ', '.join(ACTIONS) + '>", "confidence": <number 0 to 1>}.')
                        response = self.http.post('https://openrouter.ai/api/v1/chat/completions',
                            headers={'Authorization': f'Bearer {self.key}'},
                            json=dict(model=self.requested,
                                      messages=[dict(role='user', content=prompt)],
                                      response_format={'type': 'json_object'},
                                      temperature=0))
                    response.raise_for_status()
                    body = response.json()
                    d.raw_response = body
                    d.served_model = body.get('model') or self.requested
                    d.input_tokens, d.output_tokens = _usage(body)
                    if self.model == 'jev':
                        answer = body['answers']['move']
                        action, confidence = answer['choice'], answer['confidence']
                        d.probabilities = answer.get('probabilities')
                    else:
                        content = body['choices'][0]['message']['content']
                        if isinstance(content, list):
                            content = ''.join(b.get('text', '') for b in content if isinstance(b, dict))
                        answer = json.loads(content)
                        action, confidence = answer['action'], answer['confidence']
                    if (action not in ACTIONS or isinstance(confidence, bool) or
                            not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or
                            not 0 <= confidence <= 1):
                        raise ValueError('Invalid action/confidence')
                    d.action, d.confidence = action, float(confidence)
                    break
                except httpx.HTTPStatusError as e:
                    status = str(e.response.status_code)
                    retryable = e.response.status_code in (429, 500, 502, 503, 504)
                except httpx.TimeoutException:
                    status, retryable = 'timeout', True
                    d.timeout = True
                except httpx.RequestError:
                    status, retryable = 'network', True
                except (ValueError, KeyError, TypeError):
                    status, retryable = 'invalid_output', False
                    d.invalid = True
                d.errors_by_status[status] = d.errors_by_status.get(status, 0) + 1
                if not retryable or attempt == self.max_retries:
                    d.fallback = True
                    break
                time.sleep(0.1 * 2 ** attempt)
        d.latency_ms = (time.perf_counter() - start) * 1000
        return asdict(d)
