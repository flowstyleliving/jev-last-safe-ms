# Validation report — 2026-09-29

All validation used the local mock backend. No real JEV or LLM inference calls were made.

Python 3.11.15; Gymnasium 1.3.0; ale-py 0.12.1. The ALE wheel bundled the ROM; no AutoROM workaround was needed.

## Completed campaign

Five seeds in each schema slot; ten complete, naturally terminated games; 5,604 total decisions. Mock seed pairs have identical trajectories and oracle metrics. Model metadata is explicitly mock.

| Seed | Score per slot | Decisions per slot | Mixed-safety snapshots per slot |
| --- | ---: | ---: | ---: |
| 1 | 60 | 484 | 101 |
| 2 | 105 | 495 | 69 |
| 3 | 180 | 599 | 113 |
| 4 | 335 | 594 | 105 |
| 5 | 180 | 630 | 111 |

Each slot: ordinary accuracy **39.79%**, deadline-adjusted accuracy **23.55%**, 499 mixed-safety snapshots, 190 snapshots with a finite deadline inside the sweep, 12,688 safe and 4,124 unsafe immediate-action labels. All API costs and token usage are zero. No retries, timeouts, invalid responses, or fallbacks occurred in these mock games.

## Checks performed

- Ten automated tests passed (ALE state/RNG restoration, deterministic mocks, decoded state, actual mixed safety, delay holes, API wire contracts via MockTransport, retry/invalid/timeout accounting, and observation-only client inputs).
- JSON Schema validation passed for schema version 2.
- Every trace length matches its run; both accuracy metrics recompute exactly from traces.
- Three saved emulator states reproduce all recorded action/delay outcomes exactly.
- All five paired mock trajectories match on score, steps, frames, lives lost and both correctness metrics.

## Reproduce

```sh
source .venv/bin/activate
python -m pytest -q
python -m bench analyze
python -m bench verify --min-seeds 5
python -m bench run --model mock --seeds 1..5 --episodes-per-seed 1 --workers 5 --output results-new.json
```

The verification command uses local full traces under `artifacts/`. These large files are excluded from Git. The committed `validation-evidence.json` and `tests/fixtures/natural_threat.json` preserve reviewable snapshots, including emulator state. Regenerating a campaign recreates full traces.

## Interpretation

The initial 48-frame lookahead lacked action-dependent life-loss labels. It is preserved in `results-initial-48-frame.json`. The final protocol uses 180 frames and tests each delay from 0 through 12. Capped deadlines are marked censored; delays beyond the sweep conservatively fail and are counted separately. At seed 1, step 75, RIGHT and RIGHTFIRE have a last tested safe delay of 1 frame. At step 76, those actions are unsafe immediately but safe after a 10-frame wait, demonstrating non-monotonic safety.

Online scores use immediate four-frame actions; latency correctness is an offline counterfactual at each online snapshot. These results demonstrate a working harness, not a JEV-versus-Claude performance comparison. Real HTTP adapters passed simulated-contract tests but have not been exercised against live inference.

Each completed game was committed on `main`. No Git remote was configured, so no push was possible.
