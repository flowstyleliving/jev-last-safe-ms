# The Last Safe Millisecond

MK's UFA JEV Bake-Off entry: a runnable, seeded Atari benchmark for whether a correct decision arrives while it can still avoid life loss. Python 3.11; no keys needed for mock validation.

## Regenerate from scratch

```sh
git clone <your-repository-url> jev-last-safe-ms
cd jev-last-safe-ms
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
# Existing results are append-only: move results.json aside for a clean campaign.
python -m bench run --model mock --seeds 1..2 --episodes-per-seed 1
python -m bench analyze
python -m bench verify --min-seeds 2
# Five seeds in each of the decider and baseline slots (ten complete mock games):
python -m bench run --model mock --seeds 1..5 --episodes-per-seed 1 --output results-mock-new.json
```

The default mock run fills **both** schema slots, explicitly marked `provider: mock`. It does not impersonate JEV or Claude. `--mock-role decider|baseline` selects one slot. No model API is contacted. There is no wall-clock sleeping in the mock. Measured mock latency varies with the machine; its actions, observations and oracle outcomes are seeded and reproducible under the pinned dependencies. `--max-steps 100` makes an explicitly truncated smoke test; default 0 completes the game. Expect counterfactual sweeps to dominate runtime. `--workers 4` uses independent worker processes; only the parent writes results. Keep workers=1 for real latency comparisons to avoid local CPU contention.

Each completed episode atomically appends to the selected results file, commits it, and pushes `origin/main` when a remote exists. With no remote, it commits and prints an explicit push-skipped message. Set a remote with `git remote add origin <your-url>`. `--no-git` disables commit/push for experiments. Do not run simultaneous writers to the same results file. Interrupted episodes leave partial traces but do not append incomplete runs.

### ALE / ROM installation

Validated on macOS arm64 with Python 3.11.15, Gymnasium 1.3.0 and ale-py 0.12.1. **This ALE wheel includes the Space Invaders ROM**; no AutoROM download or license command was necessary. `gym.register_envs(ale_py)` is called explicitly. On an older installation missing ROMs, upgrade to the pinned requirements first. If retaining an older ALE, the historical fix is `pip install 'autorom[accept-rom-license]'`, then `AutoROM --accept-license`, then that version's `ale-import-roms <download-directory>` utility. The tested reproduction path uses the bundled current wheel.

## Protocol and ground truth

`ALE/SpaceInvaders-v5`, RAM observations, six minimal actions, four emulator frames per decision, sticky action probability 0.25, 108000-frame episode limit. No Gym wrappers are active (`unwrapped` environment). Each episode resets Gym/ALE and the mock RNG to the requested seed. A frameskip-four Gym step invokes the actual ALE transition four times. All oracle rollouts use the same ALE one-frame transition and minimal action mapping.

At every decision, capture all 128 RAM bytes and **cloneSystemState**, including emulator RNG, not RAM alone. Each branch restores that same clone; a `finally` block restores the online environment after the oracle. Traces store the serialized clone as base64, so branches can be replayed with the same ALE version. RAM alone cannot reproduce the emulator's hidden state or sticky-action RNG.

Validation found that 48 frames often sees the life-counter decrement only after an unavoidable collision. A natural 180-frame rollout produces action-dependent life-loss outcomes, so 180 frames (three game seconds) is the default; `--horizon` can change it with the shared model question updated accordingly. The initial four-game, 48-frame experiment is preserved in `results-initial-48-frame.json`.

The finite-horizon policy is explicit: over a common **180-frame horizon**, issue NOOP for `d` frames, then hold the candidate action for the remaining frames. Exhaustively test every integer delay `d=0..12` for each action. This is a conditional, finite-horizon counterfactual oracle, not an optimal full-game policy or a guarantee against later deaths. Sticky action randomness is fixed by the snapshot, not averaged over new samples. The question describes the same held-action horizon to both models.

- Outcomes record cumulative lives lost, alien population decreases, and emulator reward/score delta. Alien kills count decreases during the rollout, including the last alien before a new wave; satellite points are not mislabelled as alien kills.
- `correct` is a tie-inclusive lexicographic optimum at zero delay: minimize lives lost, then maximize score delta. If every action loses a life, the least-bad action can be ordinarily correct but cannot be deadline-correct.
- `deadline_frames` is the largest tested delay with no life loss, or -1 if none. `deadline_right_censored` means the action was still safe at the sweep limit; it is **not** a claim that it becomes unsafe on the next frame.
- Safety need not be monotonic. Store **all** safe delays, all outcomes, and a conservative `safe_prefix_frames`. Deadline-adjusted correctness requires ordinary correctness and membership of `ceil(latency_ms * 60 / 1000)` in the safe-delay set. A safety hole fails even if an even later execution succeeds.
- Latencies beyond the swept range fail conservatively and are separately counted. Use `--max-delay 179` for a wider sweep and a separate results file. Comparisons require the same protocol; mixed configurations are rejected.
- The online game applies each decision immediately with Gym's four-frame step. Wall latency is evaluated **offline on that same snapshot**. Scores are ordinary online scores, not latency-degraded asynchronous scores. Oracle compute time is excluded from model latency. This isolates response timeliness from oracle computation overhead.

The model receives only the same sorted JSON observation and shared six-choice question. Oracle labels, future rewards, emulator clones, deadlines, and prior oracle information never cross the client boundary.

## State encoding

The decoder implements the [OCAtari Space Invaders RAM map](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/spaceinvaders.py): player x at 28; alien occupancy at 18–23, origin at 26/16; shields at 43–69; enemy bullet positions at 81–84; player bullet at 85/87. Alien occupancy is a top-to-bottom, left-to-right 6×6 grid with origin and spacing. Shields retain their nine-byte bitmaps. Bullet velocities are finite differences in pixels/frame, with `null` at spawn, disappearance or discontinuity; they are estimates, not hidden physics values. Lives and accumulated score are authoritative ALE/transition values, with RAM lives and raw BCD score bytes retained for audit. The initial state has no velocity history.

## Real model runs (not executed during validation)

```sh
export TYPESAFE_API_KEY=...
# Use the price on your agreement; never guess a JEV rate.
export JEV_INPUT_USD_PER_MILLION=...
export JEV_PRICING_SOURCE='account agreement, YYYY-MM-DD'
python -m bench run --model jev --seeds 1..5 --episodes-per-seed 1 --output results-real.json
export ANTHROPIC_API_KEY=...
python -m bench run --model baseline --seeds 1..5 --episodes-per-seed 1 --output results-real.json
python -m bench analyze --output results-real.json
```

JEV (`typesafe-ai/jev`) requests `jev-latest` through `POST https://api.typesafe.ai/v1/systemone`, using the [official OpenAPI contract](https://api.typesafe.ai/docs): text state, named choice with six criteria; parse the returned choice, probabilities, confidence, usage, and actual served model. The baseline requests `claude-haiku-4-5` through Anthropic's [JSON structured-output Messages API](https://platform.claude.com/docs/en/build-with-claude/structured-outputs). It supplies the identical question and state, with an enum-constrained action and numeric confidence. Both use HTTPX directly, so metadata truthfully records HTTPX's installed version, **not** unused `typesafe-sdk 0.7.2` or `system-one-adapter 0.2.1` packages from the example schema. API adapters are tested with HTTPX MockTransport, not live credentials.

Retries cover 429, transient server errors, timeouts and network failures (two retries; exponential backoff). Total latency includes retries, response parsing, and validation. Invalid outputs and exhausted errors fall back to NOOP, receive no correctness credit, and are counted. HTTP status histograms, attempts, retries, timeout/invalid decision counts, raw successful responses, token usage and served IDs are retained. Failed requests without usage cannot have their potentially billed tokens inferred; reported cost is based on returned usage only.

## Results and analysis

`results.json` has schema version 2 and the requested `models`, `config`, `runs`, and `baseline.runs` layout, plus audit fields. The decider slot contains JEV (or explicitly marked mock); the baseline slot contains Claude (or mock). Mock and real results cannot be mixed in a slot. Results include scores, steps, actual emulator frame counts, lives lost, termination/truncation, calls/tokens, latency p50/p95/total, cost and dated pricing basis, errors/retries/fallbacks, confidence, both accuracy measures, timeout/invalid counts, calibration bins, Brier score, ECE and non-trivial-label counts.

Mock costs are exactly zero. Claude uses the published [Haiku 4.5 $1/$5 per million input/output tokens](https://www.anthropic.com/news/claude-haiku-4-5), with a recorded basis date. JEV input pricing must be supplied explicitly; output tokens are free per the official OpenAPI usage description. `analyze` prints decision-weighted accuracies, mean latency, **mean episode** latency percentiles (not pooled percentiles), total cost, binary confidence calibration and mixed-label counts. Confidence calibration is against oracle correctness of the selected action, not provider probability calibration against a unique class when optimal actions tie.

Full traces are under `artifacts/*.jsonl` (excluded from Git because emulator snapshots are large). Preserve this directory with any submitted results. `python -m bench verify --min-seeds 5` validates the JSON schema in `results.schema.json`, checks every full trace length, requires action-dependent safe/unsafe labels in each slot, and exactly replays selected serialized snapshots. It writes `validation-evidence.json` with selected non-trivial snapshots and validation counts for a small reviewable artifact. The saved test fixture is a natural gameplay snapshot, not fabricated RAM.
