#!/usr/bin/env python3
"""Live terminal demo: JEV making real Space Invaders decisions, timed."""
import json
import os
import time

from bench.clients import Client
from bench.environment import ACTIONS, make_env, snapshot


def main():
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        print("Set TYPESAFE_API_KEY first.")
        return
    env = make_env(seed=42)
    client = Client("jev", seed=42, horizon=180)
    W = 62
    print("=" * W)
    print("JEV  ·  SPACE INVADERS  ·  THE LAST SAFE MILLISECOND")
    print("=" * W)
    print("live decisions, real wall-clock latency, model jev-1.13.0\n")
    score = 0.0
    prev = None
    try:
        for step in range(1, 31):
            state, _ = snapshot(env, score, prev)
            d = client.decide(json.dumps(state, sort_keys=True, separators=(",", ":")))
            lat, act, conf = d["latency_ms"], d["action"], d["confidence"]
            _, reward, _, _, _ = env.step(ACTIONS.index(act))
            score += reward
            bar = "#" * int(round(conf * 16))
            print(f"  t={step * 4:4d}  JEV -> {act:<9}  {lat:6.0f} ms   conf {conf:4.2f}  {bar:<16}")
            prev = state
            time.sleep(0.4)
    finally:
        client.close()
        env.close()
    print()
    print("-" * W)
    print("THE GAP  (per decision, same question, same game state)")
    print("-" * W)
    print(f"  JEV  (jev-1.13.0)       ~135 ms    ~$0.00005 / decision")
    print(f"  LLM  (claude-haiku-4.5) ~12,500 ms ~$0.0195  / case")
    print(f"          (TypeSafe evals, evals.typesafe.ai)")
    print("-" * W)
    print("A correct answer delivered after impact is a wrong decision.")
    print("=" * W)


if __name__ == "__main__":
    main()
