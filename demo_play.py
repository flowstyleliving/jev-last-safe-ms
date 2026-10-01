#!/usr/bin/env python3
"""Play the REAL Space Invaders game with JEV flying the ship (rendered live).

Usage:
  # On stage: open a window and watch JEV play the actual game.
  python demo_play.py                 # needs TYPESAFE_API_KEY in the env

  # Headless test / record frames without an API key.
  python demo_play.py --mock --mode rgb --steps 60

  --mode human   render a window (default)
  --mode rgb     render off-screen (returns frames; for testing/recording)
"""
import argparse
import json
import os
import time

from bench.clients import Client
from bench.environment import ACTIONS, make_env, snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mock', action='store_true', help='use mock client (no API key)')
    ap.add_argument('--mode', default='human', choices=['human', 'rgb'])
    ap.add_argument('--steps', type=int, default=300, help='max decisions (Ctrl-C to stop early)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--delay', type=float, default=0.05,
                    help='pause between steps in human mode (s), so the game is watchable')
    args = ap.parse_args()

    model = 'mock' if args.mock else 'jev'
    render_mode = 'human' if args.mode == 'human' else 'rgb_array'
    env = make_env(args.seed, render_mode=render_mode)
    client = Client(model, seed=args.seed, horizon=180)

    W = 62
    print('=' * W)
    print('JEV  ·  SPACE INVADERS  ·  THE LAST SAFE MILLISECOND')
    print('=' * W)
    print(f'model={model}  render={args.mode}  seed={args.seed}\n')

    score = 0.0
    prev = None
    frames = 0
    try:
        for step in range(1, args.steps + 1):
            state, _ = snapshot(env, score, prev)
            d = client.decide(json.dumps(state, sort_keys=True, separators=(',', ':')))
            lat, act, conf = d['latency_ms'], d['action'], d['confidence']
            _, reward, term, trunc, _info = env.step(ACTIONS.index(act))
            score += reward
            frame = env.render()  # draw the actual game (window or array)
            if args.mode == 'rgb':
                frames += 1
            bar = '#' * int(round(conf * 16))
            print(f"\r  t={step * 4:4d}  score={score:6.0f}  lives={env.ale.lives()}  "
                  f"JEV->{act:<9}  {lat:5.0f} ms  conf {conf:4.2f}  {bar:<16}", end='', flush=True)
            prev = state
            if term or trunc:
                print(f"\n[episode over]  final score={score:.0f}  lives={env.ale.lives()}")
                break
            if args.mode == 'human':
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print(f"\n[stopped]  score={score:.0f}  lives={env.ale.lives()}")
    finally:
        client.close()
        env.close()

    print()
    print('-' * W)
    print('THE GAP  (same question, same game state)')
    print('-' * W)
    print('  JEV       ~135 ms   $0.00005   57.8% accurate')
    print('  DeepSeek  ~793 ms   $0.00021    0.0% accurate  (does NOOP)')
    print('-' * W)
    print('A correct answer delivered after impact is a wrong decision.')
    print('=' * W)
    if args.mode == 'rgb':
        print(f'(rendered {frames} frames headlessly; use --mode human to watch live)')


if __name__ == '__main__':
    main()
