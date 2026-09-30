"""Finite-horizon, common-RNG counterfactual oracle, measured in emulator frames."""
from .environment import ACTIONS

def alien_count(ale):
    return sum(int(v).bit_count() for v in ale.getRAM()[18:24])

def sweep(env, system, horizon=48, max_delay=12):
    ale = env.ale
    labels = {}
    try:
        for idx, name in enumerate(ACTIONS):
            outcomes = []
            for delay in range(max_delay + 1):
                ale.restoreSystemState(system)
                lives, aliens = ale.lives(), alien_count(ale)
                reward, lost, killed = 0, 0, 0
                last_lives, last_aliens = lives, aliens
                # Common absolute horizon: NOOP during delay, chosen action held thereafter.
                for frame in range(horizon):
                    if ale.game_over():
                        break
                    reward += ale.act(env._action_set[0 if frame < delay else idx])
                    current_lives, current_aliens = ale.lives(), alien_count(ale)
                    lost += max(0, last_lives-current_lives)
                    killed += max(0, last_aliens-current_aliens)
                    last_lives, last_aliens = current_lives, current_aliens
                outcomes.append(dict(delay_frames=delay, life_lost=bool(lost), lives_lost=lost,
                                     alien_killed=bool(killed), aliens_killed=killed, score_delta=float(reward)))
            safe = [o['delay_frames'] for o in outcomes if not o['life_lost']]
            # Prefix deadline is the conservative scalar for non-monotonic safety.
            prefix = -1
            for o in outcomes:
                if o['life_lost']: break
                prefix = o['delay_frames']
            labels[name] = dict(deadline_frames=max(safe,default=-1), safe_prefix_frames=prefix,
                                deadline_right_censored=max_delay in safe,
                                safe_delays=safe, outcomes=outcomes)
        best = max((-v['outcomes'][0]['lives_lost'],v['outcomes'][0]['score_delta']) for v in labels.values())
        for v in labels.values():
            o = v['outcomes'][0]
            v['correct'] = (-o['lives_lost'],o['score_delta']) == best
        return labels
    finally:
        ale.restoreSystemState(system)

def grade(labels, action, delay, valid=True):
    label = labels[action]
    ordinary = bool(valid and label['correct'])
    # Membership, not delay <= max(safe), handles safety holes exactly.
    return ordinary, bool(ordinary and delay in label['safe_delays'])
