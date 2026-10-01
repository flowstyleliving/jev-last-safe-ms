import base64
import gymnasium as gym
import ale_py

ACTIONS = ('NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE')
CONFIG = dict(env_id='ALE/SpaceInvaders-v5', frameskip=4,
              repeat_action_probability=0.25, full_action_space=False,
              max_num_frames_per_episode=108000, obs_type='ram', wrappers=[],
              decision_interval_steps=1,
              state_encoding='RAM bytes decoded to ship x, alien grid, bullets, shields, lives as JSON',
              ale_py_version=ale_py.__version__, gymnasium_version=gym.__version__)

def make_env(seed, render_mode=None):
    gym.register_envs(ale_py)
    kwargs = {k: CONFIG[k] for k in ('frameskip', 'repeat_action_probability', 'full_action_space', 'max_num_frames_per_episode', 'obs_type')}
    if render_mode:
        kwargs['render_mode'] = render_mode
    env = gym.make(CONFIG['env_id'], **kwargs).unwrapped
    env.reset(seed=seed)
    env.action_space.seed(seed)
    assert tuple(env.get_action_meanings()) == ACTIONS
    assert env.ale.getInt('frame_skip') == 1
    return env

def decode(ram, lives, score, previous=None, elapsed=4):
    r = list(map(int, ram))
    # RAM map independently implemented from OCAtari's documented SpaceInvaders map.
    empty = 6 - max(r[18:24]).bit_length()
    grid = [[bool(r[23-i] & (1 << (j+empty))) if j+empty < 6 else False for j in range(6)] for i in range(6)]
    bullets = []
    for slot, (xi, yi) in enumerate(((83,81),(84,82),(87,85))):
        x, y = r[xi]-2, 2*r[yi]+3
        active = 20 < y < 195
        old = previous['bullets'][slot] if previous else None
        continuous = active and old and old['active'] and abs(y-old['y']) < 40 and abs(x-old['x']) < 20
        bullets.append(dict(slot=slot, owner='enemy' if slot < 2 else 'player', x=x, y=y, active=active,
                            vx=(x-old['x'])/elapsed if continuous else None,
                            vy=(y-old['y'])/elapsed if continuous else None))
    return dict(ram=r, ship=dict(x=r[28]-1,y=185), alien_grid=grid,
                alien_origin=dict(x=r[26]-1-empty*16,y=31+2*r[16]), alien_spacing=[16,18],
                bullets=bullets, shields=[dict(x=42+32*i,y=157,bitmap=r[43+i*9:52+i*9]) for i in range(3)],
                lives=lives, ram_lives=r[73], score=score,
                score_bcd_bytes=[r[102],r[104]], velocity_units='pixels/frame; null on spawn/discontinuity')

def snapshot(env, score, previous=None):
    state = decode(env.ale.getRAM(), env.ale.lives(), score, previous)
    system = env.ale.cloneSystemState()
    return state, system

def encode_system(system):
    return base64.b64encode(system.serialize()).decode()
