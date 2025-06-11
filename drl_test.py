import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pystk2_gymnasium import AgentSpec
import numpy as np

# Fonction pour détecter un bug (vitesse quasi nulle prolongée)
def detect_bug(state, prev_states, threshold=10):
    speed = state["continuous"][0]  # suppose que c'est la vitesse dans l'obs dict
    prev_states.append(speed)
    if len(prev_states) > threshold:
        prev_states.pop(0)
    if len(prev_states) == threshold and np.mean(prev_states) < 0.01:
        return True
    return False

# Création de l'env STK
def make_env():
    return gym.make("supertuxkart/flattened-v0", render_mode="human", agent=AgentSpec(use_ai=False))

# Vectorisation pour stable-baselines3
env = make_vec_env(make_env, n_envs=1)

# Instanciation du modèle PPO avec MultiInputPolicy
model = PPO("MultiInputPolicy", env, verbose=1)

prev_speeds = []

# Reset env
obs = env.reset()

done = [False]
while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, infos = env.step(action)
    done = np.logical_or(terminated, truncated)

    # Detect bug on obs (obs is a dict of np arrays, for vectorized env obs[0] gives first env obs)
    if detect_bug(obs[0], prev_speeds):
        print("⚠️ Bug potentiel détecté : vitesse quasi nulle prolongée")

env.close()
