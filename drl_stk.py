# drl_stk.py
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from pystk2_gymnasium import AgentSpec
import datetime
from typing import List
from gymnasium import spaces
from pystk2_gymnasium.wrappers import FlattenerWrapper
from bug_detection.detectors.fps_detector import FPSDetectionWrapper
from bug_detection.detectors.stuck_detector import StuckDetectionWrapper
from stable_baselines3.common.callbacks import EvalCallback


class AgentVariant:
    def __init__(self, name: str, learning_rate: float, n_steps: int, ent_coef: float, clip_range: float):
        self.name = name
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.clip_range = clip_range  # (PPO)contrôle jusqu'à quel point la nouvelle politique (policy) peut s'écarter de l'ancienne


class STKActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(7)

    def action(self, act):
        action = {
            "acceleration": np.array([0.0]),
            "steer": np.array([0.0]),
            "brake": 0,
            "drift": 0,
            "fire": 0,
            "nitro": 0,
            "rescue": 0
        }

        if act == 0:
            action["acceleration"] = np.array([0.2])
        elif act == 1:
            action["acceleration"] = np.array([0.5])
        elif act == 2:
            action["acceleration"] = np.array([0.8])
        elif act == 3:
            action["acceleration"] = np.array([1.0])
        elif act == 4:
            action["acceleration"] = np.array([0.5])
            action["steer"] = np.array([-1.0])
        elif act == 5:
            action["acceleration"] = np.array([0.5])
            action["steer"] = np.array([1.0])
        elif act == 6:
            action["brake"] = 1

        return action


class STKObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'continuous': spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32),
            'discrete': spaces.MultiDiscrete([10, 2, 4, 11])
        })

    def observation(self, obs):
        continuous = []
        discrete = []

        if 'velocity' in obs:
            velocity = obs['velocity'][0]
            speed = np.linalg.norm(velocity)
            continuous.append(speed)
            continuous.extend(velocity.flatten())

        if 'front' in obs:
            continuous.extend(obs['front'][0].flatten())

        if 'center_path' in obs:
            continuous.extend(obs['center_path'][0].flatten())

        if 'distance_down_track' in obs:
            continuous.append(float(obs['distance_down_track'][0]))
        if 'center_path_distance' in obs:
            continuous.append(float(obs['center_path_distance'][0]))

        if 'paths_distance' in obs and len(obs['paths_distance'][0]) > 0:
            continuous.extend(obs['paths_distance'][0].flatten())
        if 'paths_width' in obs and len(obs['paths_width'][0]) > 0:
            continuous.append(float(obs['paths_width'][0].item()))

        if 'energy' in obs:
            continuous.append(float(obs['energy'][0]))
        if 'shield_time' in obs:
            continuous.append(float(obs['shield_time'][0]))
        if 'skeed_factor' in obs:
            continuous.append(float(obs['skeed_factor'][0]))
        if 'max_steer_angle' in obs:
            continuous.append(float(obs['max_steer_angle'][0]))

        if 'attachment' in obs:
            discrete.append(obs['attachment'])
        if 'jumping' in obs:
            discrete.append(obs['jumping'])
        if 'phase' in obs:
            discrete.append(obs['phase'])
        if 'powerup' in obs:
            discrete.append(obs['powerup'])

        while len(continuous) < 20:
            continuous.append(0.0)

        return {
            'continuous': np.array(continuous[:20], dtype=np.float32),
            'discrete': np.array(discrete, dtype=np.int32)
        }


def create_agent_variants() -> List[AgentVariant]:
    return [
        AgentVariant(
            "explorer",
            learning_rate=3e-4,
            n_steps=1024,
            ent_coef=0.01,   # encourager l'exploration, plus il est élévé plus il explore
            clip_range=0.2   # limite l'ampleur des maj de politiques afin d'éviter des changements trop brusques
        )
    ]


def train_single_agent(env: gym.Env, agent_variant: AgentVariant, total_timesteps: int, save_path: str, track: str):
    print(f"✨ Nouveau modèle : {agent_variant.name}")
    model = None

    # Chargement si modèle existe déjà
    # if os.path.exists(save_path + ".zip"):
    #     print(f"Chargement du modèle existant : {save_path}")
    #     model = PPO.load(save_path, env=env)
    #else:
    model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=agent_variant.learning_rate,
            n_steps=agent_variant.n_steps,
            ent_coef=agent_variant.ent_coef,
            clip_range=agent_variant.clip_range,
            tensorboard_log=f"./logs/{agent_variant.name}_{track}",
            verbose=1
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"./models/{agent_variant.name}_{track}",
        log_path=f"./logs/{agent_variant.name}_{track}",
        n_eval_episodes=1,
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="PPO",
        callback=eval_callback,
        progress_bar=True
    )

    # Sauvegarde après entraînement
    model.save(save_path)

    # Évaluation simple après entraînement
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Évaluation finale sur 1 épisode : reward total = {total_reward}")


from gymnasium import RewardWrapper

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_overall_distances = [0.0] * env.unwrapped.num_kart
        self.last_obs = None
        self.kart_ix = 0
        self.visited_positions = set()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        self.last_info = info  # stocker info pour reward()
        return obs, reward, terminated, truncated, info

    def reward(self, reward):
        if self.last_obs is None:
            return reward

        kart = self.env.unwrapped.world.karts[self.kart_ix]
        K = self.env.unwrapped.num_kart
        d_t = max(0, kart.overall_distance)
        d_t_1 = self.last_overall_distances[self.kart_ix]
        delta_d = d_t - d_t_1
        pos_t = kart.position
        f_t = 1 if kart.has_finished_race else 0

        # Progression
        progression_reward = 0.1 * delta_d
        classement_reward = (1 - pos_t / K) * (3 + 7 * f_t)
        finish_reward = 10 * f_t
        time_penalty = -0.1

        # Exploration
        pos = tuple(np.round(kart.position, 1))
        if not hasattr(self, "visited_positions"):
            self.visited_positions = set()
        exploration_bonus = 0.0
        if pos not in self.visited_positions:
            self.visited_positions.add(pos)
            exploration_bonus = 0.2  # à ajuster

        # Bugs
        bug_bonus = 0.0
        if hasattr(self, "last_info") and self.last_info is not None:
            if self.last_info.get('bug_detected', False):
                bug_bonus += 50.0  # stuck
            if self.last_info.get('collision_detected', False):
                bug_bonus += 50.0  # collision (si tu ajoutes ce wrapper)

        self.last_overall_distances[self.kart_ix] = d_t

        return (
            progression_reward +
            classement_reward +
            finish_reward +
            time_penalty +
            exploration_bonus +
            bug_bonus
        )



def main():
    os.makedirs("data_ia", exist_ok=True)
    total_timesteps = 10000
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    agent_variants = create_agent_variants()

    temp_env = gym.make("supertuxkart/full-v0", agent=AgentSpec(use_ai=False))
    all_tracks = temp_env.unwrapped.TRACKS
    all_tracks = ['abyss']
    temp_env.close()

    for track in all_tracks:
        print(f"\n🏁 Entraînement sur le track : {track}")
        for agent_variant in agent_variants:
            print(f"🚗 Agent : {agent_variant.name}")
            env = gym.make(
                "supertuxkart/full-v0",
                render_mode="human",
                agent=AgentSpec(use_ai=False),
                track=track  
            )
            env = StuckDetectionWrapper(env, track_name=track)
            # env = FPSDetectionWrapper(env, track_name=track)
            env = STKObservationWrapper(env)
            env = STKActionWrapper(env)
            #env = CustomRewardWrapper(env)
            env = Monitor(env)

            save_path = f"data_ia/data_PPO_{track}"

            train_single_agent(env, agent_variant, total_timesteps, save_path, track)
            env.close()


if __name__ == "__main__":
    main()