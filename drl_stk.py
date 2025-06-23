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
from block_detection_wrapper import FPSDetectionWrapper


class AgentVariant:
    def __init__(self, name: str, learning_rate: float, n_steps: int, ent_coef: float, clip_range: float):
        self.name = name
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.clip_range = clip_range


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
            learning_rate=1e-4,
            n_steps=2048,
            ent_coef=0.01,
            clip_range=0.2
        )
    ]


def train_single_agent(env: gym.Env, agent_variant: AgentVariant, total_timesteps: int, save_path: str):
    print(f"✨ Nouveau modèle : {agent_variant.name}")
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=agent_variant.learning_rate,
        n_steps=agent_variant.n_steps,
        ent_coef=agent_variant.ent_coef,
        clip_range=agent_variant.clip_range,
        verbose=1
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)


import pystk2_gymnasium

def main():
    save_path = "./ppo_stk_model"
    total_timesteps = 2000
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    agent_variants = create_agent_variants()

    temp_env = gym.make("supertuxkart/full-v0", agent=AgentSpec(use_ai=True))
    all_tracks = temp_env.unwrapped.TRACKS
    temp_env.close()

    for track in all_tracks:
        print(f"\n🏁 Entraînement sur le track : {track}")
        for agent_variant in agent_variants:
            print(f"🚗 Agent : {agent_variant.name}")
            env = gym.make(
                "supertuxkart/full-v0",
                render_mode=None,
                agent=AgentSpec(use_ai=True),
                track=track
            )
            env = FPSDetectionWrapper(env, track_name=track)
            env = STKObservationWrapper(env)
            env = STKActionWrapper(env)
            env = Monitor(env)

            train_single_agent(env, agent_variant, total_timesteps, f"{save_path}_{track}")
            env.close()

if __name__ == "__main__":
    main()