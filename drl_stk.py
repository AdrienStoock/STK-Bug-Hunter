# drl_stk.py
from logging import info
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
        self.clip_range = clip_range  


class STKActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(12)

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
            action["acceleration"] = np.array([0.3])
            action["steer"] = np.array([-0.5])
        elif act == 5:
            action["acceleration"] = np.array([0.5])
            action["steer"] = np.array([-0.5])
        elif act == 6:
            action["acceleration"] = np.array([0.7])
            action["steer"] = np.array([-1.0])
        
        
        elif act == 7:
            action["acceleration"] = np.array([0.3])
            action["steer"] = np.array([0.5])
        elif act == 8:
            action["acceleration"] = np.array([0.5])
            action["steer"] = np.array([0.5])
        elif act == 9:
            action["acceleration"] = np.array([0.7])
            action["steer"] = np.array([1.0])
        
        
        elif act == 10:
            action["brake"] = 1
        elif act == 11:
            action["drift"] = 1
            action["acceleration"] = np.array([0.6])

        return action


class STKObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'continuous': spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32),
            'discrete': spaces.MultiDiscrete([2])
        })

    def observation(self, obs):
        continuous = []
        discrete = []

        
        if 'center_path_distance' in obs:
            continuous.extend(obs['center_path_distance'])

        if 'paths_width' in obs and len(obs['paths_width']) > 0:
            continuous.append(float(obs['paths_width'][0].item()))

        if 'velocity' in obs:
            velocity = obs['velocity']
            speed = np.linalg.norm(velocity)
            continuous.append(speed)
            continuous.extend(velocity.flatten())

        if 'distance_down_track' in obs:
            continuous.append(float(obs['distance_down_track'][0]))

       
        if 'front' in obs:
            continuous.extend(obs['front'].flatten())


        if 'max_steer_angle' in obs:
            continuous.append(float(obs['max_steer_angle'][0]))

        if len(discrete) == 0:
            discrete = [0]

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
            ent_coef=0.02,   # encourage exploration
            clip_range=0.2   
        )
    ]



from gymnasium import RewardWrapper
import time

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_overall_distances = [0.0] * env.unwrapped.num_kart
        self.last_obs = None
        self.prev_obs = self.last_obs
        self.kart_ix = 0
        self.last_velocity = None


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.prev_obs = self.last_obs
        self.last_obs = obs  
        self.last_info = info  
        self.last_time = time.time()
        custom_reward = self.reward(reward)
        return obs, custom_reward, terminated, truncated, info

    def reward(self, reward):

        # Reward officielle SuperTuxKart (basée sur le code source)
        # r_t = (d_t - d_{t-1})/10 + (1 - pos_t/K) × (3 + 7f_t) - 0.1 + 10*f_t
        
        print(f"obs : {self.last_obs}")
        print(f"info : {self.last_info}")
        
        # Récupération des données depuis les infos
        # current_distance = self.last_info.get('distance', 0.0)
        # current_position = self.last_info.get('position', 1)
        # # Correction: récupérer le vrai nombre de karts depuis l'environnement
        # total_karts = self.env.unwrapped.num_kart if hasattr(self.env.unwrapped, 'num_kart') else 3
        # finished = self.last_info.get('finished', False)
        
        
        # # Calcul de la progression (d_t - d_{t-1})
        # if not hasattr(self, "last_distance"):
        #     self.last_distance = current_distance
        
        # progress_delta = current_distance - self.last_distance
        # self.last_distance = current_distance
        
        # # 1) Progression normalisée par 10
        # progress_reward = progress_delta / 10.0
        # reward += progress_reward
        
        # # 2) Bonus de position avec facteur de fin
        # f_t = 1 if finished else 0
        # position_bonus = (1.0 - current_position / total_karts) * (3 + 7 * f_t)
        # reward += position_bonus
        
        # # 3) Pénalité constante
        # reward -= 0.1
        
        # # 4) Bonus de fin
        # if finished:
        #     reward += 10.0

        # # 5) Bonus de vitesse (encourage à se débloquer naturellement)
        # speed_xy = self.last_info.get('speed_xy', 0.0)
        # speed_bonus = 0.0  # Initialisation
        # if speed_xy > 0.1:  # Bonus si vitesse > 0.1 (très accessible)
        #     speed_bonus = speed_xy * 2.0  # Bonus plus fort pour encourager
        #     reward += speed_bonus

        # # Debug détaillé de la reward
        # #print(f"DEBUG - progress_delta: {progress_delta:.3f}, position_bonus: {position_bonus:.3f}, speed_bonus: {speed_bonus:.3f}, final_reward: {reward:.3f}")
            
        return reward

        

        # Récupère la distance au centre et la largeur de la piste
        # center_dist = float(self.last_obs['continuous'][0])
        # track_width = float(self.last_obs['continuous'][1])
        # finished = 1 if self.env.unwrapped.world.karts[self.kart_ix].has_finished_race else 0

        # # Progression sur la piste (delta distance)
        # d_t = max(0, self.env.unwrapped.world.karts[self.kart_ix].overall_distance)
        # d_t_1 = self.last_overall_distances[self.kart_ix]
        # delta_d = d_t - d_t_1
        # self.last_overall_distances[self.kart_ix] = d_t

        # # Récompense de centrage
        # m1 = 0.1 * track_width
        # m2 = 0.25 * track_width
        # m3 = 0.5 * track_width

        # if abs(center_dist) <= m1:
        #     center_reward = 1.0
        # elif abs(center_dist) <= m2:
        #     center_reward = 0.5
        # elif abs(center_dist) <= m3:
        #     center_reward = 0.1
        # else:
        #     center_reward = 1e-3

        # # Reward finale : progression + centrage + bonus arrivée
        # # Pondération : progression (1.0), centrage (0.5), bonus arrivée (2.0)
        # reward = 0.5 * center_reward + (delta_d/10) + 2.0 * finished
        # return reward

        
        # if self.last_obs is None or self.prev_obs is None:
        #     return reward 

        # try:
        #     d_t = max(0, self.env.unwrapped.world.karts[self.kart_ix].overall_distance)
        #     d_t_1 = self.last_overall_distances[self.kart_ix]
        #     delta_d = d_t - d_t_1

        #     speed = self.last_obs['continuous'][0]
        #     prev_speed = self.prev_obs['continuous'][0]
        #     delta_speed = speed - prev_speed
        #     speed_bonus = delta_speed if delta_speed > 0 else 0.0

        #     finished = 1 if self.env.unwrapped.world.karts[self.kart_ix].has_finished_race else 0

        #     #print(f"[DEBUG] Δdist={delta_d:.3f} | speed={speed:.2f} | +Δv={speed_bonus:.3f}")

        #     reward = (
        #         (delta_d / 10) +  # progression
        #         speed_bonus +
        #         (3 + 7 * finished) -
        #         0.1
        #     )

        #     self.last_overall_distances[self.kart_ix] = d_t

        #     return reward / 100  # lissage

        # except Exception as e:
        #     print(f"[ERROR] Exception dans reward(): {e}")
        #     return reward



def train_single_agent(env: gym.Env, agent_variant: AgentVariant, total_timesteps: int, save_path: str, track: str):
    model = None

    # Chargement si modèle existe déjà
    if os.path.exists(save_path + ".zip"):
        print(f"Chargement du modèle existant : {save_path}")
        model = PPO.load(save_path, env=env)
    else:
        print(f"✨ Nouveau modèle : {agent_variant.name} sur le track {track}")
        #network neural
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO(
                "MultiInputPolicy",
                env,
                learning_rate=agent_variant.learning_rate,
                n_steps=agent_variant.n_steps,
                ent_coef=agent_variant.ent_coef,
                clip_range=agent_variant.clip_range,
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"./logs/{agent_variant.name}_{track}",
                verbose=1
        )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"./data_ia/{agent_variant.name}_{track}_best_model",
        log_path=f"./logs/{agent_variant.name}_{track}",
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="PPO",
        callback=eval_callback,
        progress_bar=True
    )

    model.save(save_path)

    # Evaluation 1 episode
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Évaluation finale sur 1 épisode : reward total = {total_reward}")


def play_best_model(track="olivermath", agent_name="explorer"):
    env = gym.make(
        "supertuxkart/full-v0",
        render_mode="human",      
        agent=AgentSpec(use_ai=False),
        track=track,
        laps=4
    )
    env = StuckDetectionWrapper(env, track_name=track)
    # env = FPSDetectionWrapper(env, track_name=track)
    env = STKObservationWrapper(env)
    env = CustomRewardWrapper(env)
    env = STKActionWrapper(env)
    

    model_path = f"./data_ia/{agent_name}_{track}_best_model/best_model.zip"
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé : {model_path}")
        return
    model = PPO.load(model_path)
    model.set_env(env)

    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    try:
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            print(f"Action choisie par la policy : {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step_count += 1
    finally:
        env.close()

    print(f"✅ Total reward avec best model : {total_reward} en {step_count} étapes")





def main():
    os.makedirs("data_ia", exist_ok=True)
    total_timesteps = 20000
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    agent_variants = create_agent_variants()

    temp_env = gym.make("supertuxkart/full-v0", agent=AgentSpec(use_ai=False))
    all_tracks = temp_env.unwrapped.TRACKS
    all_tracks = ['olivermath']
    temp_env.close()

    for track in all_tracks:
        print(f"\n🏁 Entraînement sur le track : {track}")
        for agent_variant in agent_variants:
            print(f"🚗 Agent : {agent_variant.name}")
            env = gym.make(
                "supertuxkart/full-v0",
                render_mode='human',
                agent=AgentSpec(use_ai=False),
                track=track,
                laps=4
            )
            env = StuckDetectionWrapper(env, track_name=track)
            # env = FPSDetectionWrapper(env, track_name=track)
            env = STKObservationWrapper(env)
            env = CustomRewardWrapper(env)
            env = STKActionWrapper(env)
            env = Monitor(env)

            save_path = f"data_ia/{agent_variant.name}_{track}/data_PPO_{track}"

            train_single_agent(env, agent_variant, total_timesteps, save_path, track)
            env.close()


if __name__ == "__main__":
    main()
    #play_best_model()