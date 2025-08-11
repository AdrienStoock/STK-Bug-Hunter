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
import argparse
from gymnasium.wrappers import TimeLimit


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
        self.action_space = spaces.Discrete(16)

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

        elif act == 12: 
            action["acceleration"] = np.array([0.2])
            action["steer"] = np.array([-1.0])
        elif act == 13: 
            action["acceleration"] = np.array([0.2])
            action["steer"] = np.array([1.0])
        elif act == 14:  
            action["drift"] = 1
            action["steer"] = np.array([-1.0])
            action["acceleration"] = np.array([0.5])
        elif act == 15:  
            action["drift"] = 1
            action["steer"] = np.array([1.0])
            action["acceleration"] = np.array([0.5])

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
            ent_coef=0.02,   
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
        self.prev_distance = None


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.prev_obs = self.last_obs
        self.last_obs = obs  
        self.last_info = info  
        self.last_time = time.time()
        custom_reward = self.reward()
        return obs, custom_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reset internal state
        try:
            num_karts = self.env.unwrapped.num_kart
        except Exception:
            num_karts = 1
        self.last_overall_distances = [0.0] * num_karts
        self.last_obs = obs
        self.prev_obs = None
        self.last_info = info if isinstance(info, dict) else {}
        
        # Initialize previous track distance from observation if available
        self._prev_track_distance = 0.0
        if isinstance(obs, dict) and 'continuous' in obs and len(obs['continuous']) > 7:
            try:
                self._prev_track_distance = float(obs['continuous'][7])
            except Exception:
                self._prev_track_distance = 0.0
        return obs, info

    def reward(self):
        # Use world progress and speed to avoid index-order coupling
        try:
            d_t = max(0.0, float(self.env.unwrapped.world.karts[self.kart_ix].overall_distance))
        except Exception:
            d_t = 0.0
        d_t_1 = self.last_overall_distances[self.kart_ix]
        delta_d = d_t - d_t_1
        self.last_overall_distances[self.kart_ix] = d_t

        # Compute speed from observation: prefer scalar speed at index 3 when full layout is present,
        # otherwise fall back to norm of velocity components at indices 3..5
        speed = 0.0
        cont = self.last_obs['continuous'] if (self.last_obs is not None and 'continuous' in self.last_obs) else None
        if cont is not None:
            if len(cont) > 6:
                # Standard layout: [.., paths_width(2), speed(3), vx(4), vy(5), vz(6), distance(7), ..]
                speed = float(cont[3])
            elif len(cont) >= 6:
                # Fallback: paths_width missing -> velocity likely at 3..5
                speed = float(np.linalg.norm(np.array(cont[3:6], dtype=np.float32)))
        
        progress_reward = delta_d / 3
        speed_reward = speed * 0.05

        stuck_penalty = 0
        if self.last_info.get('bug_detected', False):
            stuck_penalty = 0.5

        reward = progress_reward + speed_reward - stuck_penalty
        return reward


def train_single_agent(env: gym.Env, agent_variant: AgentVariant, total_timesteps: int, save_path: str, track: str):
    model = None

    # Ensure required directories exist
    log_dir = f"./logs/{agent_variant.name}_{track}"
    best_model_dir = f"./data_ia/{agent_variant.name}_{track}_best_model"
    os.makedirs("./logs", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
                tensorboard_log=log_dir,
                verbose=1
        )
    
    eval_env = gym.make(
            "supertuxkart/full-v0",
            render_mode=None,
            agent=AgentSpec(use_ai=False),
            track=track,
            laps=1,
    )
    
    # Match training wrapper order so obs/action/reward are identical
    eval_env = TimeLimit(eval_env, max_episode_steps=2000)
    eval_env = StuckDetectionWrapper(eval_env, track_name=track)
    # eval_env = FPSDetectionWrapper(eval_env, track_name=track)
    eval_env = STKObservationWrapper(eval_env)
    eval_env = CustomRewardWrapper(eval_env)
    eval_env = STKActionWrapper(eval_env)
    eval_env = Monitor(eval_env)


    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        n_eval_episodes=8,
        eval_freq=10000,
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

    # Evaluate best checkpoint if available, otherwise evaluate current model
    best_model_path = os.path.join(best_model_dir, "best_model.zip")
    eval_model = model
    if os.path.exists(best_model_path):
        try:
            eval_model = PPO.load(best_model_path, env=env)
            print(f"✅ Loading best model for evaluation : {best_model_path}")
        except Exception as e:
            print(f"⚠️ Impossible de charger le best model, fallback sur le modèle courant. Raison: {e}")

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _states = eval_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Final episode on 1 episode : final reward  = {total_reward}")



def play_best_model(track, agent_name="explorer"):

    env = gym.make(
        "supertuxkart/full-v0",
        render_mode='human',
        agent=AgentSpec(use_ai=False),
        track=track,
        laps=1
    )
    env = TimeLimit(env, max_episode_steps=3000)
    env = StuckDetectionWrapper(env, track_name=track)
    # env = FPSDetectionWrapper(env, track_name=track)
    env = STKObservationWrapper(env)
    env = CustomRewardWrapper(env)
    env = STKActionWrapper(env)

    model_path = f"./data_ia/{agent_name}_{track}_best_model/best_model.zip"
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé : {model_path}")
        return
    model = PPO.load(model_path, env=env)

    obs, info = env.reset()
    done = False
    step_count = 0

    try:
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render() 
            done = terminated or truncated
            step_count += 1
    finally:
        env.close()

    print(f"✅ END !")



def train(track,render_mode=None):
    os.makedirs("data_ia", exist_ok=True)
    total_timesteps = 10000
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    agent_variants = create_agent_variants()

    if isinstance(track, str):
        tracks = [track]
    else:
        tracks = track  # if we want to choose multiple tracks 

    for track in tracks:
        print(f"\n🏁 Entraînement sur le track : {track}")
        for agent_variant in agent_variants:
            print(f"🚗 Agent : {agent_variant.name}")
            env = gym.make(
                "supertuxkart/full-v0",
                render_mode=render_mode,
                agent=AgentSpec(use_ai=False),
                track=track,
                laps=1
            )
            env = TimeLimit(env, max_episode_steps=2000)
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
    import sys
    print("Récupération des circuits...")
    def get_valid_tracks():
        temp_env = gym.make("supertuxkart/full-v0", agent=AgentSpec(use_ai=False))
        tracks = temp_env.unwrapped.TRACKS
        temp_env.close()
        return tracks

    parser = argparse.ArgumentParser(description="Select mode and track for STK training or playing.")
    parser.add_argument("mode", choices=["train", "play_best_model"],help="Execution mode")
    parser.add_argument("--track", required=True, help="Name of the track to use")
    parser.add_argument("--render_mode",choices=["None", "human"],default="None",help="Render mode [None, 'human'] (default None)")
    args = parser.parse_args()

    valid_tracks = get_valid_tracks()
    if args.track not in valid_tracks:
        print(f"❌ Error: Track '{args.track}' is not valid. Valid tracks are:")
        for t in valid_tracks:
            print(f" - {t}")
        sys.exit(1)

    render_mode = None if args.render_mode == "None" else args.render_mode

    if args.mode == "train":
        train(args.track, render_mode=render_mode)
    elif args.mode == "play_best_model":
        play_best_model(args.track)
    