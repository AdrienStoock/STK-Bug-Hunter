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
    model = None

    # Chargement si modèle existe déjà
    if os.path.exists(save_path + ".zip"):
        print(f"Chargement du modèle existant : {save_path}")
        model = PPO.load(save_path, env=env)
    else:
        print(f"✨ Nouveau modèle : {agent_variant.name}")
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
        self.prev_obs = self.last_obs
        self.kart_ix = 0
        self.visited_positions = set()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.prev_obs = self.last_obs
        self.last_obs = obs
        self.last_info = info  # stocker info pour reward()
        #print('obs : ', obs)
        custom_reward = self.reward(reward)
        return obs, custom_reward, terminated, truncated, info

    def reward(self, reward):
        # On ne peut pas calculer la reward sans observations précédentes
        if self.last_obs is None or self.prev_obs is None:
            return reward  # Pas assez de données pour comparer

        try:
            # Récupération de la distance totale parcourue (>= 0)
            d_t = max(0, self.env.unwrapped.world.karts[self.kart_ix].overall_distance)
            # Distance mémorisée lors du dernier calcul
            d_t_1 = self.last_overall_distances[self.kart_ix]
            # Différence de distance (progression entre 2 étapes)
            delta_d = d_t - d_t_1

            #print(f"[DEBUG] Distance actuelle: {d_t}, distance précédente: {d_t_1}, delta: {delta_d}")

            # On tente de récupérer le pas de temps dt, sinon valeur par défaut
            try:
                dt = self.env.get_wrapper_attr('dt')
            except AttributeError:
                try:
                    dt = self.env.unwrapped.dt
                except AttributeError:
                    dt = 0.1  # Valeur par défaut si impossible à récupérer

            #speed_now = obs_now['continuous'][0]
            #speed_before = obs_prev['continuous'][0]

            #accel = (speed_now - speed_before) / dt if dt > 0 else 0

            #center_dist = abs(obs_now['continuous'][8])

            # Etat indiquant si le kart a terminé la course : 1 si oui, 0 sinon
            finished = 1 if self.env.unwrapped.world.karts[self.kart_ix].has_finished_race else 0

            # Nombre total de karts dans la course
            K = self.env.unwrapped.num_kart
            # Position actuelle du kart (1 = premier)
            pos = self.env.unwrapped.world.karts[self.kart_ix].position
            # Si la position est inconnue (None), on la considère comme la dernière
            if pos is None:
                pos = K - 1

            # Calcul de la reward
            # - Contribution positive liée à la progression (delta_d / 10)
            # - Contribution liée à la position dans la course, augmentée si la course est terminée
            # - Bonus final pour avoir terminé la course
            # - Petit malus fixe pour éviter la stagnation
            reward = (
                (delta_d / 10.0) +
                (1.0 - (pos / K)) * (3 + 7 * finished) +
                10 * finished -
                0.1
            )

            # Bonus/malus supplémentaires basés sur la vitesse, la distance au centre et l’accélération (désactivés)
            # reward *= np.exp(-(speed_now - 20.0) ** 2 / 50.0)    # Plus la vitesse est proche de 20, mieux c’est
            # reward *= np.exp(-(center_dist) ** 2 / 2.0)          # Plus on reste proche du centre de la piste, mieux c’est
            # reward *= np.exp(-(accel) ** 2 / 30.0)               # Moins l’accélération brutale est forte, mieux c’est

            # Mise à jour de la distance mémorisée pour la prochaine étape
            self.last_overall_distances[self.kart_ix] = d_t

            return reward/100

        except Exception as e:
            print(f"[ERROR] Exception dans reward(): {e}")
            return reward






def main():
    os.makedirs("data_ia", exist_ok=True)
    total_timesteps = 100000
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    agent_variants = create_agent_variants()

    temp_env = gym.make("supertuxkart/full-v0", agent=AgentSpec(use_ai=False))
    all_tracks = temp_env.unwrapped.TRACKS
    all_tracks = ['snowmountain']
    temp_env.close()

    for track in all_tracks:
        print(f"\n🏁 Entraînement sur le track : {track}")
        for agent_variant in agent_variants:
            print(f"🚗 Agent : {agent_variant.name}")
            env = gym.make(
                "supertuxkart/full-v0",
                render_mode=None,
                agent=AgentSpec(use_ai=False),
                track=track  
            )
            env = StuckDetectionWrapper(env, track_name=track)
            # env = FPSDetectionWrapper(env, track_name=track)
            env = STKObservationWrapper(env)
            env = STKActionWrapper(env)
            env = CustomRewardWrapper(env)
            env = Monitor(env)

            save_path = f"data_ia/data_PPO_{track}"

            train_single_agent(env, agent_variant, total_timesteps, save_path, track)
            env.close()


if __name__ == "__main__":
    main()