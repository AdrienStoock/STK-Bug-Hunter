import time
import numpy as np
from .base_detector import BaseDetectionWrapper  # adapte l'import si besoin


import re

class StuckDetectionWrapper(BaseDetectionWrapper):
    def __init__(self, env, stuck_time=5, speed_threshold=0.1, grace_period=6,track_name=None):
        super().__init__(env)
        self.stuck_time = stuck_time
        self.speed_threshold = speed_threshold
        self.grace_period = grace_period  # secondes à ignorer au début
        self.positions_history = []
        self.positions_history_maxlen = stuck_time * 30  # 30 fps (adapte si besoin)
        self.start_time = None
        self.track_name=track_name
        self.start_time = None 

    def reset(self, **kwargs):
        self.start_time = time.time()
        self.positions_history = []
        return self.env.reset(**kwargs)



    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        kart_pos = obs["karts_position"][0]
        self.positions_history.append(np.array(kart_pos))
        if len(self.positions_history) > self.positions_history_maxlen:
            self.positions_history.pop(0)

        # Initialisation du timer
        if self.start_time is None:
            self.start_time = time.time()

        # Période de grâce (ex. 3 secondes après le début)
        if time.time() - self.start_time < self.grace_period:
            info['bug_detected'] = False
            return obs, reward, terminated, truncated, info

        positions = np.array(self.positions_history)
        if len(positions) >= 30:
            window = positions[-30:]

            # déplacement XY max entre frames
            xy_disp = np.linalg.norm(window[0, :2] - window[-1, :2])
            z_disp = np.max(np.abs(window[:, 2] - window[-1, 2]))

            # vitesse horizontale instantanée
            velocity = obs.get("velocity")
            if velocity is not None:
                v = np.array(velocity)
                speed_xy = np.linalg.norm(v[0][:2] if v.ndim == 2 else v[:2])
            else:
                speed_xy = 0.0

            # compteur de frames bloquées
            if speed_xy < 0.2 and xy_disp < 1.0:
                self.stuck_counter = getattr(self, 'stuck_counter', 0) + 1
            else:
                self.stuck_counter = 0

            stuck = self.stuck_counter > 3
            #jump_on_spot = xy_disp < 1.0 and z_disp > 10.0
            
            info["speed_xy"] = speed_xy
            info["xy_disp"] = xy_disp
            info["stuck_counter"]= self.stuck_counter
            info["stuck"] = stuck
            #info["jump_on_spot"] = self.jump_on_spot
            info["bug_detected"] = stuck # or jump_on_spot

            if info["bug_detected"]:
                print(f"🚨 BUG détecté: stuck={stuck}, jump_on_spot, "
                    f"speed_xy={speed_xy:.2f}, xy_disp={xy_disp:.2f}, z_disp={z_disp:.2f}", )
        else:
            info["bug_detected"] = False
            info["stuck"] = False
            info["jump_on_spot"] = False

        return obs, reward, terminated, truncated, info


