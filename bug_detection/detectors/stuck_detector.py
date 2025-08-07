import time
from turtle import speed
import numpy as np
from .base_detector import BaseDetectionWrapper 

class StuckDetectionWrapper(BaseDetectionWrapper):
    def __init__(self, env, stuck_time=5, speed_threshold=0.1, grace_period=10,track_name=None):
        super().__init__(env)
        self.stuck_time = stuck_time
        self.speed_threshold = speed_threshold
        self.grace_period = grace_period  # seconds to ignore at the beginning
        self.positions_history = []
        self.positions_history_maxlen = stuck_time * 30  # 30 fps 
        self.start_time = None
        self.track_name=track_name
        self.start_time = None 
        self.stuck_counter = 0

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

        # Initialisation of timer
        if self.start_time is None:
            self.start_time = time.time()

        # grace period (ex. 3 seconds after start)
        if time.time() - self.start_time < self.grace_period:
            info['bug_detected'] = False
            return obs, reward, terminated, truncated, info

        positions = np.array(self.positions_history)
        if len(positions) >= 30:
            window = positions[-30:]

            # move XY max each frames
            #xy_disp = np.linalg.norm(window[0, :2] - window[-1, :2])
            z_disp = np.max(np.abs(window[:, 2] - window[-1, 2]))

            distances = np.linalg.norm(window[1:, :2] - window[:-1, :2], axis=1)
            total_cumulated_disp = distances.sum()

            # horizontal speed
            velocity = obs.get("velocity")
            if velocity is not None:
                v = np.array(velocity)
                speed_xy = np.linalg.norm(v[0][:2] if v.ndim == 2 else v[:2])
            else:
                speed_xy = 0.0

            if (speed_xy < 0.2 and total_cumulated_disp < 30) or speed_xy < 0.1:
                self.stuck_counter = getattr(self, 'stuck_counter', 0) + 1
            else:
                self.stuck_counter = 0

            stuck = self.stuck_counter > 10
            #jump_on_spot = xy_disp < 1.0 and z_disp > 10.0
            
            info["speed_xy"] = speed_xy
            info["total_cumulated_disp"] = total_cumulated_disp
            info["stuck_counter"]= self.stuck_counter
            info["stuck"] = stuck
            #info["jump_on_spot"] = self.jump_on_spot
            info["bug_detected"] = stuck # or jump_on_spot

            #if info["bug_detected"]:
                #print(f"🚨 BUG détecté: stuck={stuck}, jump_on_spot, "
                   # f"speed_xy={speed_xy:.2f}, cumul_disp={total_cumulated_disp:.2f}, z_disp={z_disp:.2f}", )
        else:
            info["bug_detected"] = False
            info["stuck"] = False
            info["jump_on_spot"] = False

        return obs, reward, terminated, truncated, info


