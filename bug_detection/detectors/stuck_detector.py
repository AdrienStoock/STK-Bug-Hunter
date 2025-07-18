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

    def reset(self, **kwargs):
        self.start_time = time.time()
        self.positions_history = []
        return self.env.reset(**kwargs)



    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        kart_pos = obs["karts_position"][0]
        self.positions_history.append(np.array(kart_pos))

       # print("clef de obs= ", list(obs.keys()))
        #print("clef de info= ", list(info.keys()))
        #print("env dir: ", dir(self.env))
        #print("track_name in env: ", getattr(self.env, "track_name", None))

        if len(self.positions_history) > self.positions_history_maxlen:
            self.positions_history.pop(0)

        # Ignore la détection pendant la période de grâce
        if self.start_time is not None and (time.time() - self.start_time) < self.grace_period:
            return obs, reward, terminated, truncated, info

        positions = np.array(self.positions_history)
        xy_disp = np.max(np.linalg.norm(positions[-5:, :2] - positions[-1, :2], axis=1))  # dernier mouvement
        z_disp = np.max(np.abs(positions[:, 2] - positions[0, 2]))

        # Vitesse horizontale
        velocity = obs.get("velocity")
        if velocity is not None:
            velocity = np.array(velocity)
            if velocity.ndim == 1 and velocity.shape[0] >= 2:
                speed_xy = np.linalg.norm(velocity[:2])
                vx, vy, vz = velocity
            elif velocity.ndim == 2 and velocity.shape[1] >= 2:
                speed_xy = np.linalg.norm(velocity[0][:2])
                vx, vy, vz = velocity[0]
            else:
                speed_xy = 0.0
                vx = vy = vz = 0.0
        else:
            speed_xy = 0.0
            vx = vy = vz = 0.0

        # Distance au centre du chemin
        center_dist = abs(float(obs.get("center_path_distance", [0.0])[0]))

        # Différence de hauteur entre le kart et le chemin
        try:
            kart_y = float(kart_pos[1])
            path_y = float(obs["center_path"][1])
            delta_z = abs(kart_y - path_y)
        except Exception:
            delta_z = 0.0

        # Debug complet
        print("\n=== INFO FRAME ===")
        print(f"Kart position → x: {kart_pos[0]:.2f}, y: {kart_pos[1]:.2f}, z: {kart_pos[2]:.2f}")
        print(f"Velocity      → vx: {vx:.3f}, vy: {vy:.3f}, vz: {vz:.3f}")
        print(f"Speed XY      → {speed_xy:.3f}")
        print(f"XY Disp       → {xy_disp:.2f}")
        print(f"Z Disp        → {z_disp:.2f}")
        print(f"Center Dist   → {center_dist:.2f}")
        #print(f"Δ Hauteur vs Path → {delta_z:.2f}")

        # Critères
        stuck = speed_xy < 0.2 and xy_disp < 1
        jump_on_spot = xy_disp < 1.0 and z_disp > 2.0
        #off_track = center_dist > 10.0
        #flying = delta_z > 3.0

        # Résultats
        info["bug_detected"] = stuck or jump_on_spot # or off_track or flying
        info["stuck"] = stuck
        info["jump_on_spot"] = jump_on_spot
        #info["off_track"] = off_track
        #info["flying"] = flying
        

        if info["bug_detected"]:
            print("INFO complet :", info)
            print(
                f"⚠️  Bug détecté : "
                f"{ {k: v for k, v in info.items() if (isinstance(v, (bool, np.bool_)) and v)} } "
                f"à la position x={kart_pos[0]:.2f}, y={kart_pos[1]:.2f}, z={kart_pos[2]:.2f} "
                f"track name={self.track_name}"
            )

        return obs, reward, terminated, truncated, info

