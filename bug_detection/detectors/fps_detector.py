import time
import numpy as np
from datetime import datetime
from .base_detector import BaseDetectionWrapper
from bug_logger import BugLogger
import os


class FPSDetectionWrapper(BaseDetectionWrapper):
    def __init__(self, env, fps_threshold=20, min_duration_seconds=5, track_name=None):
        super().__init__(env)
        self.fps_threshold = fps_threshold
        self.min_duration = min_duration_seconds
        self.low_fps_start_time = None
        self.bug_logged_for_track = False
        self.track_name = track_name
        self.low_movement_logged = False
        self.last_movement_log_time = None
        
        os.makedirs('bug_logs', exist_ok=True)
        # Nouveau fichier de log à chaque lancement
        log_filename = f"bug_logs/bug_log_fps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.logger = BugLogger(log_file=log_filename)
        self.last_step_time = None
        self.last_log_time = None
        self.positions_history = []
        self.positions_history_maxlen = 60  # Surveille les 60 dernières frames

    def step(self, action):
        current_time = time.time()

        if self.last_step_time is None:
            self.last_step_time = current_time
            fps = None
        else:
            elapsed = current_time - self.last_step_time
            self.last_step_time = current_time
            fps = 1.0 / elapsed if elapsed > 0 else None

        obs, reward, terminated, truncated, info = self.env.step(action)
        #print("obs :", obs)
        
        #print("info : ", info)

        # Récupération de la position du kart
        kart_pos = obs["karts_position"][0]
        self.positions_history.append(np.array(kart_pos))
        if len(self.positions_history) > self.positions_history_maxlen:
            self.positions_history.pop(0)

        if fps is not None:
            if fps < self.fps_threshold:
                if self.low_fps_start_time is None:
                    self.low_fps_start_time = datetime.now()
                else:
                    duration = (datetime.now() - self.low_fps_start_time).total_seconds()
                    if duration >= self.min_duration:
                        now = time.time()
                        if self.last_log_time is None or (now - self.last_log_time) >= 5:
                            self.last_log_time = now
                            kart_position = {
                                "x": float(kart_pos[0]),
                                "y": float(kart_pos[1]),
                                "z": float(kart_pos[2])
                            }
                            self.log_bug(fps, duration, kart_position)
                            print(f"\n🚨 BUG FPS détecté : FPS < {self.fps_threshold} pendant {duration:.1f} secondes")

                        # Calcul du déplacement total
                        total_distance = 0.0
                        for i in range(1, len(self.positions_history)):
                            total_distance += np.linalg.norm(self.positions_history[i] - self.positions_history[i - 1])

                        if total_distance < 5.0:
                            if self.last_movement_log_time is None or (now - self.last_movement_log_time) > 5:
                                print("⚠️ Pendant le bug FPS, le kart n'a presque pas bougé !")
                                print(f"Position actuelle : {self.positions_history[-1]}")
                                print(f"Déplacement total pendant le bug FPS : {total_distance:.2f}")
                                self.last_movement_log_time = now
                        else:
                            # Si le kart recommence à bouger
                            if self.last_movement_log_time is not None:
                                print("✅ Le kart a recommencé à bouger.")
                                self.last_movement_log_time = None
            else:
                # FPS revenu à la normale : reset
                self.low_fps_start_time = None
                self.bug_logged_for_track = False
                self.last_log_time = None
                self.positions_history = []
                self.last_movement_log_time = None

        info['fps'] = fps
        info['fps_bug_detected'] = self.bug_logged_for_track
        return obs, reward, terminated, truncated, info




    def reset(self, **kwargs):
        self.low_fps_start_time = None
        self.bug_logged_for_track = False
        self.last_step_time = None
        self.last_log_time = None
        return self.env.reset(**kwargs)

    def log_bug(self, fps, duration, kart_position):
        bug_info = {
            'fps_threshold': self.fps_threshold,
            'fps_measured': fps,
            'duration_seconds': duration,
            'track': self.track_name
        }

        self.logger.log_bug(
            bug_type="Low FPS",
            kart_position=kart_position,
            additional_info=bug_info
        )
        