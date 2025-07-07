import os
import json
import numpy as np
from gymnasium import Wrapper
from datetime import datetime
import time
from bug_logger import BugLogger


class BaseDetectionWrapper(Wrapper):
    """Classe de base pour la détection d'événements"""
    def __init__(self, env):
        super().__init__(env)
        self.is_detected = False
        self.detection_info = {
            'position': None,
            'duration': 0,
            'start_time': None,
            'last_action_time': None
        }
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.is_detected = False
        self.detection_info = {
            'position': None,
            'duration': 0,
            'start_time': None,
            'last_action_time': None
        }
        return obs

class StuckDetectionWrapper(BaseDetectionWrapper):
    """Détection des blocages du kart"""
    def __init__(self, env, block_steps=180, min_progress=0.1):
        super().__init__(env)
        self.block_steps = block_steps
        self.min_progress = min_progress
        self.movement_history = []
        self.attempted_actions = []
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Récupérer les informations de position et vitesse
        current_position = obs['continuous'][0:3]
        current_velocity = obs['continuous'][3:6]
        
        # Calculer le mouvement
        if len(self.movement_history) > 0:
            last_position = self.movement_history[-1]
            movement = np.linalg.norm(np.array(current_position) - np.array(last_position))
            self.detection_info['movement_history'].append({
                'position': current_position.tolist(),
                'velocity': current_velocity.tolist(),
                'movement': float(movement)
            })
        self.movement_history.append(current_position)
        
        # Garder l'historique à jour
        if len(self.movement_history) > self.block_steps:
            self.movement_history.pop(0)
        
        # Vérifier le blocage
        if len(self.movement_history) == self.block_steps:
            total_movement = sum(entry['movement'] for entry in self.detection_info['movement_history'][-self.block_steps:])
            
            if total_movement < self.min_progress and current_velocity[0] < 0.1:
                if not self.is_detected:
                    self.is_detected = True
                    self.detection_info['position'] = current_position.tolist()
                    self.detection_info['start_time'] = self.detection_info['duration']
                    self.attempted_actions = []
                else:
                    self.attempted_actions.append(action)
                    self.detection_info['duration'] += 1
                    self.detection_info['last_action_time'] = self.detection_info['duration']
                    
                    if len(set(self.attempted_actions)) > 5 and self.detection_info['duration'] > 300:
                        print(f"\n🚨 BLOCAGE DÉTECTÉ:")
                        print(f"Position: {current_position}")
                        print(f"Vitesse: {current_velocity}")
                        print(f"Durée: {self.detection_info['duration']/60:.2f} secondes")
                        print(f"Actions tentées: {self.attempted_actions}")
                        print(f"Mouvement total: {total_movement:.2f}")
            else:
                if self.is_detected:
                    print(f"\n✅ Blocage résolu après {self.detection_info['duration']/60:.2f} secondes")
                    self.is_detected = False
                    self.attempted_actions = []
                    self.detection_info = {
                        'position': None,
                        'duration': 0,
                        'start_time': None,
                        'last_action_time': None,
                        'movement_history': []
                    }
        
        info['is_stuck'] = self.is_detected
        info['stuck_info'] = self.detection_info if self.is_detected else None
        
        return obs, reward, terminated, truncated, info

class CollisionDetectionWrapper(BaseDetectionWrapper):
    """Détection des collisions"""
    def __init__(self, env):
        super().__init__(env)
        self.collision_points = []
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_position = obs['continuous'][0:3]
        current_velocity = obs['continuous'][3:6]
        
        if info.get('collision', False):
            if not self.is_detected:
                self.is_detected = True
                self.detection_info['position'] = current_position.tolist()
                self.detection_info['start_time'] = self.detection_info['duration']
                print(f"\n💥 COLLISION DÉTECTÉE:")
                print(f"Position: {current_position}")
                print(f"Vitesse: {current_velocity}")
            
            self.collision_points.append({
                'position': current_position.tolist(),
                'velocity': current_velocity.tolist(),
                'time': self.detection_info['duration']
            })
            self.detection_info['duration'] += 1
        else:
            if self.is_detected:
                self.is_detected = False
                self.collision_points = []
                self.detection_info = {
                    'position': None,
                    'duration': 0,
                    'start_time': None,
                    'last_action_time': None
                }
        
        info['is_collision'] = self.is_detected
        info['collision_info'] = self.detection_info if self.is_detected else None
        
        return obs, reward, terminated, truncated, info

class OffTrackDetectionWrapper(BaseDetectionWrapper):
    """Détection des sorties de piste normales et anormales"""
    def __init__(self, env, threshold=2.0, max_duration=300, position_threshold=1000, velocity_threshold=0.1):
        super().__init__(env)
        self.threshold = threshold
        self.max_duration = max_duration  # 5 secondes à 60 FPS
        self.position_threshold = position_threshold
        self.velocity_threshold = velocity_threshold
        self.last_on_track_position = None
        self.off_track_history = []
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_position = obs['continuous'][0:3]
        current_velocity = obs['continuous'][3:6]
        center_path_distance = obs['continuous'][7]
        
        # Mettre à jour la dernière position sur la piste
        if abs(center_path_distance) <= self.threshold:
            self.last_on_track_position = current_position.tolist()
            if self.is_detected:
                print(f"\n✅ Retour sur piste après {self.detection_info['duration']/60:.2f} secondes")
                self.is_detected = False
                self.off_track_history = []
                self.detection_info = {
                    'position': None,
                    'duration': 0,
                    'start_time': None,
                    'last_action_time': None,
                    'is_abnormal': False,
                    'reasons': []
                }
        else:  # Off-track
            if not self.is_detected:
                self.is_detected = True
                self.detection_info['position'] = current_position.tolist()
                self.detection_info['start_time'] = self.detection_info['duration']
                print(f"\n🚨 HORS PISTE DÉTECTÉ:")
                print(f"Position: {current_position}")
                print(f"Distance du centre: {center_path_distance:.2f}")
            
            
            is_abnormal = False
            abnormal_reason = []
            
            # Durée trop longue
            if self.detection_info['duration'] > self.max_duration:
                is_abnormal = True
                abnormal_reason.append("Durée off-track excessive")
            
            # Position anormale (hors limites)
            if any(abs(p) > self.position_threshold for p in current_position):
                is_abnormal = True
                abnormal_reason.append("Position hors limites")
            
            # Mouvement étrange après la sortie
            if np.linalg.norm(current_velocity) < self.velocity_threshold:
                is_abnormal = True
                abnormal_reason.append("Mouvement anormal")
            
            # Distance excessive par rapport à la dernière position sur piste
            if self.last_on_track_position is not None:
                distance = np.linalg.norm(np.array(current_position) - np.array(self.last_on_track_position))
                if distance > self.position_threshold:
                    is_abnormal = True
                    abnormal_reason.append("Distance excessive depuis la piste")
            
            # Mettre à jour l'historique et les informations
            self.off_track_history.append({
                'position': current_position.tolist(),
                'velocity': current_velocity.tolist(),
                'time': self.detection_info['duration'],
                'is_abnormal': is_abnormal,
                'reasons': abnormal_reason
            })
            
            self.detection_info['duration'] += 1
            self.detection_info['is_abnormal'] = is_abnormal
            self.detection_info['reasons'] = abnormal_reason
            
            # Afficher les informations si c'est anormal
            if is_abnormal:
                print(f"\n⚠️ OFF-TRACK ANORMAL:")
                print(f"Raisons: {', '.join(abnormal_reason)}")
        
        # Mettre à jour les informations
        info['is_off_track'] = self.is_detected
        info['off_track_info'] = self.detection_info if self.is_detected else None
        info['off_track_history'] = self.off_track_history
        
        return obs, reward, terminated, truncated, info

class BugDetectionWrapper(Wrapper):
    """Wrapper principal qui combine toutes les détections"""
    def __init__(self, env, block_steps=180, min_progress=0.1, off_track_threshold=2.0):
        super().__init__(env)
        self.stuck_detector = StuckDetectionWrapper(env, block_steps, min_progress)
        self.collision_detector = CollisionDetectionWrapper(env)
        self.off_track_detector = OffTrackDetectionWrapper(env, off_track_threshold)
        
        # Créer le dossier pour les logs
        os.makedirs('bug_logs', exist_ok=True)
        self.log_file = f'bug_logs/bug_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
    def step(self, action):
        # Appliquer les détecteurs en chaîne
        obs, reward, terminated, truncated, info = self.stuck_detector.step(action)
        obs, reward, terminated, truncated, info = self.collision_detector.step(action)
        obs, reward, terminated, truncated, info = self.off_track_detector.step(action)
        
        # Combiner les informations de détection
        info['bug_detected'] = any([
            info.get('is_stuck', False),
            info.get('is_collision', False),
            info.get('is_off_track', False)
        ])
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.stuck_detector.reset(**kwargs)
        self.collision_detector.reset(**kwargs)
        self.off_track_detector.reset(**kwargs)
        return obs

class FPSDetectionWrapper(Wrapper):
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
        
