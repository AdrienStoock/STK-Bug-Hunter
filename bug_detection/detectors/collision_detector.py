#collision_detector.py 

from .base_detector import BaseDetectionWrapper

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