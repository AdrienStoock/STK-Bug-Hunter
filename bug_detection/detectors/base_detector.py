# base_detector.py


from gymnasium import Wrapper


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