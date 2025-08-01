import os
from datetime import datetime
from gymnasium import Wrapper
from .detectors.stuck_detector import StuckDetectionWrapper
from .detectors.collision_detector import CollisionDetectionWrapper
from .detectors.offtrack_detector import OffTrackDetectionWrapper


class BugDetectionWrapper(Wrapper):
    """Wrapper principal qui combine toutes les détections"""
    def __init__(self, env, block_steps=180, min_progress=0.1, off_track_threshold=2.0):
        super().__init__(env)
        self.stuck_detector = StuckDetectionWrapper(env, block_steps, min_progress)
        self.collision_detector = CollisionDetectionWrapper(env)
        self.off_track_detector = OffTrackDetectionWrapper(env, off_track_threshold)
        
        os.makedirs('bug_logs', exist_ok=True)
        self.log_file = f'bug_logs/bug_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.stuck_detector.step(action)
        obs, reward, terminated, truncated, info = self.collision_detector.step(action)
        obs, reward, terminated, truncated, info = self.off_track_detector.step(action)
        
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
