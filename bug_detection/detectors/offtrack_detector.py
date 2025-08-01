# offtrack_detector.py

from .base_detector import BaseDetectionWrapper

class OffTrackDetectionWrapper(BaseDetectionWrapper):
    """Detection of normal and abnormal off-track"""
    def __init__(self, env, threshold=2.0, max_duration=300, position_threshold=1000, velocity_threshold=0.1):
        super().__init__(env)
        self.threshold = threshold
        self.max_duration = max_duration  # 5 seconds at 60 FPS
        self.position_threshold = position_threshold
        self.velocity_threshold = velocity_threshold
        self.last_on_track_position = None
        self.off_track_history = []
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_position = obs['continuous'][0:3]
        current_velocity = obs['continuous'][3:6]
        center_path_distance = obs['continuous'][7]
        
        # Update the last position on the track
        if abs(center_path_distance) <= self.threshold:
            self.last_on_track_position = current_position.tolist()
            if self.is_detected:
                print(f"\n✅ Back on track after {self.detection_info['duration']/60:.2f} seconds")
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
                print(f"\n🚨 OFF-TRACK DETECTED:")
                print(f"Position: {current_position}")
                print(f"Distance du centre: {center_path_distance:.2f}")
            
            
            is_abnormal = False
            abnormal_reason = []
            
            # Too long duration
            if self.detection_info['duration'] > self.max_duration:
                is_abnormal = True
                abnormal_reason.append("Off-track duration excessive")
            
            # Anomalous position (outside limits)
            if any(abs(p) > self.position_threshold for p in current_position):
                is_abnormal = True
                abnormal_reason.append("Anomalous position")
            
            # Strange movement after off-track
            if np.linalg.norm(current_velocity) < self.velocity_threshold:
                is_abnormal = True
                abnormal_reason.append("Anomalous movement")
            
            # Excessive distance from the last position on the track
            if self.last_on_track_position is not None:
                distance = np.linalg.norm(np.array(current_position) - np.array(self.last_on_track_position))
                if distance > self.position_threshold:
                    is_abnormal = True
                    abnormal_reason.append("Excessive distance from the track")
            
            # Update the history and information
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
            
                # Display the information if it's abnormal
            if is_abnormal:
                print(f"\n⚠️ OFF-TRACK ANOMALOUS:")
                print(f"Raisons: {', '.join(abnormal_reason)}")
        
        # Update the information
        info['is_off_track'] = self.is_detected
        info['off_track_info'] = self.detection_info if self.is_detected else None
        info['off_track_history'] = self.off_track_history
        
        return obs, reward, terminated, truncated, info
