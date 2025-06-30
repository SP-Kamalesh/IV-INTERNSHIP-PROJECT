import numpy as np
import cv2
from scipy.spatial import distance

class EyeTracker:
    def __init__(self):
        # MediaPipe face mesh landmark indices for eyes
        self.left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Simplified eye landmarks for EAR calculation
        self.left_eye_points = [33, 160, 158, 133, 153, 144]  # Outer corner, top, bottom, inner corner, top, bottom
        self.right_eye_points = [362, 385, 387, 263, 373, 380]  # Outer corner, top, bottom, inner corner, top, bottom
        
        # EAR thresholds
        self.ear_threshold = 0.25  # Below this is considered closed/drowsy
        self.consecutive_frames = 3  # Frames to confirm drowsiness
        
        # State tracking
        self.ear_history = []
        self.drowsy_frame_count = 0
        self.last_ear = 0.0
        
    def calculate_ear(self, landmarks):
        """Calculate Eye Aspect Ratio (EAR) from facial landmarks"""
        try:
            # Get left and right eye EAR
            left_ear = self._calculate_single_eye_ear(landmarks, self.left_eye_points)
            right_ear = self._calculate_single_eye_ear(landmarks, self.right_eye_points)
            
            # Average of both eyes
            ear = (left_ear + right_ear) / 2.0
            
            # Update history
            self.ear_history.append(ear)
            if len(self.ear_history) > 10:  # Keep last 10 values
                self.ear_history.pop(0)
                
            self.last_ear = ear
            return ear
            
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return self.last_ear
            
    def _calculate_single_eye_ear(self, landmarks, eye_points):
        """Calculate EAR for a single eye"""
        try:
            # Get eye landmark coordinates
            eye_coords = []
            for point_idx in eye_points:
                if point_idx < len(landmarks):
                    eye_coords.append(landmarks[point_idx])
                else:
                    # Fallback if landmark index is out of range
                    eye_coords.append([0, 0])
                    
            if len(eye_coords) < 6:
                return 0.3  # Default EAR value
                
            # Convert to numpy array for easier calculation
            eye_coords = np.array(eye_coords)
            
            # Calculate EAR using the formula:
            # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
            # Where p1,p4 are horizontal points and p2,p3,p5,p6 are vertical points
            
            # Vertical distances
            vertical_dist1 = np.linalg.norm(eye_coords[1] - eye_coords[5])  # Top to bottom
            vertical_dist2 = np.linalg.norm(eye_coords[2] - eye_coords[4])  # Top to bottom
            
            # Horizontal distance
            horizontal_dist = np.linalg.norm(eye_coords[0] - eye_coords[3])  # Left to right
            
            # Calculate EAR
            if horizontal_dist > 0:
                ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
            else:
                ear = 0.3  # Default value if horizontal distance is 0
                
            return ear
            
        except Exception as e:
            print(f"Error in single eye EAR calculation: {e}")
            return 0.3
            
    def is_drowsy(self, ear=None):
        """Determine if the person is drowsy based on EAR"""
        if ear is None:
            ear = self.last_ear
            
        if ear < self.ear_threshold:
            self.drowsy_frame_count += 1
        else:
            self.drowsy_frame_count = 0
            
        return self.drowsy_frame_count >= self.consecutive_frames
        
    def get_drowsiness_level(self, ear=None):
        """Get drowsiness level as a percentage"""
        if ear is None:
            ear = self.last_ear
            
        if ear >= 0.3:
            return 0  # Fully awake
        elif ear >= 0.25:
            return 25  # Slightly drowsy
        elif ear >= 0.2:
            return 50  # Moderately drowsy
        elif ear >= 0.15:
            return 75  # Very drowsy
        else:
            return 100  # Extremely drowsy/eyes closed
            
    def get_average_ear(self):
        """Get average EAR from recent history"""
        if not self.ear_history:
            return 0.0
        return sum(self.ear_history) / len(self.ear_history)
        
    def draw_eye_contours(self, frame, landmarks):
        """Draw eye contours on the frame"""
        try:
            # Draw left eye
            left_eye_coords = []
            for idx in self.left_eye_landmarks:
                if idx < len(landmarks):
                    left_eye_coords.append(landmarks[idx])
                    
            if len(left_eye_coords) > 3:
                left_eye_coords = np.array(left_eye_coords, dtype=np.int32)
                cv2.polylines(frame, [left_eye_coords], True, (0, 255, 0), 1)
                
            # Draw right eye
            right_eye_coords = []
            for idx in self.right_eye_landmarks:
                if idx < len(landmarks):
                    right_eye_coords.append(landmarks[idx])
                    
            if len(right_eye_coords) > 3:
                right_eye_coords = np.array(right_eye_coords, dtype=np.int32)
                cv2.polylines(frame, [right_eye_coords], True, (0, 255, 0), 1)
                
        except Exception as e:
            print(f"Error drawing eye contours: {e}")
            
    def reset_state(self):
        """Reset the eye tracker state"""
        self.ear_history.clear()
        self.drowsy_frame_count = 0
        self.last_ear = 0.0
        
    def get_eye_status(self):
        """Get current eye status as a string"""
        ear = self.last_ear
        
        if ear >= 0.3:
            return "Wide Open"
        elif ear >= 0.25:
            return "Normal"
        elif ear >= 0.2:
            return "Slightly Closed"
        elif ear >= 0.15:
            return "Half Closed"
        else:
            return "Closed"