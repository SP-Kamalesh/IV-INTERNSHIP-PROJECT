import numpy as np
import cv2
import math

class HeadTurnDetector:
    def __init__(self):
        # Key facial landmarks for head pose estimation
        self.nose_tip = 1          # Nose tip
        self.chin = 175            # Chin
        self.left_eye_corner = 33  # Left eye outer corner
        self.right_eye_corner = 362 # Right eye outer corner
        self.left_mouth_corner = 61 # Left mouth corner
        self.right_mouth_corner = 291 # Right mouth corner
        
        # Head pose thresholds (in degrees)
        self.turn_threshold = 25    # Degrees to consider as turned away
        self.extreme_turn_threshold = 45  # Degrees for extreme turn
        
        # State tracking
        self.head_pose_history = []
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.is_looking_away = False
        self.looking_away_frames = 0
        
        # 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye corner
            (225.0, 170.0, -135.0),    # Right eye corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ])
        
    def estimate_head_pose(self, landmarks, frame_shape):
        """Estimate head pose using facial landmarks"""
        try:
            # Get 2D image points
            image_points = np.array([
                landmarks[self.nose_tip],
                landmarks[self.chin],
                landmarks[self.left_eye_corner],
                landmarks[self.right_eye_corner],
                landmarks[self.left_mouth_corner],
                landmarks[self.right_mouth_corner]
            ], dtype=np.float32)
            
            # Camera parameters (approximate)
            height, width = frame_shape[:2]
            focal_length = width
            center = (width // 2, height // 2)
            
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Assume no lens distortion
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs
            )
            
            if success:
                # Convert rotation vector to Euler angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                yaw, pitch, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
                
                # Convert to degrees
                self.current_yaw = math.degrees(yaw)
                self.current_pitch = math.degrees(pitch)
                self.current_roll = math.degrees(roll)
                
                # Update history
                self.head_pose_history.append({
                    'yaw': self.current_yaw,
                    'pitch': self.current_pitch,
                    'roll': self.current_roll
                })
                
                if len(self.head_pose_history) > 10:
                    self.head_pose_history.pop(0)
                    
                return self.current_yaw, self.current_pitch, self.current_roll
            else:
                return 0.0, 0.0, 0.0
                
        except Exception as e:
            print(f"Error estimating head pose: {e}")
            return 0.0, 0.0, 0.0
            
    def _rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles"""
        try:
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            
            singular = sy < 1e-6
            
            if not singular:
                x = math.atan2(R[2, 1], R[2, 2])
                y = math.atan2(-R[2, 0], sy)
                z = math.atan2(R[1, 0], R[0, 0])
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0
                
            return x, y, z
            
        except Exception as e:
            print(f"Error converting rotation matrix: {e}")
            return 0.0, 0.0, 0.0
            
    def detect_head_turn(self, landmarks, frame_shape):
        """Detect if head is turned away from camera"""
        yaw, pitch, roll = self.estimate_head_pose(landmarks, frame_shape)
        
        # Check if head is turned away
        head_turned = (abs(yaw) > self.turn_threshold or 
                      abs(pitch) > self.turn_threshold)
        
        if head_turned:
            self.looking_away_frames += 1
            if self.looking_away_frames >= 3:  # Confirm after 3 frames
                self.is_looking_away = True
        else:
            self.looking_away_frames = 0
            self.is_looking_away = False
            
        return self.is_looking_away
        
    def get_head_direction(self):
        """Get current head direction as string"""
        yaw = self.current_yaw
        pitch = self.current_pitch
        
        direction = "Forward"
        
        if abs(yaw) > self.extreme_turn_threshold:
            if yaw > 0:
                direction = "Far Right"
            else:
                direction = "Far Left"
        elif abs(yaw) > self.turn_threshold:
            if yaw > 0:
                direction = "Right"
            else:
                direction = "Left"
        elif abs(pitch) > self.turn_threshold:
            if pitch > 0:
                direction = "Up"
            else:
                direction = "Down"
                
        return direction
        
    def get_attention_score(self):
        """Get attention score based on head pose (0-100)"""
        yaw_score = max(0, 100 - abs(self.current_yaw) * 2)
        pitch_score = max(0, 100 - abs(self.current_pitch) * 2)
        
        # Average of yaw and pitch scores
        attention_score = (yaw_score + pitch_score) / 2
        
        return min(100, max(0, attention_score))
        
    def draw_head_pose(self, frame, landmarks):
        """Draw head pose visualization on frame"""
        try:
            # Get nose tip position
            nose_tip = landmarks[self.nose_tip]
            
            # Calculate direction vectors based on head pose
            yaw_rad = math.radians(self.current_yaw)
            pitch_rad = math.radians(self.current_pitch)
            
            # Calculate end points for direction lines
            length = 100
            end_x = int(nose_tip[0] + length * math.sin(yaw_rad))
            end_y = int(nose_tip[1] - length * math.sin(pitch_rad))
            
            # Draw direction line
            cv2.arrowedLine(frame, tuple(nose_tip), (end_x, end_y), (255, 0, 0), 3)
            
            # Draw pose information
            pose_text = f"Yaw: {self.current_yaw:.1f}°"
            cv2.putText(frame, pose_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            pitch_text = f"Pitch: {self.current_pitch:.1f}°"
            cv2.putText(frame, pitch_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            direction_text = f"Direction: {self.get_head_direction()}"
            cv2.putText(frame, direction_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        except Exception as e:
            print(f"Error drawing head pose: {e}")
            
    def is_looking_at_camera(self):
        """Check if person is looking at camera"""
        return not self.is_looking_away
        
    def get_head_pose_summary(self):
        """Get summary of head pose data"""
        if not self.head_pose_history:
            return {
                'current_yaw': 0,
                'current_pitch': 0,
                'current_roll': 0,
                'average_yaw': 0,
                'average_pitch': 0,
                'is_looking_away': False,
                'attention_score': 100
            }
            
        yaws = [pose['yaw'] for pose in self.head_pose_history]
        pitches = [pose['pitch'] for pose in self.head_pose_history]
        
        return {
            'current_yaw': self.current_yaw,
            'current_pitch': self.current_pitch,
            'current_roll': self.current_roll,
            'average_yaw': sum(yaws) / len(yaws),
            'average_pitch': sum(pitches) / len(pitches),
            'is_looking_away': self.is_looking_away,
            'attention_score': self.get_attention_score()
        }
        
    def reset_state(self):
        """Reset head turn detector state"""
        self.head_pose_history.clear()
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.is_looking_away = False
        self.looking_away_frames = 0