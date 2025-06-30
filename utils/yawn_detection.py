import numpy as np
import cv2

class YawnDetector:
    def __init__(self):
        # MediaPipe face mesh landmark indices for mouth
        self.mouth_landmarks = [
            # Outer lip landmarks
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            # Inner lip landmarks  
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415
        ]
        
        # Key points for MAR calculation
        self.mouth_points = [
            13, 14,    # Top lip center points
            78, 308,   # Left and right mouth corners
            18, 175    # Bottom lip center points
        ]
        
        # MAR thresholds and parameters
        self.mar_threshold = 0.7  # Above this indicates yawning
        self.yawn_consecutive_frames = 3  # Frames to confirm yawning
        self.max_yawn_duration = 30  # Maximum frames for a single yawn
        
        # State tracking
        self.mar_history = []
        self.yawn_frame_count = 0
        self.is_yawning = False
        self.yawn_start_time = None
        self.last_mar = 0.0
        self.yawn_events = []
        
    def calculate_mar(self, landmarks):
        """Calculate Mouth Aspect Ratio (MAR) from facial landmarks"""
        try:
            # Get mouth coordinates
            mouth_coords = []
            for point_idx in self.mouth_points:
                if point_idx < len(landmarks):
                    mouth_coords.append(landmarks[point_idx])
                else:
                    # Fallback coordinates
                    mouth_coords.append([0, 0])
                    
            if len(mouth_coords) < 6:
                return self.last_mar
                
            mouth_coords = np.array(mouth_coords)
            
            # Calculate MAR using improved formula
            # MAR = (|A-E| + |B-D| + |C-F|) / (2 * |G-H|)
            # Where A,B,C are top points, D,E,F are bottom points, G,H are corner points
            
            # Vertical distances (mouth height at different points)
            vertical_dist1 = np.linalg.norm(mouth_coords[0] - mouth_coords[4])  # Center top to center bottom
            vertical_dist2 = np.linalg.norm(mouth_coords[1] - mouth_coords[5])  # Another vertical measurement
            
            # Horizontal distance (mouth width)
            horizontal_dist = np.linalg.norm(mouth_coords[2] - mouth_coords[3])  # Left corner to right corner
            
            # Calculate MAR
            if horizontal_dist > 0:
                mar = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
            else:
                mar = 0.0
                
            # Update history
            self.mar_history.append(mar)
            if len(self.mar_history) > 15:  # Keep last 15 values
                self.mar_history.pop(0)
                
            self.last_mar = mar
            return mar
            
        except Exception as e:
            print(f"Error calculating MAR: {e}")
            return self.last_mar
            
    def detect_yawn(self, mar=None):
        """Detect if person is yawning based on MAR"""
        if mar is None:
            mar = self.last_mar
            
        # Check if MAR exceeds threshold
        if mar > self.mar_threshold:
            self.yawn_frame_count += 1
            
            # Confirm yawning after consecutive frames
            if self.yawn_frame_count >= self.yawn_consecutive_frames and not self.is_yawning:
                self.is_yawning = True
                self.yawn_start_time = len(self.mar_history)
                self.yawn_events.append({
                    'start_frame': len(self.mar_history),
                    'max_mar': mar,
                    'duration': 0
                })
                print(f"Yawn detected! MAR: {mar:.3f}")
                
        else:
            # End of yawn
            if self.is_yawning:
                self.is_yawning = False
                duration = self.yawn_frame_count
                if self.yawn_events:
                    self.yawn_events[-1]['duration'] = duration
                print(f"Yawn ended. Duration: {duration} frames")
                
            self.yawn_frame_count = 0
            
        # Prevent extremely long yawn detection (likely false positive)
        if self.yawn_frame_count > self.max_yawn_duration:
            self.is_yawning = False
            self.yawn_frame_count = 0
            
        return self.is_yawning
        
    def get_yawn_intensity(self, mar=None):
        """Get yawn intensity as a percentage"""
        if mar is None:
            mar = self.last_mar
            
        if mar <= 0.3:
            return 0  # No yawn
        elif mar <= 0.5:
            return 25  # Slight mouth opening
        elif mar <= 0.7:
            return 50  # Moderate opening
        elif mar <= 1.0:
            return 75  # Strong yawn
        else:
            return 100  # Very strong yawn
            
    def get_average_mar(self):
        """Get average MAR from recent history"""
        if not self.mar_history:
            return 0.0
        return sum(self.mar_history) / len(self.mar_history)
        
    def draw_mouth_contour(self, frame, landmarks):
        """Draw mouth contour on the frame"""
        try:
            mouth_coords = []
            for idx in self.mouth_landmarks:
                if idx < len(landmarks):
                    mouth_coords.append(landmarks[idx])
                    
            if len(mouth_coords) > 3:
                mouth_coords = np.array(mouth_coords, dtype=np.int32)
                
                # Draw mouth contour
                if self.is_yawning:
                    color = (0, 0, 255)  # Red for yawning
                    thickness = 2
                else:
                    color = (255, 0, 0)  # Blue for normal
                    thickness = 1
                    
                cv2.polylines(frame, [mouth_coords], True, color, thickness)
                
                # Draw key points
                for point_idx in self.mouth_points:
                    if point_idx < len(landmarks):
                        cv2.circle(frame, tuple(landmarks[point_idx]), 2, (0, 255, 255), -1)
                        
        except Exception as e:
            print(f"Error drawing mouth contour: {e}")
            
    def is_mouth_open(self, mar=None):
        """Check if mouth is significantly open"""
        if mar is None:
            mar = self.last_mar
        return mar > 0.5
        
    def get_mouth_status(self):
        """Get current mouth status as a string"""
        mar = self.last_mar
        
        if self.is_yawning:
            return "Yawning"
        elif mar > 0.5:
            return "Open"
        elif mar > 0.3:
            return "Slightly Open"
        else:
            return "Closed"
            
    def reset_state(self):
        """Reset the yawn detector state"""
        self.mar_history.clear()
        self.yawn_frame_count = 0
        self.is_yawning = False
        self.yawn_start_time = None
        self.last_mar = 0.0
        self.yawn_events.clear()
        
    def get_yawn_statistics(self):
        """Get statistics about yawning events"""
        if not self.yawn_events:
            return {
                'total_yawns': 0,
                'average_duration': 0,
                'max_mar': 0
            }
            
        total_yawns = len(self.yawn_events)
        durations = [event['duration'] for event in self.yawn_events if event['duration'] > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_mar = max([event['max_mar'] for event in self.yawn_events])
        
        return {
            'total_yawns': total_yawns,
            'average_duration': avg_duration,
            'max_mar': max_mar,
            'recent_yawns': self.yawn_events[-5:]  # Last 5 yawn events
        }