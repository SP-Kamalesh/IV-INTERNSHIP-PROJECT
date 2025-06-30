import cv2
import mediapipe as mp
import time

class FacePresenceDetector:
    def __init__(self, short_absence_threshold=3.0, long_absence_threshold=10.0):
        """
        Initialize the Face Presence Detector.
        
        Args:
            short_absence_threshold (float): Time in seconds for "Inactive (Face Missing)" status
            long_absence_threshold (float): Time in seconds for "Not Awake" status
        """
        self.short_absence_threshold = short_absence_threshold
        self.long_absence_threshold = long_absence_threshold
        self.face_lost_time = None
        self.last_face_detected_time = time.time()
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # Short-range model for webcam
            min_detection_confidence=0.5
        )
        
    def check_face_presence(self, image):
        """
        Check face presence and return appropriate status.
        
        Args:
            image (numpy.ndarray): Input image from camera
            
        Returns:
            str: "Active", "Inactive (Face Missing)", or "Not Awake"
        """
        if image is None:
            return self._handle_no_face()
            
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_detection.process(rgb_image)
        
        current_time = time.time()
        
        if results.detections and len(results.detections) > 0:
            # Face detected - reset timers
            self.last_face_detected_time = current_time
            self.face_lost_time = None
            return "Active"
        else:
            # No face detected
            return self._handle_no_face()
    
    def _handle_no_face(self):
        """
        Handle the case when no face is detected.
        
        Returns:
            str: Appropriate status based on absence duration
        """
        current_time = time.time()
        
        # Set face lost time if not already set
        if self.face_lost_time is None:
            self.face_lost_time = current_time
        
        # Calculate absence duration
        absence_duration = current_time - self.face_lost_time
        
        # Return status based on absence duration
        if absence_duration >= self.long_absence_threshold:
            return "Not Awake"
        elif absence_duration >= self.short_absence_threshold:
            return "Inactive (Face Missing)"
        else:
            return "Active"  # Brief absence, still considered active
    
    def get_absence_duration(self):
        """
        Get the current absence duration in seconds.
        
        Returns:
            float: Duration in seconds since face was last detected, 0 if face is present
        """
        if self.face_lost_time is None:
            return 0.0
        return time.time() - self.face_lost_time
    
    def get_time_since_last_detection(self):
        """
        Get time since last face detection.
        
        Returns:
            float: Time in seconds since last face detection
        """
        return time.time() - self.last_face_detected_time
    
    def reset(self):
        """Reset the detector state."""
        self.face_lost_time = None
        self.last_face_detected_time = time.time()
    
    def is_face_present(self, image):
        """
        Simple boolean check for face presence.
        
        Args:
            image (numpy.ndarray): Input image from camera
            
        Returns:
            bool: True if face is detected, False otherwise
        """
        if image is None:
            return False
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        return results.detections is not None and len(results.detections) > 0
    
    def get_face_confidence(self, image):
        """
        Get the confidence score of face detection.
        
        Args:
            image (numpy.ndarray): Input image from camera
            
        Returns:
            float: Confidence score (0.0 to 1.0), 0.0 if no face detected
        """
        if image is None:
            return 0.0
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        if results.detections and len(results.detections) > 0:
            # Return the highest confidence score if multiple faces
            max_confidence = max(detection.score[0] for detection in results.detections)
            return max_confidence
        
        return 0.0
    
    def get_status_info(self):
        """
        Get detailed status information for debugging.
        
        Returns:
            dict: Dictionary containing status information
        """
        current_time = time.time()
        info = {
            'face_present': self.face_lost_time is None,
            'absence_duration': self.get_absence_duration(),
            'time_since_last_detection': current_time - self.last_face_detected_time,
            'short_threshold': self.short_absence_threshold,
            'long_threshold': self.long_absence_threshold
        }
        return info

def check_face_presence(image):
    """
    Standalone function for face presence detection.
    
    Args:
        image (numpy.ndarray): Input image from camera
        
    Returns:
        str: "Active", "Inactive (Face Missing)", or "Not Awake"
    """
    # Note: This creates a new detector each time, so it won't maintain state
    # For persistent state tracking, use the class instance in your main application
    detector = FacePresenceDetector()
    return detector.check_face_presence(image)

def is_face_present(image):
    """
    Standalone function for simple face presence check.
    
    Args:
        image (numpy.ndarray): Input image from camera
        
    Returns:
        bool: True if face is detected, False otherwise
    """
    detector = FacePresenceDetector()
    return detector.is_face_present(image)