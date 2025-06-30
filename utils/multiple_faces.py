import cv2
import mediapipe as mp
import numpy as np

class MultipleFaceDetector:
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the Multiple Face Detector.
        
        Args:
            confidence_threshold (float): Minimum confidence for face detection
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # Short-range model for webcam
            min_detection_confidence=confidence_threshold
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def check_multiple_faces(self, image):
        """
        Check if multiple faces are present in the image.
        
        Args:
            image (numpy.ndarray): Input image from camera
            
        Returns:
            str or None: "Multiple Persons Detected" if more than one face found, None otherwise
        """
        if image is None:
            return None
            
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_detection.process(rgb_image)
        
        # Count faces
        if results.detections:
            num_faces = len(results.detections)
            if num_faces > 1:
                return "Multiple Persons Detected"
        
        return None
    
    def process_multiple_faces(self, image, face_landmarks_list):
        """
        Process and draw bounding boxes for multiple faces.
        Used by master_gui.py for visualization.
        
        Args:
            image (numpy.ndarray): Input image
            face_landmarks_list (list): List of face landmarks from MediaPipe face mesh
            
        Returns:
            numpy.ndarray: Image with face detection annotations
        """
        if len(face_landmarks_list) > 1:
            h, w = image.shape[:2]
            
            for i, face_landmarks in enumerate(face_landmarks_list):
                # Get bounding box for each face
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append([x, y])
                
                # Calculate bounding box
                x_coords = [p[0] for p in landmarks]
                y_coords = [p[1] for p in landmarks]
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)
                
                # Draw red bounding box for multiple faces
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                
                # Add face number label
                cv2.putText(image, f"Person {i+1}", (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add warning text
            cv2.putText(image, "MULTIPLE PERSONS DETECTED!", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        return image

def check_multiple_faces(image):
    """
    Standalone function for checking multiple faces.
    
    Args:
        image (numpy.ndarray): Input image from camera
        
    Returns:
        str or None: "Multiple Persons Detected" if more than one face found, None otherwise
    """
    detector = MultipleFaceDetector()
    return detector.check_multiple_faces(image)