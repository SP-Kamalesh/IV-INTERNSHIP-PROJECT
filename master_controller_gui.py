import sys
import cv2
import numpy as np
import time
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette
import mediapipe as mp

from utils.activity_logger import ActivityLogger
from utils.ai_feedback import AIFeedback
from utils.eye_tracking import EyeTracker
from utils.yawn_detection import YawnDetector
from utils.head_turn import HeadTurnDetector
from utils.multiple_faces import MultipleFaceDetector
from utils.face_presence import FacePresenceDetector
from utils.emergency_wakeup import EmergencyWakeup

class AttentivenessMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Based Student Attentiveness Monitoring System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Initialize detection modules 
        self.activity_logger = ActivityLogger()
        self.ai_feedback = AIFeedback()
        self.eye_tracker = EyeTracker()
        self.yawn_detector = YawnDetector()
        self.head_turn_detector = HeadTurnDetector()
        self.multiple_face_detector = MultipleFaceDetector()
        self.face_presence_detector = FacePresenceDetector()
        self.emergency_wakeup = EmergencyWakeup()
        
        # Connect emergency flash signal
        self.emergency_wakeup.flash_signal.connect(self.handle_emergency_flash)
        
        # State variables
        self.detection_active = False
        self.current_status = "Inactive"
        self.last_active_time = time.time()
        self.inactive_start_time = None
        self.emergency_triggered = False
        self.inactivity_threshold = 8.0  # 8 seconds for inactivity
        self.previous_status = "Inactive"
        self.inactive_duration_start = None
        
        # Enhanced state tracking
        self.last_ear = 0.0
        self.last_mar = 0.0
        self.drowsy_counter = 0
        self.yawn_counter = 0
        self.status_stability_counter = 0
        self.drowsy_threshold = 5  # Need 5 consecutive frames
        self.yawn_threshold = 3   # Need 3 consecutive frames
        self.stability_threshold = 3  # Status must be stable for 3 frames
        
        # Store original stylesheet for restoration
        self.original_stylesheet = ""
        
        # MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.setup_ui()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.central_widget = central_widget  # Store reference for emergency flashing
        
        # Store original stylesheet
        self.original_stylesheet = central_widget.styleSheet()
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left side - Camera feed
        left_layout = QVBoxLayout()
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(800, 600)
        self.camera_label.setStyleSheet("border: 2px solid black; background-color: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("Camera Feed")
        left_layout.addWidget(self.camera_label)
        
        # Right side - Controls and status
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)
        
        # Status display
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.status_label.setStyleSheet("color: red; padding: 10px; border: 2px solid gray;")
        right_layout.addWidget(self.status_label)
        
        # Emergency indicator
        self.emergency_label = QLabel("EMERGENCY: OFF")
        self.emergency_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.emergency_label.setStyleSheet("color: gray; padding: 10px; border: 2px solid gray; background-color: lightgray;")
        right_layout.addWidget(self.emergency_label)
        
        # Inactivity timer display
        self.inactivity_label = QLabel("Inactive Time: 0.0s")
        self.inactivity_label.setFont(QFont("Arial", 12))
        self.inactivity_label.setStyleSheet("color: black; padding: 5px;")
        right_layout.addWidget(self.inactivity_label)
        
        # EAR display
        self.ear_label = QLabel("EAR: 0.000")
        self.ear_label.setFont(QFont("Arial", 14))
        self.ear_label.setStyleSheet("color: blue; padding: 5px;")
        right_layout.addWidget(self.ear_label)
        
        # MAR display
        self.mar_label = QLabel("MAR: 0.000")
        self.mar_label.setFont(QFont("Arial", 14))
        self.mar_label.setStyleSheet("color: green; padding: 5px;")
        right_layout.addWidget(self.mar_label)
        
        # Detection counters display
        self.counters_label = QLabel("Drowsy: 0 | Yawn: 0")
        self.counters_label.setFont(QFont("Arial", 10))
        self.counters_label.setStyleSheet("color: purple; padding: 5px;")
        right_layout.addWidget(self.counters_label)
        
        # Control buttons
        self.open_camera_btn = QPushButton("Open Camera")
        self.open_camera_btn.clicked.connect(self.open_camera)
        right_layout.addWidget(self.open_camera_btn)
        
        self.close_camera_btn = QPushButton("Stop Camera")
        self.close_camera_btn.clicked.connect(self.close_camera)
        right_layout.addWidget(self.close_camera_btn)
        
        self.start_detection_btn = QPushButton("Start Detection")
        self.start_detection_btn.clicked.connect(self.start_detection)
        right_layout.addWidget(self.start_detection_btn)
        
        self.stop_detection_btn = QPushButton("Stop Detection")
        self.stop_detection_btn.clicked.connect(self.stop_detection)
        right_layout.addWidget(self.stop_detection_btn)
        
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close_application)
        right_layout.addWidget(self.exit_btn)
        
        right_layout.addStretch()
        
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)
        
    def handle_emergency_flash(self, color):
        """Handle emergency screen flashing"""
        if color == "red":
            self.central_widget.setStyleSheet("background-color: #ff4444;")
            self.emergency_label.setText("🚨 EMERGENCY: ACTIVE 🚨")
            self.emergency_label.setStyleSheet("color: white; padding: 10px; border: 2px solid white; background-color: red; font-weight: bold;")
        elif color == "blue":
            self.central_widget.setStyleSheet("background-color: #4444ff;")
            self.emergency_label.setText("🚨 WAKE UP! 🚨")
            self.emergency_label.setStyleSheet("color: white; padding: 10px; border: 2px solid white; background-color: blue; font-weight: bold;")
        else:  # normal
            self.central_widget.setStyleSheet(self.original_stylesheet)
            self.emergency_label.setText("EMERGENCY: OFF")
            self.emergency_label.setStyleSheet("color: gray; padding: 10px; border: 2px solid gray; background-color: lightgray;")
        
    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer.start(30)  # 30ms refresh rate
            self.open_camera_btn.setEnabled(False)
            self.close_camera_btn.setEnabled(True)
            
    def close_camera(self):
        if self.cap:
            self.cap.release()
            self.timer.stop()
            self.camera_label.clear()
            self.camera_label.setText("Camera Feed")
            self.open_camera_btn.setEnabled(True)
            self.close_camera_btn.setEnabled(False)
            
    def start_detection(self):
        self.detection_active = True
        self.last_active_time = time.time()
        self.inactive_start_time = None
        self.emergency_triggered = False
        self.previous_status = "Inactive"
        
        # Reset counters
        self.drowsy_counter = 0
        self.yawn_counter = 0
        self.status_stability_counter = 0
        
        # Reset detectors
        self.face_presence_detector.reset()
        
        # Reset logger's inactive tracking
        self.activity_logger.reset_inactive_tracking()
        
        # Log detection session start
        self.activity_logger.log_detection_session("started")
        
        self.start_detection_btn.setEnabled(False)
        self.stop_detection_btn.setEnabled(True)
        
    def stop_detection(self):
        self.detection_active = False
        
        # Log detection session stop
        self.activity_logger.log_detection_session("stopped")
        
        # Stop emergency protocol safely
        try:
            if hasattr(self, 'emergency_wakeup') and self.emergency_wakeup:
                self.emergency_wakeup.stop_emergency()
        except Exception as e:
            print(f"Error stopping emergency during detection stop: {e}")
            
        self.emergency_triggered = False
        self.start_detection_btn.setEnabled(True)
        self.stop_detection_btn.setEnabled(False)
        self.update_status("Inactive")
        self.inactivity_label.setText("Inactive Time: 0.0s")
        
    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror the image
                
                if self.detection_active:
                    frame = self.process_frame(frame)
                    
                # Convert frame to Qt format and display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Scale image to fit label
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.camera_label.setPixmap(scaled_pixmap)
                
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        current_time = time.time()
        
        # Check for multiple faces first - but less aggressively
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 1:
            # Only trigger multiple faces if it's been consistent
            multiple_faces_status = self.multiple_face_detector.check_multiple_faces(frame)
            if multiple_faces_status:
                self.update_status(multiple_faces_status)
                self.handle_inactivity_tracking("Multiple Faces", current_time)
                frame = self.multiple_face_detector.process_multiple_faces(frame, results.multi_face_landmarks)
                return frame
        
        # Check face presence
        if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
            face_presence_status = self.face_presence_detector.check_face_presence(frame)
            self.handle_inactivity_tracking(face_presence_status, current_time)
            self.update_status(face_presence_status)
            return frame
        
        # Single face detected - process normally
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Get landmark coordinates
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
            
        # Draw face rectangle
        face_box = self.get_face_bounding_box(landmarks)
        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 255, 0), 2)
        
        # Eye tracking with improved thresholds
        ear = self.eye_tracker.calculate_ear(landmarks)
        self.ear_label.setText(f"EAR: {ear:.3f}")
        self.last_ear = ear
        
        # Yawn detection with improved thresholds
        mar = self.yawn_detector.calculate_mar(landmarks)
        self.mar_label.setText(f"MAR: {mar:.3f}")
        self.last_mar = mar
        
        # Determine status using improved logic
        status = self.determine_status_improved(ear, mar, current_time)
        
        # Update counters display
        self.counters_label.setText(f"Drowsy: {self.drowsy_counter} | Yawn: {self.yawn_counter}")
        
        # Draw status on frame
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, status.upper(), (face_box[0], face_box[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw eye landmarks
        self.draw_eye_landmarks(frame, landmarks)
        
        # Handle inactivity tracking
        if status == "Active":
            self.reset_inactivity_tracking()
        else:
            self.handle_inactivity_tracking(status, current_time)
        
        self.update_status(status)
        
        return frame
    
    def determine_status_improved(self, ear, mar, current_time):
        """Improved status determination with stability checks"""
        
        # Check for drowsiness (eyes closed)
        if ear < 0.22:  # Adjusted threshold
            self.drowsy_counter += 1
            self.yawn_counter = 0  # Reset yawn counter
        # Check for yawning
        elif mar > 0.6:  # Adjusted threshold
            self.yawn_counter += 1
            self.drowsy_counter = 0  # Reset drowsy counter
        else:
            # Active state - reset both counters
            self.drowsy_counter = 0
            self.yawn_counter = 0
            return "Active"
        
        # Determine status based on counters
        if self.drowsy_counter >= self.drowsy_threshold:
            return "Drowsy"
        elif self.yawn_counter >= self.yawn_threshold:
            return "Yawning"
        else:
            # Not enough consecutive detections, consider as transitioning to active
            return "Active"
    
    def reset_inactivity_tracking(self):
        """Reset inactivity tracking when user becomes active"""
        current_time = time.time()
        
        # If we were tracking inactivity and now becoming active, log the transition
        if self.inactive_start_time is not None and self.current_status != "Active":
            total_inactive_duration = current_time - self.inactive_start_time
            # Get current EAR and MAR values if available
            ear = getattr(self, 'last_ear', 0.0)
            mar = getattr(self, 'last_mar', 0.0)
            
            # Log the transition from inactive to active with duration
            self.activity_logger.log_inactive_to_active_transition(
                total_inactive_duration, ear=ear, mar=mar
            )
        
        self.last_active_time = current_time
        self.inactive_start_time = None
        self.emergency_triggered = False
        self.inactivity_label.setText("Inactive Time: 0.0s")
        
        # Stop emergency if it was active
        if self.emergency_wakeup.is_emergency_active:
            self.emergency_wakeup.stop_emergency()
            
    def handle_inactivity_tracking(self, status, current_time):
        """Handle inactivity tracking and emergency triggering"""
        # Start tracking inactivity for non-active states
        if status != "Active":
            if self.inactive_start_time is None:
                self.inactive_start_time = current_time
                
            # Calculate inactive duration
            inactive_duration = current_time - self.inactive_start_time
            self.inactivity_label.setText(f"Inactive Time: {inactive_duration:.1f}s")
            
            # Trigger emergency after threshold
            if inactive_duration >= self.inactivity_threshold and not self.emergency_triggered:
                self.trigger_emergency()
                self.emergency_triggered = True
        else:
            self.reset_inactivity_tracking()
        
    def get_face_bounding_box(self, landmarks):
        x_coords = [p[0] for p in landmarks]
        y_coords = [p[1] for p in landmarks]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        
    def draw_eye_landmarks(self, frame, landmarks):
        # Left eye landmarks (approximate indices)
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye landmarks (approximate indices)
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        for idx in left_eye_indices:
            if idx < len(landmarks):
                cv2.circle(frame, tuple(landmarks[idx]), 2, (0, 255, 0), -1)
                
        for idx in right_eye_indices:
            if idx < len(landmarks):
                cv2.circle(frame, tuple(landmarks[idx]), 2, (0, 255, 0), -1)
                
    def trigger_emergency(self):
        """Trigger emergency wake-up protocol"""
        self.emergency_wakeup.trigger_emergency()
        self.activity_logger.log_event("Emergency", f"User inactive for >{self.inactivity_threshold} seconds")
        
    def update_status(self, status):
        """Update status with enhanced logging"""
        if status != self.current_status:
            old_status = self.current_status
            self.current_status = status
            self.status_label.setText(f"Status: {status}")
            
            # Update status label color based on status
            if status == "Active":
                self.status_label.setStyleSheet("color: green; padding: 10px; border: 2px solid gray;")
            elif status in ["Yawning", "Drowsy"]:
                self.status_label.setStyleSheet("color: orange; padding: 10px; border: 2px solid gray;")
            elif status in ["Inactive (Face Missing)"]:
                self.status_label.setStyleSheet("color: red; padding: 10px; border: 2px solid gray;")
            else:  # Multiple Persons Detected, etc.
                self.status_label.setStyleSheet("color: red; padding: 10px; border: 2px solid gray;")
            
            # Get current EAR and MAR values if available
            ear = getattr(self, 'last_ear', 0.0)
            mar = getattr(self, 'last_mar', 0.0)
            
            # Enhanced logging with inactive duration tracking
            self.activity_logger.log_status_change(
                old_status, status, ear=ear, mar=mar, current_time=time.time()
            )
            
            # AI feedback
            self.ai_feedback.speak_status(status)
            
    def close_application(self):
        """Properly close the application"""
        print("Closing application...")
        
        # Stop detection first
        self.detection_active = False
        
        # Stop timer first
        try:
            if hasattr(self, 'timer') and self.timer.isActive():
                self.timer.stop()
        except Exception as e:
            print(f"Error stopping timer: {e}")
        
        # Stop emergency protocol with error handling
        try:
            if hasattr(self, 'emergency_wakeup') and self.emergency_wakeup:
                self.emergency_wakeup.cleanup()
        except Exception as e:
            print(f"Error during emergency cleanup: {e}")
        
        # Close camera
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
        except Exception as e:
            print(f"Error releasing camera: {e}")
            
        # Give a small delay for cleanup
        try:
            import time
            time.sleep(0.5)
        except:
            pass
            
        print("Application closed successfully.")
        self.close()
        
    def closeEvent(self, event):
        """Handle window close event"""
        try:
            self.close_application()
            event.accept()
        except Exception as e:
            print(f"Error during close event: {e}")
            # Force accept the event even if there's an error
            event.accept()
            
    def show_log_statistics(self):
        """Display current log statistics"""
        stats = self.activity_logger.get_log_stats()
        print("\n=== Log Statistics ===")
        print(f"Total Events: {stats.get('total_events', 0)}")
        
        if 'status_counts' in stats:
            print("\nStatus Counts:")
            for status, count in stats['status_counts'].items():
                print(f"  {status}: {count}")
        
        if 'inactive_duration_stats' in stats:
            inactive_stats = stats['inactive_duration_stats']
            print(f"\nInactive Duration Statistics:")
            print(f"  Total Inactive Periods: {inactive_stats.get('total_inactive_periods', 0)}")
            print(f"  Average Duration: {inactive_stats.get('average_inactive_duration', 0):.2f}s")
            print(f"  Max Duration: {inactive_stats.get('max_inactive_duration', 0):.2f}s")
            print(f"  Min Duration: {inactive_stats.get('min_inactive_duration', 0):.2f}s")
        
        if 'latest_entry' in stats and stats['latest_entry']:
            print(f"\nLatest Entry: {stats['latest_entry']}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttentivenessMonitor()
    window.show()
    window.show_log_statistics()
    sys.exit(app.exec_())