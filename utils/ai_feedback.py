import pyttsx3
import threading
import queue
import time

class AIFeedback:
    def __init__(self):
        self.tts_engine = None
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speech_thread = None
        self.stop_speech = False
        
        self.initialize_tts()
        self.start_speech_worker()
        
        # Status messages
        self.status_messages = {
            "Active": [
                "Great! You're staying focused.",
                "Good attention level detected.",
                "Keep up the good work!"
            ],
            "Yawning": [
                "I notice you're yawning. Try to stay alert.",
                "You seem tired. Take a deep breath.",
                "Yawning detected. Please stay focused."
            ],
            "Drowsy": [
                "You appear drowsy. Please stay awake.",
                "Low eye activity detected. Please focus.",
                "Drowsiness alert! Please pay attention."
            ],
            "Face Missing": [
                "I can't see your face. Please position yourself properly.",
                "Face not detected. Please come back to the camera.",
                "Please ensure you're visible to the camera."
            ],
            "Not Awake": [
                "You've been away too long. Please return.",
                "Extended absence detected. Please come back.",
                "Long inactivity period. Please focus."
            ],
            "Multiple Faces": [
                "Multiple people detected. Please ensure only one person is monitoring.",
                "Too many faces in view. Please clear the area.",
                "Single user mode required."
            ],
            "Fake Presence": [
                "Artificial presence detected. Please use live video only.",
                "Static image detected. Please show live movement.",
                "Spoofing attempt detected."
            ]
        }
        
    def initialize_tts(self):
        """Initialize the text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                        
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 180)  # Words per minute
            self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            
            print("TTS engine initialized successfully")
            
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            self.tts_engine = None
            
    def start_speech_worker(self):
        """Start the background thread for speech processing"""
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        
    def _speech_worker(self):
        """Background worker to process speech queue"""
        while not self.stop_speech:
            try:
                # Get speech request from queue with timeout
                speech_text = self.speech_queue.get(timeout=1)
                
                if speech_text and self.tts_engine:
                    self.is_speaking = True
                    print(f"[AI FEEDBACK] Speaking: {speech_text}")
                    
                    # Clear any pending speech and speak new text
                    self.tts_engine.stop()
                    self.tts_engine.say(speech_text)
                    self.tts_engine.runAndWait()
                    
                    self.is_speaking = False
                    
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in speech worker: {e}")
                self.is_speaking = False
                
    def speak_status(self, status):
        """Speak a status-specific message"""
        if not self.tts_engine:
            return
            
        # Don't queue new speech if already speaking, unless it's critical
        critical_statuses = ["Not Awake", "Emergency", "Fake Presence"]
        
        if self.is_speaking and status not in critical_statuses:
            return
            
        # Get appropriate message for status
        messages = self.status_messages.get(status, ["Status update: " + status])
        message = messages[0]  # Use first message for consistency
        
        # Clear queue for critical messages
        if status in critical_statuses:
            self._clear_speech_queue()
            
        # Add to speech queue
        try:
            self.speech_queue.put(message, block=False)
        except queue.Full:
            pass  # Queue is full, skip this message
            
    def speak_custom_message(self, message):
        """Speak a custom message"""
        if not self.tts_engine:
            return
            
        try:
            self.speech_queue.put(message, block=False)
        except queue.Full:
            pass
            
    def speak_emergency_message(self):
        """Speak emergency wake-up message with high priority"""
        emergency_messages = [
            "Wake up! You've been inactive for too long!",
            "Attention! Please focus on your studies!",
            "Alert! Student attentiveness required!"
        ]
        
        # Clear queue and speak immediately
        self._clear_speech_queue()
        
        for message in emergency_messages:
            if self.stop_speech:
                break
            try:
                self.speech_queue.put(message, block=False)
                time.sleep(2)  # Pause between messages
            except queue.Full:
                break
                
    def _clear_speech_queue(self):
        """Clear all pending speech requests"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                break
                
    def stop_all_speech(self):
        """Stop all speech immediately"""
        self.stop_speech = True
        self._clear_speech_queue()
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
                
    def is_currently_speaking(self):
        """Check if TTS is currently speaking"""
        return self.is_speaking
        
    def get_queue_size(self):
        """Get current speech queue size"""
        return self.speech_queue.qsize()
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_all_speech()