import threading
import time
import os
import pygame
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QPalette
import pyttsx3

class EmergencyWakeup(QObject):
    flash_signal = pyqtSignal(str)  # Signal to change screen color
    
    def __init__(self):
        super().__init__()
        # Initialize all attributes first to prevent AttributeError
        self.is_emergency_active = False
        self.flash_timer = None
        self.flash_count = 0
        self.siren_thread = None
        self.tts_thread = None
        self.siren_sound = None
        self.tts_engine = None
        self.pygame_initialized = False
        
        # Initialize pygame for sound
        try:
            pygame.mixer.init()
            self.pygame_initialized = True
        except Exception as e:
            print(f"Could not initialize pygame mixer: {e}")
            self.pygame_initialized = False
        
        # Initialize text-to-speech
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 200)
            self.tts_engine.setProperty('volume', 1.0)
        except Exception as e:
            print(f"Could not initialize TTS engine: {e}")
            self.tts_engine = None
        
        # Create siren sound programmatically
        self.create_siren_sound()
        
    def create_siren_sound(self):
        """Create a siren sound using pygame"""
        try:
            import numpy as np
            
            # Create a simple beep sound
            frequency = 1000  # Hz
            duration = 0.5    # seconds
            sample_rate = 44100
            
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            
            for i in range(frames):
                time_point = float(i) / sample_rate
                # Create alternating high and low frequency for siren effect
                freq = 800 + 400 * np.sin(2 * np.pi * 2 * time_point)  # 2 Hz modulation
                arr[i] = 32767 * np.sin(2 * np.pi * freq * time_point)
                
            arr = arr.astype(np.int16)
            
            # Convert to stereo
            stereo_arr = np.zeros((frames, 2), dtype=np.int16)
            stereo_arr[:, 0] = arr
            stereo_arr[:, 1] = arr
            
            # Create pygame sound
            self.siren_sound = pygame.sndarray.make_sound(stereo_arr)
            
        except Exception as e:
            print(f"Could not create siren sound: {e}")
            self.siren_sound = None
            
    def trigger_emergency(self):
        """Trigger the emergency wake-up protocol"""
        if self.is_emergency_active:
            return
            
        self.is_emergency_active = True
        self.flash_count = 0
        print("🚨 EMERGENCY WAKE-UP PROTOCOL ACTIVATED! 🚨")
        
        # Start all emergency actions
        self.play_siren()
        self.start_screen_flash()
        self.speak_wake_up_message()
        
    def play_siren(self):
        """Play siren sound continuously until emergency stops"""
        def play_sound():
            try:
                while self.is_emergency_active:
                    if hasattr(self, 'siren_sound') and self.siren_sound:
                        self.siren_sound.play()
                        time.sleep(1)
                    else:
                        # Fallback: system beep
                        print('\a')  # ASCII bell character
                        time.sleep(0.5)
            except Exception as e:
                print(f"Error playing siren: {e}")
                
        # Stop previous siren thread if running
        try:
            if hasattr(self, 'siren_thread') and self.siren_thread and self.siren_thread.is_alive():
                return
        except Exception as e:
            print(f"Error checking siren thread: {e}")
            
        self.siren_thread = threading.Thread(target=play_sound, daemon=True)
        self.siren_thread.start()
        
    def start_screen_flash(self):
        """Start flashing the screen with red and blue colors"""
        if self.flash_timer:
            self.flash_timer.stop()
            self.flash_timer.deleteLater()
            
        self.flash_count = 0
        self.flash_timer = QTimer()
        self.flash_timer.timeout.connect(self.flash_screen)
        self.flash_timer.start(500)  # Flash every 500ms
        
    def flash_screen(self):
        """Flash the screen background"""
        if not self.is_emergency_active:
            if self.flash_timer:
                self.flash_timer.stop()
                self.flash_timer.deleteLater()
                self.flash_timer = None
            return
            
        # Emit signal to change screen color
        color = "red" if self.flash_count % 2 == 0 else "blue"
        self.flash_signal.emit(color)
        print(f"🚨 SCREEN FLASH: {color.upper()} 🚨")
        
        self.flash_count += 1
        
    def speak_wake_up_message(self):
        """Speak wake-up message using text-to-speech continuously"""
        def speak():
            try:
                messages = [
                    "Wake up! You've been inactive for too long!",
                    "Attention! Please focus on your studies!",
                    "Alert! Student attentiveness required!",
                    "Please return to your study position!"
                ]
                
                message_index = 0
                while self.is_emergency_active:
                    if hasattr(self, 'tts_engine') and self.tts_engine is not None:
                        try:
                            message = messages[message_index % len(messages)]
                            self.tts_engine.say(message)
                            self.tts_engine.runAndWait()
                            message_index += 1
                        except Exception as e:
                            print(f"Error in TTS playback: {e}")
                            break
                    time.sleep(2)  # Wait 2 seconds between messages
                    
            except Exception as e:
                print(f"Error in text-to-speech: {e}")
                
        # Stop previous TTS thread if running
        try:
            if hasattr(self, 'tts_thread') and self.tts_thread and self.tts_thread.is_alive():
                return
        except Exception as e:
            print(f"Error checking TTS thread: {e}")
            
        self.tts_thread = threading.Thread(target=speak, daemon=True)
        self.tts_thread.start()
        
    def stop_emergency(self):
        """Stop the emergency protocol"""
        if not self.is_emergency_active:
            return
            
        self.is_emergency_active = False
        
        # Stop flash timer safely
        try:
            if self.flash_timer and hasattr(self.flash_timer, 'stop'):
                self.flash_timer.stop()
                self.flash_timer.deleteLater()
                self.flash_timer = None
        except Exception as e:
            print(f"Error stopping flash timer: {e}")
            self.flash_timer = None
            
        # Emit signal to reset screen color
        try:
            self.flash_signal.emit("normal")
        except Exception as e:
            print(f"Error emitting flash signal: {e}")
        
        # Stop pygame sounds
        try:
            if self.pygame_initialized:
                pygame.mixer.stop()
        except Exception as e:
            print(f"Error stopping pygame mixer: {e}")
            
        # Stop TTS engine
        try:
            if hasattr(self, 'tts_engine') and self.tts_engine is not None:
                self.tts_engine.stop()
        except Exception as e:
            print(f"Error stopping TTS engine: {e}")
                
        print("Emergency protocol stopped.")
        
    def cleanup(self):
        """Clean up resources properly"""
        print("Starting emergency cleanup...")
        
        # Stop emergency first
        self.stop_emergency()
        
        # Wait for threads to finish with timeout
        try:
            if hasattr(self, 'siren_thread') and self.siren_thread and self.siren_thread.is_alive():
                self.siren_thread.join(timeout=1.0)
        except Exception as e:
            print(f"Error joining siren thread: {e}")
            
        try:
            if hasattr(self, 'tts_thread') and self.tts_thread and self.tts_thread.is_alive():
                self.tts_thread.join(timeout=1.0)
        except Exception as e:
            print(f"Error joining TTS thread: {e}")
            
        # Cleanup pygame
        try:
            if hasattr(self, 'pygame_initialized') and self.pygame_initialized:
                pygame.mixer.quit()
        except Exception as e:
            print(f"Error quitting pygame mixer: {e}")
            
        # Cleanup TTS
        try:
            if hasattr(self, 'tts_engine') and self.tts_engine is not None:
                del self.tts_engine
                self.tts_engine = None
        except Exception as e:
            print(f"Error deleting TTS engine: {e}")
            
        print("Emergency cleanup completed.")
                
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.cleanup()
        except Exception as e:
            print(f"Error in cleanup: {e}")