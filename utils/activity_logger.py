import csv
import os
from datetime import datetime
import threading

class ActivityLogger:
    def __init__(self, log_file="attentiveness_log.csv"):
        self.log_file = log_file
        self.lock = threading.Lock()
        self.last_inactive_start = None  # Track when inactivity started
        self.initialize_log_file()
        
    def initialize_log_file(self):
        """Initialize the CSV log file with headers if it doesn't exist"""
        file_exists = os.path.isfile(self.log_file)
        
        if not file_exists:
            with open(self.log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'Timestamp',
                    'Date',
                    'Time',
                    'Status',
                    'Description',
                    'Duration_Seconds',
                    'EAR_Value',
                    'MAR_Value',
                    'Inactive_Duration'  # New column for tracking inactive duration
                ])
                
    def log_event(self, status, description, duration=0, ear_value=0.0, mar_value=0.0, inactive_duration='-'):
        """Log an attentiveness event to the CSV file"""
        with self.lock:
            try:
                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")
                
                with open(self.log_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        timestamp,
                        date_str,
                        time_str,
                        status,
                        description,
                        duration,
                        f"{ear_value:.3f}",
                        f"{mar_value:.3f}",
                        inactive_duration
                    ])
                    
                print(f"[LOG] {timestamp} - {status}: {description}")
                
            except Exception as e:
                print(f"Error logging event: {e}")
                
    def log_status_change(self, old_status, new_status, ear=0.0, mar=0.0, current_time=None):
        """Log a status change event with inactive duration tracking"""
        description = f"Status changed from {old_status} to {new_status}"
        inactive_duration = '-'  # Default value
        
        # Handle inactive duration tracking
        if current_time is None:
            current_time = datetime.now().timestamp()
            
        # If moving from inactive to active, calculate inactive duration
        if self._is_inactive_status(old_status) and new_status == "Active":
            if self.last_inactive_start is not None:
                inactive_duration = f"{current_time - self.last_inactive_start:.1f}"
                self.last_inactive_start = None  # Reset
        
        # If moving from active to inactive, start tracking
        elif old_status == "Active" and self._is_inactive_status(new_status):
            self.last_inactive_start = current_time
            
        # If staying inactive, don't change the tracking
        elif self._is_inactive_status(old_status) and self._is_inactive_status(new_status):
            # Keep tracking, no duration to log yet
            pass
            
        # If moving from one active-like state to another, reset tracking
        elif not self._is_inactive_status(old_status) and not self._is_inactive_status(new_status):
            self.last_inactive_start = None
        
        self.log_event(new_status, description, ear_value=ear, mar_value=mar, inactive_duration=inactive_duration)
        
    def _is_inactive_status(self, status):
        """Determine if a status is considered 'inactive'"""
        inactive_statuses = [
            "Drowsy", 
            "Yawning", 
            "Inactive (Face Missing)", 
            "Not Awake", 
            "Multiple Persons Detected",
            "Inactive"
        ]
        return status in inactive_statuses
        
    def log_emergency(self, inactive_duration):
        """Log an emergency wake-up event"""
        description = f"Emergency protocol triggered after {inactive_duration:.1f} seconds of inactivity"
        self.log_event("Emergency", description, duration=inactive_duration, inactive_duration=f"{inactive_duration:.1f}")
        
    def log_detection_session(self, action):
        """Log detection session start/stop"""
        description = f"Detection session {action}"
        # Reset inactive tracking when session starts/stops
        if action == "started":
            self.last_inactive_start = None
        elif action == "stopped":
            self.last_inactive_start = None
        self.log_event("System", description)
        
    def log_inactive_to_active_transition(self, total_inactive_duration, ear=0.0, mar=0.0):
        """Specifically log when user returns to active state after being inactive"""
        description = f"User returned to active state after {total_inactive_duration:.1f} seconds of inactivity"
        self.log_event("Active", description, ear_value=ear, mar_value=mar, inactive_duration=f"{total_inactive_duration:.1f}")
        self.last_inactive_start = None  # Reset tracking
        
    def log_continuous_inactive_status(self, status, description, ear=0.0, mar=0.0):
        """Log inactive status without duration (for continuous monitoring)"""
        self.log_event(status, description, ear_value=ear, mar_value=mar, inactive_duration='-')
        
    def reset_inactive_tracking(self):
        """Reset inactive duration tracking (useful for manual resets)"""
        self.last_inactive_start = None
        
    def get_current_inactive_duration(self):
        """Get current inactive duration if tracking"""
        if self.last_inactive_start is not None:
            return datetime.now().timestamp() - self.last_inactive_start
        return 0
        
    def get_log_stats(self):
        """Get basic statistics from the log file"""
        try:
            with open(self.log_file, 'r') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                
            if not rows:
                return {"total_events": 0}
                
            status_counts = {}
            total_events = len(rows)
            inactive_durations = []
            
            for row in rows:
                status = row['Status']
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Collect inactive durations for analysis
                if row.get('Inactive_Duration', '-') != '-':
                    try:
                        duration = float(row['Inactive_Duration'])
                        inactive_durations.append(duration)
                    except ValueError:
                        pass
                        
            # Calculate inactive duration statistics
            inactive_stats = {}
            if inactive_durations:
                inactive_stats = {
                    "total_inactive_periods": len(inactive_durations),
                    "average_inactive_duration": sum(inactive_durations) / len(inactive_durations),
                    "max_inactive_duration": max(inactive_durations),
                    "min_inactive_duration": min(inactive_durations)
                }
                
            return {
                "total_events": total_events,
                "status_counts": status_counts,
                "inactive_duration_stats": inactive_stats,
                "latest_entry": rows[-1] if rows else None
            }
            
        except Exception as e:
            print(f"Error reading log stats: {e}")
            return {"total_events": 0, "error": str(e)}
            
    def clear_logs(self):
        """Clear all logs (use with caution)"""
        with self.lock:
            try:
                self.last_inactive_start = None  # Reset tracking
                self.initialize_log_file()
                print("Log file cleared successfully")
            except Exception as e:
                print(f"Error clearing logs: {e}")
                
    def export_logs(self, export_file=None):
        """Export logs to a different file"""
        if export_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = f"attentiveness_log_export_{timestamp}.csv"
            
        try:
            with open(self.log_file, 'r') as source:
                with open(export_file, 'w') as dest:
                    dest.write(source.read())
            print(f"Logs exported to {export_file}")
            return export_file
        except Exception as e:
            print(f"Error exporting logs: {e}")
            return None