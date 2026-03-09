from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import os
import shutil
from pathlib import Path
from models import User, Video, Incident, Alert, Analytics

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class Database:
    def __init__(self, data_file: str = "data.json"):
        self.data_file = data_file
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.users: List[User] = []
        self.videos: List[Video] = []
        self.incidents: List[Incident] = []
        self.alerts: List[Alert] = []
        self.settings: Dict[str, Any] = {}
        self._load_data()
        self._initialize_defaults()
        self._cleanup_old_alerts()
    
    def _load_data(self):
        """Load data from JSON file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                for u in data.get('users', []):
                    if u.get('last_login'):
                        u['last_login'] = datetime.fromisoformat(u['last_login'])
                for i in data.get('incidents', []):
                    if i.get('timestamp'):
                        i['timestamp'] = datetime.fromisoformat(i['timestamp'])
                for a in data.get('alerts', []):
                    if a.get('timestamp'):
                        a['timestamp'] = datetime.fromisoformat(a['timestamp'])
                
                self.users = [User(**u) for u in data.get('users', [])]
                self.videos = [Video(**v) for v in data.get('videos', [])]
                self.incidents = [Incident(**i) for i in data.get('incidents', [])]
                self.alerts = [Alert(**a) for a in data.get('alerts', [])]
                self.settings = data.get('settings', {})
                print(f"✅ Loaded {len(self.users)} users, {len(self.videos)} videos, {len(self.incidents)} incidents")
            except Exception as e:
                print(f"⚠️ Error loading data: {e}")
                self.users = []
                self.videos = []
                self.incidents = []
                self.alerts = []
                self.settings = {}
    
    def _save_data(self):
        """Save data to JSON file with backup"""
        try:
            if os.path.exists(self.data_file):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.backup_dir / f"data_backup_{timestamp}.json"
                shutil.copy2(self.data_file, backup_path)
                
                backups = sorted(self.backup_dir.glob("data_backup_*.json"))
                for old_backup in backups[:-10]:
                    old_backup.unlink()
            
            data = {
                'users': [u.model_dump() for u in self.users],
                'videos': [v.model_dump() for v in self.videos],
                'incidents': [i.model_dump() for i in self.incidents],
                'alerts': [a.model_dump() for a in self.alerts],
                'settings': self.settings
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, cls=DateTimeEncoder)
        except Exception as e:
            print(f"⚠️ Error saving data: {e}")
    
    def _initialize_defaults(self):
        """Initialize default users if database is empty"""
        if not self.users:
            self.users = [
                User(
                    username="admin",
                    full_name="Admin User",
                    email="admin@videoguard.ai",
                    role="admin",
                    access_level="all",
                    password="admin123",
                    status="active"
                ),
                User(
                    username="operator",
                    full_name="Operator User",
                    email="operator@videoguard.ai",
                    role="operator",
                    access_level="downtown",
                    password="operator123",
                    status="active"
                ),
                User(
                    username="supervisor",
                    full_name="Supervisor User",
                    email="supervisor@videoguard.ai",
                    role="supervisor",
                    access_level="highway",
                    password="supervisor123",
                    status="active"
                )
            ]
            self._save_data()
            print("✅ Default users created")
    
    def _cleanup_old_alerts(self):
        """Remove alerts older than 7 days"""
        cutoff = datetime.now() - timedelta(days=7)
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff]
        self._save_data()
    
    # User Operations
    def get_user_by_username(self, username: str) -> Optional[User]:
        for user in self.users:
            if user.username == username:
                return user
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        for user in self.users:
            if user.email == email:
                return user
        return None
    
    def update_user(self, user: User):
        for i, u in enumerate(self.users):
            if u.username == user.username:
                self.users[i] = user
                self._save_data()
                return
    
    def add_user(self, user: User):
        if self.get_user_by_username(user.username):
            raise ValueError(f"User {user.username} already exists")
        self.users.append(user)
        self._save_data()
    
    def delete_user(self, username: str):
        self.users = [u for u in self.users if u.username != username]
        self._save_data()
    
    def get_all_users(self) -> List[User]:
        return self.users
    
    def get_active_users(self) -> List[User]:
        return [u for u in self.users if u.status == "active"]
    
    # Video Operations
    def add_video(self, video: Video):
        self.videos.append(video)
        self._save_data()
    
    def get_video(self, video_id: str) -> Optional[Video]:
        for video in self.videos:
            if video.id == video_id:
                return video
        return None
    
    def get_all_videos(self) -> List[Video]:
        return self.videos
    
    def update_video(self, video: Video):
        for i, v in enumerate(self.videos):
            if v.id == video.id:
                self.videos[i] = video
                self._save_data()
                return
    
    def delete_video(self, video_id: str):
        self.videos = [v for v in self.videos if v.id != video_id]
        self._save_data()
    
    # Incident Operations
    def add_incident(self, incident: Incident):
        self.incidents.append(incident)
        self._save_data()
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        for inc in self.incidents:
            if inc.id == incident_id:
                return inc
        return None
    
    def get_all_incidents(self) -> List[Incident]:
        return self.incidents
    
    def update_incident(self, incident: Incident):
        for i, inc in enumerate(self.incidents):
            if inc.id == incident.id:
                self.incidents[i] = incident
                self._save_data()
                return
    
    # Alert Operations
    def add_alert(self, alert: Alert):
        self.alerts.append(alert)
        self._cleanup_old_alerts()
        self._save_data()
    
    def get_active_alerts(self) -> List[Alert]:
        cutoff = datetime.now() - timedelta(hours=24)
        return [a for a in self.alerts if not a.acknowledged and a.timestamp > cutoff]
    
    def remove_alert(self, alert_id: str):
        self.alerts = [a for a in self.alerts if a.id != alert_id]
        self._save_data()
    
    # Analytics
    def get_analytics(self, days: int = 30) -> Analytics:
        cutoff = datetime.now() - timedelta(days=days)
        recent_incidents = [i for i in self.incidents if i.timestamp > cutoff]
        
        by_type = {}
        by_day = {}
        for inc in recent_incidents:
            by_type[inc.type] = by_type.get(inc.type, 0) + 1
            day = inc.timestamp.date().isoformat()
            by_day[day] = by_day.get(day, 0) + 1
        
        if recent_incidents:
            avg_confidence = sum(
                float(inc.confidence.replace('%', '')) for inc in recent_incidents
            ) / len(recent_incidents)
            accuracy = min(avg_confidence, 98.5)
        else:
            accuracy = 94.2
        
        processed_videos = [v for v in self.videos if v.status == "processed"]
        avg_time = 45.5 if processed_videos else 0
        
        return Analytics(
            total_videos=len(self.videos),
            processed_videos=len([v for v in self.videos if v.status == "processed"]),
            total_incidents=len(recent_incidents),
            incidents_by_type=by_type,
            incidents_by_day=by_day,
            avg_processing_time=avg_time,
            detection_accuracy=accuracy
        )