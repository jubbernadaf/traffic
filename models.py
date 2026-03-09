from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

class User(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    username: str
    full_name: str
    email: Optional[str] = None
    role: str
    access_level: str
    status: str = "active"
    last_login: Optional[datetime] = None
    password: Optional[str] = None

class Video(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    name: str
    description: str
    location: str
    status: str
    duration: str
    size: str
    uploaded: str
    incidents: int
    priority: str
    tags: List[str] = []
    file_path: Optional[str] = None
    analysis_data: Optional[Dict[str, Any]] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    progress: Optional[float] = 0

class Incident(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    type: str
    video_id: str
    video_name: str
    location: str
    timestamp: datetime
    confidence: str
    description: str
    status: str
    video_time: Optional[str] = None
    operator: Optional[str] = None
    actions: List[str] = []

class Alert(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    type: str
    video_id: str
    video_name: str
    confidence: str
    description: str
    timestamp: datetime = datetime.now()
    acknowledged: bool = False

class Analytics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    total_videos: int
    processed_videos: int
    total_incidents: int
    incidents_by_type: Dict[str, int]
    incidents_by_day: Dict[str, int]
    avg_processing_time: float
    detection_accuracy: float

class LoginRequest(BaseModel):
    username: str
    password: str
    area: Optional[str] = None

class LoginResponse(BaseModel):
    success: bool
    user: User
    token: str

class VideoUploadResponse(BaseModel):
    success: bool
    video_id: str
    message: str

class UserCreate(BaseModel):
    username: str
    full_name: str
    email: str
    role: str
    access_level: str
    password: str