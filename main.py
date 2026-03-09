from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import cv2
import numpy as np
import asyncio
import uuid
import json
import math
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
import shutil
import threading
import queue
import time
import mimetypes
import traceback
from models import User, Video, Incident, Alert, Analytics, LoginRequest, LoginResponse, VideoUploadResponse, UserCreate
from algorithms import YOLODetector, VehicleTracker, AnomalyDetector, KalmanFilter, BayesianNetwork
from database import Database
from digital_twin import DigitalTwin

app = FastAPI(title="VideoGuard AI - YOLOv8 Video Analysis System")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
DETECTIONS_DIR = STATIC_DIR / "detections"

for dir_path in [UPLOADS_DIR, DETECTIONS_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Setup templates and static files
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize database
db = Database()

# Active processing queue
processing_queue = {}
active_connections = []

# Initialize Digital Twin
digital_twin = DigitalTwin()

# WebSocket connections for digital twin
digital_twin_connections: Set[WebSocket] = set()

# Global YOLO detector (load once)
print("\n" + "="*60)
print("Loading YOLOv8 model...")
print("="*60)
try:
    yolo_detector = YOLODetector()
    print("✅ YOLOv8 model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading YOLOv8 model: {e}")
    print("⚠️  System will still run but detection will be simulated")
    yolo_detector = None

# ==================== PAGE ROUTES ====================
@app.get("/", response_class=HTMLResponse)
async def get_login(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/digital-twin", response_class=HTMLResponse)
async def get_digital_twin(request: Request):
    return templates.TemplateResponse("digital-twin.html", {"request": request})

@app.get("/video-management", response_class=HTMLResponse)
async def get_video_management(request: Request):
    return templates.TemplateResponse("video-management.html", {"request": request})

@app.get("/video-view", response_class=HTMLResponse)
async def get_video_view(request: Request):
    return templates.TemplateResponse("video-view.html", {"request": request})

@app.get("/incident-history", response_class=HTMLResponse)
async def get_incident_history(request: Request):
    return templates.TemplateResponse("incident-history.html", {"request": request})

@app.get("/incident-details", response_class=HTMLResponse)
async def get_incident_details(request: Request):
    return templates.TemplateResponse("incident-details.html", {"request": request})

@app.get("/analytics", response_class=HTMLResponse)
async def get_analytics(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request})

@app.get("/user-management", response_class=HTMLResponse)
async def get_user_management(request: Request):
    return templates.TemplateResponse("user-management.html", {"request": request})

@app.get("/settings", response_class=HTMLResponse)
async def get_settings(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/analysis-alerts", response_class=HTMLResponse)
async def get_analysis_alerts(request: Request):
    return templates.TemplateResponse("analysis-alerts.html", {"request": request})

# ==================== API ROUTES ====================
@app.post("/api/login")
async def login(request: LoginRequest):
    """Authenticate user"""
    user = db.get_user_by_username(request.username)
    if not user or user.password != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user.last_login = datetime.now()
    db.update_user(user)
    return {
        "success": True,
        "user": user.model_dump(),
        "token": str(uuid.uuid4())
    }

@app.post("/api/logout")
async def logout(username: str):
    return {"success": True}

@app.post("/api/videos/upload")
async def upload_video(
    title: str = Form(...),
    description: str = Form(...),
    location: str = Form(...),
    priority: str = Form(...),
    tags: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload video for YOLOv8 analysis"""
    try:
        # Generate unique ID
        video_id = f"VID-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Save uploaded file
        file_ext = file.filename.split('.')[-1].lower()
        filename = f"{video_id}.{file_ext}"
        file_path = UPLOADS_DIR / filename
        
        # Save file
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        print(f"✅ Video saved: {file_path} ({len(content)} bytes)")
        
        # Get video properties using OpenCV
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Format duration
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        duration_str = f"{minutes}:{seconds:02d}"
        
        # Calculate file size
        file_size_mb = len(content) / (1024 * 1024)
        
        # Create video record
        video = Video(
            id=video_id,
            name=title,
            description=description,
            location=location,
            status="processing",
            duration=duration_str,
            size=f"{file_size_mb:.2f}MB",
            uploaded="Just now",
            incidents=0,
            priority=priority,
            tags=tags.split(',') if tags else [],
            file_path=str(file_path),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            progress=0
        )
        db.add_video(video)
        
        # Start real analysis in background thread
        thread = threading.Thread(target=process_video_real, args=(video_id,))
        thread.daemon = True
        thread.start()
        
        return {
            "success": True,
            "video_id": video_id,
            "message": "Video uploaded successfully. YOLOv8 analysis started."
        }
    except Exception as e:
        print(f"❌ Upload error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def process_video_real(video_id: str):
    """Real video processing with YOLOv8 detection and digital twin updates"""
    try:
        video = db.get_video(video_id)
        if not video or not video.file_path:
            print(f"❌ Video {video_id} not found")
            return
        
        print(f"\n{'='*60}")
        print(f"🚀 YOLOv8 ANALYSIS STARTED: {video_id}")
        print(f"📁 File: {video.file_path}")
        print(f"{'='*60}")
        
        # Check if file exists
        if not os.path.exists(video.file_path):
            print(f"❌ Video file not found: {video.file_path}")
            video.status = "error"
            db.update_video(video)
            return
        
        # Open video
        cap = cv2.VideoCapture(video.file_path)
        if not cap.isOpened():
            print("❌ Error: Cannot open video")
            video.status = "error"
            db.update_video(video)
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📊 Video Properties: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
        
        # Create output video with annotations (use H.264 codec for better compatibility)
        output_path = UPLOADS_DIR / f"{video_id}_analyzed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            # Fallback to mp4v if avc1 fails
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print("⚠️ Using mp4v codec fallback")
        
        # Initialize components
        vehicle_tracker = VehicleTracker()
        anomaly_detector = AnomalyDetector()
        kalman_count = KalmanFilter()
        kalman_speed = KalmanFilter()
        bayesian = BayesianNetwork()
        
        # Data storage
        frame_number = 0
        processed_frames = 0
        incidents_detected = []
        all_detections = []
        last_incident_time = 0
        fps_update_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        print("🎥 Starting frame-by-frame YOLOv8 analysis...")
        
        # Processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            timestamp = frame_number / fps if fps > 0 else frame_number / 30
            fps_counter += 1
            
            # Calculate FPS every second
            if time.time() - fps_update_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_update_time = time.time()
            
            # Run YOLO detection (every frame for accuracy)
            if yolo_detector:
                detections = yolo_detector.detect(frame, conf_threshold=0.5)
            else:
                # Fallback if YOLO not loaded
                detections = {
                    'vehicles': [],
                    'pedestrians': [],
                    'total_vehicles': 0,
                    'total_pedestrians': 0
                }
            
            vehicles = detections['vehicles']
            pedestrians = detections['pedestrians']
            vehicle_count = detections['total_vehicles']
            pedestrian_count = detections['total_pedestrians']
            
            # Track vehicles
            tracked_vehicles = vehicle_tracker.update(vehicles)
            
            # Estimate speed
            if yolo_detector:
                estimated_speed = yolo_detector.estimate_speed()
            else:
                estimated_speed = 60.0  # Default fallback
            
            # Apply Kalman filters
            smoothed_count = kalman_count.update(vehicle_count)
            smoothed_speed = kalman_speed.update(estimated_speed)
            
            # Update anomaly detector
            anomaly_detector.update(vehicle_count, pedestrian_count, smoothed_speed)
            
            # Detect anomalies
            incident_type, confidence, details = anomaly_detector.detect()
            
            # Update Bayesian probabilities
            bayesian_probs = bayesian.update(
                (incident_type, confidence, details),
                vehicle_count,
                smoothed_speed
            )
            
            # Store incident if detected
            if incident_type != 'normal' and confidence > 0.65:
                if timestamp - last_incident_time > 30:  # Min 30 seconds between incidents
                    last_incident_time = timestamp
                    incident_data = {
                        'timestamp': timestamp,
                        'frame': frame_number,
                        'type': incident_type,
                        'confidence': confidence,
                        'vehicles': vehicle_count,
                        'pedestrians': pedestrian_count,
                        'speed': smoothed_speed,
                        'details': details,
                        'probabilities': bayesian_probs
                    }
                    incidents_detected.append(incident_data)
                    print(f"\n⚠️ {incident_type.upper()} DETECTED at {timestamp:.1f}s")
                    print(f"   Confidence: {confidence:.1%}")
                    print(f"   Vehicles: {vehicle_count}, Speed: {smoothed_speed:.0f} km/h")
            
            # ========== DIGITAL TWIN UPDATE ==========
            # Update digital twin with detection data
            try:
                digital_twin.update_from_detection(
                    video_id=video_id,
                    frame_number=frame_number,
                    vehicles=tracked_vehicles,
                    pedestrians=pedestrians,
                    incident_type=incident_type,
                    incident_confidence=confidence,
                    avg_speed=smoothed_speed
                )
                
                # Broadcast digital twin state to all connected clients
                try:
                    # Create new event loop for async broadcast
                    state = digital_twin.get_scene_state()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(broadcast_digital_twin(state))
                    loop.close()
                    print(f"📤 Broadcast digital twin: {len(tracked_vehicles)} vehicles")
                except Exception as e:
                    print(f"⚠️ Could not broadcast digital twin: {e}")
            except Exception as e:
                print(f"⚠️ Digital twin update error: {e}")
                traceback.print_exc()
            # =========================================
            
            # Draw detections on frame
            # Draw vehicles (different colors by type)
            for v in tracked_vehicles:
                x, y, w, h = v['bbox']
                # Color based on vehicle type and incident
                if incident_type == 'accident':
                    color = (0, 0, 255)  # Red for accidents
                elif incident_type == 'jam':
                    color = (0, 165, 255)  # Orange for jams
                elif incident_type == 'warning':
                    color = (0, 255, 255)  # Yellow for warnings
                else:
                    color = (0, 255, 0)  # Green for normal
                
                # Different shades for different vehicle types
                if 'type' in v:
                    if v['type'] == 'truck':
                        color = (0, 100, 255)  # Orange for trucks
                    elif v['type'] == 'bus':
                        color = (0, 0, 255)  # Red for buses
                    elif v['type'] == 'motorcycle':
                        color = (0, 255, 100)  # Light green for motorcycles
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Label with vehicle type and confidence
                vehicle_type = v.get('type', 'car')
                vehicle_conf = v.get('confidence', 0.95)
                track_id = v.get('track_id', '')
                label = f"{vehicle_type} #{track_id} {vehicle_conf:.0%}"
                if 'speed' in v and v['speed'] > 0:
                    label += f" {v['speed']:.0f}km/h"
                cv2.putText(frame, label, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw pedestrians (purple)
            for p in pedestrians:
                x, y, w, h = p['bbox']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 1)
                conf = p.get('confidence', 0.9)
                cv2.putText(frame, f"person {conf:.0%}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # Draw status overlay
            # Top status bar
            cv2.rectangle(frame, (0, 0), (width, 130), (20, 20, 20), -1)
            
            # Status indicator
            status_colors = {
                'normal': (0, 255, 0),
                'warning': (0, 255, 255),
                'jam': (0, 165, 255),
                'accident': (0, 0, 255)
            }
            status_color = status_colors.get(incident_type, (0, 255, 0))
            status_text = f"STATUS: {incident_type.upper()}"
            cv2.putText(frame, status_text, (20, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Stats line 1
            stats1 = f"Vehicles: {vehicle_count} | Pedestrians: {pedestrian_count} | Speed: {smoothed_speed:.0f} km/h"
            cv2.putText(frame, stats1, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Stats line 2 (confidence and probabilities)
            stats2 = f"Conf: {confidence:.1%} | Accident: {bayesian_probs['accident']:.1%} | Jam: {bayesian_probs['jam']:.1%}"
            cv2.putText(frame, stats2, (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Stats line 3 (digital twin indicator)
            stats3 = f"Digital Twin: ACTIVE | Frame: {frame_number}/{total_frames} | FPS: {current_fps}"
            cv2.putText(frame, stats3, (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 92, 246), 1)
            
            # Stats line 4 (incident count)
            stats4 = f"Incidents: {len(incidents_detected)} | Tracked Vehicles: {len(tracked_vehicles)}"
            cv2.putText(frame, stats4, (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Progress bar
            progress = (frame_number / total_frames) * 100
            bar_width = int((width - 40) * (progress / 100))
            cv2.rectangle(frame, (20, height - 30), (width - 20, height - 10), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, height - 30), (20 + bar_width, height - 10), (0, 255, 0), -1)
            cv2.putText(frame, f"{progress:.1f}%", (width - 100, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame
            out.write(frame)
            
            # Store detection data
            all_detections.append({
                'frame': frame_number,
                'timestamp': timestamp,
                'vehicles': vehicle_count,
                'pedestrians': pedestrian_count,
                'speed': smoothed_speed,
                'incident_type': incident_type,
                'confidence': confidence
            })
            
            processed_frames += 1
            
            # Progress update
            if processed_frames % 100 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"📊 Progress: {progress:.1f}% - Vehicles: {vehicle_count}, Speed: {smoothed_speed:.0f} km/h")
                video.progress = progress
                db.update_video(video)
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"\n{'='*60}")
        print(f"✅ YOLOv8 ANALYSIS COMPLETE")
        print(f"📊 Total frames processed: {frame_number}")
        print(f"🚗 Average vehicles per frame: {np.mean([d['vehicles'] for d in all_detections]):.1f}")
        print(f"⚠️ Incidents detected: {len(incidents_detected)}")
        print(f"{'='*60}")
        
        # Create incidents from detections
        for i, inc in enumerate(incidents_detected):
            minutes = int(inc['timestamp'] // 60)
            seconds = int(inc['timestamp'] % 60)
            incident = Incident(
                id=f"INC-{datetime.now().strftime('%Y%m%d')}-{i+1:03d}",
                type=inc['type'],
                video_id=video_id,
                video_name=video.name,
                location=video.location,
                timestamp=datetime.now(),
                confidence=f"{inc['confidence']*100:.1f}%",
                description=f"{inc['type'].upper()} detected at {minutes}:{seconds:02d} with {inc['vehicles']} vehicles",
                status="pending",
                video_time=f"{minutes:02d}:{seconds:02d}"
            )
            db.add_incident(incident)
            
            alert = Alert(
                id=f"ALT-{datetime.now().strftime('%Y%m%d')}-{i+1:03d}",
                type=inc['type'].upper(),
                video_id=video_id,
                video_name=video.name,
                confidence=incident.confidence,
                description=incident.description,
                timestamp=datetime.now(),
                acknowledged=False
            )
            db.add_alert(alert)
            
            # Broadcast alert via WebSocket
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(broadcast_alert(alert))
                loop.close()
            except Exception as e:
                print(f"⚠️ Could not broadcast alert: {e}")
            
            print(f"✅ Incident created: {incident.id}")
        
        # Calculate average stats
        avg_vehicles = np.mean([d['vehicles'] for d in all_detections]) if all_detections else 0
        avg_speed = np.mean([d['speed'] for d in all_detections]) if all_detections else 0
        
        # Update video record
        video.status = "processed"
        video.incidents = len(incidents_detected)
        video.analysis_data = {
            'detections': all_detections,
            'incidents': incidents_detected,
            'total_frames': frame_number,
            'processed_frames': processed_frames,
            'fps': fps,
            'duration': frame_number / fps if fps > 0 else 0,
            'avg_vehicles': float(avg_vehicles),
            'avg_speed': float(avg_speed),
            'analyzed_video_path': str(output_path)
        }
        video.progress = 100
        db.update_video(video)
        
        print(f"\n✅ Video {video_id} analysis complete")
        print(f"📁 Analyzed video saved to: {output_path}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error processing video {video_id}: {e}")
        traceback.print_exc()
        video = db.get_video(video_id)
        if video:
            video.status = "error"
            db.update_video(video)

@app.get("/api/videos")
async def get_videos(status: Optional[str] = None):
    """Get all videos"""
    videos = db.get_all_videos()
    if status and status != 'all':
        videos = [v for v in videos if v.status == status]
    return {"videos": [v.model_dump() for v in videos]}

@app.get("/api/videos/{video_id}")
async def get_video(video_id: str):
    """Get specific video"""
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video.model_dump()

@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete video and associated files"""
    video = db.get_video(video_id)
    if video and video.file_path and os.path.exists(video.file_path):
        try:
            os.remove(video.file_path)
            print(f"✅ Deleted: {video.file_path}")
        except Exception as e:
            print(f"⚠️ Error deleting video file: {e}")
    
    # Delete analyzed video
    analyzed_path = UPLOADS_DIR / f"{video_id}_analyzed.mp4"
    if analyzed_path.exists():
        try:
            os.remove(analyzed_path)
            print(f"✅ Deleted: {analyzed_path}")
        except Exception as e:
            print(f"⚠️ Error deleting analyzed video: {e}")
    
    # Delete detection frames
    for f in DETECTIONS_DIR.glob(f"{video_id}_*.jpg"):
        try:
            os.remove(f)
        except:
            pass
    
    db.delete_video(video_id)
    return {"success": True}

@app.get("/api/video/{video_id}/stream")
async def stream_video(request: Request, video_id: str):
    """Stream video file with proper headers for HTML5 video"""
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Try analyzed version first (with annotations)
    analyzed_path = UPLOADS_DIR / f"{video_id}_analyzed.mp4"
    video_path = None
    
    if analyzed_path.exists():
        video_path = analyzed_path
        print(f"📹 Streaming analyzed video: {analyzed_path}")
    elif video.file_path and os.path.exists(video.file_path):
        video_path = video.file_path
        print(f"📹 Streaming original video: {video.file_path}")
    else:
        raise HTTPException(status_code=404, detail="Video file not found")
    
    file_size = os.path.getsize(video_path)
    
    # Get MIME type based on file extension
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type:
        mime_type = "video/mp4"
    
    # Handle range requests for seeking
    range_header = request.headers.get('range', '')
    
    if range_header:
        byte_range = range_header.replace('bytes=', '').split('-')
        start = int(byte_range[0])
        end = int(byte_range[1]) if byte_range[1] else file_size - 1
        
        def iterfile():
            with open(video_path, 'rb') as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
        
        headers = {
            'Content-Range': f'bytes {start}-{end}/{file_size}',
            'Accept-Ranges': 'bytes',
            'Content-Length': str(end - start + 1),
            'Content-Type': mime_type,
            'Cache-Control': 'no-cache'
        }
        return StreamingResponse(
            iterfile(),
            status_code=206,
            headers=headers
        )
    else:
        return FileResponse(
            video_path,
            media_type=mime_type,
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache",
                "Content-Disposition": f"inline; filename={os.path.basename(video_path)}"
            }
        )

@app.get("/api/incidents")
async def get_incidents(type: Optional[str] = None, status: Optional[str] = None):
    """Get incidents with optional filters"""
    incidents = db.get_all_incidents()
    if type and type != 'all':
        incidents = [i for i in incidents if i.type == type.lower()]
    if status and status != 'all':
        incidents = [i for i in incidents if i.status == status.lower()]
    incidents.sort(key=lambda x: x.timestamp, reverse=True)
    return {"incidents": [i.model_dump() for i in incidents]}

@app.get("/api/incidents/{incident_id}")
async def get_incident(incident_id: str):
    """Get specific incident"""
    incident = db.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident.model_dump()

@app.post("/api/incidents/{incident_id}/resolve")
async def resolve_incident(incident_id: str):
    """Resolve incident"""
    incident = db.get_incident(incident_id)
    if incident:
        incident.status = "resolved"
        db.update_incident(incident)
    return {"success": True}

@app.get("/api/alerts")
async def get_alerts():
    """Get active alerts"""
    alerts = db.get_active_alerts()
    return {"alerts": [a.model_dump() for a in alerts]}

@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge alert"""
    db.remove_alert(alert_id)
    return {"success": True}

@app.get("/api/analytics")
async def get_analytics(days: int = 30):
    """Get analytics"""
    return db.get_analytics(days).model_dump()

@app.get("/api/analytics/trends")
async def get_trends():
    """Get incident trends"""
    incidents = db.get_all_incidents()
    by_type = {}
    by_day = {}
    by_hour = {}
    for inc in incidents:
        by_type[inc.type] = by_type.get(inc.type, 0) + 1
        day = inc.timestamp.date().isoformat()
        by_day[day] = by_day.get(day, 0) + 1
        hour = inc.timestamp.strftime("%H:00")
        by_hour[hour] = by_hour.get(hour, 0) + 1
    return {
        "by_type": by_type,
        "by_day": by_day,
        "by_hour": by_hour,
        "total": len(incidents)
    }

@app.get("/api/users")
async def get_users():
    """Get all users"""
    return {"users": [u.model_dump() for u in db.get_all_users()]}

@app.post("/api/users")
async def create_user(user: UserCreate):
    """Create new user"""
    existing = db.get_user_by_username(user.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    existing_email = db.get_user_by_email(user.email)
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already exists")
    new_user = User(
        username=user.username,
        full_name=user.full_name,
        email=user.email,
        role=user.role,
        access_level=user.access_level,
        status="active",
        password=user.password
    )
    db.add_user(new_user)
    return {"success": True, "user": new_user.model_dump()}

@app.delete("/api/users/{username}")
async def delete_user(username: str):
    """Delete user"""
    if username == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete admin user")
    db.delete_user(username)
    return {"success": True}

@app.put("/api/users/{username}/reset-password")
async def reset_user_password(username: str, new_password: str):
    """Reset user password"""
    user = db.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.password = new_password
    db.update_user(user)
    return {"success": True}

# ==================== WEBSOCKET ====================
@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    print(f"🔌 WebSocket connected. Total: {len(active_connections)}")
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
            print(f"🔌 WebSocket disconnected. Total: {len(active_connections)}")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.websocket("/ws/digital-twin")
async def digital_twin_websocket(websocket: WebSocket):
    """WebSocket for digital twin real-time updates"""
    await websocket.accept()
    digital_twin_connections.add(websocket)
    print(f"🔮 Digital Twin connected. Total: {len(digital_twin_connections)}")
    
    # Send initial state
    try:
        initial_state = digital_twin.get_scene_state()
        await websocket.send_json(initial_state)
        print(f"📤 Sent initial state with {len(initial_state.get('vehicles', []))} vehicles")
    except Exception as e:
        print(f"⚠️ Error sending initial state: {e}")
    
    try:
        while True:
            # Keep connection alive and handle messages
            data = await websocket.receive_text()
            
            if data == "get_state":
                # Send current state
                await websocket.send_json(digital_twin.get_scene_state())
            elif data == "ping":
                await websocket.send_text("pong")
            else:
                # Try to parse as JSON for other commands
                try:
                    command = json.loads(data)
                    if command.get("action") == "select_video":
                        video_id = command.get("video_id")
                        print(f"📹 Digital Twin selected video: {video_id}")
                except:
                    pass
                    
    except WebSocketDisconnect:
        digital_twin_connections.discard(websocket)
        print(f"🔮 Digital Twin disconnected. Total: {len(digital_twin_connections)}")
    except Exception as e:
        print(f"⚠️ Digital Twin WebSocket error: {e}")
        digital_twin_connections.discard(websocket)

async def broadcast_alert(alert: Alert):
    """Broadcast alert to all connected clients"""
    if not active_connections:
        return
    alert_data = alert.model_dump()
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(alert_data)
        except:
            disconnected.append(connection)
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)

async def broadcast_digital_twin(state):
    """Broadcast digital twin state to all connected clients"""
    if not digital_twin_connections:
        return
    
    disconnected = set()
    for connection in digital_twin_connections:
        try:
            await connection.send_json(state)
        except Exception as e:
            print(f"⚠️ Failed to send to client: {e}")
            disconnected.add(connection)
    
    # Remove disconnected clients
    digital_twin_connections.difference_update(disconnected)

# ==================== HEALTH CHECK ====================
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "yolo_loaded": yolo_detector is not None,
        "videos_count": len(db.get_all_videos()),
        "incidents_count": len(db.get_all_incidents()),
        "active_connections": len(active_connections),
        "digital_twin_connections": len(digital_twin_connections)
    }

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    videos = db.get_all_videos()
    incidents = db.get_all_incidents()
    total_videos = len(videos)
    processed_videos = len([v for v in videos if v.status == "processed"])
    processing_videos = len([v for v in videos if v.status == "processing"])
    total_incidents = len(incidents)
    pending_incidents = len([i for i in incidents if i.status == "pending"])
    resolved_incidents = len([i for i in incidents if i.status == "resolved"])
    
    if incidents:
        avg_confidence = sum(
            float(i.confidence.replace('%', '')) for i in incidents
        ) / len(incidents)
    else:
        avg_confidence = 0
    
    return {
        "videos": {
            "total": total_videos,
            "processed": processed_videos,
            "processing": processing_videos,
            "error": len([v for v in videos if v.status == "error"])
        },
        "incidents": {
            "total": total_incidents,
            "pending": pending_incidents,
            "resolved": resolved_incidents,
            "avg_confidence": f"{avg_confidence:.1f}%"
        },
        "system": {
            "yolo_loaded": yolo_detector is not None,
            "websocket_connections": len(active_connections),
            "digital_twin_connections": len(digital_twin_connections),
            "database_file": db.data_file,
            "backup_count": len(list(db.backup_dir.glob("*.json")))
        }
    }

# ==================== DIGITAL TWIN TEST ENDPOINT ====================
@app.post("/api/digital-twin/test")
async def test_digital_twin():
    """Generate test data for digital twin (for debugging)"""
    import random
    
    # Generate test data
    video_id = f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create test vehicles
    vehicles = []
    for i in range(random.randint(5, 15)):
        vehicle_type = random.choice(['car', 'truck', 'bus', 'motorcycle'])
        vehicles.append({
            'track_id': i,
            'type': vehicle_type,
            'bbox': [
                random.randint(100, 500),
                random.randint(100, 300),
                random.randint(40, 80),
                random.randint(30, 60)
            ],
            'confidence': random.uniform(0.7, 0.99),
            'speed': random.uniform(20, 90)
        })
    
    # Create test pedestrians
    pedestrians = []
    for i in range(random.randint(0, 5)):
        pedestrians.append({
            'track_id': i + 100,
            'bbox': [
                random.randint(100, 500),
                random.randint(100, 300),
                20,
                50
            ],
            'confidence': random.uniform(0.7, 0.95)
        })
    
    # Random incident type
    incident_types = ['normal', 'warning', 'jam', 'accident']
    weights = [0.7, 0.15, 0.1, 0.05]
    incident_type = random.choices(incident_types, weights=weights)[0]
    incident_confidence = random.uniform(0.6, 0.95) if incident_type != 'normal' else 0
    
    # Calculate average speed
    avg_speed = np.mean([v['speed'] for v in vehicles]) if vehicles else 50
    
    # Update digital twin
    digital_twin.update_from_detection(
        video_id=video_id,
        frame_number=random.randint(1, 1000),
        vehicles=vehicles,
        pedestrians=pedestrians,
        incident_type=incident_type,
        incident_confidence=incident_confidence,
        avg_speed=avg_speed
    )
    
    # Broadcast update
    await broadcast_digital_twin(digital_twin.get_scene_state())
    
    return {
        "success": True,
        "message": "Test data generated",
        "video_id": video_id,
        "vehicles": len(vehicles),
        "pedestrians": len(pedestrians),
        "incident": incident_type,
        "connections": len(digital_twin_connections)
    }

@app.get("/api/digital-twin/test-broadcast")
async def test_digital_twin_broadcast():
    """Test endpoint to broadcast test data to Digital Twin"""
    
    # Generate test vehicles
    test_vehicles = []
    for i in range(10):
        test_vehicles.append({
            'track_id': i,
            'type': ['car', 'truck', 'bus', 'motorcycle'][i % 4],
            'bbox': [100 + i*30, 100 + i*20, 50, 40],
            'confidence': 0.9,
            'speed': 40 + i * 5
        })
    
    test_pedestrians = []
    for i in range(3):
        test_pedestrians.append({
            'track_id': i + 100,
            'bbox': [200 + i*40, 200, 20, 50],
            'confidence': 0.85
        })
    
    # Update digital twin
    digital_twin.update_from_detection(
        video_id="test_video",
        frame_number=1,
        vehicles=test_vehicles,
        pedestrians=test_pedestrians,
        incident_type="normal",
        incident_confidence=0,
        avg_speed=45
    )
    
    # Broadcast update
    await broadcast_digital_twin(digital_twin.get_scene_state())
    
    return {
        "success": True,
        "message": "Test data broadcast to Digital Twin",
        "vehicles": len(test_vehicles),
        "pedestrians": len(test_pedestrians),
        "connections": len(digital_twin_connections)
    }

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 VideoGuard AI - YOLOv8 Video Analysis System")
    print("="*70)
    print("\n✅ System Ready!")
    print("✅ YOLOv8 deep learning detection")
    print("✅ Vehicle/Pedestrian differentiation")
    print("✅ Accident and traffic jam detection")
    print("✅ Real-time visual feedback")
    print("✅ Digital Twin 3D visualization")
    print("\n📁 Upload directory: static/uploads")
    print("📁 Detections directory: static/detections")
    print("📁 Backups directory: backups")
    print("\n🌐 Server: http://localhost:8000")
    print("👤 Default login: admin / admin123")
    print("="*70 + "\n")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )