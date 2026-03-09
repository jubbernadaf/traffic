# digital_twin.py
import json
import numpy as np
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import asyncio
import math
import random
from collections import deque

class DigitalTwin:
    """Digital Twin for real-time traffic visualization with actual video data"""
    
    def __init__(self):
        self.vehicles = {}  # vehicle_id -> vehicle data
        self.pedestrians = {}  # pedestrian_id -> pedestrian data
        self.active_video_id = None
        self.last_update = datetime.now()
        self.update_callbacks = []
        self.track_history = deque(maxlen=100)  # Store last 100 positions for smooth animation
        
        # Initialize scene state
        self.scene_state = {
            'timestamp': datetime.now().isoformat(),
            'video_id': None,
            'frame': 0,
            'vehicles': [],
            'pedestrians': [],
            'incident': {
                'type': 'normal',
                'confidence': 0
            },
            'statistics': {
                'total_vehicles': 0,
                'total_pedestrians': 0,
                'avg_speed': 0,
                'traffic_density': 0,
                'max_speed': 0,
                'min_speed': 0
            }
        }
        
        print("✅ Digital Twin initialized")
        
    def update_from_detection(self, 
                             video_id: str, 
                             frame_number: int,
                             vehicles: List[Dict], 
                             pedestrians: List[Dict],
                             incident_type: str = 'normal',
                             incident_confidence: float = 0.0,
                             avg_speed: float = 0.0):
        """Update digital twin with YOLOv8 detection data from video processing"""
        
        try:
            self.active_video_id = video_id
            current_time = datetime.now()
            self.last_update = current_time
            
            # Clear old objects if this is a new video
            if video_id != self.scene_state.get('video_id'):
                print(f"🔄 New video detected: {video_id}, resetting digital twin")
                self.vehicles = {}
                self.pedestrians = {}
            
            # Process vehicles with proper 3D positioning
            current_vehicle_ids = set()
            processed_vehicles = []
            
            for i, v in enumerate(vehicles):
                # Generate consistent track ID
                track_id = v.get('track_id', f"v_{frame_number}_{i}")
                current_vehicle_ids.add(str(track_id))
                
                # Get bounding box (x, y, width, height)
                bbox = v.get('bbox', [100 + i*50, 100 + i*30, 50, 40])
                x, y, w, h = bbox
                
                # Calculate center position in video coordinates
                center_x = x + w/2
                center_y = y + h/2
                
                # Map to 3D space with better distribution
                # Use a grid-like layout based on video position
                # Scale down and center around origin
                grid_x = (center_x / 50) - 10  # Range approximately -10 to 10
                grid_z = (center_y / 50) - 5   # Range approximately -5 to 10
                pos_y = 0.5  # Ground level
                
                # Get or calculate speed
                speed = float(v.get('speed', avg_speed if avg_speed > 0 else 50))
                
                # Get vehicle type
                vehicle_type = v.get('type', 'car')
                if vehicle_type not in ['car', 'truck', 'bus', 'motorcycle']:
                    vehicle_type = 'car'
                
                # Get confidence
                confidence = v.get('confidence', 0.9)
                
                # Calculate movement direction for rotation
                dx = 0
                dz = 0
                if str(track_id) in self.vehicles:
                    old_pos = self.vehicles[str(track_id)]['position']
                    dx = grid_x - old_pos['x']
                    dz = grid_z - old_pos['z']
                
                # Create vehicle data for digital twin
                vehicle_data = {
                    'id': str(track_id),
                    'type': vehicle_type,
                    'position': {
                        'x': float(grid_x),
                        'y': float(pos_y),
                        'z': float(grid_z)
                    },
                    'speed': float(speed),
                    'confidence': float(confidence),
                    'color': self._get_color_by_type(vehicle_type),
                    'rotation': math.atan2(dx, dz) if dx != 0 or dz != 0 else 0,
                    'last_seen': frame_number,
                    'video_id': video_id,
                    'timestamp': current_time.isoformat(),
                    'bbox': bbox
                }
                
                # Smooth position update if vehicle exists
                if str(track_id) in self.vehicles:
                    old_pos = self.vehicles[str(track_id)]['position']
                    vehicle_data['position'] = {
                        'x': old_pos['x'] * 0.7 + grid_x * 0.3,
                        'y': old_pos['y'] * 0.7 + pos_y * 0.3,
                        'z': old_pos['z'] * 0.7 + grid_z * 0.3
                    }
                
                self.vehicles[str(track_id)] = vehicle_data
                processed_vehicles.append(vehicle_data)
            
            # Process pedestrians
            current_ped_ids = set()
            processed_pedestrians = []
            
            for i, p in enumerate(pedestrians):
                track_id = p.get('track_id', f"p_{frame_number}_{i}")
                current_ped_ids.add(str(track_id))
                
                bbox = p.get('bbox', [100 + i*60, 200, 20, 50])
                x, y, w, h = bbox
                
                center_x = x + w/2
                center_y = y + h/2
                
                grid_x = (center_x / 50) - 10
                grid_z = (center_y / 50) - 5
                pos_y = 1.0  # Pedestrians are slightly above ground
                
                pedestrian_data = {
                    'id': str(track_id),
                    'type': 'pedestrian',
                    'position': {
                        'x': float(grid_x),
                        'y': float(pos_y),
                        'z': float(grid_z)
                    },
                    'speed': 5.0,
                    'confidence': float(p.get('confidence', 0.85)),
                    'color': '#8b5cf6',
                    'last_seen': frame_number,
                    'video_id': video_id,
                    'timestamp': current_time.isoformat()
                }
                
                # Smooth update
                if str(track_id) in self.pedestrians:
                    old_pos = self.pedestrians[str(track_id)]['position']
                    pedestrian_data['position'] = {
                        'x': old_pos['x'] * 0.7 + grid_x * 0.3,
                        'y': old_pos['y'] * 0.7 + pos_y * 0.3,
                        'z': old_pos['z'] * 0.7 + grid_z * 0.3
                    }
                
                self.pedestrians[str(track_id)] = pedestrian_data
                processed_pedestrians.append(pedestrian_data)
            
            # Remove stale objects (not seen for >30 frames)
            stale_threshold = frame_number - 30
            self.vehicles = {k: v for k, v in self.vehicles.items() 
                           if v['last_seen'] > stale_threshold}
            self.pedestrians = {k: v for k, v in self.pedestrians.items() 
                               if v['last_seen'] > stale_threshold}
            
            # Calculate statistics
            vehicle_speeds = [v['speed'] for v in self.vehicles.values() if v['speed'] > 0]
            avg_speed_calc = np.mean(vehicle_speeds) if vehicle_speeds else avg_speed
            max_speed = np.max(vehicle_speeds) if vehicle_speeds else 0
            min_speed = np.min(vehicle_speeds) if vehicle_speeds else 0
            
            # Calculate traffic density
            area = 400  # Visualization area (20x20)
            density = len(self.vehicles) / area * 100 if area > 0 else 0
            
            # Update scene state
            self.scene_state = {
                'timestamp': current_time.isoformat(),
                'video_id': video_id,
                'frame': frame_number,
                'vehicles': list(self.vehicles.values()),
                'pedestrians': list(self.pedestrians.values()),
                'incident': {
                    'type': incident_type,
                    'confidence': float(incident_confidence)
                },
                'statistics': {
                    'total_vehicles': len(self.vehicles),
                    'total_pedestrians': len(self.pedestrians),
                    'avg_speed': float(avg_speed_calc),
                    'traffic_density': float(min(density, 100)),
                    'max_speed': float(max_speed),
                    'min_speed': float(min_speed)
                }
            }
            
            # Notify callbacks
            if self.update_callbacks:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._notify_callbacks())
                    else:
                        loop.run_until_complete(self._notify_callbacks())
                except Exception as e:
                    print(f"⚠️ Error notifying callbacks: {e}")
            
            # Print debug info occasionally
            if frame_number % 100 == 0:
                print(f"📊 Digital Twin: {len(self.vehicles)} vehicles, {len(self.pedestrians)} pedestrians, incident: {incident_type}")
            
        except Exception as e:
            print(f"⚠️ Error in digital twin update: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_color_by_type(self, vehicle_type: str) -> str:
        """Get color for vehicle type"""
        colors = {
            'car': '#60a5fa',      # Blue
            'truck': '#f97316',     # Orange
            'bus': '#ef4444',       # Red
            'motorcycle': '#10b981', # Green
            'pedestrian': '#8b5cf6'  # Purple
        }
        return colors.get(vehicle_type, '#8b5cf6')
    
    async def _notify_callbacks(self):
        """Notify all registered callbacks of state update"""
        for callback in self.update_callbacks:
            try:
                await callback(self.scene_state)
            except Exception as e:
                print(f"⚠️ Error in callback: {e}")
    
    def register_callback(self, callback):
        """Register callback for state updates"""
        if callback not in self.update_callbacks:
            self.update_callbacks.append(callback)
            print(f"✅ Callback registered. Total callbacks: {len(self.update_callbacks)}")
    
    def get_scene_state(self) -> Dict:
        """Get current scene state"""
        return self.scene_state
    
    def reset(self):
        """Reset digital twin"""
        self.vehicles = {}
        self.pedestrians = {}
        self.track_history.clear()
        self.scene_state = {
            'timestamp': datetime.now().isoformat(),
            'video_id': None,
            'frame': 0,
            'vehicles': [],
            'pedestrians': [],
            'incident': {
                'type': 'normal',
                'confidence': 0
            },
            'statistics': {
                'total_vehicles': 0,
                'total_pedestrians': 0,
                'avg_speed': 0,
                'traffic_density': 0,
                'max_speed': 0,
                'min_speed': 0
            }
        }
        print("✅ Digital Twin reset")
    
    def generate_test_data(self, video_id: str = "test_video"):
        """Generate test data for debugging (used when no real video is playing)"""
        import random
        
        frame_number = random.randint(1, 1000)
        vehicles = []
        
        # Generate random vehicles for testing
        for i in range(random.randint(5, 15)):
            vehicle_type = random.choice(['car', 'truck', 'bus', 'motorcycle'])
            vehicles.append({
                'track_id': f"test_{i}",
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
        
        pedestrians = []
        for i in range(random.randint(0, 5)):
            pedestrians.append({
                'track_id': f"test_ped_{i}",
                'bbox': [
                    random.randint(100, 500),
                    random.randint(100, 300),
                    20,
                    50
                ],
                'confidence': random.uniform(0.7, 0.95)
            })
        
        incident_types = ['normal', 'warning', 'jam', 'accident']
        weights = [0.7, 0.15, 0.1, 0.05]
        incident_type = random.choices(incident_types, weights=weights)[0]
        incident_confidence = random.uniform(0.6, 0.95) if incident_type != 'normal' else 0
        
        avg_speed = np.mean([v['speed'] for v in vehicles]) if vehicles else 50
        
        self.update_from_detection(
            video_id=video_id,
            frame_number=frame_number,
            vehicles=vehicles,
            pedestrians=pedestrians,
            incident_type=incident_type,
            incident_confidence=incident_confidence,
            avg_speed=avg_speed
        )