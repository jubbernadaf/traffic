import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
import cv2
from ultralytics import YOLO
import torch

class YOLODetector:
    """YOLOv8-based vehicle and pedestrian detection"""
    def __init__(self, model_path: str = 'yolov8n.pt'):
        try:
            self.model = YOLO(model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"YOLO Detector loaded on {self.device}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
            self.device = 'cpu'
        
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        self.pedestrian_class = {0: 'person'}
        self.detection_history = deque(maxlen=30)
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Dict[str, List]:
        """Run YOLO detection on frame"""
        if self.model is None:
            return {
                'vehicles': [],
                'pedestrians': [],
                'total_vehicles': 0,
                'total_pedestrians': 0
            }
        
        try:
            results = self.model(frame, verbose=False, conf=conf_threshold)[0]
            vehicles = []
            pedestrians = []
            
            for det in results.boxes.data:
                x1, y1, x2, y2, conf, cls = det.tolist()
                if conf < conf_threshold:
                    continue
                cls = int(cls)
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                detection = {
                    'bbox': bbox,
                    'center': center,
                    'confidence': conf,
                    'class_id': cls,
                    'area': (x2 - x1) * (y2 - y1)
                }
                
                if cls in self.vehicle_classes:
                    detection['type'] = self.vehicle_classes[cls]
                    vehicles.append(detection)
                elif cls in self.pedestrian_class:
                    detection['type'] = 'person'
                    pedestrians.append(detection)
            
            self.detection_history.append({
                'vehicles': vehicles,
                'pedestrians': pedestrians,
                'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
            })
            
            return {
                'vehicles': vehicles,
                'pedestrians': pedestrians,
                'total_vehicles': len(vehicles),
                'total_pedestrians': len(pedestrians)
            }
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return {
                'vehicles': [],
                'pedestrians': [],
                'total_vehicles': 0,
                'total_pedestrians': 0
            }
    
    def estimate_speed(self) -> float:
        """Estimate traffic speed based on vehicle movement"""
        if len(self.detection_history) < 10:
            return 60.0
        
        try:
            recent = list(self.detection_history)[-10:]
            displacements = []
            
            for i in range(1, len(recent)):
                prev_vehicles = recent[i-1]['vehicles']
                curr_vehicles = recent[i]['vehicles']
                
                for curr in curr_vehicles:
                    for prev in prev_vehicles:
                        dist = np.sqrt(
                            (curr['center'][0] - prev['center'][0])**2 +
                            (curr['center'][1] - prev['center'][1])**2
                        )
                        if dist < 50:
                            displacements.append(dist)
                            break
            
            if not displacements:
                return 60.0
            
            avg_displacement = np.mean(displacements)
            speed = avg_displacement * 3.0
            return min(max(speed, 0), 120)
        except Exception as e:
            print(f"Error estimating speed: {e}")
            return 60.0

class VehicleTracker:
    """Advanced vehicle tracking using YOLO detections and IOU matching"""
    def __init__(self, max_distance: float = 100, max_frames_to_skip: int = 15):
        self.max_distance = max_distance
        self.max_frames_to_skip = max_frames_to_skip
        self.tracks = {}
        self.next_id = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new YOLO detections"""
        tracked_vehicles = []
        
        if not detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['last_seen'] += 1
                if self.tracks[track_id]['last_seen'] > self.max_frames_to_skip:
                    del self.tracks[track_id]
            return []
        
        if self.tracks:
            matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(self.tracks.keys())
            
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            track_ids = list(self.tracks.keys())
            
            for i, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                for j, det in enumerate(detections):
                    iou = self._calculate_iou(track['bbox'], det['bbox'])
                    iou_matrix[i, j] = iou
            
            while iou_matrix.size > 0 and iou_matrix.shape[0] > 0 and iou_matrix.shape[1] > 0:
                max_iou_idx = np.argmax(iou_matrix)
                i, j = np.unravel_index(max_iou_idx, iou_matrix.shape)
                max_iou = iou_matrix[i, j]
                
                if max_iou < 0.3:
                    break
                
                track_id = track_ids[i]
                det_idx = j
                matches.append((track_id, det_idx))
                
                if track_id in unmatched_tracks:
                    unmatched_tracks.remove(track_id)
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
                
                iou_matrix = np.delete(iou_matrix, i, axis=0)
                iou_matrix = np.delete(iou_matrix, j, axis=1)
                track_ids = np.delete(track_ids, i).tolist()
            
            for track_id, det_idx in matches:
                det = detections[det_idx]
                track = self.tracks[track_id]
                
                if len(track['positions']) > 5:
                    prev_pos = track['positions'][-5]
                    curr_pos = det['center']
                    dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                    speed = dist * 2.0
                else:
                    speed = 0
                
                track['positions'].append(det['center'])
                track['bboxes'].append(det['bbox'])
                track['last_seen'] = 0
                track['bbox'] = det['bbox']
                track['confidence'] = det['confidence']
                track['type'] = det['type']
                track['speed'] = speed
                det['track_id'] = track_id
                det['speed'] = speed
                tracked_vehicles.append(det)
            
            for det_idx in unmatched_detections:
                det = detections[det_idx]
                self.tracks[self.next_id] = {
                    'positions': [det['center']],
                    'bboxes': [det['bbox']],
                    'last_seen': 0,
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'type': det['type'],
                    'speed': 0.0
                }
                det['track_id'] = self.next_id
                det['speed'] = 0.0
                tracked_vehicles.append(det)
                self.next_id += 1
            
            for track_id in unmatched_tracks:
                if track_id in self.tracks:
                    self.tracks[track_id]['last_seen'] += 1
                    if self.tracks[track_id]['last_seen'] > self.max_frames_to_skip:
                        del self.tracks[track_id]
        else:
            for det in detections:
                self.tracks[self.next_id] = {
                    'positions': [det['center']],
                    'bboxes': [det['bbox']],
                    'last_seen': 0,
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'type': det['type'],
                    'speed': 0.0
                }
                det['track_id'] = self.next_id
                det['speed'] = 0.0
                tracked_vehicles.append(det)
                self.next_id += 1
        
        return tracked_vehicles
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        try:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[0] + box1[2], box2[0] + box2[2])
            y2 = min(box1[1] + box1[3], box2[1] + box2[3])
            
            if x2 < x1 or y2 < y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            box1_area = box1[2] * box1[3]
            box2_area = box2[2] * box2[3]
            union = box1_area + box2_area - intersection
            
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            print(f"Error calculating IOU: {e}")
            return 0.0

class AnomalyDetector:
    """Statistical anomaly detection with adaptive thresholds"""
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.vehicle_history = deque(maxlen=window_size)
        self.pedestrian_history = deque(maxlen=window_size)
        self.speed_history = deque(maxlen=window_size)
        self.anomaly_threshold = 2.5
    
    def update(self, vehicle_count: int, pedestrian_count: int, speed: float):
        """Add new data point"""
        self.vehicle_history.append(vehicle_count)
        self.pedestrian_history.append(pedestrian_count)
        self.speed_history.append(speed)
    
    def detect(self) -> Tuple[str, float, Dict]:
        """Detect anomalies and classify"""
        if len(self.vehicle_history) < self.window_size:
            return 'normal', 0.0, {}
        
        try:
            vehicle_list = list(self.vehicle_history)
            speed_list = list(self.speed_history)
            
            vehicle_mean = np.mean(vehicle_list[:-5]) if len(vehicle_list) > 5 else np.mean(vehicle_list)
            vehicle_std = np.std(vehicle_list[:-5]) + 1e-6 if len(vehicle_list) > 5 else np.std(vehicle_list) + 1e-6
            speed_mean = np.mean(speed_list[:-5]) if len(speed_list) > 5 else np.mean(speed_list)
            speed_std = np.std(speed_list[:-5]) + 1e-6 if len(speed_list) > 5 else np.std(speed_list) + 1e-6
            
            current_vehicles = self.vehicle_history[-1]
            current_speed = self.speed_history[-1]
            prev_speed = self.speed_history[-2] if len(self.speed_history) > 1 else current_speed
            
            vehicle_zscore = (current_vehicles - vehicle_mean) / vehicle_std
            speed_zscore = (speed_mean - current_speed) / speed_std if current_speed < speed_mean else 0
            speed_drop = (prev_speed - current_speed) / (prev_speed + 1)
            
            # Detect accident
            if speed_drop > 0.7 and current_speed < 15:
                confidence = min(speed_drop * 1.2, 1.0)
                if current_vehicles > vehicle_mean * 1.3:
                    confidence = min(confidence + 0.2, 1.0)
                details = {
                    'speed_drop': float(speed_drop),
                    'current_speed': float(current_speed),
                    'vehicle_ratio': float(current_vehicles / vehicle_mean) if vehicle_mean > 0 else 1.0
                }
                return 'accident', confidence, details
            
            # Detect traffic jam
            if current_speed < 25 and current_vehicles > vehicle_mean * 1.2:
                recent_speeds = list(self.speed_history)[-10:]
                if np.mean(recent_speeds) < 30:
                    jam_severity = (25 - current_speed) / 25
                    congestion = (current_vehicles / vehicle_mean) - 1 if vehicle_mean > 0 else 0
                    confidence = min(jam_severity * 0.6 + congestion * 0.4, 1.0)
                    details = {
                        'jam_severity': float(jam_severity),
                        'congestion': float(congestion),
                        'avg_speed': float(np.mean(recent_speeds))
                    }
                    return 'jam', confidence, details
            
            # Detect warning
            if abs(vehicle_zscore) > self.anomaly_threshold or speed_zscore > self.anomaly_threshold:
                confidence = min(max(abs(vehicle_zscore), speed_zscore) / 5.0, 1.0)
                details = {
                    'vehicle_zscore': float(vehicle_zscore),
                    'speed_zscore': float(speed_zscore)
                }
                return 'warning', confidence, details
            
            return 'normal', 0.0, {}
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return 'normal', 0.0, {}

class KalmanFilter:
    """Kalman filter for noise reduction with adaptive parameters"""
    def __init__(self):
        self.x = 0
        self.P = 1
        self.Q = 0.01
        self.R = 0.1
        self.K = 0
        self.initialized = False
        self.measurements = []
    
    def update(self, measurement: float) -> float:
        """Update with new measurement"""
        self.measurements.append(measurement)
        
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return measurement
        
        try:
            if len(self.measurements) > 10:
                measurement_std = np.std(self.measurements[-10:])
                self.Q = measurement_std * 0.01
            
            x_pred = self.x
            P_pred = self.P + self.Q
            self.K = P_pred / (P_pred + self.R)
            self.x = x_pred + self.K * (measurement - x_pred)
            self.P = (1 - self.K) * P_pred
            
            return self.x
        except Exception as e:
            print(f"Error in Kalman filter: {e}")
            return measurement

class BayesianNetwork:
    """Bayesian network for incident probability"""
    def __init__(self):
        self.priors = {
            'normal': 0.95,
            'warning': 0.03,
            'jam': 0.015,
            'accident': 0.005
        }
    
    def update(self, detection_result: Tuple[str, float, Dict],
               vehicle_count: int, speed: float) -> Dict[str, float]:
        """Update probabilities based on detection"""
        try:
            incident_type, confidence, details = detection_result
            probs = self.priors.copy()
            
            if incident_type == 'accident':
                probs['accident'] = min(0.5 + confidence * 0.5, 0.95)
                probs['jam'] = probs['jam'] * 0.5
                probs['normal'] = probs['normal'] * 0.1
                probs['warning'] = probs['warning'] * 0.5
            elif incident_type == 'jam':
                probs['jam'] = min(0.4 + confidence * 0.5, 0.9)
                probs['accident'] = probs['accident'] * 0.3
                probs['normal'] = probs['normal'] * 0.2
                probs['warning'] = probs['warning'] * 0.7
            elif incident_type == 'warning':
                probs['warning'] = min(0.3 + confidence * 0.4, 0.8)
                probs['jam'] = probs['jam'] * 1.5
                probs['accident'] = probs['accident'] * 1.5
                probs['normal'] = probs['normal'] * 0.7
            else:
                probs['accident'] = max(probs['accident'] * 0.9, 0.005)
                probs['jam'] = max(probs['jam'] * 0.9, 0.01)
                probs['warning'] = max(probs['warning'] * 0.9, 0.02)
                probs['normal'] = 1 - (probs['accident'] + probs['jam'] + probs['warning'])
            
            total = sum(probs.values())
            if total > 0:
                for k in probs:
                    probs[k] /= total
            
            return probs
        except Exception as e:
            print(f"Error in Bayesian network: {e}")
            return self.priors