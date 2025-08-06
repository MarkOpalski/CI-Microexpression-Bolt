"""
High-accuracy face detection and cropping using MTCNN
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from config import VIDEO_CONFIG


class FaceDetector:
    """
    MTCNN-based face detection with confidence filtering
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize face detector
        
        Args:
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.confidence_threshold = VIDEO_CONFIG["face_detection_confidence"]
        self.min_face_size = VIDEO_CONFIG["face_min_size"]
        
        # Initialize MTCNN
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=self.min_face_size,
            thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
            factor=0.709,
            post_process=False,
            device=self.device,
            keep_all=True
        )
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of face detection results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Detect faces
        boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
        
        faces = []
        
        if boxes is not None:
            for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                if prob >= self.confidence_threshold:
                    # Convert box coordinates to integers
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure coordinates are within frame bounds
                    height, width = frame.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    # Calculate face dimensions
                    face_width = x2 - x1
                    face_height = y2 - y1
                    
                    # Skip faces that are too small
                    if face_width < self.min_face_size or face_height < self.min_face_size:
                        continue
                    
                    face_info = {
                        'id': i,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(prob),
                        'landmarks': landmark.tolist() if landmark is not None else None,
                        'width': face_width,
                        'height': face_height,
                        'area': face_width * face_height
                    }
                    
                    faces.append(face_info)
        
        # Sort faces by area (largest first)
        faces.sort(key=lambda x: x['area'], reverse=True)
        
        return faces
    
    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                  margin: float = 0.2) -> np.ndarray:
        """
        Crop face from frame with optional margin
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            margin: Additional margin around face (as fraction of face size)
            
        Returns:
            Cropped face image
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate face dimensions
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Add margin
        margin_x = int(face_width * margin)
        margin_y = int(face_height * margin)
        
        # Expand bounding box
        x1_expanded = max(0, x1 - margin_x)
        y1_expanded = max(0, y1 - margin_y)
        x2_expanded = min(frame.shape[1], x2 + margin_x)
        y2_expanded = min(frame.shape[0], y2 + margin_y)
        
        # Crop face
        face_crop = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
        
        return face_crop
    
    def get_largest_face(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Get the largest face in the frame
        
        Args:
            frame: Input frame
            
        Returns:
            Face detection result or None if no face found
        """
        faces = self.detect_faces(frame)
        return faces[0] if faces else None
    
    def track_faces(self, frames: List[np.ndarray], 
                   max_distance: float = 50.0) -> List[List[Dict[str, Any]]]:
        """
        Simple face tracking across multiple frames
        
        Args:
            frames: List of frames
            max_distance: Maximum distance for face matching
            
        Returns:
            List of face detections for each frame with tracking IDs
        """
        all_detections = []
        tracked_faces = []
        next_track_id = 0
        
        for frame_idx, frame in enumerate(frames):
            faces = self.detect_faces(frame)
            
            if frame_idx == 0:
                # Initialize tracking for first frame
                for face in faces:
                    face['track_id'] = next_track_id
                    next_track_id += 1
                    tracked_faces.append({
                        'track_id': face['track_id'],
                        'last_bbox': face['bbox'],
                        'last_seen': frame_idx
                    })
            else:
                # Match faces with existing tracks
                for face in faces:
                    best_match = None
                    best_distance = float('inf')
                    
                    face_center = self._get_bbox_center(face['bbox'])
                    
                    for tracked_face in tracked_faces:
                        if frame_idx - tracked_face['last_seen'] > 5:
                            continue  # Skip old tracks
                        
                        tracked_center = self._get_bbox_center(tracked_face['last_bbox'])
                        distance = self._calculate_distance(face_center, tracked_center)
                        
                        if distance < max_distance and distance < best_distance:
                            best_match = tracked_face
                            best_distance = distance
                    
                    if best_match:
                        # Update existing track
                        face['track_id'] = best_match['track_id']
                        best_match['last_bbox'] = face['bbox']
                        best_match['last_seen'] = frame_idx
                    else:
                        # Create new track
                        face['track_id'] = next_track_id
                        next_track_id += 1
                        tracked_faces.append({
                            'track_id': face['track_id'],
                            'last_bbox': face['bbox'],
                            'last_seen': frame_idx
                        })
            
            all_detections.append(faces)
        
        return all_detections
    
    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def visualize_detections(self, frame: np.ndarray, 
                           faces: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw face detection results on frame
        
        Args:
            frame: Input frame
            faces: List of face detections
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            confidence = face['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            label = f"Face: {confidence:.2f}"
            if 'track_id' in face:
                label = f"ID{face['track_id']}: {confidence:.2f}"
            
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw landmarks if available
            if face['landmarks']:
                landmarks = np.array(face['landmarks'], dtype=int)
                for point in landmarks:
                    cv2.circle(vis_frame, tuple(point), 2, (255, 0, 0), -1)
        
        return vis_frame


def create_face_detector(device: Optional[str] = None) -> FaceDetector:
    """
    Factory function to create face detector
    
    Args:
        device: Device to run inference on
        
    Returns:
        FaceDetector instance
    """
    return FaceDetector(device)