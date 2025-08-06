"""
Video ingestion module for live streams and file uploads
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Union, Tuple
import threading
import queue
import time

from config import VIDEO_CONFIG, SECURITY_CONFIG
from utils.helpers import (
    clean_media_metadata, validate_file_size, 
    get_video_info, generate_session_id, calculate_file_hash
)
from security.audit_log import log_video_ingest, log_security_event


class VideoIngestionError(Exception):
    """Custom exception for video ingestion errors"""
    pass


class VideoSource:
    """
    Unified interface for video sources (webcam, files, streams)
    """
    
    def __init__(self, source: Union[str, int, Path], 
                 user_id: str = "UNKNOWN",
                 session_id: Optional[str] = None):
        """
        Initialize video source
        
        Args:
            source: Video source (webcam index, file path, or stream URL)
            user_id: User identifier for audit logging
            session_id: Session identifier
        """
        self.source = source
        self.user_id = user_id
        self.session_id = session_id or generate_session_id()
        self.cap = None
        self.is_live = False
        self.is_file = False
        self.cleaned_path = None
        self._frame_queue = queue.Queue(maxsize=30)
        self._capture_thread = None
        self._stop_event = threading.Event()
        
        self._initialize_source()
    
    def _initialize_source(self):
        """Initialize the video source"""
        try:
            # Determine source type
            if isinstance(self.source, int):
                # Webcam
                self.is_live = True
                self.cap = cv2.VideoCapture(self.source)
                log_security_event(
                    f"Webcam access initiated (device {self.source})",
                    self.user_id, self.session_id, "INFO"
                )
            
            elif isinstance(self.source, (str, Path)):
                source_path = Path(self.source)
                
                if source_path.exists():
                    # Local file
                    self.is_file = True
                    self._process_file(source_path)
                else:
                    # Assume it's a stream URL
                    self.is_live = True
                    self.cap = cv2.VideoCapture(str(self.source))
                    log_security_event(
                        f"Stream access initiated: {self.source}",
                        self.user_id, self.session_id, "INFO"
                    )
            
            if self.cap and not self.cap.isOpened():
                raise VideoIngestionError(f"Failed to open video source: {self.source}")
            
            # Configure capture properties
            if self.cap:
                self._configure_capture()
        
        except Exception as e:
            log_security_event(
                f"Video source initialization failed: {str(e)}",
                self.user_id, self.session_id, "ERROR"
            )
            raise VideoIngestionError(f"Failed to initialize video source: {e}")
    
    def _process_file(self, file_path: Path):
        """Process and validate video file"""
        # Validate file size
        if not validate_file_size(file_path):
            raise VideoIngestionError(
                f"File size exceeds limit: {SECURITY_CONFIG['max_file_size_mb']}MB"
            )
        
        # Validate file format
        if file_path.suffix.lower() not in VIDEO_CONFIG["supported_formats"]:
            raise VideoIngestionError(
                f"Unsupported file format: {file_path.suffix}"
            )
        
        # Calculate file hash for audit trail
        file_hash = calculate_file_hash(file_path)
        
        # Clean metadata if required
        if SECURITY_CONFIG["strip_metadata"]:
            self.cleaned_path = clean_media_metadata(file_path)
            self.cap = cv2.VideoCapture(str(self.cleaned_path))
        else:
            self.cap = cv2.VideoCapture(str(file_path))
        
        # Log ingestion
        log_video_ingest(str(file_path), file_hash, self.user_id, self.session_id)
    
    def _configure_capture(self):
        """Configure video capture properties"""
        if not self.cap:
            return
        
        # Set buffer size for live sources
        if self.is_live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set target FPS if specified
        target_fps = VIDEO_CONFIG.get("fps_target")
        if target_fps and self.is_live:
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        
        # Set resolution limits
        max_width, max_height = VIDEO_CONFIG["max_resolution"]
        current_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        current_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if current_width > max_width or current_height > max_height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
    
    def start_capture(self):
        """Start threaded frame capture for live sources"""
        if self.is_live and not self._capture_thread:
            self._stop_event.clear()
            self._capture_thread = threading.Thread(target=self._capture_frames)
            self._capture_thread.daemon = True
            self._capture_thread.start()
    
    def _capture_frames(self):
        """Capture frames in separate thread"""
        while not self._stop_event.is_set() and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Add timestamp
                timestamp = time.time()
                
                # Add to queue (drop old frames if queue is full)
                try:
                    self._frame_queue.put_nowait((frame, timestamp))
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait((frame, timestamp))
                    except queue.Empty:
                        pass
            else:
                break
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get next frame from video source
        
        Returns:
            Tuple of (frame, timestamp) or None if no frame available
        """
        if self.is_live and self._capture_thread:
            # Get frame from queue
            try:
                return self._frame_queue.get_nowait()
            except queue.Empty:
                return None
        else:
            # Read directly for file sources
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    timestamp = time.time()
                    return frame, timestamp
        
        return None
    
    def get_frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generator that yields frames from video source
        
        Yields:
            Tuple of (frame, timestamp)
        """
        if self.is_live:
            self.start_capture()
        
        while True:
            frame_data = self.get_frame()
            if frame_data is None:
                if self.is_file:
                    # End of file
                    break
                else:
                    # Live source - wait a bit and try again
                    time.sleep(0.01)
                    continue
            
            yield frame_data
    
    def get_info(self) -> dict:
        """Get video source information"""
        if not self.cap:
            return {}
        
        info = {
            "source_type": "live" if self.is_live else "file",
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "session_id": self.session_id
        }
        
        if self.is_file:
            info["frame_count"] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if info["fps"] > 0:
                info["duration_seconds"] = info["frame_count"] / info["fps"]
        
        return info
    
    def seek(self, frame_number: int) -> bool:
        """
        Seek to specific frame (file sources only)
        
        Args:
            frame_number: Target frame number
            
        Returns:
            True if seek successful
        """
        if self.is_file and self.cap:
            return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return False
    
    def get_position(self) -> dict:
        """Get current position information"""
        if not self.cap:
            return {}
        
        position = {
            "frame_number": int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "timestamp_ms": self.cap.get(cv2.CAP_PROP_POS_MSEC)
        }
        
        return position
    
    def stop(self):
        """Stop video capture and cleanup"""
        # Stop capture thread
        if self._capture_thread:
            self._stop_event.set()
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        
        # Release video capture
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clean up temporary files
        if self.cleaned_path and self.cleaned_path.exists():
            try:
                self.cleaned_path.unlink()
            except Exception:
                pass
        
        # Clear frame queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_video_source(source: Union[str, int, Path], 
                       user_id: str = "UNKNOWN",
                       session_id: Optional[str] = None) -> VideoSource:
    """
    Factory function to create video source
    
    Args:
        source: Video source identifier
        user_id: User identifier
        session_id: Session identifier
        
    Returns:
        VideoSource instance
    """
    return VideoSource(source, user_id, session_id)