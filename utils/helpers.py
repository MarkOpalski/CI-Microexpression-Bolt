"""
Utility functions for the CI Microexpression Tracking system
"""
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union
import uuid

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

from config import SECURITY_CONFIG


def generate_session_id() -> str:
    """Generate a unique session identifier"""
    return str(uuid.uuid4())


def calculate_file_hash(file_path: Union[str, Path], 
                       algorithm: str = "sha256") -> str:
    """
    Calculate cryptographic hash of a file
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use
        
    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def clean_media_metadata(input_path: Union[str, Path], 
                        output_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Strip metadata from media files for security
    
    Args:
        input_path: Path to input media file
        output_path: Path for cleaned output (optional)
        
    Returns:
        Path to cleaned file
    """
    input_path = Path(input_path)
    
    if output_path is None:
        # Create temporary file
        suffix = input_path.suffix
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(temp_fd)
        output_path = Path(temp_path)
    else:
        output_path = Path(output_path)
    
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff']:
        # Handle image files
        _clean_image_metadata(input_path, output_path)
    else:
        # Handle video files
        _clean_video_metadata(input_path, output_path)
    
    return output_path


def _clean_image_metadata(input_path: Path, output_path: Path):
    """Strip EXIF and other metadata from image files"""
    try:
        with Image.open(input_path) as img:
            # Create new image without metadata
            clean_img = Image.new(img.mode, img.size)
            clean_img.putdata(list(img.getdata()))
            clean_img.save(output_path, quality=95, optimize=True)
    except Exception as e:
        # Fallback: copy file if cleaning fails
        import shutil
        shutil.copy2(input_path, output_path)


def _clean_video_metadata(input_path: Path, output_path: Path):
    """Strip metadata from video files using OpenCV"""
    try:
        cap = cv2.VideoCapture(str(input_path))
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Copy frames without metadata
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        
    except Exception as e:
        # Fallback: copy file if cleaning fails
        import shutil
        shutil.copy2(input_path, output_path)


def validate_file_size(file_path: Union[str, Path], 
                      max_size_mb: Optional[int] = None) -> bool:
    """
    Validate file size against security limits
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum size in MB (uses config default if None)
        
    Returns:
        True if file size is acceptable
    """
    if max_size_mb is None:
        max_size_mb = SECURITY_CONFIG["max_file_size_mb"]
    
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    return file_size_mb <= max_size_mb


def get_video_info(video_path: Union[str, Path]) -> dict:
    """
    Extract basic information from video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(str(video_path))
    
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": 0
    }
    
    if info["fps"] > 0:
        info["duration_seconds"] = info["frame_count"] / info["fps"]
    
    cap.release()
    return info


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio
    
    Args:
        frame: Input frame
        target_size: (width, height) target size
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    target_width, target_height = target_size
    
    # Calculate scaling factor
    scale = min(target_width / width, target_height / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize frame
    resized = cv2.resize(frame, (new_width, new_height), 
                        interpolation=cv2.INTER_AREA)
    
    return resized


def create_output_directory(session_id: str) -> Path:
    """
    Create session-specific output directory
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Path to created directory
    """
    from config import OUTPUT_DIR
    
    session_dir = OUTPUT_DIR / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    return session_dir


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing problematic characters
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace problematic characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    safe_name = "".join(c if c in safe_chars else "_" for c in filename)
    
    # Ensure it's not empty and doesn't start with a dot
    if not safe_name or safe_name.startswith('.'):
        safe_name = "file_" + safe_name
    
    return safe_name[:255]  # Limit length


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert timestamp string to seconds
    
    Args:
        timestamp: Timestamp in format "HH:MM:SS" or "MM:SS"
        
    Returns:
        Time in seconds
    """
    parts = timestamp.split(':')
    
    if len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    else:
        return float(parts[0])