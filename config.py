"""
Configuration settings for CI Microexpression Tracking System
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Video processing settings
VIDEO_CONFIG = {
    "fps_target": 30,
    "max_resolution": (1920, 1080),
    "face_detection_confidence": 0.9,
    "face_min_size": 40,
    "supported_formats": [".mp4", ".avi", ".mkv", ".mov", ".wmv"]
}

# Analysis settings
ANALYSIS_CONFIG = {
    "emotion_threshold": 0.7,
    "baseline_window_seconds": 30,
    "spike_detection_factor": 2.0,
    "enable_facs_au": False,  # Set to True to enable FACS Action Units
    "whisper_model": "base",  # Options: tiny, base, small, medium, large
    "chunk_duration": 30  # seconds for processing chunks
}

# Security settings
SECURITY_CONFIG = {
    "strip_metadata": True,
    "audit_log_path": OUTPUT_DIR / "audit.log",
    "hash_algorithm": "sha256",
    "air_gapped_mode": True,  # Disable external connections
    "max_file_size_mb": 500
}

# Reporting settings
REPORT_CONFIG = {
    "pdf_template": "standard",
    "include_timeline": True,
    "include_statistics": True,
    "confidence_threshold": 0.6,
    "max_triggers_display": 10
}

# UI settings
UI_CONFIG = {
    "theme": "plotly_dark",
    "port": 8050,
    "debug": False,
    "host": "127.0.0.1"  # Localhost only for security
}