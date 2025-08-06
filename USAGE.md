# CI Microexpression Tracking System - Usage Guide

## Overview
The CI Microexpression Tracking System provides both command-line and graphical interfaces for analyzing facial micro-expressions and detecting behavioral inconsistencies in video content.

## Getting Started

### 1. Installation
```bash
# Clone and setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. First Run
```bash
# Test system components
python test_demo.py

# Verify audit system
python app.py --verify-audit
```

## Command Line Interface (CLI)

### Basic Usage
```bash
# Analyze video file
python app.py --source video.mp4 --user analyst1

# Use webcam (device 0)
python app.py --source 0 --user analyst2

# Analyze RTSP stream
python app.py --source rtsp://camera/stream --user analyst3
```

### Advanced Options
```bash
# System verification
python app.py --verify-audit

# Setup development hooks (optional)
python app.py --setup-git-hooks
```

## Graphical User Interface (GUI)

### Launching the GUI
```bash
# Recommended: Launch with error checking
python launch_gui.py

# Direct launch
python gui_app.py
```

### GUI Workflow

#### 1. Analysis Tab
- **Set Analyst ID**: Enter your analyst identifier
- **Select Source**: Choose webcam or video file
- **Start Analysis**: Begin real-time processing
- **Monitor Progress**: Watch live statistics and progress

#### 2. Results Tab
- **Analysis Summary**: High-level findings and risk assessment
- **Detailed Results**: Frame-by-frame emotion data and speech segments
- **Export Options**: Generate reports and save results

#### 3. Settings Tab
- **System Configuration**: View current analysis parameters
- **Security Settings**: Verify air-gapped mode and audit settings
- **Model Information**: Check loaded AI models

### Accessibility Features

#### Keyboard Navigation
- **Tab**: Navigate between controls
- **Arrow Keys**: Navigate lists and trees
- **Space/Enter**: Activate buttons and selections

#### Keyboard Shortcuts
- **Ctrl+N**: New Analysis Session
- **Ctrl+O**: Load Video File
- **Ctrl+S**: Start Analysis
- **Ctrl+Q**: Quit Application
- **F1**: Show Help Dialog

#### Screen Reader Support
- All controls have proper labels
- ARIA landmark regions for navigation
- Status announcements for important events

## Analysis Workflow

### 1. Preparation
- Ensure video quality is adequate (minimum 480p recommended)
- Verify subject's face is clearly visible
- Check lighting conditions for optimal detection

### 2. Analysis Process
1. **Video Ingestion**: System loads and validates video source
2. **Face Detection**: MTCNN identifies and tracks faces
3. **Expression Analysis**: DeepFace analyzes micro-expressions
4. **Speech Processing**: Whisper transcribes audio content
5. **Alignment**: System correlates speech with facial expressions
6. **Mismatch Detection**: Identifies behavioral inconsistencies

### 3. Results Interpretation

#### Emotion Categories
- **Anger**: Furrowed brows, tightened lips
- **Disgust**: Wrinkled nose, raised upper lip
- **Fear**: Widened eyes, raised eyebrows
- **Happiness**: Smile, raised cheeks
- **Sadness**: Drooped eyelids, downturned mouth
- **Surprise**: Raised eyebrows, dropped jaw
- **Neutral**: Baseline emotional state

#### Risk Indicators
- **High Severity (>70%)**: Strong behavioral inconsistencies
- **Medium Severity (40-70%)**: Moderate inconsistencies
- **Low Severity (<40%)**: Minor or uncertain indicators

#### Report Sections
- **Executive Summary**: Key findings and risk assessment
- **Emotion Timeline**: Chronological emotional states
- **Speech Analysis**: Transcribed content with sentiment
- **Behavioral Inconsistencies**: Detected mismatches
- **Technical Details**: Processing metadata and chain of custody

## Security Considerations

### Air-Gapped Operation
- All processing occurs locally
- No external network connections required
- Model weights cached after first download

### Audit Trail
- Complete chain of custody logging
- Tamper-evident audit records
- Session isolation and tracking

### Data Protection
- Automatic metadata stripping
- Secure temporary file handling
- Session-based output organization

## Troubleshooting

### Common Issues

#### "No faces detected"
- Ensure adequate lighting
- Check camera angle and distance
- Verify face is clearly visible and unobstructed

#### "Audio extraction failed"
- Install ffmpeg for better audio processing
- Check video file format compatibility
- Verify audio track exists in video

#### "Model download failed"
- Check internet connection for first run
- Verify sufficient disk space (~1GB)
- Clear models directory and retry

#### GUI doesn't start
- Verify tkinter installation: `python -c "import tkinter"`
- Check Python version (3.8+ required)
- Run `python launch_gui.py` for detailed error messages

### Performance Optimization

#### For Better Accuracy
- Use high-quality video (720p or higher)
- Ensure good lighting conditions
- Minimize camera movement
- Keep subject facing camera

#### For Faster Processing
- Reduce video resolution if needed
- Use shorter video segments for testing
- Close other applications to free memory
- Consider GPU acceleration if available

## Support and Documentation

### Log Files
- **Audit Log**: `output/audit.log`
- **Session Data**: `output/session_[ID]/`
- **Reports**: `output/session_[ID]/reports/`

### Configuration
- **System Settings**: `config.py`
- **Model Cache**: `models/`
- **Output Directory**: `output/`

### Getting Help
- Run `python app.py --help` for CLI options
- Press `F1` in GUI for help dialog
- Check `README.md` for technical details
- Review audit logs for processing history

## Operational Guidelines

### Best Practices
1. **Always set proper analyst ID** for audit trail
2. **Verify system integrity** before sensitive operations
3. **Review results manually** - automated analysis requires human validation
4. **Maintain chain of custody** through proper session management
5. **Regular audit verification** to ensure system integrity

### Limitations
- Cultural variations may affect accuracy
- Environmental factors impact detection quality
- Results require human analyst validation
- System trained primarily on Western facial expressions
- False positive rate: ~5-10% for high-confidence detections

### Operational Security
- Use air-gapped systems for classified material
- Verify audit log integrity regularly
- Maintain proper access controls
- Follow organizational data handling procedures