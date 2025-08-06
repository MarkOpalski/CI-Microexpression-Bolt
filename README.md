# CI Microexpression Tracking

## Purpose
Analytic framework that fuses facial micro-expression detection with speech transcription to surface non-verbal cues of deception, stress, or concealed intent for CI analysts.

## Core Modules
| Module | Description |
| ------ | ----------- |
| video/ingestion.py | Unified interface for live streams and uploaded files |
| video/face_detection.py | High-accuracy MTCNN face detection & cropping |
| analysis/expression_analysis.py | Emotion inference (DeepFace) ± FACS AU mapping (Py-Feat) |
| analysis/transcript_alignment.py | Align Whisper transcript to emotion timeline |
| reporting/visualizations.py | Interactive Plotly dashboards |
| reporting/report_generator.py | ReportLab PDF (meta, timeline, triggers) |
| security/audit_log.py | Append-only chain-of-custody log |

## Security Model
* Air-gapped deployment recommended for classified data  
* All computation local; no external API calls unless enabled  
* Metadata stripping on ingest (`utils/helpers.py:clean_media()`)  
* Tamper-evident audit log at `output/audit.log`

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Command-line interface
python app.py --source 0              # webcam
python app.py --source ./sample.mp4   # recorded file

# GUI applications
python gui_app.py                      # launch GUI directly
python launch_gui.py                   # launch with error checking
```

Model weights download on first run (~300 MB); subsequent runs are offline.

## User Interfaces

### Command Line Interface (CLI)
```bash
# Basic analysis
python app.py --source video.mp4 --user analyst1

# Webcam analysis
python app.py --source 0 --user analyst2

# System verification
python app.py --verify-audit
python app.py --setup-git-hooks
```

### Graphical User Interface (GUI)
```bash
# Launch GUI with error checking
python launch_gui.py

# Direct launch
python gui_app.py
```

**GUI Features:**
- Real-time video analysis with live preview
- Professional tabbed interface (Analysis/Results/Settings)
- Accessibility support with keyboard navigation
- Integrated audit log verification
- Session management and results browsing
- ARIA-compliant landmark regions for screen readers

**Keyboard Shortcuts:**
- `Ctrl+N`: New Analysis
- `Ctrl+O`: Load Video File
- `Ctrl+S`: Start Analysis
- `Ctrl+Q`: Quit Application
- `F1`: Show Help

## Roadmap
* Subject baseline profiling
* GPU acceleration toggle
* Third-party sensor fusion (thermal, PPG)
* Encrypted output packages

---

#### Next Implementation Tasks  

1. Implement `security/audit_log.py`  
2. Build `analysis/expression_analysis.py`  
3. Wire modules in `app.py` and test ingest → detect → analyze → report in a sandboxed Bolt.new container