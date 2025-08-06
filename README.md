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

# GUI application
python gui_app.py                      # launch GUI
```

Model weights download on first run (~300 MB); subsequent runs are offline.

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