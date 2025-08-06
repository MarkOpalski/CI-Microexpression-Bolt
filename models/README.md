# Model Weights Directory

This directory stores downloaded model weights and cached files for the CI Microexpression Tracking system.

## Contents

- **MTCNN weights**: Face detection model weights (downloaded automatically)
- **DeepFace models**: Emotion recognition model weights (downloaded on first use)
- **Whisper models**: Speech recognition model weights (downloaded based on config)
- **Py-Feat models**: FACS Action Unit models (optional, if enabled)

## Storage Requirements

- Base models: ~300MB
- Full model suite: ~1GB
- Whisper large model: ~3GB (if selected)

## Security Notes

- All models are downloaded from official repositories
- No custom or modified weights are used
- Models are cached locally for offline operation
- Directory is excluded from version control (.gitignore)

## Model Sources

- **MTCNN**: facenet-pytorch package
- **DeepFace**: Official DeepFace repository
- **Whisper**: OpenAI official models
- **Py-Feat**: USC ICT Py-Feat package

## Offline Operation

Once models are downloaded, the system operates entirely offline with no external network dependencies, ensuring operational security for classified environments.