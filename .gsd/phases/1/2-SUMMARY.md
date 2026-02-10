# Summary: Plan 1.2 — Configuration Module & README

**Status**: ✅ Complete
**Commit**: `50f61bc`

## What Was Done
- Created `src/config.py` with centralized settings:
  - Path resolution (PROJECT_ROOT, IMAGE_DIR, OUTPUT_DIR, OUTPUT_CSV)
  - Model settings (yolo26n.pt, confidence 0.25, image size 640)
  - Supported image extensions
- Created `README.md` with project overview, setup instructions, usage guide, project structure, and config reference

## Verification
Config module verified with Python import:
```
Root: D:\Desktop\TTV
Model: yolo26n.pt
Conf: 0.25
```
