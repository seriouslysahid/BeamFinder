# DECISIONS.md — Architecture Decision Records

> **Project**: BeamFinder

## ADR Log

| ID | Date | Decision | Rationale | Status |
|----|------|----------|-----------|--------|
| ADR-001 | 2026-02-10 | Use YOLO26n (nano) variant | CPU-optimized, 43% faster on CPU, sufficient for academic prototype | Accepted |
| ADR-002 | 2026-02-10 | Use pretrained COCO model (no fine-tuning) | Simplifies Phase 1 scope; fine-tuning deferred to future milestone | Accepted |
| ADR-003 | 2026-02-10 | Output to CSV format | Simple, portable, easy to parse; can change to socket/API later | Accepted |
| ADR-004 | 2026-02-10 | Python + Ultralytics stack | Official YOLO26 support, academic standard, CPU-compatible | Accepted |
