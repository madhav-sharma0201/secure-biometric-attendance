# Secure Biometric Attendance System

A trained CNN-LSTM face anti-spoofing model integrated with ArcFace identity
verification, deployed as containerised services.

**Status:** in development. Phase 0 (architecture) and Phase 1 (repo setup) complete.

> No performance numbers appear in this README until they are produced by a logged
> experiment. Placeholders are marked `TBD` on purpose.

## Problem

Attendance systems based on face recognition alone are trivially defeated: hold up a
photo of a classmate, or replay a video of them on a phone. The ML problem here is not
"who is this person" — pretrained models solve that well. It is **"is this a real
person in front of the camera, or a presentation attack?"**

## Documents

- [Phase 0 — Requirements & Architecture](docs/00-architecture.md)

## Layout

```
data/        datasets (gitignored)
ml/          preprocessing, training, evaluation, inference
backend/     FastAPI service, decision engine, persistence
frontend/    camera capture UI
configs/     training + runtime configuration
models/      trained weights (gitignored)
kubernetes/  deployment manifests
docs/        architecture and reports
```

## Setup

TBD — see Phase 1 notes.
