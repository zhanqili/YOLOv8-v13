# YOLOv8-v13

A benchmark project for comparing multiple YOLO variants on the **TT100K traffic sign dataset**, with a focus on detection accuracy, training behavior, and deployment efficiency.

## Overview

This repository is used to benchmark different YOLO versions under a unified experimental setting.  
It currently includes experiment code and weights for:

- YOLOv8
- YOLOv9
- YOLOv10
- YOLOv11
- YOLOv12
- YOLOv13
- YOLOv26

The project is mainly designed for **traffic sign detection** experiments on **TT100K**, and can be used to compare:

- mAP50
- mAP50-95
- Precision
- Recall
- Training/validation losses
- Inference latency
- FPS
- Model size / parameters / FLOPs

## Repository Structure

```bash
YOLOv8-v13/
├── YOLOv8/
│   ├── runs/detect/
│   ├── train.py
│   ├── yolov8n.pt
│   ├── yolov10n.pt
│   ├── yolo11n.pt
│   ├── yolo12n.pt
│   └── yolo26n.pt
├── YOLOv9/
│   ├── runs/train/exp/
│   ├── train.py
│   ├── results.py
│   ├── yolov9_e2e_metrics_no_thop.py
│   └── yolov9t.pt
├── YOLOv13/
│   ├── runs/detect/
│   ├── train.py
│   ├── results.py
│   └── yolov13n.pt
└── README.md