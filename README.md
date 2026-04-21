# Satellite Video Object Detection and Tracking

<video src="demo.mov" controls width="900"></video>

## Overview

This project builds an end-to-end computer vision pipeline for **detecting and tracking objects in satellite video imagery**. It combines a rotation-aware object detector with a lightweight multi-object tracker to identify and follow ships, aircraft, and vehicles across overhead image sequences.

The system was developed using public remote-sensing datasets and is designed to demonstrate techniques relevant to:

* satellite video analytics
  n- maritime monitoring
* airfield and traffic surveillance
* geospatial intelligence systems
* computer vision for aerospace applications

---

## Features

* **YOLOv8-OBB Detection** for oriented bounding boxes
* **Multi-object tracking** with persistent IDs across frames
* **Annotated video generation** with tracked trajectories
* **Frame-by-frame JSON outputs** for downstream analytics
* Works on **satellite image sequences** such as VISO benchmark data

---

## Demo

The video above shows tracked ships in overhead satellite imagery with persistent object IDs.

---

## Model Pipeline

```text
Satellite Frames
      ↓
YOLOv8 Oriented Detection
      ↓
Association + Motion Tracking
      ↓
Tracked IDs + Rendered Frames
      ↓
Output Video + JSON Tracks
```

---

## Datasets Used

### DOTA

Used to train the oriented object detector.

Classes include:

* plane
* ship
* helicopter
* large vehicle
* small vehicle

### VISO

Used for satellite video tracking experiments.

---

## Example Usage

```bash
python scripts/tracking_pipeline.py \
  --model runs/obb/train8/weights/best.pt \
  --frames-dir data/VISO/mot/ship/045/img \
  --output-frames-dir runs/obb/viso_ship_tracks \
  --output-json runs/obb/viso_ship_tracks.json \
  --output-video demo.mov \
  --fps 10 \
  --imgsz 640 \
  --conf 0.15 \
  --max-distance 100 \
  --max-missed 15
```

---

## Results

* Stable tracking of multiple ships across image sequences
* Persistent IDs under moderate motion
* Robust oriented detections on overhead imagery
* Exportable outputs for evaluation and visualization

---

## Future Improvements

* Hungarian matching + IoU association
* Kalman filter motion model
* Fine-tuning directly on VISO frames
* Transformer-based detector comparison (DETR)
* Real-time streaming deployment

---

## Tech Stack

* Python
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* NumPy
