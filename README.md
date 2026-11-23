# 🚛 MineFleet AI - Mining Vehicle & Damage Detection System

Automated AI-powered computer vision system designed for **Mine CCTV footage analysis**.
It automatically identifies **Trucks, Dumpers, Containers & Mining Machinery**, tracks them across CCTV video frames, localizes **body damages** (Dents, Cracks, Tears, Gouges), and generates downloadable **PDF / CSV Inspection Reports**.

---

## 🌟 Key Features

1. **Two-Stage AI Vision Pipeline**:
   - **Stage 1 (Vehicle Detection & Tracking):** Ultralytics YOLOv8 / YOLOv11 with ByteTrack tracker assigns unique track IDs to each truck, dumper, and container.
   - **Stage 2 (Damage & Surface Defect Localizer):** Detects impact craters/dents, structural weld/panel cracks, deep scrapes, and body tears.
2. **CCTV Stream & Video Upload**:
   - Accepts `.mp4`, `.avi`, `.mov` footage directly from CCTV NVRs, gate cameras, or scale cameras.
3. **Interactive Web Dashboard (Streamlit)**:
   - Real-time frame processing progress & preview.
   - Side-by-side annotated video playback with HUD overlay (bounding boxes + damage alerts).
   - Defect gallery with zoom-in photo snapshots of damaged vehicles.
4. **Automated Inspection Reports**:
   - **PDF Report:** Executive summary, fleet defect percentage, defect table, and embedded high-resolution snapshots.
   - **CSV Export:** For ERP / Fleet maintenance integration.
5. **Sample Video Generator**:
   - Built-in synthetic mining CCTV video generator to test the entire system instantly.

---

## 📂 Project Structure

```text
damage-detector/
├── app.py                      # Streamlit Web UI Dashboard
├── generate_sample_video.py    # Synthetic Mining CCTV Video Generator
├── detector/
│   ├── vehicle_tracker.py      # YOLOv8 + ByteTrack vehicle tracking
│   ├── damage_detector.py      # Multi-scale surface anomaly & damage engine
│   └── video_processor.py      # Video frame pipeline, HUD & annotations
├── utils/
│   └── report_generator.py     # ReportLab PDF & CSV report exporter
├── models/                     # Custom trained weights (.pt)
├── samples/                    # Sample CCTV footage
├── data/
│   ├── uploads/                # Client uploaded videos
│   ├── outputs/                # Processed annotated videos & snapshots
│   └── reports/                # Generated PDF/CSV inspection reports
└── requirements.txt
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate a Test Mining CCTV Video (Optional)
```bash
python generate_sample_video.py
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ How to Add a Custom-Trained Damage Model
If you train a dedicated YOLO model for vehicle dents and cracks (e.g. from Roboflow or a private mine fleet dataset):
1. Save your weights as `models/damage_yolov8.pt`.
2. The `DamageDetector` will automatically load and prioritize it over the built-in CV anomaly engine.
