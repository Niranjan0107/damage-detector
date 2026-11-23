"""
Damage Detector Module for Mining Vehicles
Detects dents, cracks, scratches, flat tires, and structural deformations on vehicle bodies.
Uses fine-tuned deep learning YOLO damage models with graceful fallback.
"""
from typing import List, Dict, Any, Tuple, Optional
import os
import cv2
import numpy as np


class DamageDetector:
    CLASS_DISPLAY_NAMES = {
        "dent": "Dent / Body Impact",
        "dents": "Dent / Body Impact",
        "scratch": "Deep Scratch / Gouge",
        "scratches": "Deep Scratch / Gouge",
        "crack": "Structural Crack / Fracture",
        "glass shatter": "Broken Glass / Windshield",
        "tire flat": "Flat / Damaged Tire",
        "lamp broken": "Broken Headlamp / Indicator",
        "front-bumper-dent": "Front Bumper Dent",
        "doorouter-dent": "Door Panel Dent",
        "fender-dent": "Fender / Mudguard Dent",
    }

    def __init__(
        self,
        custom_model_path: Optional[str] = None,
        severity_threshold: float = 0.20,
    ):
        """
        Initialize damage detector.
        """
        self.severity_threshold = severity_threshold
        self.model = None

        # Look for pre-trained weights
        candidates = [
            custom_model_path,
            "models/damage_model.pt",
            "models/v1.pt",
            "models/akila_yolo11.pt",
            "models/shawn_damage.pt",
        ]

        for path in candidates:
            if path and os.path.exists(path):
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(path)
                    print(f"[DamageDetector] Successfully loaded Deep Learning Damage Model: {path}")
                    print(f"[DamageDetector] Registered Classes: {self.model.names}")
                    break
                except Exception as e:
                    print(f"[DamageDetector] Could not load {path}: {e}")

        # Auto-download damage weights if missing
        if self.model is None:
            default_target = "models/damage_model.pt"
            hf_url = "https://huggingface.co/scythe410/vehicle-damage-detection-yolo/resolve/main/v1.pt"
            try:
                os.makedirs("models", exist_ok=True)
                print(f"[DamageDetector] Downloading fine-tuned damage weights from HuggingFace...")
                import urllib.request
                urllib.request.urlretrieve(hf_url, default_target)
                from ultralytics import YOLO
                self.model = YOLO(default_target)
                print(f"[DamageDetector] Successfully downloaded and loaded {default_target}")
            except Exception as e:
                print(f"[DamageDetector] Auto-download failed ({e}); using built-in CV surface defect engine.")

    def detect_damage_in_crop(
        self,
        vehicle_crop: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
    ) -> List[Dict[str, Any]]:
        """
        Detect damage inside a cropped vehicle image.
        Returns a list of damage items with relative and absolute coordinates.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        h, w = vehicle_crop.shape[:2]
        if h < 50 or w < 50:
            return []

        # 1. Primary: Fine-tuned Deep Learning Damage YOLO Model
        if self.model is not None:
            return self._detect_with_yolo(vehicle_crop, vehicle_bbox)

        # 2. Fallback: Localized Anomaly Engine
        return self._detect_with_cv_engine(vehicle_crop, vehicle_bbox)

    def _detect_with_yolo(
        self,
        vehicle_crop: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
    ) -> List[Dict[str, Any]]:
        damages = []
        vx1, vy1, _, _ = vehicle_bbox
        vh, vw = vehicle_crop.shape[:2]

        results = self.model.predict(
            source=vehicle_crop,
            conf=self.severity_threshold,
            verbose=False,
        )

        if not results or len(results) == 0:
            return damages

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return damages

        for box in boxes:
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            dx1, dy1, dx2, dy2 = xyxy

            # Clip within vehicle crop
            dx1 = max(0, min(dx1, vw - 1))
            dy1 = max(0, min(dy1, vh - 1))
            dx2 = max(dx1 + 1, min(dx2, vw))
            dy2 = max(dy1 + 1, min(dy2, vh))

            # Filter out boxes that span the entire vehicle (false global box)
            bw = dx2 - dx1
            bh = dy2 - dy1
            if bw > (vw * 0.90) and bh > (vh * 0.90):
                continue

            raw_name = results[0].names.get(cls_id, "Damage")
            if raw_name.lower() in ["none", "background"]:
                continue

            dmg_type = self.CLASS_DISPLAY_NAMES.get(raw_name.lower(), raw_name.title())

            # Map to absolute coordinates on original CCTV frame
            ax1 = vx1 + dx1
            ay1 = vy1 + dy1
            ax2 = vx1 + dx2
            ay2 = vy1 + dy2

            score = round(conf * 100, 1)
            severity_label = self._get_severity_label(conf)

            damages.append({
                "type": dmg_type,
                "severity_score": score,
                "severity_label": severity_label,
                "local_bbox": (int(dx1), int(dy1), int(dx2), int(dy2)),
                "abs_bbox": (int(ax1), int(ay1), int(ax2), int(ay2)),
            })

        return damages

    def _detect_with_cv_engine(
        self,
        vehicle_crop: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
    ) -> List[Dict[str, Any]]:
        """
        Improved localized defect finder for vehicle sheet metal:
        - Excludes straight manufacturing seams (horizontal/vertical lines)
        - Detects localized 3D surface depressions and irregular jagged cracks
        """
        damages = []
        vh, vw = vehicle_crop.shape[:2]
        vx1, vy1, _, _ = vehicle_bbox

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

        # Restrict inspection to vehicle body panels (18% to 78% height, 5% to 95% width)
        mask = np.zeros((vh, vw), dtype=np.uint8)
        mask[int(vh * 0.18) : int(vh * 0.78), int(vw * 0.05) : int(vw * 0.95)] = 255

        # Suppress straight panel seams (door lines, bed ribs)
        edges = cv2.Canny(gray, 60, 160)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
        seams = cv2.dilate(cv2.bitwise_or(h_lines, v_lines), np.ones((5, 5), np.uint8), iterations=2)

        # Detect localized surface depression (Dent) via Black Top-Hat
        kernel_dent = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        black_tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_dent)
        black_tophat = cv2.bitwise_and(black_tophat, black_tophat, mask=mask)
        black_tophat[seams > 0] = 0

        _, dent_thresh = cv2.threshold(black_tophat, 40, 255, cv2.THRESH_BINARY)
        dent_thresh = cv2.morphologyEx(dent_thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        cnts, _ = cv2.findContours(dent_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if 200 < area < (vw * vh * 0.25):
                dx, dy, dw, dh = cv2.boundingRect(c)
                # Ensure it's not a thin slit
                if dw > 15 and dh > 15:
                    score = min(95.0, max(30.0, 45.0 + (area / 120.0)))
                    if (score / 100.0) >= self.severity_threshold:
                        damages.append({
                            "type": "Dent / Body Impact",
                            "severity_score": round(score, 1),
                            "severity_label": self._get_severity_label(score / 100.0),
                            "local_bbox": (dx, dy, dx + dw, dy + dh),
                            "abs_bbox": (vx1 + dx, vy1 + dy, vx1 + dx + dw, vy1 + dy + dh),
                        })

        return damages

    @staticmethod
    def _get_severity_label(confidence: float) -> str:
        if confidence >= 0.70:
            return "CRITICAL"
        elif confidence >= 0.45:
            return "MAJOR"
        else:
            return "MINOR"
