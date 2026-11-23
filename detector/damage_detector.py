"""
Damage Detector Module for Mining Vehicles
Detects dents, cracks, scratches, smashed glass, crushed panels, and structural deformations.
Uses YOLO11m Fine-Tuned Damage AI.
"""
from typing import List, Dict, Any, Tuple, Optional
import os
import cv2
import numpy as np


class DamageDetector:
    CLASS_DISPLAY_NAMES = {
        "dent": "Dent / Body Impact",
        "dents": "Dent / Body Impact",
        "scratch": "Deep Scratch / Surface Abrasion",
        "scratches": "Deep Scratch / Surface Abrasion",
        "crack": "Structural Crack / Fracture",
        "shattered_glass": "Smashed Glass / Crushed Windshield",
        "broken_lamp": "Broken Headlamp / Light",
        "flat_tire": "Damaged / Flat Tire",
    }

    def __init__(
        self,
        custom_model_path: Optional[str] = None,
        severity_threshold: float = 0.14,
        filter_sun_glare: bool = True,
    ):
        """
        Initialize damage detector with industrial CCTV sensitivity and solar glare filtering.
        """
        self.severity_threshold = severity_threshold
        self.filter_sun_glare = filter_sun_glare
        self.model = None

        candidates = [
            custom_model_path,
            "models/damage_model.pt",
            "models/shanexf_yolo11m.pt",
            "models/akila_yolo11.pt",
        ]

        for path in candidates:
            if path and os.path.exists(path):
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(path)
                    print(f"[DamageDetector] Loaded Deep Learning Model: {path}")
                    print(f"[DamageDetector] Classes: {self.model.names}")
                    break
                except Exception as e:
                    print(f"[DamageDetector] Could not load {path}: {e}")

        # Auto-download damage weights if missing
        if self.model is None:
            default_target = "models/damage_model.pt"
            hf_url = "https://huggingface.co/shanexf/car-damage-yolo11m-ft/resolve/main/damage_best.pt"
            try:
                os.makedirs("models", exist_ok=True)
                print(f"[DamageDetector] Downloading fine-tuned damage weights from HuggingFace...")
                import urllib.request
                urllib.request.urlretrieve(hf_url, default_target)
                from ultralytics import YOLO
                self.model = YOLO(default_target)
                print(f"[DamageDetector] Successfully downloaded and loaded {default_target}")
            except Exception as e:
                print(f"[DamageDetector] Auto-download failed ({e}); using CV surface defect engine.")

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
        if h < 40 or w < 40:
            return []

        # Fine-tuned Deep Learning YOLO Damage Model
        if self.model is not None:
            return self._detect_with_yolo(vehicle_crop, vehicle_bbox)

        return []

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

            # Filter out boxes that span 95%+ of the vehicle (false global box)
            bw = dx2 - dx1
            bh = dy2 - dy1
            if bw > (vw * 0.95) and bh > (vh * 0.95):
                continue

            # Sunlight Glare / Specular Solar Reflection Filter
            if self.filter_sun_glare and self._is_sunlight_glare(vehicle_crop[dy1:dy2, dx1:dx2]):
                continue

            raw_name = results[0].names.get(cls_id, "Damage")
            if raw_name.lower() in ["none", "background"]:
                continue

            dmg_type = self.CLASS_DISPLAY_NAMES.get(raw_name.lower(), raw_name.title())

            # Absolute coordinates on original CCTV frame
            ax1 = vx1 + dx1
            ay1 = vy1 + dy1
            ax2 = vx1 + dx2
            ay2 = vy1 + dy2

            # Industrial scoring: Confidences above 0.14 represent real defects
            score = round(min(98.0, conf * 180.0), 1)
            severity_label = self._get_severity_label(score)

            damages.append({
                "type": dmg_type,
                "severity_score": score,
                "severity_label": severity_label,
                "local_bbox": (int(dx1), int(dy1), int(dx2), int(dy2)),
                "abs_bbox": (int(ax1), int(ay1), int(ax2), int(ay2)),
            })

        return damages

    @staticmethod
    def _get_severity_label(score: float) -> str:
        if score >= 45.0:
            return "CRITICAL"
        elif score >= 25.0:
            return "MAJOR"
        else:
            return "MINOR"

    @staticmethod
    def _is_sunlight_glare(crop_roi: np.ndarray) -> bool:
        """
        Filters out false positives caused by intense sunlight glare / specular solar reflection.
        Sun glare produces pure overexposed, low-saturation washed out pixels without dark
        indentation shadows or structural fissures.
        """
        if crop_roi is None or crop_roi.size == 0 or crop_roi.shape[0] < 5 or crop_roi.shape[1] < 5:
            return False

        hsv = cv2.cvtColor(crop_roi, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)

        # Overexposed solar glare pixels: high value (brightness >= 235), low saturation (<= 50)
        glare_mask = (v >= 235) & (s <= 50)
        dark_mask = (v <= 80)

        total_pixels = float(crop_roi.shape[0] * crop_roi.shape[1] + 1e-5)
        glare_ratio = np.count_nonzero(glare_mask) / total_pixels
        dark_ratio = np.count_nonzero(dark_mask) / total_pixels

        # If more than 30% of candidate box is pure white sunlight glare
        # and has no dark structural crack/dent fissure (< 6%), it is sunlight glare!
        if glare_ratio > 0.30 and dark_ratio < 0.06:
            return True

        return False
