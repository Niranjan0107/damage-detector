"""
Vehicle Tracker Module for Mining Fleet
Detects and tracks trucks, dumpers, and containers across CCTV video frames.
"""
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from ultralytics import YOLO


class VehicleTracker:
    # Strictly Target COCO Class 7: Truck (Mining Dumpers, Haul Trucks, Tippers, Container Trucks)
    TARGET_CLASSES = {
        7: "Mining Dumper / Truck",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        target_classes: Optional[List[int]] = None,
        min_box_width: int = 90,
        min_box_height: int = 70,
    ):
        """
        Initialize YOLO-based Vehicle Tracker strictly for Trucks, Dumpers, and Containers.
        """
        self.confidence_threshold = confidence_threshold
        # Strictly Class 7 (truck) only - Ignore cars, carts, pickups, buses
        self.target_classes = [7] if target_classes is None else [c for c in target_classes if c == 7]
        self.min_box_width = min_box_width
        self.min_box_height = min_box_height
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # In-memory registry for tracked vehicles:
        # {track_id: {"class_name": str, "best_crop": np.ndarray, "best_conf": float, ...}}
        self.vehicles_registry: Dict[int, Dict[str, Any]] = {}

    def track_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Process a single frame and return tracked vehicles.
        """
        h, w = frame.shape[:2]
        detected_vehicles = []

        # Run tracking with ByteTrack
        results = self.model.track(
            source=frame,
            persist=True,
            classes=self.target_classes,
            conf=self.confidence_threshold,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        if not results or len(results) == 0:
            return detected_vehicles

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detected_vehicles

        # Iterate over detected bounding boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            # Strictly filter: Only COCO class 7 (Truck / Dumper / Container)
            if cls_id != 7:
                continue

            conf = float(box.conf[0].item())
            
            # Track ID may be None in rare single-frame glitches
            if box.id is not None:
                track_id = int(box.id[0].item())
            else:
                track_id = -1

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # Clip within frame boundaries
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            box_w = x2 - x1
            box_h = y2 - y1

            # Strictly ignore tiny distant background vehicles / passenger cars
            if box_w < self.min_box_width or box_h < self.min_box_height:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            
            # Refine mining classification based on aspect ratio & size
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h
            frame_area = w * h

            if (box_area / frame_area) > 0.08:
                class_name = "Heavy Mining Dumper"
            elif box_w > (box_h * 1.5):
                class_name = "Mining Container Truck"
            else:
                class_name = "Mining Tipper / Truck"

            vehicle_info = {
                "track_id": track_id,
                "class_name": class_name,
                "cls_id": cls_id,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "crop": crop,
                "frame_idx": frame_idx,
            }
            detected_vehicles.append(vehicle_info)

            # Update in global registry if track_id > 0
            if track_id > 0:
                if track_id not in self.vehicles_registry:
                    self.vehicles_registry[track_id] = {
                        "track_id": track_id,
                        "class_name": class_name,
                        "first_seen": frame_idx,
                        "last_seen": frame_idx,
                        "best_conf": conf,
                        "best_crop": crop,
                        "best_bbox": (x1, y1, x2, y2),
                        "damage_detections": [],
                        "max_severity": 0.0,
                    }
                else:
                    reg = self.vehicles_registry[track_id]
                    reg["last_seen"] = frame_idx
                    # Keep crop with highest confidence & good size
                    if conf > reg["best_conf"] and crop.size > 0:
                        reg["best_conf"] = conf
                        reg["best_crop"] = crop
                        reg["best_bbox"] = (x1, y1, x2, y2)

        return detected_vehicles

    def get_summary(self) -> List[Dict[str, Any]]:
        """Return list of all unique vehicles tracked during the video."""
        return list(self.vehicles_registry.values())
