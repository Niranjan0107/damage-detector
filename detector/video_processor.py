"""
Video Processor Module
Orchestrates vehicle detection, tracking, damage analysis, H.264 video encoding, and event logging.
"""
from typing import Callable, Optional, Dict, Any, List
import os
import time
import cv2
import numpy as np
import av

from .vehicle_tracker import VehicleTracker
from .damage_detector import DamageDetector


class VideoProcessor:
    def __init__(
        self,
        vehicle_tracker: VehicleTracker,
        damage_detector: DamageDetector,
        snapshots_dir: str = "data/outputs/snapshots",
    ):
        self.tracker = vehicle_tracker
        self.damage_detector = damage_detector
        self.snapshots_dir = snapshots_dir
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def process_video(
        self,
        input_video_path: str,
        output_video_path: str,
        frame_skip: int = 1,
        progress_callback: Optional[Callable[[float, int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process an input CCTV video, run vehicle tracking & damage detection,
        and write an H.264 web-compatible annotated output video.
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = max(1.0, fps / max(1, frame_skip))

        # Setup H.264 Video Encoder via PyAV (Ensures 100% web browser compatibility)
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        container = av.open(output_video_path, mode="w")
        stream = container.add_stream("libx264", rate=int(out_fps))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "23"}

        frame_idx = 0
        processed_count = 0
        start_time = time.time()

        # Track history of damages found per vehicle: {track_id: [damage_dict, ...]}
        vehicle_damages: Dict[int, List[Dict[str, Any]]] = {}
        # Stores the best annotated snapshot crop directly from the most centered, clear frame
        vehicle_best_snapshot: Dict[int, np.ndarray] = {}
        vehicle_first_seen_sec: Dict[int, float] = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if (frame_idx - 1) % frame_skip != 0:
                continue

            processed_count += 1
            timestamp_sec = round(frame_idx / fps, 2)

            # 1. Track vehicles in current frame
            tracked_vehicles = self.tracker.track_frame(frame, frame_idx)

            # 2. Inspect each vehicle for damage (skip when vehicle is cut off at frame border)
            active_frame_damages = []
            frame_damage_by_vehicle: Dict[int, List[Dict[str, Any]]] = {}

            for veh in tracked_vehicles:
                t_id = veh["track_id"]
                crop = veh["crop"]
                v_bbox = veh["bbox"]
                vx1, vy1, vx2, vy2 = v_bbox

                if t_id not in vehicle_first_seen_sec:
                    vehicle_first_seen_sec[t_id] = timestamp_sec

                # Avoid inspecting if vehicle is barely entering or exiting (< 10% on screen)
                box_w = vx2 - vx1
                if box_w < 80 or vx1 <= 5 or vx2 >= (width - 5):
                    continue

                # Run damage detector on the vehicle crop
                damages = self.damage_detector.detect_damage_in_crop(crop, v_bbox)

                if damages:
                    for d in damages:
                        d["frame_idx"] = frame_idx
                        d["timestamp"] = timestamp_sec
                        d["vehicle_id"] = t_id
                        d["vehicle_class"] = veh["class_name"]

                    active_frame_damages.extend(damages)
                    frame_damage_by_vehicle[t_id] = damages

                    if t_id not in vehicle_damages:
                        vehicle_damages[t_id] = []
                    vehicle_damages[t_id].extend(damages)

            # 3. Draw annotations on frame
            annotated_frame = self._render_annotations(
                frame.copy(),
                tracked_vehicles,
                active_frame_damages,
                frame_idx,
                total_frames,
                fps,
            )

            # 4. Capture high-quality crop of each vehicle from the most centered, full-view frame
            for veh in tracked_vehicles:
                t_id = veh["track_id"]
                x1, y1, x2, y2 = veh["bbox"]
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h

                # Check if this frame has damage for this vehicle
                dmg_here = frame_damage_by_vehicle.get(t_id, [])
                cur_sev = max([d["severity_score"] for d in dmg_here], default=0.0)

                reg = self.tracker.vehicles_registry.get(t_id, {})
                if cur_sev > reg.get("max_severity", 0.0):
                    reg["max_severity"] = cur_sev

                # Crop vehicle with generous contextual padding
                pad_x = int(box_w * 0.40) + 30
                pad_y = int(box_h * 0.25) + 20
                px1 = max(0, x1 - pad_x)
                py1 = max(0, y1 - pad_y)
                px2 = min(width, x2 + pad_x)
                py2 = min(height, y2 + pad_y)
                crop_annotated = annotated_frame[py1:py2, px1:px2].copy()

                if crop_annotated.size > 0:
                    # Centrality metric: Closer to middle of camera view (x=width/2) is better
                    center_dist = abs((x1 + x2) / 2.0 - (width / 2.0))
                    centrality_bonus = max(0, 1.0 - (center_dist / (width / 2.0)))
                    
                    has_damage = len(dmg_here) > 0
                    quality_score = (box_area * (1.0 + centrality_bonus)) * (2.0 if has_damage else 1.0)

                    prev_score = reg.get("best_snap_score", 0)
                    if quality_score > prev_score or t_id not in vehicle_best_snapshot:
                        reg["best_snap_score"] = quality_score
                        vehicle_best_snapshot[t_id] = crop_annotated

            # 5. Write frame to H.264 video stream
            av_frame = av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

            # 6. Progress callback
            if progress_callback:
                progress_pct = min(1.0, frame_idx / max(1, total_frames))
                stats = {
                    "current_frame": frame_idx,
                    "total_frames": total_frames,
                    "vehicles_scanned": len(self.tracker.vehicles_registry),
                    "damaged_count": len([v for v in self.tracker.vehicles_registry.values() if v.get("max_severity", 0) > 0]),
                    "elapsed_time": round(time.time() - start_time, 1),
                }
                progress_callback(progress_pct, frame_idx, total_frames, stats)

        # Flush H.264 video encoder
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        cap.release()

        # 7. Generate records for ALL vehicles (both damaged and clean)
        all_records = self._generate_snapshots_and_records(
            vehicle_damages,
            vehicle_best_snapshot,
            vehicle_first_seen_sec,
        )

        damaged_records = [r for r in all_records if r["has_damage"]]

        summary = {
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "duration_sec": round(total_frames / fps, 2),
            "output_video_path": output_video_path,
            "total_vehicles_detected": len(self.tracker.vehicles_registry),
            "damaged_vehicles_count": len(damaged_records),
            "incidents": damaged_records,
            "all_vehicles": all_records,
        }

        return summary

    def _render_annotations(
        self,
        frame: np.ndarray,
        tracked_vehicles: List[Dict[str, Any]],
        damages: List[Dict[str, Any]],
        frame_idx: int,
        total_frames: int,
        fps: float,
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and HUD on the video frame."""
        h, w = frame.shape[:2]

        # 1. Draw Vehicle Bounding Boxes
        for veh in tracked_vehicles:
            x1, y1, x2, y2 = veh["bbox"]
            t_id = veh["track_id"]
            c_name = veh["class_name"]
            conf = veh["confidence"]

            # Vehicle Box: Deep Sky Blue / Cyan
            cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 175, 0), 2)

            # Vehicle Label Header
            label = f"ID #{t_id} | {c_name} ({int(conf * 100)}%)"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + lw + 10, y1), (235, 175, 0), -1)
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # 2. Draw Damage Bounding Boxes
        for d in damages:
            ax1, ay1, ax2, ay2 = d["abs_bbox"]
            dmg_type = d["type"]
            score = d["severity_score"]
            sev_label = d["severity_label"]

            # Color based on severity: Red for Critical, Orange for Major, Yellow for Minor
            if sev_label == "CRITICAL":
                color = (0, 0, 240)      # Bright Red
            elif sev_label == "MAJOR":
                color = (0, 140, 255)    # Orange
            else:
                color = (0, 220, 255)    # Yellow

            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), color, 2)

            # Tag for damage
            dmg_tag = f"[!] {dmg_type} ({int(score)}%)"
            (tw, th), _ = cv2.getTextSize(dmg_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(frame, (ax1, max(0, ay1 - 20)), (ax1 + tw + 6, ay1), color, -1)
            cv2.putText(
                frame,
                dmg_tag,
                (ax1 + 3, ay1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Top HUD Bar (Mine AI Surveillance Dashboard)
        hud_h = 42
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(hud_overlay, 0.75, frame, 0.25, 0, frame)

        total_v = len(self.tracker.vehicles_registry)
        active_dmg = len(damages)
        hud_text = (
            f"MINE CCTV AI | Frame: {frame_idx}/{total_frames} | "
            f"Active: {len(tracked_vehicles)} | Total Logged: {total_v} | "
            f"Damage Alerts: {active_dmg}"
        )
        cv2.putText(
            frame,
            hud_text,
            (15, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return frame

    def _generate_snapshots_and_records(
        self,
        vehicle_damages: Dict[int, List[Dict[str, Any]]],
        vehicle_best_snapshot: Dict[int, np.ndarray],
        vehicle_first_seen_sec: Dict[int, float],
    ) -> List[Dict[str, Any]]:
        """Save high-resolution cropped snapshots for all monitored vehicles."""
        records = []

        for t_id, veh_info in self.tracker.vehicles_registry.items():
            dmg_list = vehicle_damages.get(t_id, [])
            
            # Confirm damage: Any defect or structural crumple detected
            confirmed_dmgs = [d for d in dmg_list if d["severity_score"] >= 15.0 or len(dmg_list) >= 1]
            has_damage = len(confirmed_dmgs) > 0

            unique_dmg_types = list({d["type"] for d in confirmed_dmgs}) if has_damage else ["None (Surface Intact)"]
            max_score = max([d["severity_score"] for d in confirmed_dmgs], default=0.0)
            first_seen_time = vehicle_first_seen_sec.get(t_id, 0.0)
            veh_class = veh_info.get("class_name", "Mining Vehicle")

            # Save snapshot
            snapshot_path = None
            if t_id in vehicle_best_snapshot:
                snap_crop = vehicle_best_snapshot[t_id]
                status_suffix = "damaged" if has_damage else "clean"
                snapshot_filename = f"vehicle_id_{t_id}_{status_suffix}.jpg"
                snapshot_path = os.path.join(self.snapshots_dir, snapshot_filename)
                cv2.imwrite(snapshot_path, snap_crop)

            if not has_damage:
                severity_level = "HEALTHY"
            elif max_score >= 50.0:
                severity_level = "CRITICAL"
            elif max_score >= 25.0:
                severity_level = "MAJOR"
            else:
                severity_level = "MINOR"

            records.append({
                "vehicle_id": t_id,
                "vehicle_class": veh_class,
                "has_damage": has_damage,
                "first_detected_sec": first_seen_time,
                "damage_types": unique_dmg_types,
                "max_severity_score": max_score,
                "severity_level": severity_level,
                "total_damage_points": len(confirmed_dmgs),
                "snapshot_path": snapshot_path,
            })

        # Sort: Damaged first (by severity), then clean vehicles
        records.sort(key=lambda x: (not x["has_damage"], -x["max_severity_score"]))
        return records
