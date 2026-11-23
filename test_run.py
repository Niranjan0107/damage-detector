from detector.vehicle_tracker import VehicleTracker
from detector.damage_detector import DamageDetector
from detector.video_processor import VideoProcessor
from utils.report_generator import ReportGenerator
import os

print("Initializing Mining Fleet AI Inspector...")
tracker = VehicleTracker(
    model_path="yolov8n.pt",
    confidence_threshold=0.40,
    target_classes=[7],
    min_box_width=90,
    min_box_height=70,
)
damage_detector = DamageDetector(severity_threshold=0.14, filter_sun_glare=True)
processor = VideoProcessor(vehicle_tracker=tracker, damage_detector=damage_detector)

video_in = "samples/mine_cctv_simulation.mp4"
video_out = "data/outputs/annotated_mine_cctv.mp4"

print(f"Analyzing CCTV video: {video_in}")
summary = processor.process_video(
    input_video_path=video_in,
    output_video_path=video_out,
    frame_skip=2,
)

print("\n================ INSPECTION REPORT ================")
print(f"Total Vehicles Detected: {summary['total_vehicles_detected']}")
print(f"Damaged Units Flagged: {summary['damaged_vehicles_count']}")

for v in summary.get("all_vehicles", []):
    status = "DAMAGED" if v["has_damage"] else "CLEAN"
    print(f"Vehicle #{v['vehicle_id']} ({v['vehicle_class']}) -> Status: {status} ({v['severity_level']}) | Score: {v['max_severity_score']}%")
    if v["has_damage"]:
        print(f"   Damages: {', '.join(v['damage_types'])}")
    print(f"   Snapshot: {v['snapshot_path']}")

pdf_path = "data/reports/Mine_CCTV_Report.pdf"
ReportGenerator.generate_pdf(summary, summary["incidents"], pdf_path)
print(f"\nPDF Report generated successfully at: {pdf_path} (Exists: {os.path.exists(pdf_path)})")
