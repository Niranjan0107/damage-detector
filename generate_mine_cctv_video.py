"""
Photorealistic Mining CCTV Footage Generator
Generates a multi-vehicle CCTV sequence:
- Vehicle 1: Clean Mining Transport Truck
- Vehicle 2: Damaged Mining Truck with Dent, Scrape, and Crack
Uses H.264 PyAV encoding for seamless web playback.
"""
import os
import cv2
import numpy as np
import av


def generate_mine_cctv_video(
    output_path: str = "samples/mine_cctv_simulation.mp4",
    num_frames: int = 170,
    fps: int = 25,
    width: int = 1280,
    height: int = 720,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Load clean and damaged vehicle source images
    clean_truck = cv2.imread("samples/clean_truck.jpg")
    damaged_truck = cv2.imread("samples/test_damaged.jpg")

    if clean_truck is None or damaged_truck is None:
        raise ValueError("Source truck images missing in samples/")

    # Resize trucks to realistic CCTV gate dimensions
    t_w = 420
    t_h = int(t_w * (clean_truck.shape[0] / clean_truck.shape[1]))
    truck1 = cv2.resize(clean_truck, (t_w, t_h))

    t2_h = int(t_w * (damaged_truck.shape[0] / damaged_truck.shape[1]))
    truck2 = cv2.resize(damaged_truck, (t_w, t2_h))

    # 2. Mine Quarry Background (Gate 04 Weighbridge)
    np.random.seed(42)
    base_bg = np.zeros((height, width, 3), dtype=np.uint8)
    # Sky / Rim
    base_bg[:280, :] = [140, 165, 185]
    # Rock benches / terraces
    for c in range(width):
        h1 = int(220 + 20 * np.sin(c / 80.0) + 10 * np.cos(c / 35.0))
        base_bg[h1:320, c] = [100, 125, 140]
        h2 = int(270 + 15 * np.cos(c / 95.0))
        base_bg[h2:360, c] = [80, 110, 130]
    # Sandy Dirt Haul Road
    base_bg[360:, :] = [75, 120, 155]

    # Add realistic ground noise / gravel texture
    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    base_bg = np.clip(base_bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Gate & Checkpoint Signboard
    cv2.rectangle(base_bg, (40, 180), (300, 270), (30, 30, 30), -1)
    cv2.rectangle(base_bg, (40, 180), (300, 270), (0, 215, 255), 2)
    cv2.putText(base_bg, "MINE ZONE 04", (55, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 230, 255), 2)
    cv2.putText(base_bg, "SPEED LIMIT: 15 KM/H", (55, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Road guide markings
    cv2.line(base_bg, (0, 580), (width, 580), (50, 85, 115), 2)
    cv2.line(base_bg, (0, 595), (width, 595), (180, 180, 180), 2)

    # Setup H.264 PyAV stream
    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "23"}

    print(f"Generating realistic CCTV mining sequence ({num_frames} frames)...")

    # Vehicle 1 (Clean): Frames 0 to 85
    # Vehicle 2 (Damaged): Frames 80 to 170

    for f in range(num_frames):
        frame = base_bg.copy()

        # Vehicle 1: Clean Truck
        if 0 <= f <= 85:
            p1 = f / 85.0
            x1 = int(-t_w + p1 * (width + t_w + 50))
            y1 = 340
            _paste_vehicle(frame, truck1, x1, y1, width, height)

        # Vehicle 2: Damaged Truck
        if 80 <= f <= 170:
            p2 = (f - 80) / 90.0
            x2 = int(-t_w + p2 * (width + t_w + 50))
            y2 = 340
            _paste_vehicle(frame, truck2, x2, y2, width, height)

        # Camera grain
        grain = np.random.randint(-5, 5, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + grain, 0, 255).astype(np.uint8)

        # CCTV HUD Overlay
        sec = f / float(fps)
        if int(f / 12) % 2 == 0:
            cv2.circle(frame, (35, 35), 8, (0, 0, 240), -1)
        cv2.putText(frame, "REC  [CCTV-04 QUARRY PIT GATE]", (52, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        ts = f"2026-11-23 15:10:{int(sec):02d}.{int((sec % 1) * 100):02d} [25 FPS]"
        cv2.putText(frame, ts, (width - 450, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.putText(frame, f"CAM SENSOR: SONY-STARVIS | FRAME: {f:04d}/{num_frames}", (35, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(f"H.264 Mining CCTV video successfully created: {output_path}")
    return output_path


def _paste_vehicle(frame, veh_img, x, y, frame_w, frame_h):
    vh, vw = veh_img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + vw)
    y2 = min(frame_h, y + vh)

    crop_x1 = max(0, -x)
    crop_y1 = max(0, -y)
    crop_x2 = crop_x1 + (x2 - x1)
    crop_y2 = crop_y1 + (y2 - y1)

    if x2 > x1 and y2 > y1 and crop_x2 > crop_x1 and crop_y2 > crop_y1:
        frame[y1:y2, x1:x2] = veh_img[crop_y1:crop_y2, crop_x1:crop_x2]


if __name__ == "__main__":
    generate_mine_cctv_video()
