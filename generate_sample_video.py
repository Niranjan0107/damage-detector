"""
Photorealistic Mining CCTV Video Generator
Uses real truck imagery moving through a mine camera view, with realistic body damage patches.
"""
import os
import cv2
import numpy as np


def generate_sample_mine_video(
    output_path: str = "samples/sample_mine_cctv.mp4",
    num_frames: int = 120,
    fps: int = 25,
    width: int = 1280,
    height: int = 720,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    truck_img_path = "samples/truck_0.jpg"

    if not os.path.exists(truck_img_path):
        import urllib.request
        url = "https://images.unsplash.com/photo-1601584115197-04ecc0da31d7?w=800"
        urllib.request.urlretrieve(url, truck_img_path)

    truck_raw = cv2.imread(truck_img_path)
    if truck_raw is None:
        raise ValueError("Could not load truck image")

    # Resize truck to realistic CCTV size
    target_truck_w = 480
    aspect = truck_raw.shape[0] / truck_raw.shape[1]
    target_truck_h = int(target_truck_w * aspect)
    truck = cv2.resize(truck_raw, (target_truck_w, target_truck_h))

    # Add realistic damage artifacts onto the truck body
    # 1. Heavy dent (dark shadow contour + specular highlight)
    dent_cx = int(target_truck_w * 0.45)
    dent_cy = int(target_truck_h * 0.55)
    cv2.ellipse(truck, (dent_cx, dent_cy), (35, 22), 30, 0, 360, (20, 35, 50), -1)
    cv2.ellipse(truck, (dent_cx - 3, dent_cy - 2), (26, 14), 30, 0, 360, (5, 15, 25), -1)
    cv2.ellipse(truck, (dent_cx + 4, dent_cy + 4), (28, 15), 30, 0, 180, (180, 220, 240), 2)

    # 2. Structural weld crack / fissure
    crack_pts = np.array([
        [dent_cx + 60, dent_cy - 40],
        [dent_cx + 70, dent_cy - 20],
        [dent_cx + 65, dent_cy + 5],
        [dent_cx + 80, dent_cy + 30],
    ], np.int32)
    cv2.polylines(truck, [crack_pts], False, (10, 10, 15), 3)

    # 3. Deep metallic scrape
    cv2.line(truck, (dent_cx - 90, dent_cy + 15), (dent_cx - 20, dent_cy + 35), (25, 25, 25), 4)
    cv2.line(truck, (dent_cx - 88, dent_cy + 16), (dent_cx - 22, dent_cy + 34), (210, 210, 210), 1)

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Rendering {num_frames} frames of CCTV footage...")

    # Create Quarry Site Background
    np.random.seed(42)
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    # Ground: Sandy ochre/dirt
    bg[320:, :] = [75, 120, 155]
    # Rocky backdrop
    bg[:320, :] = [110, 135, 150]
    
    # Add quarry rock textures
    noise = np.random.randint(-12, 12, (height, width, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Mine Zone Signboard
    cv2.rectangle(bg, (40, 220), (220, 300), (35, 35, 35), -1)
    cv2.putText(bg, "MINE PIT 02", (55, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
    cv2.putText(bg, "WEIGHBRIDGE", (55, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # Moving truck animation across mine yard
    truck_y = 350
    for f_idx in range(num_frames):
        frame = bg.copy()
        progress = f_idx / float(num_frames)
        truck_x = int(-target_truck_w + progress * (width + target_truck_w + 120))

        # Composite truck onto frame with boundary clipping
        x1 = max(0, truck_x)
        y1 = max(0, truck_y)
        x2 = min(width, truck_x + target_truck_w)
        y2 = min(height, truck_y + target_truck_h)

        crop_x1 = max(0, -truck_x)
        crop_y1 = max(0, -truck_y)
        crop_x2 = crop_x1 + (x2 - x1)
        crop_y2 = crop_y1 + (y2 - y1)

        if x2 > x1 and y2 > y1 and crop_x2 > crop_x1 and crop_y2 > crop_y1:
            frame[y1:y2, x1:x2] = truck[crop_y1:crop_y2, crop_x1:crop_x2]

        # CCTV timestamp and camera watermark
        sec = f_idx / float(fps)
        cctv_time = f"CAM-04 [WEIGHBRIDGE] | 2026-11-21 16:45:{int(sec):02d}.{int((sec % 1) * 100):02d}"
        cv2.putText(frame, cctv_time, (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"Sample mine CCTV video ready: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_sample_mine_video()
