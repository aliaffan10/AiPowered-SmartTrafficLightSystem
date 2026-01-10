"""
Video Traffic Analytics (YOLOv5) + Congestion Estimation

What this script does:
- Runs YOLOv5 on each video frame
- Counts vehicles per frame
- Computes smoothed traffic "congestion level" using:
    (1) rolling average vehicle count
    (2) rolling average occupancy (sum of bbox areas / frame area)
- Overlays metrics on the output video
- Writes annotated video to OUTPUT_VIDEO

Notes:
- This version is "lane-agnostic" on purpose, since you said intersections vary (2/3/4/6 lanes).
  It gives a robust overall congestion score for the whole frame.
- Later you can extend to lane-specific ROIs (manual or auto ROI) without changing the core logic.
"""

import cv2
import torch
import numpy as np
import warnings
from collections import deque
from pathlib import Path

# --------- CONFIG ----------
WEIGHTS = "yolov5s.pt"  # pretrained (COCO). For custom later: "yolov5/runs/train/<run>/weights/best.pt"
SOURCE_VIDEO = "test_vids/vid4.mp4"
OUTPUT_VIDEO = "test_vids/output_detected4.mp4"

IMG_SIZE = 416
CONF = 0.30

# Rolling window length (frames). If you export at 30 FPS, 30 frames ~ 1 second smoothing.
ROLLING_WINDOW = 30

# Which classes count as "vehicles" for COCO pretrained yolov5s:
# COCO ids: 2=car, 3=motorcycle, 5=bus, 7=truck
COCO_VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# Congestion thresholds (tune later with logs)
# avg_count thresholds:
COUNT_LOW_MAX = 10
COUNT_MED_MAX = 15

# avg_occupancy thresholds (fraction of frame area occupied by vehicle boxes)
OCC_LOW_MAX = 0.10
OCC_MED_MAX = 0.20
# ---------------------------

# Optional: silence the YOLOv5 autocast FutureWarning spam
warnings.filterwarnings("ignore", message=".*autocast.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_model():
    """
    Loads YOLOv5 model from local 'yolov5' repo via torch.hub.
    If WEIGHTS is yolov5s.pt (COCO), this will work.
    If WEIGHTS is your custom best.pt, it will also work.
    """
    model = torch.hub.load("yolov5", "custom", path=WEIGHTS, source="local")
    model.conf = CONF
    model.iou = 0.45
    model.max_det = 200
    return model


def classify_congestion(avg_count: float, avg_occ: float) -> str:
    """
    Hybrid congestion classification using both count and occupancy.
    """
    # HIGH if either count or occupancy is high
    if avg_occ >= OCC_MED_MAX and avg_count >= COUNT_MED_MAX:
        return "HIGH"
    # MED if either count or occupancy is medium
    if avg_occ >= OCC_LOW_MAX and avg_count >= COUNT_LOW_MAX:
        return "MED"
    return "LOW"


def extract_vehicle_detections(results):
    """
    Extract detections from YOLO results and return list of vehicle boxes.

    Returns:
        vehicles: list of dicts:
          {"x1": int, "y1": int, "x2": int, "y2": int, "conf": float, "cls": int, "name": str}
    """
    vehicles = []

    # YOLOv5 results have .xyxy[0] tensor: [x1,y1,x2,y2,conf,cls]
    det = results.xyxy[0]
    if det is None or len(det) == 0:
        return vehicles

    names = results.names  # class id -> name mapping

    for row in det.cpu().numpy():
        x1, y1, x2, y2, conf, cls = row
        cls = int(cls)

        # If using pretrained COCO weights, filter vehicle classes:
        # If later you use a custom dataset with only vehicles, you can remove this filter.
        if WEIGHTS == "yolov5s.pt" and cls not in COCO_VEHICLE_CLASS_IDS:
            continue

        vehicles.append(
            {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "conf": float(conf),
                "cls": cls,
                "name": names.get(cls, str(cls)) if isinstance(names, dict) else str(cls),
            }
        )

    return vehicles


def compute_frame_metrics(vehicles, frame_w, frame_h):
    """
    Compute:
    - vehicle_count: number of vehicles detected
    - occupancy: sum of vehicle bbox areas / frame area (clipped to avoid crazy boxes)
    """
    vehicle_count = len(vehicles)
    frame_area = max(1, frame_w * frame_h)

    area_sum = 0.0
    for v in vehicles:
        bw = max(0, v["x2"] - v["x1"])
        bh = max(0, v["y2"] - v["y1"])
        # Clip individual box area to something reasonable (optional safeguard)
        box_area = min(bw * bh, frame_area)
        area_sum += box_area

    occupancy = float(area_sum / frame_area)
    # Occupancy can exceed 1.0 if overlapping boxes; cap for stability
    occupancy = min(occupancy, 1.0)

    return vehicle_count, occupancy


def draw_overlay(frame_bgr, vehicle_count, avg_count, occ, avg_occ, level, fps, frame_idx):
    """
    Draws metrics overlay on the frame.
    """
    # Simple HUD background
    hud_x, hud_y = 10, 10
    hud_w, hud_h = 520, 140
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)
    alpha = 0.35
    frame_bgr[:] = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

    # Text lines
    lines = [
        f"Frame: {frame_idx}",
        f"FPS (input): {fps:.1f}",
        f"Vehicles (frame): {vehicle_count} | Avg({ROLLING_WINDOW}): {avg_count:.2f}",
        f"Occupancy (frame): {occ:.3f} | Avg({ROLLING_WINDOW}): {avg_occ:.3f}",
        f"Congestion Level: {level}",
    ]

    y = hud_y + 28
    for i, t in enumerate(lines):
        # Emphasize congestion level
        if "Congestion Level" in t:
            scale = 0.9
            thickness = 2
        else:
            scale = 0.75
            thickness = 2

        cv2.putText(
            frame_bgr,
            t,
            (hud_x + 12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += 26


def run_video(model, src, out):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {src}")

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output FPS (fixed to 30 as you wanted)
    target_fps = 30.0

    # Ensure output directory exists
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, target_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter. Try a different codec or output path.")

    # Rolling history for stable congestion estimates
    count_hist = deque(maxlen=ROLLING_WINDOW)
    occ_hist = deque(maxlen=ROLLING_WINDOW)

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Inference expects RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = model(rgb, size=IMG_SIZE)

        # Extract only vehicle detections
        vehicles = extract_vehicle_detections(results)

        # Compute metrics
        vehicle_count, occ = compute_frame_metrics(vehicles, w, h)
        count_hist.append(vehicle_count)
        occ_hist.append(occ)

        avg_count = float(np.mean(count_hist)) if len(count_hist) else 0.0
        avg_occ = float(np.mean(occ_hist)) if len(occ_hist) else 0.0

        level = classify_congestion(avg_count, avg_occ)

        # Render YOLO boxes on frame
        results.render()
        rendered_rgb = results.ims[0]          # RGB with boxes
        frame_out = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)

        # Overlay analytics
        draw_overlay(frame_out, vehicle_count, avg_count, occ, avg_occ, level, fps, frame_idx)

        writer.write(frame_out)

        if frame_idx % 30 == 0:
            print(
                f"Processed {frame_idx} frames | "
                f"count={vehicle_count}, avg_count={avg_count:.2f}, avg_occ={avg_occ:.3f}, level={level}"
            )

    cap.release()
    writer.release()
    print(f"\n✅ Saved output video to: {out}")


if __name__ == "__main__":
    model = load_model()
    run_video(model, SOURCE_VIDEO, OUTPUT_VIDEO)
