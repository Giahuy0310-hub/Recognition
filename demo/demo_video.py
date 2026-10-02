from ultralytics import YOLO
import cv2
import os
import time

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "traffic_sign_18class_hardcase" / "weights" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "input" / "compare_video.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "output" / "hardcase_demo_result.mp4"
CONF = 0.35
IMG_SIZE = 640

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Khong tim thay model:\n{MODEL_PATH}"
    )

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Khong tim thay video:\n{VIDEO_PATH}"
    )

os.makedirs(
    os.path.dirname(OUTPUT_VIDEO),
    exist_ok=True
)

print("=" * 70)
print("CREATE DEMO VIDEO - HARDCASE MODEL")
print("=" * 70)
print("MODEL:")
print(MODEL_PATH)
print()
print("VIDEO:")
print(VIDEO_PATH)
print()
print("CONF :", CONF)
print("IMG  :", IMG_SIZE)
print("=" * 70)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        "Khong mo duoc video"
    )

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("Resolution :", width, "x", height)
print("FPS input  :", fps)
print("Frames     :", total_frames)
print("=" * 70)

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

frame_id = 0
start_time = time.time()
class_count = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    results = model.predict(
        frame,
        conf=CONF,
        imgsz=IMG_SIZE,
        verbose=False
    )

    result = results[0]

    if result.boxes:
        for box in result.boxes:
            cls = int(
                box.cls[0]
            )

            name = model.names[cls]

            class_count[name] = (
                class_count.get(name, 0)
                + 1
            )

    annotated = result.plot()

    elapsed = time.time() - start_time

    real_fps = (
        frame_id / elapsed
    )

    cv2.putText(
        annotated,
        f"FPS: {real_fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        "YOLOv8 Traffic Sign Detection - Hardcase",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    writer.write(
        annotated
    )

    if frame_id % 20 == 0:
        print(
            f"Processed {frame_id}/{total_frames}"
        )

cap.release()
writer.release()

print()
print("=" * 70)
print("DONE")
print("=" * 70)
print(
    "Processed frames:",
    frame_id
)

print()
print("Detection summary:")

for k, v in sorted(
    class_count.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(
        f"{k:40s}: {v}"
    )

print()
print("Output video:")
print(OUTPUT_VIDEO)
print("=" * 70)