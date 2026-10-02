from ultralytics import YOLO
import cv2
import os


# ==============================
# PATH
# ==============================

MODEL_PATH = r"C:\Users\Gia Huy\OneDrive - ut.edu.vn\Desktop\TTTN\runs\detect\traffic_sign_18class_hardcase\weights\best.pt"

VIDEO_PATH = r"C:\Users\Gia Huy\OneDrive - ut.edu.vn\Desktop\TTTN\input\compare_video.mp4"

OUTPUT_DIR = r"C:\Users\Gia Huy\OneDrive - ut.edu.vn\Desktop\TTTN\output\hardcase_video_test"


# ==============================
# CONFIG
# ==============================

CONF = 0.18
IMG_SIZE = 640


# ==============================
# LOAD MODEL
# ==============================

print("="*70)
print("TEST VIDEO - HARDCASE MODEL")
print("="*70)

print("Model :", MODEL_PATH)
print("Video :", VIDEO_PATH)

model = YOLO(MODEL_PATH)


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# OPEN VIDEO
# ==============================

cap = cv2.VideoCapture(VIDEO_PATH)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)


print("Frames:", total_frames)
print("FPS   :", fps)
print("Conf  :", CONF)
print("ImgSz :", IMG_SIZE)

print("="*70)


count = 0

class_count = {}


while True:

    ret, frame = cap.read()

    if not ret:
        break

    count += 1


    results = model.predict(
        frame,
        conf=CONF,
        imgsz=IMG_SIZE,
        verbose=False
    )


    labels = []


    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            name = model.names[cls]

            labels.append(
                f"{name.upper()} ({conf:.2f})"
            )
            if count <= 36 or (171 <= count <= 204):
                print(
                    ">>> HARD CASE",
                    count,
                    name,
                    round(conf,3)
    )

            class_count[name] = class_count.get(name,0)+1


    if labels:

        print(
            f"[FRAME {count:06d}] "
            + " | ".join(labels)
        )

    else:

        print(
            f"[FRAME {count:06d}] NONE"
        )


    # lưu ảnh bbox mỗi frame

    annotated = results[0].plot()

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            f"frame_{count:04d}.jpg"
        ),
        annotated
    )


cap.release()


print()
print("="*70)
print("HOAN TAT")
print("="*70)

print("Tong frame:", count)

print()
print("So luong detection:")

for k,v in class_count.items():

    print(
        f"{k:30s}: {v}"
    )


print()
print("Anh luu tai:")
print(OUTPUT_DIR)
print("="*70)