import os
import csv
import cv2
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
OLD_MODEL = "models/best_old.pt"   # YOLO 15 classes
NEW_MODEL = "models/best.pt"       # YOLO 18 classes

VIDEO_PATH = "input/compare_video.mp4"

OUTPUT_DIR = "output/model_compare"
CSV_PATH = os.path.join(OUTPUT_DIR, "model_comparison.csv")

IMG_SIZE = 640
CONF_THRESHOLD = 0.18
IOU_THRESHOLD = 0.60
MAX_DET = 50

# ============================================================
# CLASS MAP
# ============================================================
# Tên hiển thị của model cũ.
OLD_NAMES = {
    0: "CAM DO XE",
    1: "CAM DUNG/DO XE",
    2: "CAM DI NGUOC CHIEU",
    3: "CAM XE 2/3 BANH",
    4: "GIAO NHAU DUONG KHONG UU TIEN",
    5: "CHU Y NGUOI DI BO",
    6: "DI CHAM",
    7: "CAM O TO TAI/XE KHACH",
    8: "TOC DO TOI DA 60",
    9: "LAN DUONG THEO VACH",
    10: "CAM O TO",
    11: "TOC DO TOI DA 80",
    12: "TOC DO TOI DA 50",
    13: "CAM RE PHAI",
    14: "DUONG MOT CHIEU",
}

# Model mới dùng đúng thứ tự 18 class trong data.yaml.
NEW_NAMES = {
    0: "CAM DO XE",
    1: "CAM DUNG/DO XE",
    2: "CAM DI NGUOC CHIEU",
    3: "CAM XE 2/3 BANH",
    4: "GIAO NHAU DUONG KHONG UU TIEN",
    5: "CHU Y NGUOI DI BO",
    6: "DI CHAM",
    7: "CAM O TO TAI/XE KHACH",
    8: "TOC DO TOI DA 50",
    9: "TOC DO TOI DA 60",
    10: "TOC DO TOI DA 80",
    11: "LAN DUONG THEO VACH",
    12: "CAM O TO",
    13: "CAM RE PHAI",
    14: "DUONG MOT CHIEU",
    15: "CAM RE TRAI",
    16: "CAM O TO RE PHAI",
    17: "CAM O TO RE TRAI",
}

# ============================================================
# HELPERS
# ============================================================
def detections_from_result(result, names):
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        xyxy = box.xyxy[0].tolist()

        detections.append({
            "class_id": cls_id,
            "class_name": names.get(cls_id, str(cls_id)),
            "confidence": conf,
            "x1": xyxy[0],
            "y1": xyxy[1],
            "x2": xyxy[2],
            "y2": xyxy[3],
        })

    return detections


def compare_detections(old_dets, new_dets):
    """
    Ghép detection gần nhau theo IoU để so sánh class.
    Không nhằm đánh giá mAP; chỉ nhằm tìm trường hợp model
    cũ và model mới nhìn cùng một biển nhưng phân loại khác nhau.
    """
    comparisons = []
    used_new = set()

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter

        return inter / union if union > 0 else 0

    for old in old_dets:
        best_idx = None
        best_iou = 0

        for idx, new in enumerate(new_dets):
            if idx in used_new:
                continue

            score = iou(old, new)

            if score > best_iou:
                best_iou = score
                best_idx = idx

        if best_idx is not None and best_iou >= 0.30:
            new = new_dets[best_idx]
            used_new.add(best_idx)

            same_class = old["class_name"] == new["class_name"]

            comparisons.append({
                "match_type": "MATCH",
                "old_class": old["class_name"],
                "old_conf": round(old["confidence"], 4),
                "new_class": new["class_name"],
                "new_conf": round(new["confidence"], 4),
                "iou": round(best_iou, 4),
                "class_same": same_class,
            })
        else:
            comparisons.append({
                "match_type": "OLD_ONLY",
                "old_class": old["class_name"],
                "old_conf": round(old["confidence"], 4),
                "new_class": "",
                "new_conf": "",
                "iou": "",
                "class_same": False,
            })

    for idx, new in enumerate(new_dets):
        if idx not in used_new:
            comparisons.append({
                "match_type": "NEW_ONLY",
                "old_class": "",
                "old_conf": "",
                "new_class": new["class_name"],
                "new_conf": round(new["confidence"], 4),
                "iou": "",
                "class_same": False,
            })

    return comparisons


# ============================================================
# CHECK FILES
# ============================================================
if not os.path.exists(OLD_MODEL):
    raise FileNotFoundError(f"Không tìm thấy model cũ: {OLD_MODEL}")

if not os.path.exists(NEW_MODEL):
    raise FileNotFoundError(f"Không tìm thấy model mới: {NEW_MODEL}")

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy video: {VIDEO_PATH}\n"
        "Hãy đổi VIDEO_PATH ở đầu file thành đúng đường dẫn video."
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD MODELS
# ============================================================
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(f"Model cũ : {OLD_MODEL}")
print(f"Model mới : {NEW_MODEL}")
print(f"Video     : {VIDEO_PATH}")
print(f"Confidence: {CONF_THRESHOLD}")
print(f"Image size: {IMG_SIZE}")
print(f"IoU       : {IOU_THRESHOLD}")
print()

old_model = YOLO(OLD_MODEL)
new_model = YOLO(NEW_MODEL)

# ============================================================
# VIDEO
# ============================================================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Không mở được video.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS video     : {fps:.2f}")
print(f"Tổng số frame : {total_frames}")
print()

rows = []
frame_no = 0
different_frames = 0
different_pairs = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    frame_no += 1

    old_result = old_model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_det=MAX_DET,
        verbose=False,
        device="cpu",
    )[0]

    new_result = new_model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_det=MAX_DET,
        verbose=False,
        device="cpu",
    )[0]

    old_dets = detections_from_result(old_result, OLD_NAMES)
    new_dets = detections_from_result(new_result, NEW_NAMES)

    comparisons = compare_detections(old_dets, new_dets)

    frame_has_difference = False

    for item in comparisons:
        if item["match_type"] == "MATCH" and not item["class_same"]:
            frame_has_difference = True
            different_pairs += 1

        elif item["match_type"] in ("OLD_ONLY", "NEW_ONLY"):
            frame_has_difference = True

        rows.append({
            "frame": frame_no,
            "time_sec": round((frame_no - 1) / fps, 3) if fps else "",
            **item,
        })

    if frame_has_difference:
        different_frames += 1

    if frame_no % 20 == 0 or frame_no == total_frames:
        print(
            f"\rĐang xử lý: {frame_no}/{total_frames} "
            f"({frame_no / total_frames * 100:.1f}%)",
            end="",
        )

cap.release()

print("\n")
print("=" * 70)
print(f"Hoàn tất: {frame_no} frames")
print(f"Frame có khác biệt: {different_frames}")
print(f"Cặp detection khác class: {different_pairs}")
print("=" * 70)

# ============================================================
# SAVE CSV
# ============================================================
fieldnames = [
    "frame",
    "time_sec",
    "match_type",
    "old_class",
    "old_conf",
    "new_class",
    "new_conf",
    "iou",
    "class_same",
]

with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nĐã lưu: {CSV_PATH}")

# ============================================================
# SUMMARY
# ============================================================
class_confusions = {}

for row in rows:
    if row["match_type"] == "MATCH" and not row["class_same"]:
        key = (row["old_class"], row["new_class"])
        class_confusions[key] = class_confusions.get(key, 0) + 1

print("\nCác cặp class bị nhầm nhiều nhất:")
if class_confusions:
    for (old_class, new_class), count in sorted(
        class_confusions.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:15]:
        print(f"  {old_class}  ->  {new_class}: {count} lần")
else:
    print("  Không phát hiện cặp class khác nhau.")

print("\nXong.")
