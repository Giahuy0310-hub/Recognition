import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL = "models/best.pt"
DEFAULT_SOURCE = "input"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
WINDOW_NAME = "Traffic Sign Detection"

CLASS_DISPLAY = {
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

WARNING_CLASSES = {0, 1, 2, 6, 8, 9, 10, 13, 15, 16, 17}

DETECTION_HEADER = [
    "timestamp", "run_id", "source", "frame", "video_time_s",
    "class_id", "class_name", "confidence", "is_warning",
    "x1", "y1", "x2", "y2", "crop_path",
]
EVENT_HEADER = [
    "timestamp", "run_id", "source", "frame", "video_time_s",
    "class_id", "class_name", "confidence", "event", "crop_path",
]


def is_warning(class_id):
    return class_id in WARNING_CLASSES


def source_name_of(source):
    return os.path.basename(str(source))


def detect_source_type(source):
    if str(source).lower() in {"webcam", "0"}:
        return "webcam"
    if os.path.isdir(source):
        return "folder"
    if os.path.isfile(source):
        extension = os.path.splitext(source)[1].lower()
        if extension in IMAGE_EXTENSIONS:
            return "image"
        if extension in VIDEO_EXTENSIONS:
            return "video"
    return "unknown"


class EventTracker:
    def __init__(self, min_confidence, timeout_s):
        self.min_confidence = min_confidence
        self.timeout_s = timeout_s
        self.last_seen = {}

    def update(self, detections, t):
        best = {}
        for detection in detections:
            if detection["confidence"] < self.min_confidence:
                continue
            class_id = detection["class_id"]
            if class_id not in best or detection["confidence"] > best[class_id]["confidence"]:
                best[class_id] = detection

        new_events = []
        for class_id, detection in best.items():
            if class_id not in self.last_seen:
                new_events.append(detection)
            self.last_seen[class_id] = t

        expired = [c for c, seen in self.last_seen.items() if t - seen > self.timeout_s]
        for class_id in expired:
            del self.last_seen[class_id]

        return new_events


class CsvLogger:
    def __init__(self, output_dir, run_id, fresh):
        self.run_id = run_id
        self.det_file, self.det_writer = self._open(
            os.path.join(output_dir, "detection_log.csv"), DETECTION_HEADER, fresh
        )
        self.evt_file, self.evt_writer = self._open(
            os.path.join(output_dir, "event_log.csv"), EVENT_HEADER, fresh
        )

    @staticmethod
    def _open(path, header, fresh):
        needs_header = fresh or not os.path.exists(path) or os.path.getsize(path) == 0

        if not needs_header:
            with open(path, encoding="utf-8-sig", newline="") as f:
                current = next(csv.reader(f), [])
            if current != header:
                os.replace(path, f"{path}.bak_{datetime.now():%Y%m%d_%H%M%S}")
                needs_header = True

        handle = open(path, "w" if needs_header else "a", newline="", encoding="utf-8-sig")
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(header)
            handle.flush()
        return handle, writer

    def log_frame(self, source, frame_number, video_time, detections, events):
        if not detections and not events:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_time_text = f"{video_time:.3f}"

        for d in detections:
            self.det_writer.writerow([
                timestamp, self.run_id, source, frame_number, video_time_text,
                d["class_id"], d["class_name"], f'{d["confidence"]:.4f}',
                int(is_warning(d["class_id"])),
                d["x1"], d["y1"], d["x2"], d["y2"], d.get("crop_path", ""),
            ])

        for d in events:
            self.evt_writer.writerow([
                timestamp, self.run_id, source, frame_number, video_time_text,
                d["class_id"], d["class_name"], f'{d["confidence"]:.4f}',
                "DETECTED", d.get("crop_path", ""),
            ])

        self.det_file.flush()
        self.evt_file.flush()

    def close(self):
        self.det_file.close()
        self.evt_file.close()


class RunStats:
    def __init__(self):
        self.frames = 0
        self.detections = 0
        self.infer_times = []
        self.total_times = []

    def add(self, n_detections, infer, total):
        self.frames += 1
        self.detections += n_detections
        self.infer_times.append(infer)
        self.total_times.append(total)

    @staticmethod
    def _avg_fps(times):
        if not times:
            return 0.0, 0.0
        avg = sum(times) / len(times)
        return (1.0 / avg if avg > 0 else 0.0), avg

    def print_summary(self, title):
        infer_fps, infer_avg = self._avg_fps(self.infer_times)
        total_fps, total_avg = self._avg_fps(self.total_times)
        print(f"\n===== {title} =====")
        print(f"So frame/anh: {self.frames}")
        print(f"Tong bien bao phat hien: {self.detections}")
        print(f"Infer FPS trung binh: {infer_fps:.2f} (latency {infer_avg * 1000:.1f} ms)")
        print(f"Pipeline FPS trung binh: {total_fps:.2f} (latency {total_avg * 1000:.1f} ms)")


def ui_scale(image):
    return min(3.0, max(0.6, max(image.shape[:2]) / 1280))


def draw_text_box(image, text, x, y, s, base_scale=0.5, base_thickness=2):
    font_scale = base_scale * s
    thickness = max(1, round(base_thickness * s))
    pad = max(3, int(5 * s))

    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    x1 = max(0, x - pad)
    y1 = max(0, y - h - baseline - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + pad)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def draw_detection(image, detection, s):
    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    color = (0, 0, 255) if is_warning(detection["class_id"]) else (0, 200, 0)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, max(2, round(3 * s)))

    label = f'{detection["class_name"]}  {detection["confidence"] * 100:.0f}%'
    text_y = y1 - int(8 * s)
    if text_y < int(20 * s):
        text_y = y1 + int(22 * s)

    draw_text_box(image, label, x1, text_y, s)


def draw_panel(image, infer_time, detections, source_name, conf_threshold, s):
    x0, y0 = int(15 * s), int(15 * s)
    shown = sorted(detections, key=lambda d: d["confidence"], reverse=True)[:5]
    panel_w = int(360 * s)
    panel_h = int((155 + len(shown) * 24 - (24 if not shown else 0)) * s)

    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)

    def put(text, dx, dy, font_scale, thickness):
        cv2.putText(
            image, text, (x0 + int(dx * s), y0 + int(dy * s)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale * s, (255, 255, 255),
            max(1, round(thickness * s)), cv2.LINE_AA,
        )

    infer_fps = 1.0 / infer_time if infer_time > 0 else 0.0

    put("TRAFFIC SIGN DETECTION", 12, 25, 0.62, 2)
    put(f"INFER FPS: {infer_fps:.1f}", 12, 53, 0.52, 1)
    put(f"LATENCY: {infer_time * 1000:.1f} ms", 175, 53, 0.52, 1)
    put(f"BIEN BAO: {len(detections)}", 12, 80, 0.52, 1)
    put(f"CONF: {conf_threshold:.2f}", 175, 80, 0.52, 1)

    y = 108
    for d in shown:
        prefix = "CANH BAO: " if is_warning(d["class_id"]) else ""
        put(f'{prefix}{d["class_name"]} {d["confidence"] * 100:.0f}%'[:48], 12, y, 0.43, 1)
        y += 23

    source_text = source_name[:70]
    font_scale = 0.48 * s
    thickness = max(1, round(s))
    (text_w, text_h), _ = cv2.getTextSize(source_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    sx = max(10, image.shape[1] - text_w - int(12 * s))
    sy = image.shape[0] - int(12 * s)

    cv2.rectangle(image, (sx - 5, sy - text_h - 7), (sx + text_w + 5, sy + 5), (0, 0, 0), -1)
    cv2.putText(image, source_text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


class Demo:
    def __init__(self, args):
        self.args = args
        self.show = not args.no_show
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(args.output, exist_ok=True)
        self.crops_dir = os.path.join(args.output, "crops")
        os.makedirs(self.crops_dir, exist_ok=True)

        print("Dang load model YOLO...")
        self.model = YOLO(args.model)
        print("Load model thanh cong.")

        self.logger = CsvLogger(args.output, self.run_id, args.fresh)
        self.warm_up()

    def warm_up(self):
        blank = np.zeros((self.args.imgsz, self.args.imgsz, 3), dtype=np.uint8)
        self.model.predict(source=blank, imgsz=self.args.imgsz, verbose=False)

    def close(self):
        self.logger.close()

    def infer(self, frame):
        start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.args.conf,
            imgsz=self.args.imgsz,
            iou=self.args.iou,
            max_det=self.args.max_det,
            verbose=False,
        )
        elapsed = time.perf_counter() - start

        detections = []
        if not results or results[0].boxes is None:
            return detections, elapsed

        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": class_id,
                "class_name": CLASS_DISPLAY.get(
                    class_id, str(self.model.names.get(class_id, f"CLASS {class_id}"))
                ),
                "confidence": float(box.conf[0].item()),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "crop_path": "",
            })

        return detections, elapsed

    def save_crop(self, frame, detection, source_name, frame_number):
        h, w = frame.shape[:2]
        pad = int(0.1 * max(detection["x2"] - detection["x1"], detection["y2"] - detection["y1"]))

        x1 = max(0, detection["x1"] - pad)
        y1 = max(0, detection["y1"] - pad)
        x2 = min(w, detection["x2"] + pad)
        y2 = min(h, detection["y2"] + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        stem = os.path.splitext(source_name)[0]
        name = f'{self.run_id}_{stem}_f{frame_number:06d}_c{detection["class_id"]}.jpg'
        path = os.path.join(self.crops_dir, name)
        cv2.imwrite(path, crop)
        return path.replace("\\", "/")

    def process_frame(self, frame, source_name, frame_number, video_time, tracker):
        start = time.perf_counter()

        detections, infer_time = self.infer(frame)
        events = tracker.update(detections, video_time)

        for event in events:
            event["crop_path"] = self.save_crop(frame, event, source_name, frame_number)

        s = ui_scale(frame)
        for detection in detections:
            draw_detection(frame, detection, s)
        draw_panel(frame, infer_time, detections, source_name, self.args.conf, s)

        self.logger.log_frame(source_name, frame_number, video_time, detections, events)

        return frame, detections, infer_time, time.perf_counter() - start

    def new_tracker(self):
        return EventTracker(self.args.conf, self.args.event_timeout)

    def process_images(self, files, title):
        stats = RunStats()
        print(f"Tim thay {len(files)} anh. Bat dau detect...")

        for index, path in enumerate(files, start=1):
            name = source_name_of(path)
            try:
                image = cv2.imread(path)
                if image is None:
                    print(f"[{index}/{len(files)}] Khong doc duoc anh: {name}")
                    continue

                result, detections, infer, total = self.process_frame(
                    image, name, 1, 0.0, self.new_tracker()
                )
                cv2.imwrite(os.path.join(self.args.output, f"result_{name}"), result)
                stats.add(len(detections), infer, total)

                print(
                    f"[{index}/{len(files)}] {name} -> {len(detections)} bien bao | "
                    f"{1.0 / infer if infer > 0 else 0:.2f} FPS"
                )
            except Exception as error:
                print(f"[{index}/{len(files)}] Loi khi xu ly {name}: {error}")

        stats.print_summary(title)

    def open_writer(self, output_path, frame, fps):
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            print(f"Khong tao duoc file video ket qua: {output_path}")
            return None
        return writer

    def process_capture(self, cap, source_name, output_path, fps_source, title):
        tracker = self.new_tracker()
        stats = RunStats()
        start = time.perf_counter()
        frame_number = 0
        writer = None

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_number += 1

                if output_path and writer is None:
                    writer = self.open_writer(output_path, frame, fps_source or 25)
                    if writer is None:
                        output_path = None

                if fps_source:
                    position = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    video_time = (
                        position if position > 0 or frame_number == 1
                        else (frame_number - 1) / fps_source
                    )
                else:
                    video_time = time.perf_counter() - start

                result, detections, infer, total = self.process_frame(
                    frame, source_name, frame_number, video_time, tracker
                )
                stats.add(len(detections), infer, total)

                if writer is not None:
                    writer.write(result)

                if self.show:
                    cv2.imshow(WINDOW_NAME, result)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                elif frame_number % 100 == 0:
                    print(f"  da xu ly {frame_number} frame...")
        except KeyboardInterrupt:
            print("\nDa dung theo yeu cau.")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if self.show:
                cv2.destroyAllWindows()

        stats.print_summary(title)
        return output_path

    def process_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Khong mo duoc video: {path}")
            return

        fps_source = cap.get(cv2.CAP_PROP_FPS)
        if fps_source <= 0:
            fps_source = 25

        name = source_name_of(path)
        output_path = os.path.join(
            self.args.output, f"result_{os.path.splitext(name)[0]}.mp4"
        )

        saved = self.process_capture(cap, name, output_path, fps_source, "TONG KET VIDEO")
        if saved:
            print(f"Video ket qua: {saved}")

    def process_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Khong mo duoc webcam.")
            return
        self.process_capture(cap, "webcam", None, None, "TONG KET WEBCAM")

    def run(self, source):
        source_type = detect_source_type(source)
        print(f"Source type: {source_type}")

        if source_type == "image":
            self.process_images([source], "TONG KET ANH")
        elif source_type == "folder":
            names = sorted(os.listdir(source))
            images = [
                os.path.join(source, f) for f in names
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ]
            videos = [
                os.path.join(source, f) for f in names
                if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
            ]
            if not images and not videos:
                print("Khong tim thay anh hoac video trong thu muc.")
                return

            print(f"Tim thay {len(images)} anh va {len(videos)} video.")
            if images:
                self.process_images(images, "TONG KET THU MUC ANH")
            for index, video in enumerate(videos, start=1):
                print(f"\n[Video {index}/{len(videos)}] {source_name_of(video)}")
                self.process_video(video)
        elif source_type == "video":
            self.process_video(source)
        elif source_type == "webcam":
            self.process_webcam()
        else:
            print("SOURCE khong hop le. Hay dung anh, thu muc, video hoac 'webcam'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic sign detection demo")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="anh, thu muc, video hoac 'webcam'")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--output", default="output")
    parser.add_argument("--event-timeout", type=float, default=1.0,
                        help="so giay bien bao vang mat truoc khi tinh la su kien moi")
    parser.add_argument("--no-show", action="store_true", help="khong mo cua so hien thi")
    parser.add_argument("--fresh", action="store_true", help="xoa log cu, bat dau log moi")
    return parser.parse_args()


def print_config(demo):
    args = demo.args
    names = demo.model.names

    print(f"So class cua model: {len(names)}")
    if len(names) != 18:
        print("CANH BAO: Model hien tai khong phai model 18 class.")

    print("\n===== CAU HINH DEMO =====")
    print(f"RUN_ID: {demo.run_id}")
    print(f"MODEL: {args.model}")
    print(f"SOURCE: {args.source}")
    print(f"CONF: {args.conf} | IOU: {args.iou} | IMGSZ: {args.imgsz} | MAX_DET: {args.max_det}")
    print(f"OUTPUT: {args.output}")
    print("=========================")

    for class_id, class_name in names.items():
        display = CLASS_DISPLAY.get(int(class_id), str(class_name).upper())
        print(f"{int(class_id):02d}: {class_name} -> {display}")
    print("=========================\n")


def main():
    args = parse_args()
    demo = Demo(args)

    try:
        print_config(demo)
        demo.run(args.source)
    finally:
        demo.close()

    print("\n===== HOAN THANH =====")
    print(f"Run ID: {demo.run_id}")
    print(f"Detection log: {os.path.join(args.output, 'detection_log.csv')}")
    print(f"Event log: {os.path.join(args.output, 'event_log.csv')}")
    print(f"Anh crop: {demo.crops_dir}")
    print("=======================")


if __name__ == "__main__":
    main()