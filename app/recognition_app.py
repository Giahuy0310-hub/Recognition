import argparse
import glob
import os
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.detection_engine import CsvLogger, Demo, EventTracker, is_warning

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARDCASE_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "traffic_sign_18class_hardcase"
    / "weights"
    / "best.pt"
)

IMG_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VID_TYPES = ["mp4", "avi", "mov", "mkv", "wmv"]

MODE_IMAGE = "📷 Ảnh"
MODE_VIDEO = "🎥 Video"
MODE_WEBCAM = "📹 Webcam"

NUM_CLASSES = 18
IOU = 0.7
MAX_DET = 300
EVENT_TIMEOUT = 1.0
BOX_KEYS = ("x1", "y1", "x2", "y2")
VIDEO_TITLE = "YOLOv8 Traffic Sign Detection - Hardcase"

CSS = """
<style>
.block-container { padding-top: 2rem; }

.hero {
    padding: 22px 28px; border-radius: 18px; margin-bottom: 20px;
    background: linear-gradient(120deg, #4F8BF9 0%, #7B5CFA 100%); color: white;
}
.hero h1 { margin: 0; font-size: 30px; font-weight: 700; color: white; }
.hero p  { margin: 4px 0 0 0; opacity: .85; font-size: 14px; }

.kpi-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-top: 12px; }
.kpi {
    padding: 14px 16px; border-radius: 14px;
    background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.18);
}
.kpi .label { font-size: 13px; opacity: .7; }
.kpi .value { font-size: 24px; font-weight: 700; }

.feed-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; margin-bottom: 6px; border-radius: 10px;
    background: rgba(128,128,128,.08);
}
.feed-item .name { font-weight: 600; }

.badge { padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.badge.hi  { background: rgba(46,204,113,.18); color: #2ecc71; }
.badge.mid { background: rgba(241,196,15,.18); color: #f1c40f; }
.badge.lo  { background: rgba(231,76,60,.18);  color: #e74c3c; }

.muted { opacity: .55; padding: 8px 2px; }
</style>
"""


class Kpi(NamedTuple):
    fps: float
    objects: int
    infer_ms: float
    size: str
    conf: float


@dataclass
class Settings:
    mode: str
    uploads: object
    model_path: str
    conf: float
    imgsz: int
    step: int
    log_enabled: bool
    start: bool


@st.cache_resource(show_spinner="Đang load model...")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def clamp_box(det, frame_shape):
    h, w = frame_shape[:2]
    x1, x2 = (min(max(int(det[k]), 0), w) for k in ("x1", "x2"))
    y1, y2 = (min(max(int(det[k]), 0), h) for k in ("y1", "y2"))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


class NullLogger:
    def log_frame(self, *args, **kwargs):
        pass

    def close(self):
        pass


class WebDemo(Demo):

    def __init__(self, model, args, log_enabled):
        self.model = model
        self.args = args
        self.show = False
        self.log_enabled = log_enabled
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.crops_dir = os.path.join(args.output, "crops")
        os.makedirs(self.crops_dir, exist_ok=True)

        self.logger = (
            CsvLogger(args.output, self.run_id, False) if log_enabled else NullLogger()
        )

        self.model.predictor = None

    def _parse(self, result):
        detections = []
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": cls,
                "class_name": self.model.names[cls],
                "confidence": float(box.conf[0]),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        return detections

    def _attach_boxes_and_crops(self, events, detections, frame, source_name, frame_number):
        for event in events:
            match = next(
                (
                    d for d in detections
                    if d["class_id"] == event.get("class_id")
                    and abs(d["confidence"] - event.get("confidence", 0)) < 0.1
                ),
                None,
            )
            if match:
                event.update(clamp_box(match, frame.shape))

            if all(k in event for k in BOX_KEYS):
                event["crop_path"] = self.save_crop(frame, event, source_name, frame_number)

    def step(self, frame, source_name, frame_number, video_time, tracker):
        t0 = time.perf_counter()

        result = self.model.predict(
            frame,
            conf=self.args.conf,
            imgsz=self.args.imgsz,
            iou=self.args.iou,
            max_det=self.args.max_det,
            verbose=False,
        )[0]
        infer_time = time.perf_counter() - t0

        detections = self._parse(result)
        events = tracker.update(detections, video_time)

        if self.log_enabled:
            self._attach_boxes_and_crops(events, detections, frame, source_name, frame_number)

        self.logger.log_frame(source_name, frame_number, video_time, detections, events)

        annotated = result.plot()
        return annotated, detections, infer_time, time.perf_counter() - t0


def draw_overlay(frame, fps):
    height = frame.shape[0]
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, VIDEO_TITLE, (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def conf_badge(conf: float) -> str:
    level = "hi" if conf >= 0.75 else "mid" if conf >= 0.5 else "lo"
    return f'<span class="badge {level}">{conf:.0%}</span>'


def render_detections(target, detections):
    if not detections:
        target.markdown(
            '<div class="muted">Chưa phát hiện biển báo nào.</div>',
            unsafe_allow_html=True,
        )
        return

    rows = "".join(
        f"""
        <div class="feed-item">
            <div class="name">{"🔴" if is_warning(d["class_id"]) else "🟢"} {d["class_name"]}</div>
            {conf_badge(d["confidence"])}
        </div>
        """
        for d in sorted(detections, key=lambda d: d["confidence"], reverse=True)
    )
    target.markdown(rows, unsafe_allow_html=True)


def render_kpis(target, kpi: Kpi):
    items = [
        ("⚡ FPS", f"{kpi.fps:.1f}"),
        ("🎯 Objects", f"{kpi.objects}"),
        ("⏱️ Inference", f"{kpi.infer_ms:.0f} ms"),
        ("📐 Image size", kpi.size),
        ("🎚️ Confidence", f"{kpi.conf:.2f}"),
    ]
    html = "".join(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in items
    )
    target.markdown(f'<div class="kpi-row">{html}</div>', unsafe_allow_html=True)


def render_results():
    summary = st.session_state.get("summary")
    if summary:
        st.subheader("📊 Thống kê phát hiện")
        df = pd.DataFrame(
            sorted(summary.items(), key=lambda kv: kv[1], reverse=True),
            columns=["Class", "Số lượng"],
        )
        st.dataframe(df, hide_index=True, width="stretch")

    video = st.session_state.get("video_out")
    if video and os.path.exists(video):
        st.download_button(
            "⬇️ Tải video kết quả",
            data=Path(video).read_bytes(),
            file_name=Path(video).name,
            mime="video/mp4",
        )
        st.caption(f"Đã lưu tại: {video}")


class Panel:

    def __init__(self):
        self.metrics = st.empty()
        self.frame = None
        self.dets = None

    def new_view(self):
        left, right = st.columns([3, 2])
        self.frame = left.empty()
        right.subheader("Detections")
        self.dets = right.empty()

    def update(self, frame, detections, kpi: Kpi):
        self.frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
        render_detections(self.dets, detections)
        render_kpis(self.metrics, kpi)
        st.session_state["last"] = {"frame": frame, "dets": detections, "kpi": kpi}


def build_sidebar() -> Settings:
    st.sidebar.header("🎥 Nguồn đầu vào")
    mode = st.sidebar.radio(
        "Nguồn", [MODE_IMAGE, MODE_VIDEO, MODE_WEBCAM], label_visibility="collapsed"
    )

    uploads = None
    if mode == MODE_IMAGE:
        uploads = st.sidebar.file_uploader(
            "Chọn ảnh", type=IMG_TYPES, accept_multiple_files=True
        )
    elif mode == MODE_VIDEO:
        uploads = st.sidebar.file_uploader("Chọn video", type=VID_TYPES)
    else:
        st.sidebar.caption("Webcam chỉ hoạt động khi chạy trực tiếp trên máy.")

    st.sidebar.header("🧠 Model")
    models = sorted(glob.glob("models/*.pt"))
    if HARDCASE_MODEL not in models:
        models.append(HARDCASE_MODEL)

    choice = st.sidebar.selectbox(
        "Model", models + ["Khác…"], index=models.index(HARDCASE_MODEL)
    )
    model_path = (
        st.sidebar.text_input("Đường dẫn .pt", HARDCASE_MODEL)
        if choice == "Khác…"
        else choice
    )

    st.sidebar.header("⚙️ Tham số")
    conf = st.sidebar.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
    imgsz = st.sidebar.select_slider("Image size", [320, 416, 640, 960, 1280], value=640)

    step = 1
    if mode != MODE_IMAGE:
        step = st.sidebar.slider("Xử lý mỗi N khung hình", 1, 6, 1)

    log_enabled = st.sidebar.checkbox("Ghi vào log (Dashboard)", value=True)

    col_start, col_stop = st.sidebar.columns(2)
    start = col_start.button("▶ Bắt đầu", type="primary", width="stretch")
    col_stop.button("⏹ Dừng", width="stretch")

    return Settings(mode, uploads, model_path, conf, imgsz, step, log_enabled, start)


def render_header(settings: Settings):
    st.markdown(
        """
        <div class="hero">
            <h1>🎥 Nhận diện biển báo</h1>
            <p>YOLO Traffic Sign Detection - Hardcase Model</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Model: `{settings.model_path}` | "
        f"Confidence: {settings.conf:.2f} | "
        f"Image size: {settings.imgsz}"
    )


def remove_quietly(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            pass


def process_images(demo: WebDemo, settings: Settings, panel: Panel) -> Counter:
    summary = Counter()

    for upload in settings.uploads:
        img = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        tracker = EventTracker(settings.conf, EVENT_TIMEOUT)
        result, dets, infer, total = demo.step(img, upload.name, 1, 0.0, tracker)

        summary.update(d["class_name"] for d in dets)

        panel.new_view()
        panel.update(
            result, dets,
            Kpi(1 / total if total else 0, len(dets), infer * 1000, f"{w}x{h}", settings.conf),
        )

    return summary


def open_source(settings: Settings):
    if settings.mode == MODE_WEBCAM:
        return cv2.VideoCapture(0), "webcam", None

    upload = settings.uploads
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as tmp:
        tmp.write(upload.getbuffer())

    return cv2.VideoCapture(tmp.name), upload.name, tmp.name


def process_stream(demo: WebDemo, settings: Settings, panel: Panel) -> Counter:
    is_video = settings.mode == MODE_VIDEO
    cap, name, tmp_path = open_source(settings)
    writer = None
    out_path = None
    summary = Counter()

    try:
        if not cap.isOpened():
            st.error("Không mở được nguồn.")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if is_video else 0

        if is_video:
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
            out_path = Path(OUTPUT_DIR) / f"{Path(name).stem}_{demo.run_id}_result.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps_src / settings.step,
                (width, height),
            )

        panel.new_view()
        progress = st.progress(0.0) if n_total > 0 else None
        tracker = EventTracker(settings.conf, EVENT_TIMEOUT)

        t_start = time.perf_counter()
        frame_id = processed = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1
            if (frame_id - 1) % settings.step:
                continue

            video_time = (
                cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if is_video
                else time.perf_counter() - t_start
            )

            annotated, dets, infer, _ = demo.step(frame, name, frame_id, video_time, tracker)

            processed += 1
            summary.update(d["class_name"] for d in dets)

            fps = processed / (time.perf_counter() - t_start)
            annotated = draw_overlay(annotated, fps)

            if writer:
                writer.write(annotated)

            panel.update(
                annotated, dets,
                Kpi(fps, len(dets), infer * 1000, f"{width}x{height}", settings.conf),
            )

            if progress:
                progress.progress(min(1.0, frame_id / n_total))

    finally:
        cap.release()
        if writer:
            writer.release()
        remove_quietly(tmp_path)

    elapsed = time.perf_counter() - t_start
    st.success(
        f"Hoàn thành: {processed} frame | "
        f"{sum(summary.values())} detections | "
        f"{processed / elapsed:.1f} FPS"
    )
    st.session_state["video_out"] = str(out_path) if out_path else None

    return summary


def main():
    st.set_page_config(page_title="Nhận diện biển báo", page_icon="🎥", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    settings = build_sidebar()
    render_header(settings)

    if not os.path.exists(settings.model_path):
        st.error(f"Không tìm thấy model: {settings.model_path}")
        st.stop()

    model = load_model(settings.model_path)
    if len(model.names) != NUM_CLASSES:
        st.warning(f"Model có {len(model.names)} class (kỳ vọng {NUM_CLASSES})")

    panel = Panel()

    if not settings.start:
        last = st.session_state.get("last")
        if last:
            panel.new_view()
            panel.update(last["frame"], last["dets"], last["kpi"])
            render_results()
        else:
            st.info("Chọn nguồn và bấm Bắt đầu.")
        st.stop()

    if settings.mode != MODE_WEBCAM and not settings.uploads:
        st.warning("Hãy chọn ảnh." if settings.mode == MODE_IMAGE else "Hãy chọn video.")
        st.stop()

    args = argparse.Namespace(
        conf=settings.conf,
        iou=IOU,
        imgsz=settings.imgsz,
        max_det=MAX_DET,
        output=OUTPUT_DIR,
        event_timeout=EVENT_TIMEOUT,
    )
    demo = WebDemo(model, args, settings.log_enabled)

    st.session_state["summary"] = {}
    st.session_state["video_out"] = None

    try:
        if settings.mode == MODE_IMAGE:
            summary = process_images(demo, settings, panel)
        else:
            summary = process_stream(demo, settings, panel)
        st.session_state["summary"] = dict(summary)
    finally:
        demo.logger.close()

    render_results()

    if settings.log_enabled:
        st.caption(f"Run ID: {demo.run_id} | Log: {OUTPUT_DIR}/detection_log.csv")


main()
