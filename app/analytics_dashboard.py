import inspect
import os
import re
import time

import pandas as pd
import plotly.express as px
import streamlit as st

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
DETECTION_LOG = os.path.join(OUTPUT_DIR, "detection_log.csv")
EVENT_LOG = os.path.join(OUTPUT_DIR, "event_log.csv")

IMAGE_COLS = ("crop_path", "image_path", "frame_path", "image", "path")
PALETTE = [
    "#4F8BF9", "#F9A24F", "#3DD6A3", "#F95F7E", "#A57BFF", "#F9DB4F",
    "#4FD1F9", "#F97BC7", "#8BD64F", "#B0B7C3",
]

RESTRICTED_NAMES = {
    n.upper()
    for n in [
        "CAM DO XE",
        "CAM DUNG/DO XE",
        "CAM DI NGUOC CHIEU",
        "DI CHAM",
        "TOC DO TOI DA 50",
        "TOC DO TOI DA 60",
        "TOC DO TOI DA 80",
        "CAM O TO",
        "CAM RE PHAI",
        "CAM RE TRAI",
        "CAM O TO RE PHAI",
        "CAM O TO RE TRAI",
    ]
}

st.set_page_config(page_title="Traffic Sign Dashboard", page_icon="ðŸš¦", layout="wide")

_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
_DF_HAS_WIDTH = "width" in inspect.signature(st.dataframe).parameters


PLOT_CONFIG = {"displayModeBar": False}


def show_plot(fig):
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    else:
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def show_df(df, **kwargs):
    if _DF_HAS_WIDTH:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def style_fig(fig, height=420, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        font=dict(size=13),
        colorway=PALETTE,
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False)
    return fig


st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .hero {
        padding: 22px 28px; border-radius: 18px; margin-bottom: 20px;
        background: linear-gradient(120deg, #4F8BF9 0%, #7B5CFA 100%);
        color: white;
    }
    .hero h1 {margin: 0; font-size: 30px; font-weight: 700; color: white;}
    .hero p {margin: 4px 0 0 0; opacity: .85; font-size: 14px;}
    .kpi {
        padding: 16px 18px; border-radius: 14px;
        background: rgba(128,128,128,.10);
        border: 1px solid rgba(128,128,128,.18);
        transition: transform .15s ease, border-color .15s ease;
    }
    .kpi:hover {transform: translateY(-2px); border-color: #4F8BF9;}
    .kpi .label {font-size: 13px; opacity: .7;}
    .kpi .value {font-size: 28px; font-weight: 700; line-height: 1.3;}
    .kpi .delta {font-size: 12px; font-weight: 600;}
    .kpi .delta.up {color: #2ecc71;}
    .kpi .delta.down {color: #e74c3c;}
    .kpi .delta.flat {opacity: .5;}
    .feed-item {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 14px; margin-bottom: 6px; border-radius: 10px;
        background: rgba(128,128,128,.08);
    }
    .feed-item .name {font-weight: 600;}
    .feed-item .meta {font-size: 12px; opacity: .6;}
    .badge {padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 700;}
    .badge.hi {background: rgba(46,204,113,.18); color: #2ecc71;}
    .badge.mid {background: rgba(241,196,15,.18); color: #f1c40f;}
    .badge.lo {background: rgba(231,76,60,.18); color: #e74c3c;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _read_csv(path, mtime, size):
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    stat = os.stat(path)
    return _read_csv(path, stat.st_mtime, stat.st_size).copy()


def prepare_detections(df):
    if df.empty or "class_name" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["class_name"]).copy()
    df["class_name"] = df["class_name"].astype(str).str.strip()
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def warning_mask(df):
    if "is_warning" in df.columns:
        return pd.to_numeric(df["is_warning"], errors="coerce").fillna(0).astype(bool)
    return df["class_name"].str.upper().isin(RESTRICTED_NAMES)


def resolve_path(p):
    if pd.isna(p) or not str(p).strip():
        return None
    p = str(p)
    if os.path.exists(p):
        return p
    alt = os.path.join(OUTPUT_DIR, p)
    return alt if os.path.exists(alt) else None


def kpi(col, icon, label, value, delta=None):
    delta_html = ""
    if delta is not None:
        if abs(delta) < 0.005:
            delta_html = '<div class="delta flat">â€” khÃ´ng Ä‘á»•i so vá»›i ká»³ trÆ°á»›c</div>'
        else:
            cls = "up" if delta > 0 else "down"
            arrow = "â–²" if delta > 0 else "â–¼"
            delta_html = f'<div class="delta {cls}">{arrow} {abs(delta):.0%} so vá»›i ká»³ trÆ°á»›c</div>'
    col.markdown(
        f'<div class="kpi"><div class="label">{icon} {label}</div>'
        f'<div class="value">{value}</div>{delta_html}</div>',
        unsafe_allow_html=True,
    )


def short_source(name, limit=42):
    name = re.sub(r"\.rf\.[0-9a-fA-F]{16,}", "", str(name))
    if len(name) <= limit:
        return name
    return name[: limit - 15] + "â€¦" + name[-14:]


def render_feed(df):
    if "timestamp" in df.columns:
        latest = df.sort_values("timestamp", ascending=False).head(6)
    else:
        latest = df.tail(6).iloc[::-1]
    rows = []
    for _, r in latest.iterrows():
        t = ""
        if "timestamp" in latest.columns and pd.notna(r["timestamp"]):
            t = f"{r['timestamp']:%H:%M:%S}"
        src = f" Â· {short_source(r['source'])}" if has_source and pd.notna(r["source"]) else ""
        conf = conf_badge(r["confidence"]) if has_conf else ""
        rows.append(
            f'<div class="feed-item"><div><div class="name">{r["class_name"]}</div>'
            f'<div class="meta">{t}{src}</div></div>{conf}</div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def conf_badge(c):
    if pd.isna(c):
        return ""
    cls = "hi" if c >= 0.75 else "mid" if c >= 0.5 else "lo"
    return f'<span class="badge {cls}">{c:.0%}</span>'


st.sidebar.header("âš™ï¸ CÃ i Ä‘áº·t")
auto_refresh = st.sidebar.toggle("Tá»± Ä‘á»™ng lÃ m má»›i", value=False)
refresh_secs = st.sidebar.slider("Chu ká»³ (giÃ¢y)", 2, 60, 5, disabled=not auto_refresh)
if st.sidebar.button("ðŸ”„ LÃ m má»›i ngay"):
    st.cache_data.clear()
    st.rerun()


def maybe_autorefresh():
    if auto_refresh:
        time.sleep(refresh_secs)
        st.rerun()


detections = prepare_detections(load_csv(DETECTION_LOG))
events = load_csv(EVENT_LOG)

if detections.empty:
    st.markdown(
        '<div class="hero"><h1>ðŸš¦ Traffic Sign Detection Dashboard</h1>'
        "<p>YOLO 18-class traffic sign recognition</p></div>",
        unsafe_allow_html=True,
    )
    st.warning(
        "ChÆ°a cÃ³ dá»¯ liá»‡u trong `output/detection_log.csv` "
        "(hoáº·c file thiáº¿u cá»™t `class_name`). HÃ£y cháº¡y `recognition_app.py` trÆ°á»›c."
    )
    maybe_autorefresh()
    st.stop()

has_conf = "confidence" in detections.columns
has_source = "source" in detections.columns
has_time = "timestamp" in detections.columns and detections["timestamp"].notna().any()

st.sidebar.header("ðŸ”Ž Bá»™ lá»c")

if "run_id" in detections.columns:
    runs = sorted(detections["run_id"].dropna().astype(str).unique().tolist(), reverse=True)
    run_choice = st.sidebar.selectbox(
        "Láº§n cháº¡y", ["Táº¥t cáº£"] + runs, index=1 if runs else 0
    )
    if run_choice != "Táº¥t cáº£":
        detections = detections[detections["run_id"].astype(str) == run_choice]
        if "run_id" in events.columns:
            events = events[events["run_id"].astype(str) == run_choice]

classes = sorted(detections["class_name"].unique().tolist())
selected_classes = st.sidebar.multiselect("Loáº¡i biá»ƒn bÃ¡o (trá»‘ng = táº¥t cáº£)", classes)

min_conf = st.sidebar.slider("Confidence tá»‘i thiá»ƒu", 0.0, 1.0, 0.30, 0.05) if has_conf else 0.0

selected_sources = []
if has_source:
    sources = sorted(detections["source"].dropna().astype(str).unique().tolist())
    selected_sources = st.sidebar.multiselect("Nguá»“n (trá»‘ng = táº¥t cáº£)", sources)

start = end = None
tmax_ts = None
if has_time:
    tmax_ts = detections["timestamp"].max()
    tmin_ts = detections["timestamp"].min()
    preset = st.sidebar.radio(
        "Khoáº£ng thá»i gian (tÃ­nh tá»« báº£n ghi má»›i nháº¥t)",
        ["Táº¥t cáº£", "15 phÃºt qua", "1 giá» qua", "HÃ´m nay", "7 ngÃ y qua", "Tuá»³ chá»n"],
    )
    if preset == "15 phÃºt qua":
        start = tmax_ts - pd.Timedelta(minutes=15)
    elif preset == "1 giá» qua":
        start = tmax_ts - pd.Timedelta(hours=1)
    elif preset == "HÃ´m nay":
        start = tmax_ts.normalize()
    elif preset == "7 ngÃ y qua":
        start = tmax_ts - pd.Timedelta(days=7)
    elif preset == "Tuá»³ chá»n":
        picked = st.sidebar.date_input(
            "Chá»n ngÃ y",
            value=(tmin_ts.date(), tmax_ts.date()),
            min_value=tmin_ts.date(),
            max_value=tmax_ts.date(),
        )
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            start = pd.Timestamp(picked[0])
            end = pd.Timestamp(picked[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

base = detections
if selected_classes:
    base = base[base["class_name"].isin(selected_classes)]
if has_conf:
    base = base[base["confidence"].fillna(0) >= min_conf]
if has_source and selected_sources:
    base = base[base["source"].astype(str).isin(selected_sources)]

filtered = base
prev = None
if start is not None:
    ts = base["timestamp"]
    mask = ts >= start
    if end is not None:
        mask &= ts <= end
    filtered = base[mask]
    window = (end or tmax_ts) - start
    prev = base[(ts >= start - window) & (ts < start)]

subtitle = f"{len(detections):,} báº£n ghi"
if has_time:
    subtitle += f" Â· {tmin_ts:%d/%m/%Y %H:%M} â†’ {tmax_ts:%d/%m/%Y %H:%M}"
st.markdown(
    f'<div class="hero"><h1>ðŸš¦ Traffic Sign Detection Dashboard</h1>'
    f"<p>YOLO 18-class traffic sign recognition Â· {subtitle}</p></div>",
    unsafe_allow_html=True,
)

total = len(filtered)
avg_conf = filtered["confidence"].mean() if has_conf and total else 0.0
restricted_count = int(warning_mask(filtered).sum())
n_sources = filtered["source"].nunique() if has_source else 0
delta_total = None
if prev is not None and len(prev) > 0:
    delta_total = (total - len(prev)) / len(prev)

c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "ðŸ”", "Tá»•ng phÃ¡t hiá»‡n", f"{total:,}", delta_total)
kpi(c2, "ðŸš¦", "Loáº¡i biá»ƒn bÃ¡o", f"{filtered['class_name'].nunique()}")
kpi(c3, "ðŸ–¼ï¸", "Nguá»“n Ä‘Ã£ xá»­ lÃ½", f"{n_sources:,}")
kpi(c4, "âš ï¸", "Biá»ƒn cáº¥m/giá»›i háº¡n", f"{restricted_count:,}")
kpi(c5, "ðŸŽ¯", "Confidence TB", f"{avg_conf * 100:.1f}%")

st.write("")

if filtered.empty:
    st.info("KhÃ´ng cÃ³ dá»¯ liá»‡u phÃ¹ há»£p vá»›i bá»™ lá»c.")
    maybe_autorefresh()
    st.stop()

if has_conf:
    low_share = (filtered["confidence"] < 0.5).mean()
    if low_share > 0.25:
        st.warning(
            f"{low_share:.0%} sá»‘ phÃ¡t hiá»‡n cÃ³ confidence dÆ°á»›i 50%. "
            "NÃªn kiá»ƒm tra cháº¥t lÆ°á»£ng áº£nh nguá»“n hoáº·c bá»• sung dá»¯ liá»‡u huáº¥n luyá»‡n."
        )

img_col = next((c for c in IMAGE_COLS if c in filtered.columns), None)

labels = ["ðŸ“Š Tá»•ng quan", "â±ï¸ Thá»i gian", "ðŸ–¼ï¸ Nguá»“n", "ðŸ“‹ Sá»± kiá»‡n", "ðŸ”Ž Dá»¯ liá»‡u"]
if img_col:
    labels.insert(2, "ðŸ“· HÃ¬nh áº£nh")
tabs = dict(zip(labels, st.tabs(labels)))

with tabs["ðŸ“Š Tá»•ng quan"]:
    max_n = filtered["class_name"].nunique()
    top_n = st.slider("Hiá»ƒn thá»‹ top N loáº¡i biá»ƒn", 3, max_n, min(10, max_n)) if max_n > 3 else max_n

    left, right = st.columns([3, 2])
    chart_h = max(340, 34 * min(top_n, max_n))

    with left:
        st.subheader("Sá»‘ láº§n phÃ¡t hiá»‡n theo loáº¡i")
        counts = (
            filtered["class_name"]
            .value_counts()
            .head(top_n)
            .rename_axis("class_name")
            .reset_index(name="count")
            .sort_values("count")
        )
        fig = px.bar(
            counts, x="count", y="class_name", orientation="h", text="count",
            color_discrete_sequence=[PALETTE[0]],
        )
        fig.update_traces(textposition="outside", cliponaxis=False, marker_cornerradius=6)
        fig.update_layout(xaxis_title="", yaxis_title="")
        show_plot(style_fig(fig, height=chart_h, legend=False))

    with right:
        st.subheader("CÆ¡ cáº¥u nhÃ³m biá»ƒn")
        is_restricted = warning_mask(filtered)
        split = pd.DataFrame(
            {
                "NhÃ³m": ["Cáº¥m / giá»›i háº¡n", "KhÃ¡c"],
                "Sá»‘ lÆ°á»£ng": [int(is_restricted.sum()), int((~is_restricted).sum())],
            }
        )
        fig = px.pie(split, names="NhÃ³m", values="Sá»‘ lÆ°á»£ng", hole=0.6,
                     color_discrete_sequence=[PALETTE[3], PALETTE[0]])
        fig.update_traces(
            textinfo="percent+value",
            textfont=dict(size=14, color="white"),
            marker=dict(line=dict(width=2, color="rgba(0,0,0,0)")),
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0.5, xanchor="center"))
        show_plot(style_fig(fig, height=chart_h))

    if has_conf:
        feed_col, hist_col = st.columns(2)
        with feed_col:
            st.subheader("Má»›i nháº¥t")
            render_feed(filtered)
        with hist_col:
            st.subheader("PhÃ¢n bá»‘ Confidence")
            fig = px.histogram(filtered, x="confidence", nbins=20, range_x=[0, 1],
                               color_discrete_sequence=[PALETTE[4]])
            fig.update_traces(marker_line_width=0)
            fig.update_layout(xaxis_title="Confidence", yaxis_title="Sá»‘ detection", bargap=0.08)
            show_plot(style_fig(fig, height=420, legend=False))

        st.subheader("Confidence trung bÃ¬nh theo loáº¡i")
        per_class = (
            filtered.groupby("class_name")["confidence"]
            .agg(["count", "mean", "min"])
            .reset_index()
            .sort_values("mean")
        )
        fig = px.bar(
            per_class, x="mean", y="class_name", orientation="h",
            color="mean", color_continuous_scale="RdYlGn", range_color=[0.3, 1.0],
            hover_data={"count": True, "min": ":.2f", "mean": ":.2f"},
        )
        fig.update_layout(xaxis_title="", yaxis_title="", xaxis_range=[0, 1],
                          coloraxis_showscale=False)
        show_plot(style_fig(fig, height=max(340, 28 * len(per_class)), legend=False))
    else:
        st.subheader("Má»›i nháº¥t")
        render_feed(filtered)

with tabs["â±ï¸ Thá»i gian"]:
    has_vt = "video_time_s" in filtered.columns and pd.to_numeric(
        filtered["video_time_s"], errors="coerce"
    ).notna().any()
    axis = (
        st.radio("Trá»¥c thá»i gian", ["Giá» cháº¡y", "Thá»i gian video"], horizontal=True)
        if has_vt
        else "Giá» cháº¡y"
    )

    if axis == "Thá»i gian video":
        vdata = filtered.assign(
            video_time_s=pd.to_numeric(filtered["video_time_s"], errors="coerce")
        ).dropna(subset=["video_time_s"])
        bin_s = st.select_slider("Gá»™p theo (giÃ¢y)", [1, 2, 5, 10, 30, 60], value=5)
        vdata = vdata.assign(bin=(vdata["video_time_s"] // bin_s) * bin_s)
        by_video = vdata.groupby(["bin", "class_name"]).size().reset_index(name="count")
        fig = px.bar(by_video, x="bin", y="count", color="class_name",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(xaxis_title="GiÃ¢y trong video", yaxis_title="Sá»‘ detection",
                          legend_title="Loáº¡i biá»ƒn")
        show_plot(style_fig(fig, height=440))
    else:
        tdata = filtered.dropna(subset=["timestamp"]) if "timestamp" in filtered.columns else pd.DataFrame()
        if tdata.empty:
            st.info("KhÃ´ng cÃ³ dá»¯ liá»‡u timestamp há»£p lá»‡.")
        else:
            options = {"Tá»± Ä‘á»™ng": None, "10 giÃ¢y": "10s", "1 phÃºt": "1min",
                       "5 phÃºt": "5min", "1 giá»": "1h", "1 ngÃ y": "1D"}
            choice = st.radio("Gá»™p theo", list(options), horizontal=True)
            freq = options[choice]
            if freq is None:
                span = (tdata["timestamp"].max() - tdata["timestamp"].min()).total_seconds()
                freq = ("10s" if span <= 900 else "1min" if span <= 10800
                        else "5min" if span <= 86400 else "1h" if span <= 7 * 86400 else "1D")

            by_time = (
                tdata.groupby([pd.Grouper(key="timestamp", freq=freq), "class_name"])
                .size()
                .reset_index(name="count")
            )
            fig = px.bar(by_time, x="timestamp", y="count", color="class_name",
                         color_discrete_sequence=PALETTE)
            fig.update_layout(xaxis_title="", yaxis_title="Sá»‘ detection", legend_title="Loáº¡i biá»ƒn")
            show_plot(style_fig(fig, height=440))

            st.subheader("Máº­t Ä‘á»™ theo giá» trong ngÃ y")
            heat = tdata.assign(
                day=tdata["timestamp"].dt.date.astype(str),
                hour=tdata["timestamp"].dt.hour,
            ).groupby(["day", "hour"]).size().reset_index(name="count")
            fig = px.density_heatmap(heat, x="hour", y="day", z="count", nbinsx=24,
                                     color_continuous_scale="Blues")
            fig.update_layout(xaxis_title="Giá»", yaxis_title="")
            show_plot(style_fig(fig, height=320, legend=False))

if img_col:
    with tabs["ðŸ“· HÃ¬nh áº£nh"]:
        gallery = filtered.copy()
        gallery["_resolved"] = gallery[img_col].map(resolve_path)
        gallery = gallery.dropna(subset=["_resolved"])
        if "timestamp" in gallery.columns:
            gallery = gallery.sort_values("timestamp", ascending=False)
        if gallery.empty:
            st.info("KhÃ´ng tÃ¬m tháº¥y file áº£nh nÃ o trong cá»™t nÃ y.")
        else:
            n_show = st.select_slider("Sá»‘ áº£nh hiá»ƒn thá»‹", [4, 8, 12, 16, 24, 32], value=12)
            cols = st.columns(4)
            for i, (_, r) in enumerate(gallery.head(n_show).iterrows()):
                caption = r["class_name"]
                if has_conf and pd.notna(r["confidence"]):
                    caption += f" Â· {r['confidence']:.0%}"
                cols[i % 4].image(r["_resolved"], caption=caption)

with tabs["ðŸ–¼ï¸ Nguá»“n"]:
    if not has_source:
        st.info("detection_log.csv khÃ´ng cÃ³ cá»™t `source`.")
    else:
        agg = {"class_name": ["size", "nunique"]}
        if has_conf:
            agg["confidence"] = "mean"
        if "timestamp" in filtered.columns:
            agg["timestamp"] = ["min", "max"]
        src = filtered.groupby("source").agg(agg)
        src.columns = ["_".join(c).strip("_") for c in src.columns]
        src = src.reset_index().rename(columns={
            "class_name_size": "Sá»‘ detection",
            "class_name_nunique": "Sá»‘ loáº¡i biá»ƒn",
            "confidence_mean": "Confidence TB",
            "timestamp_min": "Láº§n Ä‘áº§u",
            "timestamp_max": "Láº§n cuá»‘i",
        }).sort_values("Sá»‘ detection", ascending=False)
        config = None
        if "Confidence TB" in src.columns:
            config = {"Confidence TB": st.column_config.ProgressColumn(
                "Confidence TB", min_value=0.0, max_value=1.0, format="%.2f")}
        show_df(src, column_config=config)

with tabs["ðŸ“‹ Sá»± kiá»‡n"]:
    if events.empty:
        st.info("ChÆ°a cÃ³ `event_log.csv` hoáº·c chÆ°a ghi nháº­n sá»± kiá»‡n.")
    else:
        ev = events.copy()
        if "timestamp" in ev.columns:
            ev["timestamp"] = pd.to_datetime(ev["timestamp"], errors="coerce")
            ev = ev.sort_values("timestamp", ascending=False)

        f1, f2 = st.columns([2, 3])
        type_col = next((c for c in ("event", "event_type", "type") if c in ev.columns), None)
        if type_col:
            types = sorted(ev[type_col].dropna().astype(str).unique().tolist())
            chosen = f1.multiselect(f"Lá»c theo `{type_col}`", types)
            if chosen:
                ev = ev[ev[type_col].astype(str).isin(chosen)]
        keyword = f2.text_input("TÃ¬m kiáº¿m", placeholder="Nháº­p tá»« khoÃ¡...")
        if keyword:
            mask = ev.astype(str).apply(lambda col: col.str.contains(keyword, case=False, na=False)).any(axis=1)
            ev = ev[mask]

        st.caption(f"{len(ev):,} sá»± kiá»‡n")
        show_df(ev)
        st.download_button(
            "â¬‡ï¸ Táº£i sá»± kiá»‡n (CSV)",
            data=ev.to_csv(index=False).encode("utf-8-sig"),
            file_name="events_filtered.csv",
            mime="text/csv",
        )

with tabs["ðŸ”Ž Dá»¯ liá»‡u"]:
    view = filtered.sort_values("timestamp", ascending=False) if "timestamp" in filtered.columns else filtered
    st.caption(f"{len(view):,} dÃ²ng sau khi lá»c")
    show_df(view)
    st.download_button(
        "â¬‡ï¸ Táº£i CSV (Ä‘Ã£ lá»c)",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name="detections_filtered.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Traffic Sign Recognition â€” YOLO 18 classes | "
    f"Nguá»“n dá»¯ liá»‡u: {DETECTION_LOG}, {EVENT_LOG}"
)

maybe_autorefresh()
