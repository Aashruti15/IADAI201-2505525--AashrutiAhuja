"""
app.py
------
ParkVision AI - Streamlit web app (v2 - polished UI).

Upload a parking lot image -> the app detects every parking slot,
marks it green (empty) or red (occupied), shows availability metrics,
charts, and a recommendation.

HOW TO RUN LOCALLY:
    streamlit run app.py

REQUIRES:
    - best.pt (your trained model) sitting in this same folder.
    - requirements.txt libraries installed.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from insights import compute_insights

MODEL_PATH = "best.pt"

st.set_page_config(
    page_title="ParkVision AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Light custom styling ----------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9a9a9a;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(120,120,120,0.08);
        padding: 12px 16px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def classify_label(label_name: str) -> str:
    """
    Maps whatever your dataset calls its classes (e.g. 'space-occupied',
    'occupied', 'car') into a simple 'occupied' or 'empty' bucket.
    """
    name = label_name.lower()
    if "occ" in name or "car" in name or "busy" in name:
        return "occupied"
    return "empty"


def draw_boxes(image_np, results, model_names):
    """Draws green boxes on empty slots and red boxes on occupied slots."""
    occupied_count = 0
    empty_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label_name = model_names[cls_id]
        status = classify_label(label_name)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if status == "occupied":
            color = (0, 0, 255)  # red (BGR)
            occupied_count += 1
        else:
            color = (0, 200, 0)  # green
            empty_count += 1

        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

    return image_np, occupied_count, empty_count


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🚗 ParkVision AI")
        st.markdown(
            "Intelligent parking analytics that detects free and "
            "occupied slots from a single image using a custom-trained "
            "YOLOv8 object detection model."
        )

        st.markdown("---")
        st.markdown("### How to use")
        st.markdown(
            "1. Upload a parking lot photo (aerial / elevated view works best)\n"
            "2. The model scans it and marks every slot\n"
            "3. 🟩 Green = empty  🟥 Red = occupied\n"
            "4. Check the summary panel for occupancy stats and a recommendation"
        )

        st.markdown("---")
        st.markdown("### Model info")
        st.markdown(
            "- **Architecture:** YOLOv8n (object detection)\n"
            "- **Dataset:** PKLot (parking lot surveillance images)\n"
            "- **Classes:** space-empty, space-occupied\n"
            "- **Validation mAP50:** ~96.9%"
        )

        st.markdown("---")
        st.markdown("### Congestion scale")
        st.markdown(
            "🟢 **Low** — under 40% occupied\n\n"
            "🟡 **Moderate** — 40–75% occupied\n\n"
            "🔴 **High** — over 75% occupied"
        )

        st.markdown("---")
        st.caption("Built for the AI/ML & Deep Learning CRS Assignment.")


def render_charts(info):
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Slot Breakdown")
        chart_df = pd.DataFrame({
            "Status": ["Occupied", "Available"],
            "Slots": [info["occupied_slots"], info["available_slots"]]
        })
        st.bar_chart(chart_df.set_index("Status"), color="#FF4B4B")

    with col_b:
        st.markdown("#### Occupancy Gauge")
        occ = info["occupancy_percent"]
        gauge_df = pd.DataFrame({
            "Occupied %": [occ],
            "Free %": [100 - occ]
        }, index=["Lot"])
        st.bar_chart(gauge_df, horizontal=True)
        st.caption(f"{occ}% of slots are currently occupied")


def main():
    render_sidebar()

    st.markdown('<p class="main-title">🚗 ParkVision AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Upload a parking lot image to detect slot '
        'availability in real time.</p>',
        unsafe_allow_html=True
    )

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload a parking lot image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing parking slots..."):
            results = model.predict(image_bgr, conf=0.4, verbose=False)[0]
            annotated_bgr, occupied_count, empty_count = draw_boxes(
                image_bgr.copy(), results, model.names
            )
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        total_slots = occupied_count + empty_count
        info = compute_insights(total_slots, occupied_count)

        st.markdown("---")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(annotated_rgb, caption="Detected Parking Slots", use_container_width=True)

            # Downloadable annotated image
            success, encoded_img = cv2.imencode(".png", annotated_bgr)
            if success:
                st.download_button(
                    label="⬇ Download annotated image",
                    data=encoded_img.tobytes(),
                    file_name="parkvision_result.png",
                    mime="image/png"
                )

        with col2:
            st.markdown("#### Parking Summary")
            m1, m2 = st.columns(2)
            m1.metric("Total Slots", info["total_slots"])
            m2.metric("Occupancy %", f"{info['occupancy_percent']}%")

            m3, m4 = st.columns(2)
            m3.metric("Occupied", info["occupied_slots"])
            m4.metric("Available", info["available_slots"])

            level = info["congestion_level"]
            if level == "Low":
                st.success(f"Congestion Level: {level}")
            elif level == "Moderate":
                st.warning(f"Congestion Level: {level}")
            elif level == "High":
                st.error(f"Congestion Level: {level}")
            else:
                st.info(f"Congestion Level: {level}")

            st.markdown("#### Recommendation")
            st.info(info["recommendation"])

        st.markdown("---")
        render_charts(info)

    else:
        st.info("👆 Upload a parking lot image above to get started.")
        st.markdown(
            "Don't have one handy? Try an image from your dataset's "
            "`test` folder for the most reliable results."
        )


if __name__ == "__main__":
    main()