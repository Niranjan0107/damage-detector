"""
MineFleet AI - Mining Vehicle & Body Damage Detection Dashboard
Streamlit Web Application for Mine CCTV Video Inspection.
"""
import os
import shutil
import tempfile
import streamlit as st
import pandas as pd
from PIL import Image

from detector.vehicle_tracker import VehicleTracker
from detector.damage_detector import DamageDetector
from detector.video_processor import VideoProcessor
from utils.report_generator import ReportGenerator

# ----------------- Streamlit Page Configuration -----------------
st.set_page_config(
    page_title="MineFleet AI - Mining Vehicle & Damage Detector",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #F7FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2B6CB0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2D3748;
    }
    .badge-critical {
        background-color: #FED7D7;
        color: #9B2C2C;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-major {
        background-color: #FEEBC8;
        color: #C05621;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-minor {
        background-color: #FEFCBF;
        color: #744210;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Sidebar Configuration -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dump-truck.png", width=72)
    st.title("Settings & Controls")
    st.markdown("Configure Detection & Inspection Parameters:")

    model_option = st.selectbox(
        "Vehicle Detection Model",
        options=["yolov8n.pt (Fast / Recommended)", "yolov8s.pt (Balanced)", "yolov8m.pt (High Accuracy)"],
        index=0,
    )
    model_weight = model_option.split(" ")[0]

    veh_conf = st.slider("Vehicle Confidence Threshold", 0.15, 0.90, 0.35, 0.05)
    dmg_sens = st.slider("Damage Sensitivity Threshold", 0.05, 0.70, 0.14, 0.01)
    frame_skip = st.select_slider(
        "Processing Speed / Frame Sampling",
        options=[1, 2, 3, 5],
        value=2,
        help="1 = Inspect every frame (slowest, highest detail); 2 = Process every 2nd frame (2x faster, recommended)",
    )

    st.divider()
    st.markdown("### AI Models Active")
    st.success("✅ **Vehicle Filter:** Strictly Mining Trucks & Dumpers")
    st.success("✅ **Damage Model:** YOLO11m AI (Dents, Scratches, Cracks & Cabin Crumple)")
    
    st.markdown("### Mine CCTV Info")
    mine_location = st.text_input("Mine Site / Zone", "Quarry Pit Alpha - Gate 03")
    st.info("System strictly tracks Mining Trucks, Dumpers & Containers (cars/carts ignored), then inspects for surface damages.")

# ----------------- Main Dashboard UI -----------------
st.markdown('<div class="main-header">🚛 MineFleet AI: CCTV Vehicle & Damage Inspection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated Deep Learning pipeline to detect Mining Trucks, Dumpers & Containers, localize body damage (dents, cracks, gouges), and generate maintenance reports.</div>',
    unsafe_allow_html=True,
)

tab_upload, tab_sample = st.tabs(["📤 Upload CCTV Video", "🧪 Use Sample Mining CCTV Video"])

video_source_path = None
uploaded_video_name = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload Mine CCTV Video (.mp4, .avi, .mov)",
        type=["mp4", "avi", "mov"],
        help="Upload footage recorded at gate scales, quarry pits, or dump yards.",
    )
    if uploaded_file is not None:
        save_dir = "data/uploads"
        os.makedirs(save_dir, exist_ok=True)
        video_source_path = os.path.join(save_dir, uploaded_file.name)
        with open(video_source_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_video_name = uploaded_file.name
        st.success(f"Video uploaded: `{uploaded_video_name}`")
        st.video(uploaded_file.getvalue(), format="video/mp4")

with tab_sample:
    st.markdown("Don't have a mining CCTV video handy? Use the built-in photorealistic mining CCTV video:")
    sample_path = "samples/mine_cctv_simulation.mp4"
    
    col_gen, col_view = st.columns([1, 2])
    with col_gen:
        if st.button("Generate / Reset Mine CCTV Video", type="secondary"):
            with st.spinner("Generating multi-vehicle CCTV footage with quarry pit background..."):
                from generate_mine_cctv_video import generate_mine_cctv_video
                generate_mine_cctv_video(sample_path)
            st.success("Mine CCTV video ready!")
            
    if os.path.exists(sample_path):
        with col_view:
            with open(sample_path, "rb") as f_samp:
                st.video(f_samp.read(), format="video/mp4")
            if st.button("Select Mine Video for Inspection", type="primary"):
                video_source_path = sample_path
                uploaded_video_name = "mine_cctv_simulation.mp4"
                st.session_state["selected_sample"] = True

if st.session_state.get("selected_sample") and not video_source_path:
    video_source_path = "samples/mine_cctv_simulation.mp4"
    uploaded_video_name = "mine_cctv_simulation.mp4"

# ----------------- Execution & Processing -----------------
st.divider()

if video_source_path and os.path.exists(video_source_path):
    st.markdown(f"**Target Video:** `{video_source_path}`")
    
    start_btn = st.button("🚀 Start AI Damage Inspection", type="primary", use_container_width=True)

    if start_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        preview_box = st.empty()

        # Initialize detector components strictly for Trucks, Dumpers, Containers
        tracker = VehicleTracker(
            model_path=model_weight,
            confidence_threshold=veh_conf,
            target_classes=[7],
            min_box_width=90,
            min_box_height=70,
        )
        damage_detector = DamageDetector(severity_threshold=dmg_sens)
        processor = VideoProcessor(
            vehicle_tracker=tracker,
            damage_detector=damage_detector,
            snapshots_dir="data/outputs/snapshots",
        )

        out_video_name = f"annotated_{os.path.basename(video_source_path)}"
        out_video_path = os.path.join("data/outputs", out_video_name)

        def on_progress(pct: float, cur_frame: int, total_frames: int, stats: dict):
            progress_bar.progress(pct)
            status_text.markdown(
                f"**Processing Frame:** `{cur_frame}/{total_frames}` | "
                f"**Vehicles Scanned:** `{stats['vehicles_scanned']}` | "
                f"**Damaged Units Flagged:** `{stats['damaged_count']}` | "
                f"**Elapsed:** `{stats['elapsed_time']}s`"
            )

        with st.spinner("Analyzing CCTV footage, tracking mining fleet and scanning for structural defects..."):
            summary = processor.process_video(
                input_video_path=video_source_path,
                output_video_path=out_video_path,
                frame_skip=frame_skip,
                progress_callback=on_progress,
            )

        progress_bar.progress(1.0)
        status_text.success("Inspection Complete!")
        st.session_state["inspection_summary"] = summary

# ----------------- Display Results -----------------
if "inspection_summary" in st.session_state:
    summary = st.session_state["inspection_summary"]
    incidents = summary.get("incidents", [])

    st.markdown("### 📊 Inspection Analytics & Results")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    tot_veh = summary.get("total_vehicles_detected", 0)
    dmg_veh = summary.get("damaged_vehicles_count", 0)
    critical_count = sum(1 for i in incidents if i["severity_level"] == "CRITICAL")
    clean_veh = max(0, tot_veh - dmg_veh)

    with m1:
        st.metric("Total Vehicles Scanned", tot_veh)
    with m2:
        st.metric("Damaged Units Flagged", dmg_veh, delta=f"{round(dmg_veh/max(1, tot_veh)*100, 1)}%", delta_color="inverse")
    with m3:
        st.metric("Critical Defects (Immediate Action)", critical_count)
    with m4:
        st.metric("Clean / Undamaged Fleet", clean_veh)

    # Video Player
    st.markdown("#### 🎥 Annotated Video Inspection Playback")
    out_video = summary.get("output_video_path")
    if out_video and os.path.exists(out_video):
        with open(out_video, "rb") as vid_file:
            st.video(vid_file.read(), format="video/mp4")

    # Defect Gallery & Evidence
    st.markdown("#### 🔍 Fleet Inspection Gallery & Visual Evidence")
    all_vehicles = summary.get("all_vehicles", incidents)
    if not all_vehicles:
        st.info("No vehicles tracked in this footage.")
    else:
        for veh_record in all_vehicles:
            has_dmg = veh_record.get("has_damage", False)
            sev_level = veh_record.get("severity_level", "HEALTHY")
            max_score = veh_record.get("max_severity_score", 0.0)

            if not has_dmg or sev_level == "HEALTHY":
                sev_badge = '<span style="background-color:#C6F6D5;color:#22543D;padding:4px 8px;border-radius:4px;font-weight:bold;">✅ HEALTHY / CLEAN</span>'
            elif sev_level == "CRITICAL":
                sev_badge = f'<span class="badge-critical">🚨 CRITICAL ({int(max_score)}%)</span>'
            elif sev_level == "MAJOR":
                sev_badge = f'<span class="badge-major">⚠️ MAJOR ({int(max_score)}%)</span>'
            else:
                sev_badge = f'<span class="badge-minor">⚠️ MINOR ({int(max_score)}%)</span>'

            card_title = f"Vehicle #{veh_record['vehicle_id']} - {veh_record['vehicle_class']} | Status: {sev_level}"
            with st.expander(card_title, expanded=has_dmg):
                col_img, col_details = st.columns([1.2, 1.5])
                with col_img:
                    snap_path = veh_record.get("snapshot_path")
                    if snap_path and os.path.exists(snap_path):
                        st.image(snap_path, caption=f"CCTV Capture at {veh_record['first_detected_sec']}s", use_container_width=True)
                    else:
                        st.info("No image capture available")

                with col_details:
                    st.markdown(f"**Vehicle ID:** `#{veh_record['vehicle_id']}`")
                    st.markdown(f"**Classification:** `{veh_record['vehicle_class']}`")
                    st.markdown(f"**Inspection Status:** {sev_badge}", unsafe_allow_html=True)
                    
                    if has_dmg:
                        st.markdown(f"**Identified Defects / Damages:**")
                        for d_type in veh_record.get("damage_types", []):
                            st.markdown(f"- ⚠️ **{d_type}**")
                        st.markdown(f"**Damage Severity Score:** `{int(max_score)}%`")
                        st.markdown(f"**Defect Clusters Count:** `{veh_record.get('total_damage_points', 1)}`")
                        if sev_level == "CRITICAL":
                            st.error("🚨 Action Required: Flag for scale stop & chassis/weld inspection.")
                        else:
                            st.warning("⚠️ Action Required: Schedule routine body maintenance.")
                    else:
                        st.success("✅ Clean Inspection: Surface intact, no dents or structural cracks detected.")

    # ----------------- Download Reports -----------------
    st.divider()
    st.markdown("### 📥 Export Official Inspection Reports")

    col_pdf, col_csv = st.columns(2)
    
    # Generate PDF
    pdf_path = "data/reports/Mining_Fleet_Inspection_Report.pdf"
    ReportGenerator.generate_pdf(summary, incidents, pdf_path)
    
    with col_pdf:
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f_pdf:
                st.download_button(
                    label="📄 Download Official PDF Report (with Photos)",
                    data=f_pdf,
                    file_name="Mining_Damage_Inspection_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    # Generate CSV
    csv_path = "data/reports/Defect_Log.csv"
    ReportGenerator.generate_csv(incidents, csv_path)
    
    with col_csv:
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f_csv:
                st.download_button(
                    label="📊 Export Defect Data (CSV)",
                    data=f_csv,
                    file_name="Mining_Damage_Log.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
