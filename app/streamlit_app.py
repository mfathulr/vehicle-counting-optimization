"""
Professional Vehicle Counting & Traffic Light Optimization System
Modern Streamlit web application with Upload and Realtime detection modes
"""

import torch
import streamlit as st
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import tempfile
import time
import platform
import os

# Suppress FFmpeg/OpenCV verbose warnings (TLS, H264 errors are normal for live streams)
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # Suppress most FFmpeg logs
cv2.setLogLevel(0)  # Suppress OpenCV warnings

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    PAGE_TITLE,
    LAYOUT,
    NUM_CLASSES,
    MODEL_PATH,
)
from src.models import create_faster_rcnn_resnet18
from src.inference import ImageDetector, VideoDetector
from src.optimization import TrafficLightOptimizer
from src.utils import count_classes


def setup_page():
    """Configure Streamlit page settings with professional styling."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🚗",
        layout=LAYOUT,
        initial_sidebar_state="expanded",
    )

    # Modern professional CSS styling
    st.markdown(
        """
        <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* Keep header visible for sidebar toggle */
        header {visibility: visible;}
        
        /* Sidebar styling - Dark Blue Gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 50%, #2563eb 100%);
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        /* Primary button styling */
        .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 14px 28px;
            font-weight: 700;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #2563eb 0%, #1e40af 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
        }
        
        .stButton > button:active {
            transform: translateY(0px);
        }
        
        /* Title styling */
        h1 {
            text-align: center;
            color: #1e3a8a;
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        h2 {
            color: #1e40af;
            font-weight: 700;
        }
        
        h3 {
            color: #2563eb;
            font-weight: 600;
        }
        
        /* Image styling */
        [data-testid="stImage"] {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background: rgba(59, 130, 246, 0.1);
            border-radius: 10px;
            padding: 20px;
            border: 2px dashed #3b82f6;
        }
        
        /* Slider */
        .stSlider > div > div > div {
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        }

        /* Sidebar cards */
        .sidebar-card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        .sidebar-label {
            font-weight: 700;
            letter-spacing: 0.3px;
            text-transform: uppercase;
            font-size: 12px;
            color: #bfdbfe;
            margin-bottom: 6px;
        }

        .sidebar-help {
            color: #e5e7eb;
            font-size: 12px;
            opacity: 0.8;
            margin-top: -4px;
            margin-bottom: 6px;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .main .block-container {
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 32px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_model(device: str) -> torch.nn.Module:
    """Load the trained Faster R-CNN model."""
    model = create_faster_rcnn_resnet18(num_classes=NUM_CLASSES)
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def get_youtube_stream_url(youtube_url: str) -> str:
    """Extract direct stream URL from YouTube video/livestream with retry fallback."""
    try:
        import yt_dlp
    except ImportError:
        st.error(
            "❌ yt-dlp is not installed. Please install it using: pip install yt-dlp"
        )
        return ""

    max_retries = 2
    for attempt in range(max_retries):
        try:
            ydl_opts = {
                "format": "best[ext=mp4][height<=480]/best[height<=480]/best[height<=720]/best",
                "quiet": True,
                "no_warnings": False,
                "extract_flat": False,
                # Prefer native HLS handling and bypass minor geo restrictions
                "hls_prefer_native": True,
                "geo_bypass": True,
                "geo_bypass_country": "US",
                "noplaylist": True,
                "live_from_start": False,
                # Try multiple player clients for compatibility
                "compat_opts": ["no-ejs", "no-youtube-prefer-utc-upload-date"],
                "extractor_args": {"youtube": {"player_client": ["android", "web", "mweb"]}},
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                "socket_timeout": 30,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)

                # Handle playlists
                if "entries" in info:
                    if not info["entries"]:
                        st.error("❌ Playlist is empty or unavailable.")
                        return ""
                    info = info["entries"][0]

                # Check if it's a live stream that's not currently live
                if (not info.get("is_live")) and info.get("was_live"):
                    st.warning(
                        "⚠️ This is a past livestream. Some streams may not be available."
                    )

                # Try to get direct URL
                if "url" in info:
                    return info["url"]

                # Search through formats (prioritize HLS/DASH)
                if "formats" in info:
                    # Prefer HLS M3U8 that OpenCV can read natively
                    for fmt in info["formats"]:
                        if fmt.get("url") and fmt.get("protocol") in [
                            "m3u8",
                            "m3u8_native",
                        ]:
                            return fmt["url"]
                    # Fallback to MP4/WebM formats
                    for fmt in info["formats"]:
                        if (
                            fmt.get("url")
                            and fmt.get("ext") in ["mp4", "webm"]
                            and fmt.get("vcodec") != "none"
                        ):
                            return fmt["url"]

                st.error("❌ No compatible video format found. Try a different video.")
                return ""
        except Exception as e:
            error_msg = str(e)
            # Log attempt for debugging
            if st.session_state.debug_realtime:
                st.warning(f"Extraction attempt {attempt + 1}/{max_retries} failed: {error_msg}")
            # Retry once more if it's a temporary error
            if attempt < max_retries - 1 and ("technical" in error_msg.lower() or "timeout" in error_msg.lower()):
                time.sleep(2)
                continue
            # Final error handling
            if "not available" in error_msg.lower():
                st.error("❌ This video/stream is not available. Please try:")
                st.info(
                    "💡 Use a regular YouTube video (not expired livestream)\n💡 Check if video is public and playable\n💡 Try a different traffic camera or livestream"
                )
            elif "javascript" in error_msg.lower() or "js" in error_msg.lower():
                st.warning(
                    "⚠️ This stream requires JS extraction. Try a simpler video or a public livestream with direct HLS access."
                )
            else:
                st.error(f"❌ Error: {error_msg}")
            return ""
    return ""


def upload_mode(image_detector, video_detector, optimizer, device, threshold):
    """Handle Upload mode for images and videos."""
    st.markdown("## 📤 Upload Mode")
    st.markdown("Upload an image or video file for vehicle detection and counting.")

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Choose a file",
        type=["jpg", "jpeg", "png", "mp4"],
        help="Supported formats: JPG, JPEG, PNG for images | MP4 for videos",
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type.startswith("image"):
            # Image processing
            st.markdown("### 📷 Image Detection")
            display_placeholder = st.empty()
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            display_placeholder.image(
                image, use_container_width=True, caption="Original"
            )

            # Detection button
            if st.button(
                "🚀 Detect Vehicles", type="primary", use_container_width=True
            ):
                with st.spinner("🔍 Detecting vehicles..."):
                    pred_image, boxes, scores, pred_classes = image_detector.detect(
                        image_np, threshold=threshold, show_progress=False
                    )
                    display_placeholder.image(
                        pred_image,
                        channels="RGB",
                        use_container_width=True,
                        caption="Detection Result",
                    )

                    counts = count_classes(pred_classes)
                    duration = optimizer.optimize(counts["mobil"], counts["motor"])

                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Detection Results")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🚗 Cars", counts["mobil"])
                    with col2:
                        st.metric("🏍️ Motorcycles", counts["motor"])
                    with col3:
                        st.metric("⏱️ Green Light Duration", f"{duration} sec")

                    st.success(
                        f"✅ Detection complete! Found {counts['mobil']} cars and {counts['motor']} motorcycles."
                    )

        elif file_type.startswith("video"):
            # Video processing
            st.markdown("### 🎬 Video Detection")

            # Save uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name

            video_placeholder = st.empty()
            video_placeholder.video(temp_video_path)

            # Detection button
            if st.button("🚀 Process Video", type="primary", use_container_width=True):
                st.markdown("---")
                st.markdown("### 🎞️ Processing")
                progress_bar = st.progress(0, text="🔄 Processing video...")

                # Debug: Check video properties first
                try:
                    cap_check = cv2.VideoCapture(temp_video_path)
                    if cap_check.isOpened():
                        w = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps_val = int(cap_check.get(cv2.CAP_PROP_FPS))
                        total = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
                        st.info(f"📹 Video: {w}x{h} @ {fps_val}fps, {total} frames")
                        cap_check.release()
                    else:
                        st.error(
                            "❌ Could not open video file. May be corrupted or unsupported format."
                        )
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Video check failed: {e}")
                    st.stop()

                def update_progress(progress: float, frame: np.ndarray):
                    progress_bar.progress(
                        progress, text=f"🔄 Processing: {int(progress * 100)}%"
                    )
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(
                        frame_rgb,
                        channels="RGB",
                        caption="Processing...",
                        use_container_width=True,
                    )

                # Process video
                output_path = temp_video_path.replace(".mp4", "_output.mp4")
                try:
                    final_counts, avg_fps = video_detector.process_video(
                        temp_video_path,
                        output_path,
                        threshold=threshold,
                        progress_callback=update_progress,
                    )

                    # Debug: Check output file
                    import os

                    if os.path.exists(output_path):
                        size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        st.success(f"✅ Output file created ({size_mb:.1f} MB)")
                    else:
                        st.error(
                            "❌ Output file was not created. Check codec or disk space."
                        )
                        st.stop()
                except Exception as ve:
                    st.error(f"❌ Video processing error: {ve}")
                    import traceback

                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.stop()

                progress_bar.empty()

                # Display results
                st.markdown("---")
                st.markdown("### 📊 Detection Results")

                duration = optimizer.optimize(
                    final_counts["mobil"], final_counts["motor"]
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🚗 Cars", final_counts["mobil"])
                with col2:
                    st.metric("🏍️ Motorcycles", final_counts["motor"])
                with col3:
                    st.metric("⏱️ Green Light Duration", f"{duration} sec")

                st.success(
                    f"✅ Video processed successfully! Average FPS: {avg_fps:.2f}"
                )

                # Display processed video in the same placeholder
                if os.path.exists(output_path):
                    video_placeholder.video(output_path)
                else:
                    st.warning("⚠️ Processed video could not be displayed.")
    else:
        st.info("👆 Please upload an image or video file to begin.")


def realtime_mode(image_detector, optimizer, device, threshold):
    """Handle Realtime mode for webcam and YouTube streams."""
    # CRITICAL: Initialize session state FIRST before any UI rendering
    if "realtime_running" not in st.session_state:
        st.session_state.realtime_running = False
    if "video_source" not in st.session_state:
        st.session_state.video_source = None
    if "source_name" not in st.session_state:
        st.session_state.source_name = ""
    if "youtube_url" not in st.session_state:
        st.session_state.youtube_url = ""
    if "last_url_refresh" not in st.session_state:
        st.session_state.last_url_refresh = 0
    if "realtime_error" not in st.session_state:
        st.session_state.realtime_error = None
    if "realtime_error_time" not in st.session_state:
        st.session_state.realtime_error_time = None
    if "debug_realtime" not in st.session_state:
        st.session_state.debug_realtime = False

    # CRITICAL: Show any stored errors BEFORE everything else
    # Use st.stop() to halt rendering if error exists
    if st.session_state.realtime_error:
        error_age = (
            time.time() - st.session_state.realtime_error_time
            if st.session_state.realtime_error_time
            else 0
        )
        # Display error prominently
        st.markdown("## ❌ Realtime Detection Failed")
        st.error(
            f"**Error (occurred {error_age:.1f}s ago):**\n```\n{st.session_state.realtime_error}\n```"
        )
        st.info(
            "**Troubleshooting checklist:**\n\n"
            "✓ For YouTube: Try a different public traffic camera or regular video\n"
            "✓ Check your internet connection and firewall\n"
            "✓ For webcam: Ensure camera is connected and permissions are granted\n"
            "✓ Enable '🔍 Enable realtime debug logs' to see backend details"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Error & Try Again", key="clear_error_btn"):
                st.session_state.realtime_error = None
                st.session_state.realtime_error_time = None
                st.session_state.realtime_running = False
                st.session_state.video_source = None
                st.rerun()
        with col2:
            if st.button("🛑 Stop & Close", key="stop_realtime_btn"):
                # DON'T clear error - only on explicit clear button
                st.session_state.realtime_running = False
                st.session_state.video_source = None
                st.rerun()
        st.stop()  # Stop rendering other UI elements - keep error visible

    st.markdown("## 🎥 Realtime Mode")
    st.markdown("Live vehicle detection from YouTube stream.")

    # Check if running in Streamlit Cloud and warn about resource constraints
    is_cloud = (
        "STREAMLIT_SERVER_HEADLESS" in os.environ or "STREAMLIT_RUNTIME" in os.environ
    )
    if is_cloud and device == "cpu":
        st.info(
            "💡 Running on Streamlit Cloud with CPU. Performance may be slower. If inference fails, try:\n1. Lower resolution (50%)\n2. Increase frame skip (5+)\n3. Close other apps"
        )

    st.markdown("---")

    # YouTube stream as default
    st.markdown("### 🎬 YouTube Video Stream")

    # Default YouTube URL - using a stable public traffic camera livestream
    # Option 1: CCTV traffic camera (more reliable for testing)
    default_url = "https://www.youtube.com/watch?v=cYt0zb1F3U8"  # Sample public traffic camera

    youtube_url = st.text_input(
        "YouTube URL",
        value=default_url,
        help="Enter YouTube video or livestream URL (prefer public traffic cameras or regular videos)",
    )

    col1, col2 = st.columns(2)
    with col1:
        youtube_button = st.button(
            "▶️ Start Detection", type="primary", use_container_width=True
        )
    with col2:
        webcam_button = st.button("📹 Use Webcam Instead", use_container_width=True)

    st.markdown("---")

    # Performance settings
    with st.expander("⚡ Performance Settings"):
        col1, col2 = st.columns(2)
        with col1:
            frame_skip = st.slider(
                "Process every N frames",
                min_value=1,
                max_value=10,
                value=3,
                help="Higher value = faster but less accurate",
            )
        with col2:
            resize_factor = st.select_slider(
                "Resolution",
                options=[0.5, 0.75, 1.0],
                value=0.75,
                format_func=lambda x: f"{int(x * 100)}%",
            )

    # Debug checkbox
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.session_state.debug_realtime = st.checkbox(
        "🔍 Enable realtime debug logs",
        value=st.session_state.debug_realtime,
        help="Show backend, URL, and capture status for troubleshooting in production",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Handle start buttons
    if webcam_button:
        st.session_state.realtime_running = True
        st.session_state.video_source = 0
        st.session_state.source_name = "Webcam"
        st.rerun()

    if youtube_button:
        if not youtube_url:
            st.error("❌ Please enter a YouTube URL.")
        else:
            with st.spinner("🔄 Extracting YouTube stream URL..."):
                stream_url = get_youtube_stream_url(youtube_url)
                if stream_url:
                    st.session_state.realtime_running = True
                    st.session_state.video_source = stream_url
                    st.session_state.source_name = "YouTube Stream"
                    st.session_state.youtube_url = youtube_url
                    st.session_state.last_url_refresh = time.time()
                    st.rerun()
                else:
                    st.error("❌ Failed to extract YouTube stream URL.")

    # Realtime processing
    if st.session_state.realtime_running:
        st.markdown(f"### 📡 Live Detection - {st.session_state.source_name}")

        # Stop button
        if st.button("⏹️ Stop Detection", type="primary", use_container_width=True):
            st.session_state.realtime_running = False
            # DON'T clear error - let user see it before clicking "Clear Error & Retry"
            st.rerun()

        try:
            col1, col2 = st.columns([2, 1])

            with col1:
                frame_holder = st.empty()

            with col2:
                st.markdown("#### 📊 Live Stats")
                status_display = st.empty()
                fps_display = st.empty()
                frame_count_display = st.empty()
                processed_display = st.empty()

                st.markdown("#### 🚦 Traffic Optimization")
                mobil_metric = st.empty()
                motor_metric = st.empty()
                duration_metric = st.empty()

            def open_capture(source):
                """Open video capture with fallback backends for URL streams."""
                if isinstance(source, str):
                    # For URLs, try multiple backends with fallback
                    # In cloud/production, prefer CAP_ANY first due to limited ffmpeg
                    backends_to_try = (
                        [
                            (cv2.CAP_ANY, "ANY"),
                            (cv2.CAP_FFMPEG, "FFMPEG"),
                        ]
                        if is_cloud
                        else [
                            (cv2.CAP_FFMPEG, "FFMPEG"),
                            (cv2.CAP_ANY, "ANY"),
                        ]
                    )
                    cap_obj = None
                    for backend, name in backends_to_try:
                        try:
                            cap_obj = cv2.VideoCapture(source, backend)
                            if cap_obj.isOpened():
                                # Test if we can actually read a frame
                                ret, _ = cap_obj.read()
                                if ret:
                                    cap_obj.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset
                                    cap_obj.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                                    if st.session_state.debug_realtime:
                                        st.info(f"✅ Using backend: {name}")
                                    return cap_obj
                                cap_obj.release()
                        except Exception as e:
                            if cap_obj:
                                cap_obj.release()
                            if st.session_state.debug_realtime:
                                st.warning(f"Backend {name} failed: {e}")
                            continue

                    # If all backends fail, try Streamlink fallback (production only)
                    if is_cloud:
                        try:
                            import streamlink

                            session = streamlink.Streamlink()
                            streams = session.streams(source)
                            # Prefer 'best' stream
                            best = (
                                streams.get("best")
                                or streams.get("720p")
                                or streams.get("480p")
                            )
                            if best:
                                url = best.to_url()
                                if st.session_state.debug_realtime:
                                    st.info("🧪 Streamlink fallback URL acquired")
                                cap_obj = cv2.VideoCapture(url, cv2.CAP_ANY)
                                if cap_obj.isOpened():
                                    return cap_obj
                        except Exception as e:
                            if st.session_state.debug_realtime:
                                st.warning(f"Streamlink fallback failed: {e}")
                    # Final fallback
                    return cv2.VideoCapture(source, cv2.CAP_ANY)
                else:
                    # Webcam
                    sys_name = platform.system()
                    if sys_name == "Windows":
                        backend = cv2.CAP_DSHOW
                    elif sys_name == "Darwin":
                        backend = cv2.CAP_AVFOUNDATION
                    else:
                        backend = cv2.CAP_V4L2
                    cap_obj = cv2.VideoCapture(source, backend)
                    if st.session_state.debug_realtime:
                        st.info(f"🎥 Webcam backend: {backend}")
                    return cap_obj

            # Open video capture
            if st.session_state.debug_realtime:
                st.info(f"🔗 Source: {st.session_state.video_source}")
                st.info(f"📺 Mode: {st.session_state.source_name}")
            cap = open_capture(st.session_state.video_source)

            if not cap.isOpened():
                if st.session_state.source_name == "Webcam":
                    err_msg = "❌ Webcam not available. Check camera permissions, index value, or connect a camera."
                else:
                    err_msg = f"❌ Could not open {st.session_state.source_name}. Stream may be region-restricted, expired, or require a different format. Try a different YouTube video or check your network."
                # Store error in session state (will be displayed by banner on next rerun)
                st.session_state.realtime_error = err_msg
                st.session_state.realtime_error_time = time.time()
                st.session_state.realtime_running = False
                # Stop immediately - error will be shown by banner on next rerun
                st.stop()

            # Set properties for webcam
            if st.session_state.video_source == 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)

            frame_count = 0
            processed_count = 0
            prev_time = time.time()
            last_pred_image = None
            last_counts = {"mobil": 0, "motor": 0}
            retry_count = 0
            max_retries = 3
            # In production, refresh URLs more aggressively (1–2 minutes)
            url_refresh_interval = 120 if is_cloud else 300

            # Processing loop
            while st.session_state.realtime_running:
                try:
                    # Check if we need to refresh YouTube URL (for livestreams)
                    if (
                        st.session_state.source_name == "YouTube Stream"
                        and st.session_state.youtube_url
                    ):
                        time_since_refresh = (
                            time.time() - st.session_state.last_url_refresh
                        )
                        if time_since_refresh > url_refresh_interval:
                            status_display.info("🔄 Refreshing stream URL...")
                            new_stream_url = get_youtube_stream_url(
                                st.session_state.youtube_url
                            )
                            if new_stream_url:
                                cap.release()
                                st.session_state.video_source = new_stream_url
                                cap = open_capture(new_stream_url)
                                st.session_state.last_url_refresh = time.time()
                                retry_count = 0
                                status_display.success("✅ Stream URL refreshed")
                                time.sleep(1)
                                status_display.empty()
                            else:
                                status_display.warning(
                                    "⚠️ Failed to refresh URL, continuing..."
                                )

                    ret, frame = cap.read()
                    if st.session_state.debug_realtime and frame is None:
                        st.warning(
                            "⚠️ Read returned no frame (None). Will retry/fallback if configured."
                        )

                    if not ret:
                        if (
                            st.session_state.source_name == "YouTube Stream"
                            and retry_count < max_retries
                        ):
                            retry_count += 1
                            status_display.warning(
                                f"⚠️ Connection lost. Retry {retry_count}/{max_retries}..."
                            )
                            time.sleep(2)
                            cap.release()
                            # Try to re-extract fresh URL
                            if st.session_state.youtube_url:
                                new_stream_url = get_youtube_stream_url(
                                    st.session_state.youtube_url
                                )
                                if new_stream_url:
                                    st.session_state.video_source = new_stream_url
                                    cap = open_capture(new_stream_url)
                                    st.session_state.last_url_refresh = time.time()
                                    if st.session_state.debug_realtime:
                                        st.info(
                                            f"🔁 Reconnected with new URL: {new_stream_url}"
                                        )
                                    status_display.success("✅ Reconnected")
                                    time.sleep(1)
                                    status_display.empty()
                                    continue
                            else:
                                cap = open_capture(st.session_state.video_source)
                                continue
                        else:
                            err_msg = f"❌ Stream ended or connection lost after {max_retries} retries."
                            st.error(err_msg)
                            st.session_state.realtime_error = err_msg
                            st.session_state.realtime_error_time = time.time()
                            st.session_state.realtime_running = False
                            st.stop()

                    frame_count += 1

                    # Process every N frames
                    if frame_count % frame_skip == 0:
                        try:
                            # Resize if needed
                            if resize_factor < 1.0:
                                width = int(frame.shape[1] * resize_factor)
                                height = int(frame.shape[0] * resize_factor)
                                resized_frame = cv2.resize(frame, (width, height))
                            else:
                                resized_frame = frame

                            # Detect - most likely place for errors with timing diagnostic
                            detect_start = time.time()
                            pred_image, boxes, scores, pred_classes = (
                                image_detector.detect(
                                    resized_frame,
                                    threshold=threshold,
                                    show_progress=False,
                                )
                            )
                            detect_time = time.time() - detect_start

                            # Warn if detection is taking too long (might indicate issues)
                            if detect_time > 5.0 and st.session_state.debug_realtime:
                                st.warning(
                                    f"⚠️ Inference slow: {detect_time:.2f}s (GPU/resource issue?)"
                                )

                            # Resize back if needed
                            if resize_factor < 1.0:
                                pred_image = cv2.resize(
                                    pred_image, (frame.shape[1], frame.shape[0])
                                )

                            last_pred_image = pred_image
                            last_counts = count_classes(pred_classes)
                            processed_count += 1

                        except RuntimeError as mem_error:
                            # GPU/memory error - common in cloud
                            if (
                                "cuda" in str(mem_error).lower()
                                or "out of memory" in str(mem_error).lower()
                            ):
                                st.session_state.realtime_running = False
                                st.session_state.realtime_error = f"GPU Memory Error: {str(mem_error)}\n\nTry: Use CPU instead of CUDA, or reduce resolution"
                                st.session_state.realtime_error_time = time.time()
                                cap.release()
                                # Do NOT rerun automatically; keep error visible
                                st.stop()
                            else:
                                raise
                        except Exception as detect_error:
                            import traceback

                            st.session_state.realtime_running = False
                            error_trace = traceback.format_exc()
                            st.session_state.realtime_error = f"Detection Error: {type(detect_error).__name__}: {str(detect_error)}\n\nTraceback:\n{error_trace}"
                            st.session_state.realtime_error_time = time.time()
                            if st.session_state.debug_realtime:
                                st.warning(f"Detection failed: {detect_error}")
                            cap.release()
                            # Do NOT rerun automatically; keep error visible
                            st.stop()

                    # Calculate FPS
                    current_time = time.time()
                    fps = 1 / (current_time - prev_time + 1e-6)
                    prev_time = current_time

                    # Display frame
                    display_frame = (
                        last_pred_image if last_pred_image is not None else frame
                    )
                    # Ensure BGR to RGB conversion if needed
                    if display_frame is not None:
                        if (
                            len(display_frame.shape) == 3
                            and display_frame.shape[2] == 3
                        ):
                            # Check if it needs BGR to RGB conversion
                            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        else:
                            display_rgb = display_frame

                        try:
                            frame_holder.image(
                                display_rgb,
                                channels="RGB",
                                caption=f"FPS: {fps:.1f}",
                                use_container_width=True,
                            )
                        except Exception as e:
                            if st.session_state.debug_realtime:
                                st.warning(f"⚠️ Frame display error: {e}")

                    # Update stats
                    if status_display and not status_display._is_empty:
                        pass
                    else:
                        status_display.success("🟢 Live")
                    fps_display.metric("🎯 FPS", f"{fps:.1f}")
                    frame_count_display.metric("📊 Total Frames", frame_count)
                    processed_display.metric("✅ Processed", processed_count)

                    # Update detection results
                    duration = optimizer.optimize(
                        last_counts["mobil"], last_counts["motor"]
                    )
                    mobil_metric.metric("🚗 Cars", last_counts["mobil"])
                    motor_metric.metric("🏍️ Motorcycles", last_counts["motor"])
                    duration_metric.metric("⏱️ Duration", f"{duration} sec")

                    time.sleep(0.001)

                except Exception as loop_error:
                    import traceback

                    st.session_state.realtime_running = False
                    error_trace = traceback.format_exc()
                    st.session_state.realtime_error = f"Loop Error: {type(loop_error).__name__}: {str(loop_error)}\n\nTraceback:\n{error_trace}"
                    st.session_state.realtime_error_time = time.time()
                    cap.release()
                    # Do NOT rerun automatically; keep error visible
                    st.stop()

            cap.release()
            st.session_state.realtime_running = False
            st.success(
                f"✅ Session ended. Total: {frame_count} frames | Processed: {processed_count} frames"
            )

        except Exception as rt_error:
            import traceback

            error_trace = traceback.format_exc()

            # Store error in session state for persistence
            st.session_state.realtime_running = False
            st.session_state.realtime_error = f"Runtime Error: {type(rt_error).__name__}: {str(rt_error)}\n\nTraceback:\n{error_trace}"
            st.session_state.realtime_error_time = time.time()

            # Log to console for debugging
            st.write(f"[ERROR CAPTURED] {st.session_state.realtime_error}")

            # Display error in UI
            st.error(
                f"❌ Realtime loop failed:\n```\n{st.session_state.realtime_error}\n```"
            )

            if st.button("🔄 Retry Realtime Mode"):
                st.session_state.realtime_error = None
                st.session_state.realtime_running = False
                st.rerun()

            # Retry same source without clearing error
            if st.session_state.video_source and st.button(
                "🔁 Retry Same Source", key="retry_same_source_btn"
            ):
                st.session_state.realtime_running = True
                st.rerun()

    else:
        st.info("👆 Select a source above to start realtime detection.")


def main():
    """Main application entry point."""
    # Setup page
    setup_page()

    # Header
    st.title("🚗 Vehicle Counting System")
    st.markdown(
        """
        <p style='text-align: center; color: #666; font-size: 18px; margin-top: -10px;'>
            Traffic Light Optimization with Faster R-CNN & Fuzzy Logic
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Sidebar - redesigned
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")

        # Mode selection card
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-label'>Mode</div>", unsafe_allow_html=True)
        mode = st.radio(
            "Select Mode",
            ["🎥 Realtime", "📤 Upload"],
            index=0,
            label_visibility="collapsed",
            help="Realtime: YouTube/Webcam | Upload: static files",
        )
        st.markdown(
            "<div class='sidebar-help'>Upload for static files; Realtime for YouTube/Webcam live streams.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Device & threshold card
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sidebar-label'>Performance</div>", unsafe_allow_html=True
        )

        col_dev, col_thr = st.columns([1, 1])
        with col_dev:
            # Auto-fallback to CPU if CUDA not available in production
            cuda_available = False
            try:
                import torch

                cuda_available = torch.cuda.is_available()
            except Exception:
                cuda_available = False

            device = st.radio(
                "Device",
                ("cpu", "cuda"),
                index=0 if not cuda_available else 1,
                label_visibility="collapsed",
                help="CUDA requires an NVIDIA GPU",
            )
            if device == "cuda" and not cuda_available:
                st.warning("CUDA not available. Falling back to CPU.")
                device = "cpu"
        with col_thr:
            threshold = st.slider(
                "Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                label_visibility="collapsed",
                help="Higher = more selective",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Location card
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sidebar-label'>📍 Optimized For</div>", unsafe_allow_html=True
        )
        st.markdown("**Location:** Simpang Pingit 1 (Selatan)", unsafe_allow_html=True)
        st.markdown("**Source:** Jogja City CCTV", unsafe_allow_html=True)
        st.markdown(
            "<div class='sidebar-help'>Model trained specifically for this intersection's camera angle and traffic patterns.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Info card
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sidebar-label'>Model Info</div>", unsafe_allow_html=True
        )
        st.markdown(
            "**Faster R-CNN (ResNet18)**<br>Classes: Cars & Motorcycles",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Validate CUDA
    if device == "cuda" and not torch.cuda.is_available():
        st.warning("⚠️ CUDA not available. Using CPU.")
        device = "cpu"

    # Load model with diagnostics
    with st.spinner("🔄 Loading model..."):
        try:
            # Try to track memory usage, but don't fail if psutil not available
            try:
                import psutil

                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                has_psutil = True
            except ImportError:
                has_psutil = False
                mem_before = 0

            model = load_model(device)
            image_detector = ImageDetector(model, device)
            video_detector = VideoDetector(model, device)
            optimizer = TrafficLightOptimizer()

            if has_psutil:
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                mem_used = mem_after - mem_before

                if st.session_state.get("debug_realtime"):
                    st.info(
                        f"💾 Memory used for model: {mem_used:.1f}MB | Total: {mem_after:.1f}MB"
                    )

        except RuntimeError as e:
            if "cuda" in str(e).lower():
                st.error(f"❌ CUDA Error: {e}\nFalling back to CPU...")
                device = "cpu"
                model = load_model(device)
                image_detector = ImageDetector(model, device)
                video_detector = VideoDetector(model, device)
                optimizer = TrafficLightOptimizer()
            else:
                raise
        except Exception as e:
            st.error(f"❌ Model loading failed: {type(e).__name__}: {str(e)}")
            st.info("This might be a memory or GPU issue. Try reloading the page.")
            return

    st.success(f"✅ Model loaded! Device: {device.upper()}")

    st.markdown("---")

    # Route to selected mode
    if mode == "📤 Upload":
        upload_mode(image_detector, video_detector, optimizer, device, threshold)
    else:
        realtime_mode(image_detector, optimizer, device, threshold)


if __name__ == "__main__":
    main()
