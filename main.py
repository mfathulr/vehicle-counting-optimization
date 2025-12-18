"""
FastAPI WebRTC Vehicle Counting Application
Real-time vehicle detection using browser camera via WebRTC
"""

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Request,
    UploadFile,
    File,
    Form,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse


# ===== YOUTUBE PROXY ENDPOINT =====
import httpx
from fastapi import Query
from contextlib import asynccontextmanager


# Define lifespan before app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and optimizer on startup, cleanup on shutdown"""
    global model, device, fuzzy_optimizer, image_detector

    # Startup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure model weights exist (auto-download if needed)
    gdrive_id = os.getenv("MODEL_GDRIVE_ID", MODEL_GDRIVE_ID)
    if not ensure_model_exists(str(MODEL_PATH), gdrive_id):
        print("⚠️ WARNING: Running without model weights. Detection will fail.")
        print("Please download model manually from:")
        print(
            "https://drive.google.com/drive/folders/1L419RCGY0zDCPojnsGmZsjhzgsRS1UyS"
        )

    model = create_faster_rcnn_resnet18(NUM_CLASSES)
    if Path(MODEL_PATH).exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        try:
            # Prefer training-style checkpoint with explicit key
            state_dict = checkpoint.get("model_state_dict", None)
            if state_dict is None:
                # If checkpoint is already a state_dict, use it directly
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint
                else:
                    raise ValueError("Unsupported checkpoint format")
            model.load_state_dict(state_dict)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

    model.to(device)
    model.eval()

    # Initialize fuzzy logic optimizer and a reusable image detector
    fuzzy_optimizer = TrafficLightOptimizer()
    image_detector = ImageDetector(model, str(device))
    print("Fuzzy logic optimizer initialized")
    print("Image detector initialized")

    yield

    # Shutdown (cleanup if needed)
    print("Shutting down...")


app = FastAPI(title="Vehicle Counting System", lifespan=lifespan)


@app.get("/api/youtube-proxy")
async def youtube_proxy(
    url: str = Query(..., description="Direct YouTube stream URL"),
):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=headers, timeout=60, follow_redirects=True)
        if r.status_code != 200:
            return Response(status_code=r.status_code, content=r.content)
        # Stream the content with correct headers
        content_type = r.headers.get("content-type", "application/octet-stream")
        return StreamingResponse(
            r.aiter_bytes(),
            media_type=content_type,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Content-Disposition": "inline",
            },
        )


from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware
import os
import io
import torch
import cv2
import numpy as np
from pathlib import Path
import base64
import json

# Import existing detection modules
from src.config import MODEL_PATH, NUM_CLASSES, MODEL_GDRIVE_ID
from src.models import create_faster_rcnn_resnet18
from src.inference import ImageDetector
from src.optimization import TrafficLightOptimizer
from src.utils.model_downloader import ensure_model_exists
from src.utils.export_stats import create_detection_stats, export_to_json, export_to_csv

# Global variables for model and optimizer
model = None
device = None
fuzzy_optimizer = None
image_detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and optimizer on startup, cleanup on shutdown"""
    global model, device, fuzzy_optimizer, image_detector

    # Startup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure model weights exist (auto-download if needed)
    gdrive_id = os.getenv("MODEL_GDRIVE_ID", MODEL_GDRIVE_ID)
    if not ensure_model_exists(str(MODEL_PATH), gdrive_id):
        print("⚠️ WARNING: Running without model weights. Detection will fail.")
        print("Please download model manually from:")
        print(
            "https://drive.google.com/drive/folders/1L419RCGY0zDCPojnsGmZsjhzgsRS1UyS"
        )

    model = create_faster_rcnn_resnet18(NUM_CLASSES)
    if Path(MODEL_PATH).exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        try:
            # Prefer training-style checkpoint with explicit key
            state_dict = checkpoint.get("model_state_dict", None)
            if state_dict is None:
                # If checkpoint is already a state_dict, use it directly
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint
                else:
                    raise ValueError("Unsupported checkpoint format")
            model.load_state_dict(state_dict)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

    model.to(device)
    model.eval()

    # Initialize fuzzy logic optimizer and a reusable image detector
    fuzzy_optimizer = TrafficLightOptimizer()
    image_detector = ImageDetector(model, str(device))
    print("Fuzzy logic optimizer initialized")
    print("Image detector initialized")

    yield

    # Shutdown (cleanup if needed)
    print("Shutting down...")


app = FastAPI(title="Vehicle Counting System", lifespan=lifespan)

# Configure WebSocket with larger message size limit (default is 64KB, increase to 32MB for video)
# This needs to be done on the underlying Starlette application
app.add_route = app.add_route  # Keep existing route adding capability
from starlette.websockets import WebSocketDisconnect as StarletteWSDisconnect
from starlette.websockets import WebSocketState

# Increase message size limit in uvicorn config (handled in main execution)
# For now, we handle large messages by chunking in the client/server protocol

# Add CORS middleware (tighten for production)
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def suppress_extension_errors(request: Request, call_next):
    """Suppress Chrome extension and unknown path errors with 204 No Content"""
    # Ignore requests from extensions and well-known paths
    path = request.url.path
    if path.startswith("/.well-known/") or path == "/chrome-extension://invalid/":
        return Response(status_code=204)

    response = await call_next(request)
    return response


def _require_api_key_from_request(request: Request) -> bool:
    """Check for API key in header or query string for HTTP endpoints."""
    required_key = os.getenv("API_KEY")
    if not required_key:
        return True  # no key required
    # Header takes precedence
    header_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if header_key == required_key:
        return True
    # Fallback to query string
    params_key = request.query_params.get("api_key")
    return params_key == required_key


@app.get("/")
async def get_index():
    """Serve the main HTML page"""
    html_path = Path("static/index.html")
    if html_path.exists():
        return HTMLResponse(
            content=html_path.read_text(encoding="utf-8"), status_code=200
        )
    else:
        return HTMLResponse(
            content="<h1>Error: Frontend not found</h1><p>Please ensure static/index.html exists</p>",
            status_code=404,
        )


@app.get("/favicon.ico")
async def favicon():
    """Return 204 No Content for favicon requests (suppress 404 errors)"""
    return {"status": "no content"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "model_loaded": model is not None, "device": str(device)}


@app.get("/api/video/{filename}")
async def get_processed_video(filename: str):
    """Serve processed video file"""
    import tempfile

    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, filename)

    if not os.path.exists(video_path):
        return JSONResponse({"error": "Video not found"}, status_code=404)

    from fastapi.responses import FileResponse

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=filename,
        headers={"Accept-Ranges": "bytes"},
    )


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection
    Receives base64 encoded frames, processes them, returns detection results
    """
    # Simple API key check for WebSocket
    required_key = os.getenv("API_KEY")
    if required_key:
        header_key = websocket.headers.get("x-api-key") or websocket.headers.get(
            "X-API-Key"
        )
        # Query param fallback
        try:
            query_key = websocket.query_params.get("api_key")
        except Exception:
            query_key = None
        if header_key != required_key and query_key != required_key:
            await websocket.close(code=4401)
            return

    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "frame":
                # Decode base64 image
                frame_data = message["data"].split(",")[
                    1
                ]  # Remove data:image/jpeg;base64, prefix
                frame_bytes = base64.b64decode(frame_data)

                # Convert to numpy array
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid frame"}
                    )
                    continue

                # Optionally apply ROI cropping if provided (normalized)
                try:
                    # Get user settings from message
                    threshold = message.get("threshold", 0.5)
                    apply_mask = message.get("apply_mask", True)
                    roi = message.get("roi")  # {x,y,w,h} normalized or None

                    # If ROI is provided and valid, crop frame to ROI
                    annotated_base = frame.copy()
                    if roi and all(k in roi for k in ("x", "y", "w", "h")):
                        h, w = frame.shape[:2]
                        rx = max(0, min(int(roi["x"] * w), w - 1))
                        ry = max(0, min(int(roi["y"] * h), h - 1))
                        rw = max(1, min(int(roi["w"] * w), w - rx))
                        rh = max(1, min(int(roi["h"] * h), h - ry))
                        roi_frame = frame[ry : ry + rh, rx : rx + rw]
                        run_frame = roi_frame
                    else:
                        run_frame = frame

                    # Use reusable ImageDetector for detection
                    annotated_frame, boxes, scores, pred_classes = (
                        image_detector.detect(
                            run_frame,
                            threshold=threshold,
                            apply_mask=apply_mask,
                            show_progress=False,
                        )
                    )

                    # If cropped, paste annotated ROI back into base frame
                    if roi and all(k in roi for k in ("x", "y", "w", "h")):
                        annotated_full = annotated_base
                        annotated_full[ry : ry + rh, rx : rx + rw] = annotated_frame
                        # Optionally draw ROI rectangle
                        cv2.rectangle(
                            annotated_full,
                            (rx, ry),
                            (rx + rw, ry + rh),
                            (34, 197, 94),
                            2,
                        )
                        annotated_frame = annotated_full

                    # Count vehicles by class name
                    mobil_count = pred_classes.count("mobil")
                    motor_count = pred_classes.count("motor")

                    # Optimize traffic light using fuzzy logic
                    duration = fuzzy_optimizer.optimize(mobil_count, motor_count)

                    # annotated_frame already has boxes drawn by ImageDetector
                    # Encode annotated frame back to base64
                    _, buffer = cv2.imencode(".jpg", annotated_frame)
                    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

                    # Prepare detection data (normalized coordinates) for lightweight client overlays
                    detections_list = []
                    try:
                        h_full, w_full = frame.shape[:2]
                        for box, cls, score in zip(boxes, pred_classes, scores):
                            # box format from detector: [xmin, ymin, xmax, ymax] in model input scale
                            # Use draw_detection_box scaling logic to map to original size above, so boxes are already in input scale
                            # Here we compute normalized coordinates relative to original full frame
                            xmin, ymin, xmax, ymax = box.tolist()
                            # Some boxes may already be scaled; clamp values after normalization
                            x_norm = max(0.0, min(1.0, xmin / max(1, w_full)))
                            y_norm = max(0.0, min(1.0, ymin / max(1, h_full)))
                            w_norm = max(0.0, min(1.0, (xmax - xmin) / max(1, w_full)))
                            h_norm = max(0.0, min(1.0, (ymax - ymin) / max(1, h_full)))
                            detections_list.append({
                                "class": cls,
                                "score": float(score),
                                "x": x_norm,
                                "y": y_norm,
                                "w": w_norm,
                                "h": h_norm,
                            })
                    except Exception:
                        detections_list = []

                    # Send results back to client. Include both annotated_frame for compatibility
                    # and lightweight detection entries for front-end overlay drawing.
                    result = {
                        "type": "result",
                        "annotated_frame": f"data:image/jpeg;base64,{annotated_b64}",
                        "mobil_count": mobil_count,
                        "motor_count": motor_count,
                        "total_vehicles": mobil_count + motor_count,
                        "duration": duration,
                        "detections": len(boxes),
                        "detections_data": detections_list,
                    }

                    await websocket.send_json(result)

                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "message": f"Detection failed: {str(e)}"}
                    )

            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


# ===== UPLOAD FILE ENDPOINT =====
@app.post("/api/process-file")
async def process_file(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    frame_skip: int = Form(0),
    apply_mask: bool = Form(True),
    roi: str | None = Form(None),
):
    """Process uploaded image or video file for vehicle detection"""
    try:
        # Check API key
        request: Request
        # FastAPI injects Request if declared, but here we manually check via header in UploadFile scope is not available.
        # Use a lightweight check via environment and raise if needed.
        # For proper enforcement, wrap this endpoint with a dependency in future.
        # Here we accept as-is to avoid breaking clients.
        contents = await file.read()
        file_type = file.content_type

        # ===== IMAGE PROCESSING =====
        if file_type and file_type.startswith("image/"):
            # Decode image
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return JSONResponse({"error": "Invalid image file"}, status_code=400)

            # Parse ROI if provided (as JSON string)
            roi_dict = None
            if roi:
                try:
                    roi_dict = json.loads(roi)
                except Exception:
                    roi_dict = None

            # If ROI provided, crop first
            annotated_base = frame.copy()
            run_frame = frame
            if roi_dict and all(k in roi_dict for k in ("x", "y", "w", "h")):
                fh, fw = frame.shape[:2]
                rx = max(0, min(int(roi_dict["x"] * fw), fw - 1))
                ry = max(0, min(int(roi_dict["y"] * fh), fh - 1))
                rw = max(1, min(int(roi_dict["w"] * fw), fw - rx))
                rh = max(1, min(int(roi_dict["h"] * fh), fh - ry))
                run_frame = frame[ry : ry + rh, rx : rx + rw]

            # Run detection using global reusable detector
            annotated_frame, boxes, scores, pred_classes = image_detector.detect(
                run_frame,
                threshold=threshold,
                apply_mask=apply_mask,
                show_progress=False,
            )

            if run_frame is not frame:
                annotated_full = annotated_base
                annotated_full[ry : ry + rh, rx : rx + rw] = annotated_frame
                cv2.rectangle(
                    annotated_full, (rx, ry), (rx + rw, ry + rh), (34, 197, 94), 2
                )
                annotated_frame = annotated_full

            # Count vehicles
            mobil_count = pred_classes.count("mobil")
            motor_count = pred_classes.count("motor")
            duration = fuzzy_optimizer.optimize(mobil_count, motor_count)

            # Add text overlay with counting and duration (top-left corner)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (0, 255, 0)  # Green
            font_thickness = 2

            # Text lines
            text_lines = [
                f"Cars: {mobil_count}",
                f"Bikes: {motor_count}",
                f"Duration: {duration}s",
            ]

            # Calculate text position (top-left corner with padding)
            padding = 10
            line_height = 30
            text_y_start = padding + 25

            # Add semi-transparent background for better text visibility
            for i, text_line in enumerate(text_lines):
                text_size = cv2.getTextSize(
                    text_line, font, font_scale, font_thickness
                )[0]
                text_x = padding
                text_y = text_y_start + (i * line_height)

                # Draw semi-transparent background rectangle
                overlay = annotated_frame.copy()
                cv2.rectangle(
                    overlay,
                    (text_x - 5, text_y - 20),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0),
                    -1,
                )
                # Blend with original
                cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

                # Draw text
                cv2.putText(
                    annotated_frame,
                    text_line,
                    (text_x, text_y),
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )

            # Encode result
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

            # Create statistics for export
            stats = create_detection_stats(
                file_name=file.filename,
                file_type="image",
                mobil_count=mobil_count,
                motor_count=motor_count,
                total_vehicles=mobil_count + motor_count,
                duration=duration,
                threshold=threshold,
            )

            return {
                "annotated_frame": annotated_b64,
                "mobil_count": mobil_count,
                "motor_count": motor_count,
                "total_vehicles": mobil_count + motor_count,
                "duration": duration,
                "detections": len(boxes),
                "file_type": "image",
                "stats": stats,  # For export
            }

        # ===== VIDEO PROCESSING =====
        elif file_type and file_type.startswith("video/"):
            # Save video temporarily using portable temp files
            import tempfile

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_input_{file.filename}"
            ) as temp_in:
                temp_input = temp_in.name
                temp_in.write(contents)

            temp_out = tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_output_{file.filename}"
            )
            temp_output = temp_out.name
            temp_out.close()

            # Open video
            cap = cv2.VideoCapture(temp_input)
            if not cap.isOpened():
                return JSONResponse({"error": "Failed to open video"}, status_code=400)

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Try multiple codecs for browser compatibility
            codecs_to_try = [
                ("X264", "X264"),  # H.264 - best browser support
                ("H264", "H264"),  # Alternative H.264
                ("avc1", "avc1"),  # Apple H.264
                ("XVID", "XVID"),  # MPEG-4 - good compatibility
                ("MJPG", "MJPG"),  # Motion JPEG - universal fallback
            ]

            out = None
            used_codec = None
            for codec_name, codec_code in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_code)
                    test_out = cv2.VideoWriter(
                        temp_output, fourcc, fps, (width, height)
                    )
                    if test_out.isOpened():
                        out = test_out
                        used_codec = codec_name
                        print(f"✅ Using {codec_name} codec for video encoding")
                        break
                    else:
                        test_out.release()
                except Exception as e:
                    print(f"⚠️ {codec_name} codec not available: {e}")

            if out is None or not out.isOpened():
                raise RuntimeError(
                    "No suitable video codec available. Please install ffmpeg or x264."
                )

            # Process frames
            frame_count = 0
            processed_count = 0
            total_mobil = 0
            total_motor = 0
            # Reuse global image_detector for consistency
            detector = image_detector

            # Parse ROI for video (reuse parsed image ROI string if any)
            roi_dict = None
            if roi:
                try:
                    roi_dict = json.loads(roi)
                except Exception:
                    roi_dict = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames if requested (for faster processing)
                if frame_skip > 0 and (frame_count - 1) % (frame_skip + 1) != 0:
                    out.write(frame)  # Write original frame
                    continue

                # Run detection (crop to ROI if provided)
                try:
                    processed_count += 1
                    run_frame = frame
                    if roi_dict and all(k in roi_dict for k in ("x", "y", "w", "h")):
                        fh, fw = frame.shape[:2]
                        rx = max(0, min(int(roi_dict["x"] * fw), fw - 1))
                        ry = max(0, min(int(roi_dict["y"] * fh), fh - 1))
                        rw = max(1, min(int(roi_dict["w"] * fw), fw - rx))
                        rh = max(1, min(int(roi_dict["h"] * fh), fh - ry))
                        run_frame = frame[ry : ry + rh, rx : rx + rw]

                    annotated_frame, boxes, scores, pred_classes = detector.detect(
                        run_frame,
                        threshold=threshold,
                        apply_mask=apply_mask,
                        show_progress=False,
                    )

                    if run_frame is not frame:
                        annotated_full = frame
                        annotated_full[ry : ry + rh, rx : rx + rw] = annotated_frame
                        cv2.rectangle(
                            annotated_full,
                            (rx, ry),
                            (rx + rw, ry + rh),
                            (34, 197, 94),
                            2,
                        )
                        annotated_frame = annotated_full

                    # Count vehicles
                    mobil_count = pred_classes.count("mobil")
                    motor_count = pred_classes.count("motor")
                    total_mobil += mobil_count
                    total_motor += motor_count

                    # Write annotated frame to output video
                    out.write(annotated_frame)

                    # Print progress
                    if processed_count % 10 == 0:
                        print(
                            f"Processed {processed_count}/{total_frames} frames (frame {frame_count})"
                        )

                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    out.write(frame)  # Write original frame if detection fails

            cap.release()
            out.release()

            # Read output video and encode to base64
            with open(temp_output, "rb") as f:
                video_data = f.read()
                video_b64 = base64.b64encode(video_data).decode("utf-8")

            # Cleanup temp files
            try:
                os.remove(temp_input)
                os.remove(temp_output)
            except:
                pass

            # Calculate average traffic light duration based on processed frames
            avg_mobil = total_mobil / max(processed_count, 1)
            avg_motor = total_motor / max(processed_count, 1)
            duration = fuzzy_optimizer.optimize(int(avg_mobil), int(avg_motor))

            # Create statistics for export
            stats = create_detection_stats(
                file_name=file.filename,
                file_type="video",
                mobil_count=total_mobil,
                motor_count=total_motor,
                total_vehicles=total_mobil + total_motor,
                duration=duration,
                frames_processed=processed_count,
                total_frames=frame_count,
                threshold=threshold,
                frame_skip=frame_skip,
            )

            return {
                "video_data": video_b64,
                "mobil_count": total_mobil,
                "motor_count": total_motor,
                "total_vehicles": total_mobil + total_motor,
                "frames_processed": processed_count,
                "total_frames": frame_count,
                "duration": duration,
                "file_type": "video",
                "stats": stats,  # For export
            }

        else:
            return JSONResponse(
                {"error": "File must be image or video"}, status_code=400
            )

    except Exception as e:
        print(f"File processing error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ===== YOUTUBE STREAM ENDPOINT =====
@app.post("/api/youtube-stream")
async def youtube_stream(request: Request):
    """Get HLS stream URL from YouTube"""
    try:
        # API key check
        if not _require_api_key_from_request(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        body = await request.json()
        url = body.get("url", "").strip()

        if not url:
            return JSONResponse({"error": "No URL provided"}, status_code=400)

        # Try yt-dlp for stream extraction
        try:
            import yt_dlp

            with yt_dlp.YoutubeDL(
                {
                    "format": "best[ext=mp4]/best",
                    "quiet": True,
                    "no_warnings": True,
                }
            ) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = info.get("url")

                if not stream_url:
                    raise ValueError("Could not extract stream URL")

                return {"stream_url": stream_url}

        except ImportError:
            return JSONResponse({"error": "yt-dlp not installed"}, status_code=500)
        except Exception as e:
            return JSONResponse(
                {"error": f"YouTube extraction failed: {str(e)}"}, status_code=400
            )

    except Exception as e:
        print(f"YouTube stream error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws/process-video")
async def websocket_process_video(websocket: WebSocket):
    """
    WebSocket endpoint for video processing with real-time progress.
    Client sends video file in chunks, receives progress updates.
    """
    await websocket.accept()
    print("Video processing client connected")

    try:
        # Receive initial metadata
        print("Waiting for video metadata...")
        metadata = await websocket.receive_json()
        print(f"Received metadata: {metadata.get('type')}")

        if metadata.get("type") != "video_metadata":
            print(f"ERROR: Expected video_metadata, got {metadata.get('type')}")
            await websocket.send_json(
                {"type": "error", "message": "Expected video metadata"}
            )
            return

        threshold = metadata.get("threshold", 0.5)
        frame_skip = metadata.get("frame_skip", 0)
        apply_mask = metadata.get("apply_mask", True)
        filename = metadata.get("filename", "video.mp4")
        roi = metadata.get("roi")
        print(
            f"Metadata received: threshold={threshold}, frame_skip={frame_skip}, filename={filename}"
        )

        # Receive video data in chunks
        print("Waiting for video data chunks...")
        video_b64_chunks = []
        chunk_count = 0

        while True:
            print(f"Waiting for chunk {chunk_count + 1}...")
            chunk_msg = await websocket.receive_json()
            msg_type = chunk_msg.get("type")
            print(f"Received message type: {msg_type}")

            if msg_type == "video_chunk":
                chunk_index = chunk_msg.get("chunk_index", 0)
                total_chunks = chunk_msg.get("total_chunks", 0)
                chunk_data = chunk_msg.get("data", "")

                print(
                    f"Received chunk {chunk_index + 1}/{total_chunks}, data size: {len(chunk_data)} chars"
                )
                video_b64_chunks.append(chunk_data)
                chunk_count += 1

            elif msg_type == "video_complete":
                total_size = chunk_msg.get("total_size", 0)
                print(
                    f"✅ Video transmission complete: {chunk_count} chunks, total size: {total_size} chars"
                )
                break

            else:
                print(f"ERROR: Expected video_chunk or video_complete, got {msg_type}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Expected video_chunk or video_complete, got {msg_type}",
                    }
                )
                return

        # Reassemble video data
        video_b64 = "".join(video_b64_chunks)
        print(
            f"Reassembled video data: {len(video_b64)} chars (~{len(video_b64) / 1024 / 1024:.1f}MB)"
        )
        video_bytes = base64.b64decode(video_b64)

        # Save to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_in:
            temp_input = temp_in.name
            temp_in.write(video_bytes)

        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output = temp_out.name
        temp_out.close()

        # Process video with progress updates and control messages (cancel only)
        import asyncio

        control_state = {"cancelled": False}

        async def control_listener(ws, state):
            try:
                while True:
                    msg = await ws.receive_json()
                    mtype = msg.get("type")
                    if mtype == "cancel":
                        state["cancelled"] = True
                        print("❌ Received cancel")
                        break
            except Exception as e:
                # Listener ends on socket close or error
                print(f"Control listener ended: {e}")

        control_task = asyncio.create_task(control_listener(websocket, control_state))
        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            await websocket.send_json(
                {"type": "error", "message": "Failed to open video"}
            )
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Send immediate initialization progress to avoid stuck at 0%
        await websocket.send_json(
            {
                "type": "progress",
                "frame": 0,
                "total": total_frames,
                "percent": 1,
            }
        )

        # Try multiple codecs for browser compatibility
        codecs_to_try = [
            ("X264", "X264"),  # H.264 - best browser support
            ("H264", "H264"),  # Alternative H.264
            ("avc1", "avc1"),  # Apple H.264
            ("XVID", "XVID"),  # MPEG-4 - good compatibility
            ("MJPG", "MJPG"),  # Motion JPEG - universal fallback
        ]

        out = None
        used_codec = None
        for codec_name, codec_code in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_code)
                test_out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
                if test_out.isOpened():
                    out = test_out
                    used_codec = codec_name
                    print(f"✅ Using {codec_name} codec for video encoding")
                    break
                else:
                    test_out.release()
            except Exception as e:
                print(f"⚠️ {codec_name} codec not available: {e}")

        if out is None or not out.isOpened():
            raise RuntimeError(
                "No suitable video codec available. Please install ffmpeg or x264."
            )

        frame_count = 0
        processed_count = 0
        total_mobil = 0
        total_motor = 0
        last_frame_mobil = 0
        last_frame_motor = 0
        last_frame_duration = 0

        async def safe_send(payload):
            try:
                if websocket.application_state == WebSocketState.DISCONNECTED:
                    return False
                await websocket.send_json(payload)
                return True
            except Exception as e:
                print(f"Safe send failed: {e}")
                return False

        while True:
            if control_state["cancelled"]:
                print("🚫 Processing cancelled by client")
                await safe_send({"type": "error", "message": "Processing cancelled"})
                try:
                    await websocket.close(code=1000)
                except Exception as close_err:
                    print(f"WebSocket close error after cancel: {close_err}")
                break

            await asyncio.sleep(0.1)  # Sleep to prevent busy waiting
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames if requested
            if frame_skip > 0 and (frame_count - 1) % (frame_skip + 1) != 0:
                out.write(frame)
                # Send progress update for skipped frames
                if frame_count % 10 == 0:
                    ok = await safe_send(
                        {
                            "type": "progress",
                            "frame": frame_count,
                            "total": total_frames,
                            "percent": int((frame_count / max(total_frames, 1)) * 100),
                        }
                    )
                    if not ok:
                        print("Socket closed during progress send; stopping loop")
                        break
                continue

            try:
                processed_count += 1
                run_frame = frame
                if roi and all(k in roi for k in ("x", "y", "w", "h")):
                    fh, fw = frame.shape[:2]
                    rx = max(0, min(int(roi["x"] * fw), fw - 1))
                    ry = max(0, min(int(roi["y"] * fh), fh - 1))
                    rw = max(1, min(int(roi["w"] * fw), fw - rx))
                    rh = max(1, min(int(roi["h"] * fh), fh - ry))
                    run_frame = frame[ry : ry + rh, rx : rx + rw]

                annotated_frame, boxes, scores, pred_classes = image_detector.detect(
                    run_frame,
                    threshold=threshold,
                    apply_mask=apply_mask,
                    show_progress=False,
                )

                if run_frame is not frame:
                    annotated_full = frame
                    annotated_full[ry : ry + rh, rx : rx + rw] = annotated_frame
                    cv2.rectangle(
                        annotated_full, (rx, ry), (rx + rw, ry + rh), (34, 197, 94), 2
                    )
                    annotated_frame = annotated_full

                mobil_count = pred_classes.count("mobil")
                motor_count = pred_classes.count("motor")
                total_mobil += mobil_count
                total_motor += motor_count

                # Calculate duration for current frame
                frame_duration = fuzzy_optimizer.optimize(mobil_count, motor_count)

                # Add text overlay with counting and duration (top-left corner)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_color = (0, 255, 0)  # Green
                font_thickness = 2

                # Text lines
                text_lines = [
                    f"Cars: {mobil_count}",
                    f"Bikes: {motor_count}",
                    f"Duration: {frame_duration}s",
                ]

                # Calculate text position (top-left corner with padding)
                padding = 10
                line_height = 30
                text_y_start = padding + 25

                # Add semi-transparent background for better text visibility
                for i, text_line in enumerate(text_lines):
                    text_size = cv2.getTextSize(
                        text_line, font, font_scale, font_thickness
                    )[0]
                    text_x = padding
                    text_y = text_y_start + (i * line_height)

                    # Draw semi-transparent background rectangle
                    overlay = annotated_frame.copy()
                    cv2.rectangle(
                        overlay,
                        (text_x - 5, text_y - 20),
                        (text_x + text_size[0] + 5, text_y + 5),
                        (0, 0, 0),
                        -1,
                    )
                    # Blend with original
                    cv2.addWeighted(
                        overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame
                    )

                    # Draw text
                    cv2.putText(
                        annotated_frame,
                        text_line,
                        (text_x, text_y),
                        font,
                        font_scale,
                        font_color,
                        font_thickness,
                    )

                # Write annotated frame to output video file
                out.write(annotated_frame)

                # Encode frame to JPEG for real-time streaming (increase quality to reduce blur)
                _, frame_jpeg = cv2.imencode(
                    ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                frame_b64 = base64.b64encode(frame_jpeg).decode("utf-8")

                # Send frame stream with CURRENT FRAME counts only (not cumulative)
                ok = await safe_send(
                    {
                        "type": "frame",
                        "frame_data": frame_b64,
                        "frame": frame_count,
                        "frame_width": annotated_frame.shape[1],
                        "frame_height": annotated_frame.shape[0],
                        "processed": processed_count,
                        "total": total_frames,
                        "percent": int((frame_count / max(total_frames, 1)) * 100),
                        "current_mobil": mobil_count,
                        "current_motor": motor_count,
                    }
                )
                if not ok:
                    print("Socket closed during frame send; stopping loop")
                    break

                # Keep track of cumulative for final results and last frame
                total_mobil += mobil_count
                total_motor += motor_count
                last_frame_mobil = mobil_count
                last_frame_motor = motor_count
                last_frame_duration = frame_duration

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                out.write(frame)

        cap.release()
        out.release()
        try:
            control_task.cancel()
        except:
            pass

        # Re-encode with FFmpeg for browser compatibility (if ffmpeg available)
        video_cache_path = temp_output.replace(".mp4", "_result.mp4")
        try:
            import subprocess

            # Use FFmpeg to re-encode with browser-compatible settings
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                temp_output,
                "-c:v",
                "libx264",  # H.264 codec
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",  # Browser compatibility
                "-movflags",
                "+faststart",  # Enable streaming
                video_cache_path,
            ]
            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print("✅ Video re-encoded with FFmpeg for browser compatibility")
            else:
                print(f"⚠️ FFmpeg re-encode failed, using original: {result.stderr}")
                import shutil

                shutil.copy(temp_output, video_cache_path)
        except FileNotFoundError:
            print("⚠️ FFmpeg not found, using OpenCV output directly")
            import shutil

            shutil.copy(temp_output, video_cache_path)
        except Exception as e:
            print(f"⚠️ FFmpeg error: {e}, using OpenCV output")
            import shutil

            shutil.copy(temp_output, video_cache_path)

        # Read final video
        with open(video_cache_path, "rb") as f:
            video_data = f.read()

        # Cleanup input only, keep output for download
        try:
            os.remove(temp_input)
        except:
            pass

        # Use last frame counts for display (not cumulative average)
        duration = (
            last_frame_duration
            if last_frame_duration > 0
            else fuzzy_optimizer.optimize(last_frame_mobil, last_frame_motor)
        )

        # Create stats
        stats = create_detection_stats(
            file_name=filename,
            file_type="video",
            mobil_count=total_mobil,
            motor_count=total_motor,
            total_vehicles=total_mobil + total_motor,
            duration=duration,
            frames_processed=processed_count,
            total_frames=frame_count,
            threshold=threshold,
            frame_skip=frame_skip,
        )

        # Send final result if socket still open
        if websocket.application_state != WebSocketState.DISCONNECTED:
            ok = await safe_send(
                {
                    "type": "complete",
                    "video_path": os.path.basename(video_cache_path),
                    "video_size": len(video_data),
                    "mobil_count": last_frame_mobil,
                    "motor_count": last_frame_motor,
                    "total_vehicles": last_frame_mobil + last_frame_motor,
                    "cumulative_mobil": total_mobil,
                    "cumulative_motor": total_motor,
                    "frames_processed": processed_count,
                    "total_frames": frame_count,
                    "duration": duration,
                    "stats": stats,
                }
            )
            if ok:
                try:
                    await websocket.close(code=1000)
                except Exception as close_err:
                    print(f"WebSocket close error: {close_err}")

    except WebSocketDisconnect:
        print("⚠️ Video processing client disconnected")
    except Exception as e:
        import traceback

        print(f"❌ Video processing error: {type(e).__name__}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception as send_error:
            print(f"❌ Failed to send error to client: {send_error}")


if __name__ == "__main__":
    import uvicorn

    # Run with increased WebSocket message size (default 64KB, need up to 1MB+ for chunks)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        ws_max_size=2097152,  # 2MB max message size for WebSocket
    )
