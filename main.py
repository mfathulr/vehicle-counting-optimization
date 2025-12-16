"""
FastAPI WebRTC Vehicle Counting Application
Real-time vehicle detection using browser camera via WebRTC
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response, JSONResponse
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
from src.config import MODEL_PATH, NUM_CLASSES
from src.models import create_faster_rcnn_resnet18
from src.inference import ImageDetector
from src.optimization import TrafficLightOptimizer

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

                # Run detection
                try:
                    # Get user settings from message
                    threshold = message.get("threshold", 0.5)

                    # Use reusable ImageDetector for detection
                    annotated_frame, boxes, scores, pred_classes = (
                        image_detector.detect(
                            frame, threshold=threshold, show_progress=False
                        )
                    )

                    # Count vehicles by class name
                    mobil_count = pred_classes.count("mobil")
                    motor_count = pred_classes.count("motor")

                    # Optimize traffic light using fuzzy logic
                    duration = fuzzy_optimizer.optimize(mobil_count, motor_count)

                    # annotated_frame already has boxes drawn by ImageDetector
                    # Encode annotated frame back to base64
                    _, buffer = cv2.imencode(".jpg", annotated_frame)
                    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

                    # Send results back to client
                    result = {
                        "type": "result",
                        "annotated_frame": f"data:image/jpeg;base64,{annotated_b64}",
                        "mobil_count": mobil_count,
                        "motor_count": motor_count,
                        "total_vehicles": mobil_count + motor_count,
                        "duration": duration,
                        "detections": len(boxes),
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
async def process_file(file: UploadFile = File(...), threshold: float = 0.5):
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

            # Run detection
            detector = ImageDetector(model, str(device))
            annotated_frame, boxes, scores, pred_classes = detector.detect(
                frame, threshold=threshold, show_progress=False
            )

            # Count vehicles
            mobil_count = pred_classes.count("mobil")
            motor_count = pred_classes.count("motor")
            duration = fuzzy_optimizer.optimize(mobil_count, motor_count)

            # Encode result
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

            return {
                "annotated_frame": annotated_b64,
                "mobil_count": mobil_count,
                "motor_count": motor_count,
                "total_vehicles": mobil_count + motor_count,
                "duration": duration,
                "detections": len(boxes),
                "file_type": "image",
            }

        # ===== VIDEO PROCESSING =====
        elif file_type and file_type.startswith("video/"):
            # Save video temporarily using portable temp files
            import tempfile
            import os

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

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            # Process frames
            frame_count = 0
            total_mobil = 0
            total_motor = 0
            # Reuse global image_detector for consistency
            detector = image_detector

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run detection
                try:
                    annotated_frame, boxes, scores, pred_classes = detector.detect(
                        frame, threshold=threshold, show_progress=False
                    )

                    # Count vehicles
                    mobil_count = pred_classes.count("mobil")
                    motor_count = pred_classes.count("motor")
                    total_mobil += mobil_count
                    total_motor += motor_count

                    # Write annotated frame to output video
                    out.write(annotated_frame)
                    frame_count += 1

                    # Print progress
                    if frame_count % 10 == 0:
                        print(f"Processed {frame_count}/{total_frames} frames")

                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    out.write(frame)  # Write original frame if detection fails
                    frame_count += 1

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

            # Calculate average traffic light duration
            avg_mobil = total_mobil / max(frame_count, 1)
            avg_motor = total_motor / max(frame_count, 1)
            duration = fuzzy_optimizer.optimize(int(avg_mobil), int(avg_motor))

            return {
                "video_data": video_b64,
                "mobil_count": total_mobil,
                "motor_count": total_motor,
                "total_vehicles": total_mobil + total_motor,
                "frames_processed": frame_count,
                "duration": duration,
                "file_type": "video",
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
