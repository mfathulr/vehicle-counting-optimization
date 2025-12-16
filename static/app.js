// Suppress Chrome extension console errors
const originalError = console.error;
console.error = function(...args) {
    const message = args[0]?.toString() || '';
    if (message.includes('chrome-extension') || 
        message.includes('Denying load') ||
        message.includes('web_accessible_resources') ||
        message.includes('devtools') ||
        message.includes('GET chrome-extension://invalid/')) {
        return;
    }
    originalError.apply(console, args);
};

// ===== REALTIME MODE =====
let videoElement, canvasElement, ctx;
let stream = null;
let ws = null;
let isRunning = false;
let processingFps = 10;
let threshold = 0.5;
let lastFrameTime = 0;
let lastSendTime = 0;
let fpsHistory = [];
let inFlight = false; // backpressure flag
let lastLatencyMs = 0; // round-trip latency

// ===== UPLOAD MODE =====
let selectedFile = null;
let uploadThreshold = 0.5;

// DOM Elements - Realtime
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDisplay = document.getElementById('statusDisplay');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const fpsSlider = document.getElementById('fpsSlider');
const fpsValue = document.getElementById('fpsValue');
const fpsCounter = document.getElementById('fpsCounter');
const mobilCount = document.getElementById('mobilCount');
const motorCount = document.getElementById('motorCount');
const durationValue = document.getElementById('durationValue');
const sourceSelect = document.getElementById('sourceSelect');
const youtubeGroup = document.getElementById('youtubeGroup');
const youtubeUrl = document.getElementById('youtubeUrl');

// DOM Elements - Upload
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const downloadBtn = document.getElementById('downloadBtn');
const uploadStatusDisplay = document.getElementById('uploadStatusDisplay');
const uploadThresholdSlider = document.getElementById('uploadThresholdSlider');
const uploadThresholdValue = document.getElementById('uploadThresholdValue');
const uploadPreview = document.getElementById('uploadPreview');
const uploadMobilCount = document.getElementById('uploadMobilCount');
const uploadMotorCount = document.getElementById('uploadMotorCount');
const uploadDurationValue = document.getElementById('uploadDurationValue');

// Mode Selector
const modeBtns = document.querySelectorAll('.mode-btn');
const modeContents = document.querySelectorAll('.mode-content');

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    // Realtime mode init
    videoElement = document.getElementById('sourceVideo');
    canvasElement = document.getElementById('videoCanvas');
    ctx = canvasElement.getContext('2d');
    
    // Event listeners - Realtime
    startBtn.addEventListener('click', startDetection);
    stopBtn.addEventListener('click', stopDetection);
    
    thresholdSlider.addEventListener('input', (e) => {
        threshold = parseFloat(e.target.value);
        thresholdValue.textContent = threshold.toFixed(2);
    });
    
    fpsSlider.addEventListener('input', (e) => {
        processingFps = parseInt(e.target.value);
        fpsValue.textContent = processingFps;
    });

    sourceSelect.addEventListener('change', (e) => {
        if (e.target.value === 'youtube') {
            youtubeGroup.style.display = 'block';
        } else {
            youtubeGroup.style.display = 'none';
        }
    });

    // Event listeners - Upload
    fileInput.addEventListener('change', handleFileSelect);
    
    uploadThresholdSlider.addEventListener('input', (e) => {
        uploadThreshold = parseFloat(e.target.value);
        uploadThresholdValue.textContent = uploadThreshold.toFixed(2);
    });

    processBtn.addEventListener('click', processUploadedFile);
    downloadBtn.addEventListener('click', downloadResult);

    // Mode switcher
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            switchMode(mode);
        });
    });
});

// ===== MODE SWITCHER =====
function switchMode(mode) {
    // Update buttons
    modeBtns.forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
    
    // Update content
    modeContents.forEach(content => content.classList.remove('active'));
    document.getElementById(`${mode}-mode`).classList.add('active');
    
    // Stop detection if switching from realtime
    if (mode !== 'realtime' && isRunning) {
        stopDetection();
    }
}

// ===== REALTIME MODE FUNCTIONS =====
async function startDetection() {
    try {
        updateStatus('Initializing...', 'processing');
        const source = sourceSelect.value;
        
        if (source === 'webcam') {
            await startWebcamDetection();
        } else if (source === 'youtube') {
            await startYoutubeDetection();
        }
    } catch (error) {
        console.error('Error starting detection:', error);
        updateStatus(`❌ Error: ${error.message}`, 'disconnected');
        stopDetection();
    }
}

async function startWebcamDetection() {
    updateStatus('Requesting camera access...', 'processing');
    
    stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
        },
        audio: false
    });
    
    console.log('Camera stream acquired');
    videoElement.srcObject = stream;
    
    await new Promise((resolve) => {
        videoElement.onloadedmetadata = () => {
            console.log(`Video loaded: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
            videoElement.play();
            resolve();
        };
    });
    
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    
    connectWebSocket();
    
    startBtn.disabled = true;
    stopBtn.disabled = false;
    isRunning = true;
    
    updateStatus('✅ Connected & Processing', 'connected');
}

async function startYoutubeDetection() {
    const url = youtubeUrl.value.trim();
    if (!url) {
        updateStatus('❌ Please enter YouTube URL', 'disconnected');
        return;
    }
    
    updateStatus('Connecting to YouTube...', 'processing');
    
    try {
        const response = await fetch('/api/youtube-stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });
        
        if (!response.ok) throw new Error('Failed to connect to YouTube');
        
        const data = await response.json();
        videoElement.src = data.stream_url;
        
        await new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                resolve();
            };
        });
        
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        
        connectWebSocket();
        
        startBtn.disabled = true;
        stopBtn.disabled = false;
        isRunning = true;
        
        updateStatus('✅ Connected & Processing', 'connected');
    } catch (error) {
        throw new Error(`YouTube connection failed: ${error.message}`);
    }
}

function stopDetection() {
    isRunning = false;
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    if (ws) {
        ws.close();
        ws = null;
    }
    
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    updateStatus('⚠️ Disconnected', 'disconnected');
    
    mobilCount.textContent = '0';
    motorCount.textContent = '0';
    durationValue.textContent = '0s';
    fpsCounter.textContent = 'FPS: 0';
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/detect`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected - Starting frame processing loop...');
        updateStatus('✅ Connected & Processing', 'connected');
        requestAnimationFrame(processFrame);
    };
    
    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        if (message.type === 'result') {
            console.log('Detection result received:', {
                mobil_count: message.mobil_count,
                motor_count: message.motor_count,
            });
            // measure client-side round-trip latency
            if (message.sent_ts) {
                lastLatencyMs = Math.round(performance.now() - message.sent_ts);
            }
            handleDetectionResult(message);
            inFlight = false; // allow next frame
        } else if (message.type === 'error') {
            console.error('Detection error:', message.message);
            updateStatus(`⚠️ ${message.message}`, 'disconnected');
        inFlight = false;
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('❌ Connection error', 'disconnected');
    };
    
    ws.onclose = () => {
        console.log('WebSocket closed');
        if (isRunning) {
            updateStatus('⚠️ Connection lost', 'disconnected');
            stopDetection();
        }
    };
}

function processFrame(timestamp) {
    if (!isRunning) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    
    try {
        ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    } catch (error) {
        console.error('Error drawing to canvas:', error);
        requestAnimationFrame(processFrame);
        return;
    }
    
    const frameInterval = 1000 / processingFps;
    const elapsed = timestamp - lastSendTime;
    
    if (!inFlight && elapsed >= frameInterval && ws.readyState === WebSocket.OPEN) {
        lastSendTime = timestamp;
        
        const frameData = canvasElement.toDataURL('image/jpeg', 0.8);
        
        try {
            inFlight = true;
            ws.send(JSON.stringify({
                type: 'frame',
                data: frameData,
                threshold: threshold,
                sent_ts: performance.now()
            }));
        } catch (error) {
            console.error('Error sending frame:', error);
            inFlight = false;
        }
    }
    
    requestAnimationFrame(processFrame);
}

function handleDetectionResult(result) {
    const img = new Image();
    img.onload = () => {
        ctx.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
    };
    img.src = result.annotated_frame;
    
    mobilCount.textContent = result.mobil_count;
    motorCount.textContent = result.motor_count;
    durationValue.textContent = `${result.duration}s`;
}

function updateStatus(message, statusClass) {
    statusDisplay.textContent = message;
    statusDisplay.className = `status ${statusClass}`;
}

function updateFPS(timestamp) {
    fpsHistory.push(timestamp);
    
    if (fpsHistory.length > 30) {
        fpsHistory.shift();
    }
    
    if (fpsHistory.length >= 2) {
        const timeDiff = fpsHistory[fpsHistory.length - 1] - fpsHistory[0];
        const fps = Math.round((fpsHistory.length - 1) / (timeDiff / 1000));
        const latency = lastLatencyMs;
        fpsCounter.textContent = `FPS: ${fps} | RT: ${latency}ms`;
    }
}

// ===== UPLOAD MODE FUNCTIONS =====
function handleFileSelect(event) {
    selectedFile = event.target.files[0];
    
    if (!selectedFile) {
        updateUploadStatus('❌ No file selected', 'disconnected');
        processBtn.disabled = true;
        return;
    }
    
    updateUploadStatus('✅ File selected', 'processing');
    processBtn.disabled = false;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        if (selectedFile.type.startsWith('image/')) {
            uploadPreview.src = e.target.result;
            uploadPreview.classList.add('active');
            document.getElementById('videoResultContainer').style.display = 'none';
            document.getElementById('uploadCanvas').style.display = 'none';
        } else if (selectedFile.type.startsWith('video/')) {
            const video = document.createElement('video');
            video.src = e.target.result;
            video.onloadedmetadata = () => {
                const canvas = document.getElementById('uploadCanvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const canvasCtx = canvas.getContext('2d');
                canvasCtx.drawImage(video, 0, 0);
                canvas.style.display = 'block';
                uploadPreview.classList.remove('active');
                document.getElementById('videoResultContainer').style.display = 'none';
            };
        }
    };
    reader.readAsDataURL(selectedFile);
}

async function processUploadedFile() {
    if (!selectedFile) return;
    
    updateUploadStatus('Processing...', 'processing');
    processBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('threshold', uploadThreshold);
    
    try {
        const response = await fetch('/api/process-file', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Processing failed');
        
        const data = await response.json();
        
        // Handle IMAGE result
        if (data.file_type === 'image') {
            // Display result image
            const resultImg = document.createElement('img');
            resultImg.src = `data:image/jpeg;base64,${data.annotated_frame}`;
            uploadPreview.src = resultImg.src;
            uploadPreview.classList.add('active');
            document.getElementById('videoResultContainer').style.display = 'none';
            document.getElementById('uploadCanvas').style.display = 'none';
            
            // Update stats
            uploadMobilCount.textContent = data.mobil_count;
            uploadMotorCount.textContent = data.motor_count;
            uploadDurationValue.textContent = `${data.duration}s`;
            
            updateUploadStatus('✅ Image processing complete', 'connected');
            downloadBtn.disabled = false;
            downloadBtn.textContent = 'Download Image';
            
            // Store result for download
            window.lastResult = data;
        }
        // Handle VIDEO result
        else if (data.file_type === 'video') {
            // Hide preview, show video player
            uploadPreview.classList.remove('active');
            document.getElementById('uploadCanvas').style.display = 'none';
            
            // Show video result container and set video source
            const videoResultContainer = document.getElementById('videoResultContainer');
            const videoPlayer = document.getElementById('videoPlayer');
            videoPlayer.src = `data:video/mp4;base64,${data.video_data}`;
            videoResultContainer.style.display = 'block';
            
            // Update stats
            uploadMobilCount.textContent = data.mobil_count;
            uploadMotorCount.textContent = data.motor_count;
            uploadDurationValue.textContent = `${data.duration}s`;
            
            // Update status with frame info
            updateUploadStatus(`✅ Video processed (${data.frames_processed} frames)`, 'connected');
            downloadBtn.disabled = false;
            downloadBtn.textContent = 'Download Video';
            
            // Store result for download
            window.lastResult = data;
        }
        
    } catch (error) {
        updateUploadStatus(`❌ ${error.message}`, 'disconnected');
    }
    
    processBtn.disabled = false;
}

function downloadResult() {
    if (!window.lastResult) return;
    
    const link = document.createElement('a');
    
    // Download based on file type
    if (window.lastResult.file_type === 'image') {
        link.href = `data:image/jpeg;base64,${window.lastResult.annotated_frame}`;
        link.download = `detection_result_${Date.now()}.jpg`;
    } else if (window.lastResult.file_type === 'video') {
        link.href = `data:video/mp4;base64,${window.lastResult.video_data}`;
        link.download = `detection_result_${Date.now()}.mp4`;
    }
    
    link.click();
}

function updateUploadStatus(message, statusClass) {
    uploadStatusDisplay.textContent = message;
    uploadStatusDisplay.className = `status ${statusClass}`;
}
