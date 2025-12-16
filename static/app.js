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
let roiCanvas, roiCtx; // realtime ROI overlay
let uploadRoiCanvas, uploadRoiCtx; // upload ROI overlay
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

// ROI State
let isDrawingRoi = false;
let roiStart = null;
let roiRealtime = null; // {x,y,w,h} normalized [0,1]
let roiUpload = null; // {x,y,w,h} normalized [0,1]

// ===== UPLOAD MODE =====
let selectedFile = null;
let uploadThreshold = 0.5;
const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500MB in bytes

// DOM Elements - Realtime
const previewBtn = document.getElementById('previewBtn');
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
const frameSkipSlider = document.getElementById('frameSkipSlider');
const frameSkipValue = document.getElementById('frameSkipValue');
const frameSkipGroup = document.getElementById('frameSkipGroup');
const progressContainer = document.getElementById('progressContainer');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const screenshotBtn = document.getElementById('screenshotBtn');
const exportJsonBtn = document.getElementById('exportJsonBtn');
const exportCsvBtn = document.getElementById('exportCsvBtn');
const maskToggle = document.getElementById('maskToggle');
const uploadMaskToggle = document.getElementById('uploadMaskToggle');
// Dynamic processing dots spinner
let processingDotsIndex = 0;
const PROCESSING_DOTS = ['.', '..', '...', '....', '.....'];
function nextProcessingDots() {
    processingDotsIndex = (processingDotsIndex + 1) % PROCESSING_DOTS.length;
    return PROCESSING_DOTS[processingDotsIndex];
}

// Simple fuzzy logic traffic light duration calculator (matches backend)
// Config from src/config.py
const FUZZY_CONFIG = {
    // Motor ranges: [min, mid, max]
    motor_low: [0, 0, 15],
    motor_medium: [0, 15, 30],
    motor_high: [15, 30, 30],
    // Mobil ranges: [min, mid, max]
    mobil_low: [0, 0, 5],
    mobil_medium: [0, 5, 10],
    mobil_high: [5, 10, 10],
    // Duration ranges: [min, mid, max] or [min, mid1, mid2, max]
    duration_short: [0, 0, 20],
    duration_medium: [10, 20, 40, 50],
    duration_long: [40, 60, 60],
};

// Triangular membership function
function trimf(x, points) {
    // points = [a, b, c] where b is peak
    const [a, b, c] = points;
    if (x <= a || x >= c) return 0;
    if (x <= b) return (x - a) / (b - a);
    return (c - x) / (c - b);
}

// Trapezoidal membership function
function trapmf(x, points) {
    // points = [a, b, c, d]
    const [a, b, c, d] = points;
    if (x <= a || x >= d) return 0;
    if (x < b) return (x - a) / (b - a);
    if (x <= c) return 1;
    return (d - x) / (d - c);
}

// Calculate fuzzy traffic light duration based on vehicle counts
function calculateTrafficLightDuration(numMobil, numMotor) {
    const mobil = Math.max(0, Math.min(10, numMobil)); // Clamp to range
    const motor = Math.max(0, Math.min(30, numMotor)); // Clamp to range
    
    // Membership levels for inputs
    const motor_low_level = trimf(motor, FUZZY_CONFIG.motor_low);
    const motor_medium_level = trimf(motor, FUZZY_CONFIG.motor_medium);
    const motor_high_level = trimf(motor, FUZZY_CONFIG.motor_high);
    
    const mobil_low_level = trimf(mobil, FUZZY_CONFIG.mobil_low);
    const mobil_medium_level = trimf(mobil, FUZZY_CONFIG.mobil_medium);
    const mobil_high_level = trimf(mobil, FUZZY_CONFIG.mobil_high);
    
    // Calculate output membership levels based on rules
    let duration_short = 0;
    let duration_medium = 0;
    let duration_long = 0;
    
    // Single input rules
    duration_short = Math.max(duration_short, motor_low_level, mobil_low_level);
    duration_medium = Math.max(duration_medium, motor_medium_level, mobil_medium_level);
    duration_long = Math.max(duration_long, motor_high_level, mobil_high_level);
    
    // Combined rules
    duration_short = Math.max(duration_short, motor_low_level * mobil_low_level);
    duration_medium = Math.max(duration_medium, 
        motor_low_level * mobil_medium_level,
        motor_medium_level * mobil_low_level,
        motor_medium_level * mobil_medium_level
    );
    duration_long = Math.max(duration_long,
        motor_low_level * mobil_high_level,
        motor_medium_level * mobil_high_level,
        motor_high_level * mobil_low_level,
        motor_high_level * mobil_medium_level,
        motor_high_level * mobil_high_level
    );
    
    // Defuzzification using center of mass
    let numerator = 0;
    let denominator = 0;
    
    // For each output level, calculate weighted sum
    for (let y = 0; y <= 60; y += 0.1) {
        const short_mem = (y <= 20) ? trimf(y, FUZZY_CONFIG.duration_short) * duration_short : 0;
        const medium_mem = trapmf(y, FUZZY_CONFIG.duration_medium) * duration_medium;
        const long_mem = (y >= 40) ? trimf(y, FUZZY_CONFIG.duration_long) * duration_long : 0;
        
        const mem_level = Math.max(short_mem, medium_mem, long_mem);
        numerator += y * mem_level;
        denominator += mem_level;
    }
    
    const output = denominator > 0 ? numerator / denominator : 30; // Default to 30s if no output
    return Math.round(output * 100) / 100; // Round to 2 decimals
}

// ROI DOM Elements
const drawRoiBtn = document.getElementById('drawRoiBtn');
const clearRoiBtn = document.getElementById('clearRoiBtn');
const drawUploadRoiBtn = document.getElementById('drawUploadRoiBtn');
const clearUploadRoiBtn = document.getElementById('clearUploadRoiBtn');

// History DOM
const historyList = document.getElementById('historyList');
const exportHistoryBtn = document.getElementById('exportHistoryBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');

// Mode Selector
const modeBtns = document.querySelectorAll('.mode-btn');
const modeContents = document.querySelectorAll('.mode-content');

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    // Realtime mode init
    videoElement = document.getElementById('sourceVideo');
    canvasElement = document.getElementById('videoCanvas');
    ctx = canvasElement.getContext('2d');
    roiCanvas = document.getElementById('roiCanvas');
    roiCtx = roiCanvas.getContext('2d');
    uploadRoiCanvas = document.getElementById('uploadRoiCanvas');
    uploadRoiCtx = uploadRoiCanvas.getContext('2d');
    
    // Event listeners - Realtime
    previewBtn.addEventListener('click', startPreview);
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
        const isYoutube = e.target.value === 'youtube';
        youtubeGroup.style.display = isYoutube ? 'block' : 'none';

        // If user was previewing webcam and switches to YouTube, stop the webcam stream
        if (isYoutube && !isRunning && stream) {
            stopStreamOnly();
            previewBtn.disabled = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            screenshotBtn.disabled = true;
            drawRoiBtn.disabled = true;
            clearRoiBtn.disabled = true;
            canvasElement.style.display = 'block';
            roiCanvas.style.display = 'block';
            videoElement.style.display = 'none';
            updateStatus('⚠️ Webcam stopped (switched to YouTube)', 'disconnected');
        }
    });

    // Event listeners - Upload
    fileInput.addEventListener('change', handleFileSelect);
    
    uploadThresholdSlider.addEventListener('input', (e) => {
        uploadThreshold = parseFloat(e.target.value);
        uploadThresholdValue.textContent = uploadThreshold.toFixed(2);
    });

    frameSkipSlider.addEventListener('input', (e) => {
        frameSkipValue.textContent = e.target.value;
    });

    processBtn.addEventListener('click', processUploadedFile);
    downloadBtn.addEventListener('click', downloadResult);
    screenshotBtn.addEventListener('click', takeScreenshot);
    exportJsonBtn.addEventListener('click', () => exportStats('json'));
    exportCsvBtn.addEventListener('click', () => exportStats('csv'));

    // ROI buttons
    drawRoiBtn.addEventListener('click', () => toggleRealtimeRoiDrawing());
    clearRoiBtn.addEventListener('click', () => clearRealtimeRoi());
    drawUploadRoiBtn.addEventListener('click', () => toggleUploadRoiDrawing());
    clearUploadRoiBtn.addEventListener('click', () => clearUploadRoi());

    // History buttons
    exportHistoryBtn.addEventListener('click', exportHistoryCSV);
    clearHistoryBtn.addEventListener('click', clearHistory);

    // Initial history render
    renderHistory();

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
    
    // Stop detection or preview stream when leaving realtime
    if (mode !== 'realtime') {
        if (isRunning) {
            stopDetection();
        } else if (stream) {
            stopStreamOnly();
            previewBtn.disabled = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            screenshotBtn.disabled = true;
            drawRoiBtn.disabled = true;
            clearRoiBtn.disabled = true;
            canvasElement.style.display = 'block';
            roiCanvas.style.display = 'block';
            videoElement.style.display = 'none';
            updateStatus('⚠️ Webcam stopped (switched mode)', 'disconnected');
        }
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

async function startPreview() {
    try {
        const source = sourceSelect.value;
        if (source !== 'webcam') {
            updateStatus('Preview is only available for webcam source', 'disconnected');
            return;
        }

        await ensureWebcamStream();

        // Show raw video for preview (hide canvases)
        videoElement.style.display = 'block';
        canvasElement.style.display = 'none';
        roiCanvas.style.display = 'none';

        previewBtn.disabled = true;
        startBtn.disabled = false;
        stopBtn.disabled = false;
        screenshotBtn.disabled = true;
        drawRoiBtn.disabled = true;
        clearRoiBtn.disabled = true;
        updateStatus('✅ Preview running. Start detection to process frames.', 'connected');
    } catch (error) {
        console.error('Error starting preview:', error);
        updateStatus(`❌ Preview error: ${error.message}`, 'disconnected');
    }
}

async function ensureWebcamStream() {
    if (stream && stream.getTracks().some(t => t.readyState === 'live')) {
        return; // already have an active stream
    }

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
    roiCanvas.width = canvasElement.width;
    roiCanvas.height = canvasElement.height;
}

async function startWebcamDetection() {
    await ensureWebcamStream();
    // if preview was running, reuse the stream and proceed to websocket

    // Switch back to canvas rendering for detection
    videoElement.style.display = 'none';
    canvasElement.style.display = 'block';
    roiCanvas.style.display = 'block';
    
    connectWebSocket();
    
    previewBtn.disabled = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    screenshotBtn.disabled = false;
    isRunning = true;
    
    updateStatus('✅ Connected & Processing', 'connected');
    // Enable ROI drawing controls when running
    drawRoiBtn.disabled = false;
    clearRoiBtn.disabled = roiRealtime === null;
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
        roiCanvas.width = canvasElement.width;
        roiCanvas.height = canvasElement.height;
        
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
    stopStreamOnly();
    
    if (ws) {
        ws.close();
        ws = null;
    }
    
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    screenshotBtn.disabled = true;
    previewBtn.disabled = false;
    // Reset displays to default (canvas visible, video hidden)
    canvasElement.style.display = 'block';
    roiCanvas.style.display = 'block';
    videoElement.style.display = 'none';
    updateStatus('⚠️ Disconnected', 'disconnected');
    

function stopStreamOnly() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    if (videoElement) {
        videoElement.srcObject = null;
    }
}
    mobilCount.textContent = '0';
    motorCount.textContent = '0';
    durationValue.textContent = '0s';
    fpsCounter.textContent = 'FPS: 0';
    // disable ROI controls
    drawRoiBtn.disabled = true;
    clearRoiBtn.disabled = true;
    clearRealtimeRoi();
}

function takeScreenshot() {
    if (!canvasElement) return;
    
    // Create download link
    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    link.download = `detection_${timestamp}.png`;
    link.href = canvasElement.toDataURL('image/png');
    link.click();
    
    // Show feedback
    updateStatus('✅ Screenshot saved!', 'connected');
    setTimeout(() => {
        if (isRunning) {
            updateStatus('✅ Connected & Processing', 'connected');
        }
    }, 2000);
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
                apply_mask: maskToggle.checked,
                sent_ts: performance.now(),
                roi: roiRealtime ? {...roiRealtime} : null
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
    
    // Update FPS display
    updateFPS(performance.now());

    // Save compact history entry for realtime
    try {
        const preview = createCanvasThumbnail(canvasElement, 320);
        addHistoryEntry({
            ts: Date.now(),
            mode: 'realtime',
            mobil: result.mobil_count,
            motor: result.motor_count,
            duration: result.duration,
            threshold,
            mask: maskToggle.checked,
            roi: roiRealtime,
            fileType: 'frame',
            framesProcessed: 1,
            totalFrames: 1,
            preview
        });
    } catch (e) {
        // ignore history errors
    }
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
        frameSkipGroup.style.display = 'none';
        return;
    }
    
    // Validate file size (500MB limit)
    if (selectedFile.size > MAX_FILE_SIZE) {
        const maxMB = Math.round(MAX_FILE_SIZE / (1024 * 1024));
        const fileMB = (selectedFile.size / (1024 * 1024)).toFixed(2);
        updateUploadStatus(`❌ File too large: ${fileMB}MB (max ${maxMB}MB)`, 'error');
        processBtn.disabled = true;
        frameSkipGroup.style.display = 'none';
        selectedFile = null;
        alert(`File size exceeds limit!\n\nYour file: ${fileMB}MB\nMaximum allowed: ${maxMB}MB\n\nPlease select a smaller file.`);
        return;
    }
    
    // Show frame skip only for videos
    if (selectedFile.type.startsWith('video/')) {
        frameSkipGroup.style.display = 'block';
    } else {
        frameSkipGroup.style.display = 'none';
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
            // Wait for layout to update, then size ROI canvas to preview
            setTimeout(() => {
                uploadRoiCanvas.width = uploadPreview.clientWidth || uploadPreview.naturalWidth || 0;
                uploadRoiCanvas.height = uploadPreview.clientHeight || uploadPreview.naturalHeight || 0;
                drawUploadRoiBtn.disabled = false;
                clearUploadRoiBtn.disabled = roiUpload === null;
            }, 0);
        } else if (selectedFile.type.startsWith('video/')) {
            // Show playable video preview
            const uploadVideoPreview = document.getElementById('uploadVideoPreview');
            uploadVideoPreview.src = e.target.result;
            uploadVideoPreview.style.display = 'block';
            uploadPreview.classList.remove('active');
            document.getElementById('videoResultContainer').style.display = 'none';
            document.getElementById('uploadCanvas').style.display = 'none';
            
            // Setup ROI canvas to match video dimensions
            uploadVideoPreview.onloadedmetadata = () => {
                uploadRoiCanvas.width = uploadVideoPreview.videoWidth;
                uploadRoiCanvas.height = uploadVideoPreview.videoHeight;
                updateUploadStatus(`✅ Video uploaded`, 'connected');
                drawUploadRoiBtn.disabled = false;
                clearUploadRoiBtn.disabled = roiUpload === null;
            };
        }
    };
    reader.readAsDataURL(selectedFile);
}

async function processUploadedFile() {
    if (!selectedFile) return;
    
    updateUploadStatus('Processing...', 'processing');
    processBtn.disabled = true;
    downloadBtn.disabled = true;
    
    // Show progress bar
    progressContainer.style.display = 'block';
    progressBar.style.width = '10%';
    if (progressText) progressText.textContent = '10%';
    
    // Use WebSocket for video (real progress) or HTTP for image (instant)
    if (selectedFile.type.startsWith('video/')) {
        processVideoWithWebSocket();
    } else {
        processImageWithHTTP();
    }
}

async function processImageWithHTTP() {
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('threshold', uploadThreshold);
    formData.append('apply_mask', uploadMaskToggle.checked);
    if (roiUpload) {
        formData.append('roi', JSON.stringify(roiUpload));
    }
    
    try {
        progressBar.style.width = '50%';
        if (progressText) progressText.textContent = '50%';
        
        const response = await fetch('/api/process-file', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Processing failed');
        
        const data = await response.json();
        progressBar.style.width = '100%';
        if (progressText) progressText.textContent = '100%';
        
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
            exportJsonBtn.disabled = false;
            exportCsvBtn.disabled = false;
            
            // Store result for download
            window.lastResult = data;

            // Save to history
            try {
                const preview = uploadPreview.src || `data:image/jpeg;base64,${data.annotated_frame}`;
                addHistoryEntry({
                    ts: Date.now(),
                    mode: 'upload-image',
                    mobil: data.mobil_count,
                    motor: data.motor_count,
                    duration: data.duration,
                    threshold: uploadThreshold,
                    mask: uploadMaskToggle.checked,
                    roi: roiUpload,
                    fileType: 'image',
                    preview
                });
            } catch (e) {}
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
            
            // Save to history (no preview for video)
            try {
                addHistoryEntry({
                    ts: Date.now(),
                    mode: 'upload-video',
                    mobil: data.mobil_count,
                    motor: data.motor_count,
                    duration: data.duration,
                    threshold: uploadThreshold,
                    mask: uploadMaskToggle.checked,
                    roi: roiUpload,
                    fileType: 'video',
                    framesProcessed: data.frames_processed,
                    totalFrames: data.total_frames,
                });
            } catch (e) {}
        }
    } catch (error) {
        progressBar.style.width = '0%';
        progressBar.textContent = '';
        updateUploadStatus(`❌ ${error.message}`, 'disconnected');
    }
    
    processBtn.disabled = false;
    
    // Hide progress bar after 2 seconds
    setTimeout(() => {
        progressContainer.style.display = 'none';
        progressBar.style.width = '0%';
        progressBar.textContent = '';
    }, 2000);
}

async function processVideoWithWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/process-video`;
    
    const videoWs = new WebSocket(wsUrl);
    window.videoWs = videoWs; // expose for control handlers
    window.videoProcessingDone = false;
    window.uploadProcessingCancelled = false;

    // Enable control buttons
    const cancelBtn = document.getElementById('cancelBtn');
    cancelBtn.disabled = false;
    cancelBtn.onclick = () => {
        if (!window.videoWs) return;
        window.uploadProcessingCancelled = true;
        try { window.videoWs.send(JSON.stringify({ type: 'cancel' })); } catch {}
        updateUploadStatus('❌ Cancelling...', 'disconnected');
        processBtn.disabled = false;
        // Force auto-reload after cancel
        setTimeout(() => { window.location.reload(); }, 500);
    };
    
    videoWs.onopen = async () => {
        console.log('✅ Video processing WebSocket connected');
        updateUploadStatus('📤 Uploading video...', 'processing');
        // Ensure progress container visible and start at 0%
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        if (progressText) progressText.textContent = 'Uploading... 0%';

        // Prepare streaming canvas over the preview area
        const canvas = document.getElementById('uploadCanvas');
        const videoPreview = document.getElementById('uploadVideoPreview');
        const imgPreview = document.getElementById('uploadPreview');
        // Show canvas and match size to preview element
        canvas.style.display = 'block';
        const targetWidth = (videoPreview && !videoPreview.paused && !videoPreview.ended) ? videoPreview.clientWidth : imgPreview.clientWidth;
        const targetHeight = (videoPreview && !videoPreview.paused && !videoPreview.ended) ? videoPreview.clientHeight : imgPreview.clientHeight;
        if (targetWidth && targetHeight) {
            canvas.width = targetWidth;
            canvas.height = targetHeight;
        }
        // Hide video/image preview while streaming annotated frames
        if (videoPreview) videoPreview.style.display = 'none';
        if (imgPreview) imgPreview.classList.remove('active');
        
        try {
            // Send metadata
            const metadata = {
                type: 'video_metadata',
                threshold: uploadThreshold,
                frame_skip: parseInt(frameSkipSlider.value),
                apply_mask: uploadMaskToggle.checked,
                filename: selectedFile.name,
                roi: roiUpload ? {...roiUpload} : null
            };
            console.log('📤 Sending metadata:', metadata);
            videoWs.send(JSON.stringify(metadata));
            
            // Read and send video data with chunking (WebSocket has message size limits)
            const reader = new FileReader();
            reader.onprogress = (e) => {
                if (e.lengthComputable) {
                    const percentComplete = Math.round((e.loaded / e.total) * 100);
                    progressBar.style.width = percentComplete + '%';
                    if (progressText) progressText.textContent = `Uploading... ${percentComplete}%`;
                }
            };
            reader.onload = (e) => {
                try {
                    const videoB64 = e.target.result.split(',')[1];
                    console.log(`📤 Video file size: ${videoB64.length} chars (~${Math.round(videoB64.length / 1024 / 1024)}MB)`);
                    
                    // Reset for processing phase
                    progressBar.style.width = '0%';
                    if (progressText) progressText.textContent = 'Processing... 0%';
                    updateUploadStatus('⏳ Processing video frames...', 'processing');
                    
                    // Send video data in chunks (max 1MB per chunk to avoid WebSocket message size limits)
                    const CHUNK_SIZE = 1024 * 1024; // 1MB chunks
                    const totalChunks = Math.ceil(videoB64.length / CHUNK_SIZE);
                    console.log(`📤 Sending ${totalChunks} chunks of ~${Math.round(CHUNK_SIZE / 1024)}KB each`);
                    
                    let chunksSent = 0;
                    const sendChunk = () => {
                        if (window.uploadProcessingCancelled) { console.log('🚫 Chunk sending cancelled'); return; }
                        if (chunksSent < totalChunks) {
                            const start = chunksSent * CHUNK_SIZE;
                            const end = Math.min(start + CHUNK_SIZE, videoB64.length);
                            const chunk = videoB64.slice(start, end);
                            
                            const message = {
                                type: 'video_chunk',
                                chunk_index: chunksSent,
                                total_chunks: totalChunks,
                                data: chunk
                            };
                            
                            console.log(`📤 Sending chunk ${chunksSent + 1}/${totalChunks}`);
                            videoWs.send(JSON.stringify(message));
                            chunksSent++;
                            
                            // Send next chunk after small delay to avoid overwhelming server
                            setTimeout(sendChunk, 10);
                        } else {
                            console.log(`✅ All ${totalChunks} chunks sent`);
                            // Send completion signal
                            videoWs.send(JSON.stringify({
                                type: 'video_complete',
                                total_size: videoB64.length
                            }));
                        }
                    };
                    
                    // Start sending chunks
                    sendChunk();
                    
                } catch (e) {
                    console.error('❌ Error preparing video data:', e);
                    updateUploadStatus(`❌ Error preparing video: ${e.message}`, 'disconnected');
                    videoWs.close();
                }
            };
            reader.onerror = (e) => {
                console.error('❌ FileReader error:', e);
                updateUploadStatus('❌ Error reading file', 'disconnected');
                videoWs.close();
                processBtn.disabled = false;
            };
            reader.readAsDataURL(selectedFile);
        } catch (e) {
            console.error('❌ Error in onopen:', e);
            updateUploadStatus(`❌ Error: ${e.message}`, 'disconnected');
            videoWs.close();
            processBtn.disabled = false;
        }
    };
    
    videoWs.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        if (message.type === 'frame') {
            // Display processed frame in real-time on canvas (render at native resolution when possible)
            const frameCanvas = document.getElementById('uploadCanvas');
            if (frameCanvas) {
                const ctx = frameCanvas.getContext('2d');
                const img = new Image();
                img.onload = () => {
                    // Prefer native frame size so boxes remain sharp
                    const fw = Number(message.frame_width) || img.naturalWidth || img.width;
                    const fh = Number(message.frame_height) || img.naturalHeight || img.height;
                    if (fw && fh) { frameCanvas.width = fw; frameCanvas.height = fh; }
                    ctx.clearRect(0, 0, frameCanvas.width, frameCanvas.height);
                    ctx.imageSmoothingEnabled = false; // keep sharp edges
                    ctx.drawImage(img, 0, 0, frameCanvas.width, frameCanvas.height);
                };
                img.onerror = () => {
                    console.error('Failed to decode frame image');
                };
                img.src = `data:image/jpeg;base64,${message.frame_data}`;
            }
            
            // Update progress bar
            const percent = Number(message.percent) || 1;
            progressBar.style.width = percent + '%';
            
            // Update progress text (static, no animated dots)
            if (progressText) {
                progressText.textContent = `Processing ${percent}% (${message.frame}/${message.total})`;
            }
            
            // Update status (animated spinner only, no redundant info)
            const dots = nextProcessingDots();
            updateUploadStatus(`⏳ Processing ${dots}`, 'processing');
            
            // Update counts - show current frame detections only
            uploadMobilCount.textContent = message.current_mobil;
            uploadMotorCount.textContent = message.current_motor;
            
            // Calculate and update traffic light duration dynamically per frame
            const dynamicDuration = calculateTrafficLightDuration(message.current_mobil, message.current_motor);
            uploadDurationValue.textContent = `${dynamicDuration}s`;
            
        } else if (message.type === 'progress') {
            const percent = Number(message.percent) || 1; // Default to 1% if no percent sent
            progressBar.style.width = percent + '%';
            
            if (message.frame > 0) {
                // Show frame progress (static, no animated dots)
                if (progressText) progressText.textContent = `Processing ${percent}% (${message.frame}/${message.total})`;
            } else {
                // Initializing phase
                if (progressText) progressText.textContent = `Initializing ${percent}%`;
            }
            
            // Update status (animated spinner only)
            const dots = nextProcessingDots();
            updateUploadStatus(`⏳ Processing ${dots}`, 'processing');
            
            // Update counts in real-time if available
            if (message.total_mobil !== undefined) {
                uploadMobilCount.textContent = message.total_mobil;
                uploadMotorCount.textContent = message.total_motor;
            }
            
        } else if (message.type === 'complete') {
            progressBar.style.width = '100%';
            if (progressText) progressText.textContent = '100%';
            
            // Show video result - hide all other previews first
            const canvas = document.getElementById('uploadCanvas');
            const imgPreview = document.getElementById('uploadPreview');
            const uploadVideoPreview = document.getElementById('uploadVideoPreview');
            
            if (canvas) canvas.style.display = 'none';
            if (imgPreview) imgPreview.classList.remove('active');
            if (uploadVideoPreview) uploadVideoPreview.style.display = 'none';
            
            const videoResultContainer = document.getElementById('videoResultContainer');
            const videoPlayer = document.getElementById('videoPlayer');
            
            // Fetch video via URL instead of base64 for better browser support
            if (message.video_path) {
                const videoUrl = `/api/video/${message.video_path}`;
                videoPlayer.src = videoUrl;
                console.log(`✅ Loading processed video from: ${videoUrl}`);
            } else if (message.video_data) {
                // Fallback to base64 if path not provided (legacy support)
                videoPlayer.src = `data:video/mp4;base64,${message.video_data}`;
            }
            
            videoResultContainer.style.display = 'block';
            
            // Force video to load and be ready for playback
            videoPlayer.load();
            videoPlayer.play().catch(err => {
                console.log('Auto-play prevented, user can click play manually:', err);
            });
            
            // Update stats
            uploadMobilCount.textContent = message.mobil_count;
            uploadMotorCount.textContent = message.motor_count;
            uploadDurationValue.textContent = `${message.duration}s`;
            
            const skipInfo = ` (${message.frames_processed}/${message.total_frames} frames)`;
            updateUploadStatus(`✅ Video processed${skipInfo}`, 'connected');
            downloadBtn.disabled = false;
            downloadBtn.textContent = 'Download Video';
            exportJsonBtn.disabled = false;
            exportCsvBtn.disabled = false;
            
            // Store result (use video_path if available, fallback to video_data)
            window.lastResult = {
                video_path: message.video_path,
                video_data: message.video_data,
                mobil_count: message.mobil_count,
                motor_count: message.motor_count,
                total_vehicles: message.total_vehicles,
                cumulative_mobil: message.cumulative_mobil,
                cumulative_motor: message.cumulative_motor,
                frames_processed: message.frames_processed,
                total_frames: message.total_frames,
                duration: message.duration,
                file_type: 'video',
                stats: message.stats
            };

            // Save to history (no preview for video)
            try {
                addHistoryEntry({
                    ts: Date.now(),
                    mode: 'upload-video',
                    mobil: message.mobil_count,
                    motor: message.motor_count,
                    duration: message.duration,
                    threshold: uploadThreshold,
                    mask: uploadMaskToggle.checked,
                    roi: roiUpload,
                    fileType: 'video',
                    framesProcessed: message.frames_processed,
                    totalFrames: message.total_frames,
                });
            } catch (e) {}
            
            processBtn.disabled = false;
            
            // Hide progress after delay
            setTimeout(() => {
                progressContainer.style.display = 'none';
                progressBar.style.width = '0%';
                progressBar.textContent = '';
            }, 2000);
            
            // Mark completion to treat socket close as normal
            window.videoProcessingDone = true;
            // Do not force-close from client; let server close gracefully
            
        } else if (message.type === 'error') {
            progressBar.style.width = '0%';
            progressBar.textContent = '';
            updateUploadStatus(`❌ ${message.message}`, 'disconnected');
            processBtn.disabled = false;
            // If cancelled, reset UI and force reload for clean state
            if (message.message && message.message.toLowerCase().includes('cancelled')) {
                try {
                    const canvas = document.getElementById('uploadCanvas');
                    if (canvas) { canvas.style.display = 'none'; const ctx = canvas.getContext('2d'); ctx && ctx.clearRect(0,0,canvas.width,canvas.height); }
                    const videoResultContainer = document.getElementById('videoResultContainer');
                    if (videoResultContainer) videoResultContainer.style.display = 'none';
                    const videoPlayer = document.getElementById('videoPlayer');
                    if (videoPlayer) videoPlayer.src = '';
                } catch {}
                // Close socket and force page reload to ensure clean state
                try { videoWs.close(); } catch {}
                setTimeout(() => { window.location.reload(); }, 100);
                return;
            }
            try { videoWs.close(); } catch {}
        }
    };
    
    videoWs.onerror = (error) => {
        console.error('❌ Video WebSocket error:', error);
        updateUploadStatus('❌ Connection error', 'disconnected');
        processBtn.disabled = false;
    };
    
    videoWs.onclose = (event) => {
        console.log(`⚠️ Video WebSocket closed: code=${event.code}, reason=${event.reason}`);
        if (!event.wasClean && !window.videoProcessingDone) {
            console.error('❌ WebSocket closed abnormally');
            updateUploadStatus('❌ Connection closed unexpectedly', 'disconnected');
            processBtn.disabled = false;
        }
    };
}

function downloadResult() {
    if (!window.lastResult) return;
    
    const link = document.createElement('a');
    
    // Download based on file type
    if (window.lastResult.file_type === 'image') {
        link.href = `data:image/jpeg;base64,${window.lastResult.annotated_frame}`;
        link.download = `detection_result_${Date.now()}.jpg`;
    } else if (window.lastResult.file_type === 'video') {
        // Use video_path if available (direct download from server)
        if (window.lastResult.video_path) {
            link.href = `/api/video/${window.lastResult.video_path}`;
            link.download = `detection_result_${Date.now()}.mp4`;
        } else if (window.lastResult.video_data) {
            // Fallback to base64
            link.href = `data:video/mp4;base64,${window.lastResult.video_data}`;
            link.download = `detection_result_${Date.now()}.mp4`;
        }
    }
    
    link.click();
}

function updateUploadStatus(message, statusClass) {
    uploadStatusDisplay.textContent = message;
    uploadStatusDisplay.className = `status ${statusClass}`;
}

function exportStats(format) {
    if (!window.lastResult || !window.lastResult.stats) {
        alert('No statistics available to export');
        return;
    }
    
    const stats = window.lastResult.stats;
    let content, filename, mimeType;
    
    if (format === 'json') {
        content = JSON.stringify(stats, null, 2);
        filename = `detection_stats_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
        mimeType = 'application/json';
    } else if (format === 'csv') {
        // Convert stats to CSV
        const flat = {
            timestamp: stats.timestamp,
            file_name: stats.file_name,
            file_type: stats.file_type,
            mobil_count: stats.detection.mobil_count,
            motor_count: stats.detection.motor_count,
            total_vehicles: stats.detection.total_vehicles,
            traffic_light_duration_seconds: stats.optimization.traffic_light_duration_seconds,
            confidence_threshold: stats.parameters.confidence_threshold,
        };
        
        if (stats.video) {
            flat.frames_processed = stats.video.frames_processed;
            flat.total_frames = stats.video.total_frames;
            flat.frame_skip = stats.video.frame_skip;
        }
        
        // Create CSV header and row
        const headers = Object.keys(flat).join(',');
        const values = Object.values(flat).join(',');
        content = headers + '\n' + values;
        filename = `detection_stats_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
        mimeType = 'text/csv';
    }
    
    // Create download link
    const blob = new Blob([content], { type: mimeType });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    
    updateUploadStatus(`✅ Exported as ${format.toUpperCase()}`, 'connected');
    setTimeout(() => {
        updateUploadStatus(`✅ ${window.lastResult.file_type === 'video' ? 'Video' : 'Image'} processing complete`, 'connected');
    }, 2000);
}

// ===== ROI DRAWING HELPERS =====
function toggleRealtimeRoiDrawing() {
    if (roiCanvas.classList.contains('drawing')) {
        roiCanvas.classList.remove('drawing');
        isDrawingRoi = false;
        drawRoiBtn.textContent = '🟪 Draw ROI';
        return;
    }
    roiCanvas.classList.add('drawing');
    drawRoiBtn.textContent = '✅ Finish Drawing';
}

function clearRealtimeRoi() {
    roiRealtime = null;
    roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
    clearRoiBtn.disabled = true;
}

function toggleUploadRoiDrawing() {
    if (uploadRoiCanvas.classList.contains('drawing')) {
        uploadRoiCanvas.classList.remove('drawing');
        isDrawingRoi = false;
        drawUploadRoiBtn.textContent = '🟪 Draw ROI';
        return;
    }
    uploadRoiCanvas.classList.add('drawing');
    drawUploadRoiBtn.textContent = '✅ Finish Drawing';
}

function clearUploadRoi() {
    roiUpload = null;
    uploadRoiCtx.clearRect(0, 0, uploadRoiCanvas.width, uploadRoiCanvas.height);
    clearUploadRoiBtn.disabled = true;
}

// Attach mouse handlers for both ROI canvases
function attachRoiHandlers(canvas, ctx, setResult, clearBtn) {
    let start = null;

    function onDown(e) {
        if (!canvas.classList.contains('drawing')) return;
        const rect = canvas.getBoundingClientRect();
        start = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }
    function onMove(e) {
        if (!canvas.classList.contains('drawing') || !start) return;
        const rect = canvas.getBoundingClientRect();
        const curr = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        const x = Math.min(start.x, curr.x);
        const y = Math.min(start.y, curr.y);
        const w = Math.abs(curr.x - start.x);
        const h = Math.abs(curr.y - start.y);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#a855f7';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
    }
    function onUp(e) {
        if (!canvas.classList.contains('drawing') || !start) return;
        const rect = canvas.getBoundingClientRect();
        const end = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        const x = Math.max(0, Math.min(start.x, end.x));
        const y = Math.max(0, Math.min(start.y, end.y));
        const w = Math.abs(end.x - start.x);
        const h = Math.abs(end.y - start.y);
        start = null;
        canvas.classList.remove('drawing');
        // normalize
        const nx = +(x / canvas.width).toFixed(6);
        const ny = +(y / canvas.height).toFixed(6);
        const nw = +(w / canvas.width).toFixed(6);
        const nh = +(h / canvas.height).toFixed(6);
        if (nw < 0.01 || nh < 0.01) {
            // too small, clear
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            setResult(null);
            clearBtn.disabled = true;
            return;
        }
        // persist drawing
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 2;
        ctx.strokeRect(nx * canvas.width, ny * canvas.height, nw * canvas.width, nh * canvas.height);
        setResult({ x: nx, y: ny, w: nw, h: nh });
        clearBtn.disabled = false;
    }
    canvas.addEventListener('mousedown', onDown);
    canvas.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
}

// Initialize ROI handlers after DOM ready
document.addEventListener('DOMContentLoaded', () => {
    attachRoiHandlers(roiCanvas, roiCtx, (val) => { roiRealtime = val; drawRoiBtn.textContent = '🟪 Draw ROI'; }, clearRoiBtn);
    attachRoiHandlers(uploadRoiCanvas, uploadRoiCtx, (val) => { roiUpload = val; drawUploadRoiBtn.textContent = '🟪 Draw ROI'; }, clearUploadRoiBtn);
});

// ===== HISTORY HELPERS =====
const HISTORY_KEY = 'detectionHistory';
const HISTORY_LIMIT = 25;

function getHistory() {
    try {
        const raw = localStorage.getItem(HISTORY_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch (_) {
        return [];
    }
}

function setHistory(items) {
    try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_LIMIT)));
        renderHistory();
    } catch (_) {}
}

function addHistoryEntry(entry) {
    const items = getHistory();
    items.unshift(entry);
    if (items.length > HISTORY_LIMIT) items.length = HISTORY_LIMIT;
    setHistory(items);
}

function clearHistory() {
    try {
        localStorage.removeItem(HISTORY_KEY);
    } catch (_) {}
    renderHistory();
}

function renderHistory() {
    const items = getHistory();
    if (!historyList) return;
    historyList.innerHTML = '';
    if (items.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'history-item';
        empty.innerHTML = '<div class="history-meta"><div class="title">No history yet</div><div class="sub">Run a detection to see it here.</div></div>';
        historyList.appendChild(empty);
        return;
    }
    for (const item of items) {
        const el = document.createElement('div');
        el.className = 'history-item';
        const thumb = document.createElement('img');
        thumb.className = 'history-thumb';
        if (item.preview) thumb.src = item.preview; else thumb.style.display = 'none';
        const meta = document.createElement('div');
        meta.className = 'history-meta';
        const ts = new Date(item.ts).toLocaleString();
        const title = document.createElement('div');
        title.className = 'title';
        title.textContent = `${item.mode} • ${ts}`;
        const sub = document.createElement('div');
        sub.className = 'sub';
        const counts = `🚗 ${item.mobil ?? 0} | 🏍️ ${item.motor ?? 0}`;
        const dur = item.duration !== undefined ? ` • ⏱ ${item.duration}s` : '';
        sub.textContent = `${counts}${dur}`;
        meta.appendChild(title);
        meta.appendChild(sub);
        el.appendChild(thumb);
        el.appendChild(meta);
        historyList.appendChild(el);
    }
}

function exportHistoryCSV() {
    const items = getHistory();
    if (!items.length) return alert('No history to export');
    const headers = [
        'timestamp','mode','mobil','motor','duration','threshold','mask','roi_x','roi_y','roi_w','roi_h','file_type','frames_processed','total_frames'
    ];
    const rows = [headers.join(',')];
    for (const it of items) {
        const roi = it.roi || {};
        const row = [
            new Date(it.ts).toISOString(),
            it.mode || '',
            it.mobil ?? '',
            it.motor ?? '',
            it.duration ?? '',
            it.threshold ?? '',
            it.mask ?? '',
            roi.x ?? '',
            roi.y ?? '',
            roi.w ?? '',
            roi.h ?? '',
            it.fileType || '',
            it.framesProcessed ?? '',
            it.totalFrames ?? ''
        ].join(',');
        rows.push(row);
    }
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `detection_history_${new Date().toISOString().replace(/[:.]/g,'-')}.csv`;
    link.click();
}

function createCanvasThumbnail(sourceCanvas, maxWidth) {
    const scale = Math.min(1, maxWidth / sourceCanvas.width);
    const w = Math.max(1, Math.floor(sourceCanvas.width * scale));
    const h = Math.max(1, Math.floor(sourceCanvas.height * scale));
    const cnv = document.createElement('canvas');
    cnv.width = w; cnv.height = h;
    const c = cnv.getContext('2d');
    c.drawImage(sourceCanvas, 0, 0, w, h);
    return cnv.toDataURL('image/jpeg', 0.7);
}
