// ============================================
// Face Attendance System - Client Application
// ============================================

// Configuration
const CONFIG = {
    API_BASE_URL: 'http://localhost:8080',
    MODEL_URL: 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model',
    DETECTION_INTERVAL: 500, // ms
    MIN_CONFIDENCE: 0.5,
};

// Global State
const state = {
    isModelsLoaded: false,
    isCameraActive: false,
    isProcessing: false,
    stream: null,
    detectionInterval: null,
    lastDetection: null,

    lastCheckinTime : null,
    checkinCooldown : 5000,

};

// DOM Elements
const elements = {
    // Video
    video: document.getElementById('video'),
    overlay: document.getElementById('overlay'),
    noCamera: document.getElementById('noCamera'),
    
    // Buttons
    btnStartCamera: document.getElementById('btnStartCamera'),
    btnStopCamera: document.getElementById('btnStopCamera'),
    btnRefreshHistory: document.getElementById('btnRefreshHistory'),
    
    // Tabs
    tabCheckin: document.getElementById('tabCheckin'),
    tabHistory: document.getElementById('tabHistory'),
    tabStats: document.getElementById('tabStats'),
    contentCheckin: document.getElementById('contentCheckin'),
    contentHistory: document.getElementById('contentHistory'),
    contentStats: document.getElementById('contentStats'),
    
    // Status
    serverStatus: document.getElementById('serverStatus'),
    statusMessage: document.getElementById('statusMessage'),
    faceDetectionStatus: document.getElementById('faceDetectionStatus'),
    faceApiStatus: document.getElementById('faceApiStatus'),
    faceDetected: document.getElementById('faceDetected'),
    detectionConfidence: document.getElementById('detectionConfidence'),
    
    // Last Check-in
    lastCheckin: document.getElementById('lastCheckin'),
    lastEmployeeName: document.getElementById('lastEmployeeName'),
    lastEmployeeId: document.getElementById('lastEmployeeId'),
    lastTimestamp: document.getElementById('lastTimestamp'),
    lastConfidence: document.getElementById('lastConfidence'),
    
    // History
    historyList: document.getElementById('historyList'),
    filterEmployeeId: document.getElementById('filterEmployeeId'),
    filterType: document.getElementById('filterType'),
    
    // Stats
    statTotalEmployees: document.getElementById('statTotalEmployees'),
    statTodayAttendance: document.getElementById('statTodayAttendance'),
    statTotalAttendance: document.getElementById('statTotalAttendance'),
    statModel: document.getElementById('statModel'),
    statThreshold: document.getElementById('statThreshold'),
    statVersion: document.getElementById('statVersion'),
};

// ============================================
// Initialization
// ============================================

async function init() {
    console.log('Initializing Face Attendance System...');
    
    // Check server status
    await checkServerStatus();
    
    // Load Face-api.js models
    await loadModels();
    
    // Setup event listeners
    setupEventListeners();
    
    // Load initial data
    await loadStats();
    
    console.log('System initialized successfully');
}

// ============================================
// Face-api.js Model Loading
// ============================================
let facenetModel;
class Lambda extends tf.layers.Layer {
  static className = 'Lambda';
  constructor(config) {
    super(config);
    this.config = config;
  }
  build(inputShape) {
    this.built = true;
  }
  call(inputs, kwargs) {
    const { scale } = this.config.arguments || { scale: 1 };
    const [x] = inputs;
    return tf.mul(x, scale);
  }
}

tf.serialization.registerClass(Lambda);

function preprocessFace(faceTensor) {
  return tf.tidy(() => {
    const resized = tf.image.resizeBilinear(faceTensor, [160, 160]);
    const normalized = resized.div(127.5).sub(1);
    return normalized.expandDims(0); // [1,160,160,3]
  });
}

async function getDescriptor512(faceTensor) {
  const input = preprocessFace(faceTensor);
  const embedding = facenetModel.predict(input);
  return embedding.dataSync(); // Float32Array(512)
}

async function loadFaceNet512() {
    try {
    console.log('Loading FaceNet512 model...');
    facenetModel = await tf.loadLayersModel('http://localhost:8080/models/facenet512_model/model.json');
    console.log('FaceNet512 loaded successfully');
  } catch (err) {
    console.error('Error loading FaceNet512:', err);
    throw err;
  }

}


async function loadModels() {
    try {
        console.log('Loading face-api.js models...');
        elements.faceApiStatus.textContent = 'Memuat...';
        elements.faceApiStatus.className = 'font-medium text-yellow-600';
        
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(CONFIG.MODEL_URL),
            faceapi.nets.faceLandmark68Net.loadFromUri(CONFIG.MODEL_URL),
        ]);
        
        // load FaceNet512 TF.js
        await loadFaceNet512();


        state.isModelsLoaded = true;
        console.log('Models loaded successfully');
        
        elements.faceApiStatus.textContent = 'Siap';
        elements.faceApiStatus.className = 'font-medium text-green-600';

        

        showMessage('Models berhasil dimuat. Sistem siap digunakan.', 'success');
    } catch (error) {
        console.error('Error loading models:', error);
        elements.faceApiStatus.textContent = 'Error';
        elements.faceApiStatus.className = 'font-medium text-red-600';
        showMessage('Gagal memuat models. Refresh halaman.', 'error');
    }
}

// ============================================
// Camera Management
// ============================================

async function startCamera() {
    try {
        console.log('Starting camera...');
        
        state.stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            }
        });
        
        elements.video.srcObject = state.stream;
        state.isCameraActive = true;
        
        // Update UI
        elements.noCamera.classList.add('hidden');
        elements.btnStartCamera.classList.add('hidden');
        elements.btnStopCamera.classList.remove('hidden');
        elements.faceDetectionStatus.classList.remove('hidden');
        
        // Setup canvas overlay
        setupCanvas();
        
        // Start face detection
        startFaceDetection();
        
        showMessage('Kamera aktif. Posisikan wajah Anda.', 'info');
        
    } catch (error) {
        console.error('Camera error:', error);
        showMessage('Gagal mengakses kamera: ' + error.message, 'error');
    }
}

function stopCamera() {
    console.log('Stopping camera...');
    
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
    }
    
    if (state.detectionInterval) {
        clearInterval(state.detectionInterval);
        state.detectionInterval = null;
    }
    
    state.isCameraActive = false;
    
    // Update UI
    elements.noCamera.classList.remove('hidden');
    elements.btnStartCamera.classList.remove('hidden');
    elements.btnStopCamera.classList.add('hidden');
    elements.faceDetectionStatus.classList.add('hidden');
    
    // Clear canvas
    const ctx = elements.overlay.getContext('2d');
    ctx.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
    
    showMessage('Kamera dimatikan.', 'info');
}

function setupCanvas() {
    const displaySize = {
        width: elements.video.offsetWidth,
        height: elements.video.offsetHeight
    };
    
    elements.overlay.width = displaySize.width;
    elements.overlay.height = displaySize.height;
}

// ============================================
// Face Detection
// ============================================

async function startFaceDetection() {
  state.detectionInterval = setInterval(async () => {
    if (!state.isCameraActive) return;

    if (!state.lastDetection) {
        showMessage('Wajah tidak terdeteksi. Posisikan wajah Anda.', 'warning');
    }
    
    if (state.isProcessing) return;
    
    try {
      // 1. Deteksi wajah (tanpa descriptor bawaan face-api.js)
      const detection = await faceapi
        .detectSingleFace(elements.video, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks();

      if (detection) {
        state.isProcessing = true;

        state.lastDetection = detection;
        updateFaceStatus(true, detection.detection.score);
        drawFaceBox(detection);

        // 2. Crop wajah dari video
        const regions = [detection.detection.box];
        const faces = await faceapi.extractFaces(elements.video, regions);

        if (!faces || faces.length === 0) {
            console.warn("Tidak ada wajah yang berhasil dicrop");
            return;
        }

        const faceCanvas = faces[0];
        const faceTensor = tf.browser.fromPixels(faceCanvas);

        const descriptor512 = await getDescriptor512(faceTensor);

        // 3. Kirim descriptor ke server
        const now = Date.now();
        if (state.lastCheckinTime && (now - state.lastCheckinTime < state.checkinCooldown)) {
            console.log("Skip check-in, masih dalam cooldown");
            return;
        }
        

        const response = await fetch(`${CONFIG.API_BASE_URL}/api/attendance/checkin`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                descriptor: Array.from(descriptor512),
                timestamp: new Date().toISOString()
            })
        });

        const result = await response.json();
        
        if (response.ok) {
            // Success
            console.log('Check-in successful:', result);
            showMessage(
                `Berhasil! Selamat datang, ${result.employee_name}`,
                'success'
            );
            
            // Update last check-in
            updateLastCheckin(result);
            
            // Stop camera
            // stopCamera();
            
            // Refresh stats
            await loadStats();
            state.lastCheckinTime = now;

        } else {
            // Error
            console.error('Check-in failed:', result);
            showMessage(result.detail || 'Wajah tidak dikenali', 'error');
            state.lastCheckinTime = now;
        }

      } else {
        state.lastDetection = null;
        updateFaceStatus(false);
        clearCanvas();
      }

    } catch (error) {
        console.error('Detection error:', error);
    } finally {
      clearCanvas();
      state.isProcessing = false;
    }
  }, CONFIG.DETECTION_INTERVAL);
}

function updateFaceStatus(detected, confidence = 0) {
    if (detected) {
        elements.faceDetected.textContent = 'Ya';
        elements.faceDetected.className = 'font-medium text-green-600';
        elements.detectionConfidence.textContent = (confidence * 100).toFixed(1) + '%';
        elements.detectionConfidence.className = 'font-medium text-green-600';
       
    } else {
        elements.faceDetected.textContent = 'Tidak';
        elements.faceDetected.className = 'font-medium text-red-600';
        elements.detectionConfidence.textContent = '-';
        elements.detectionConfidence.className = 'font-medium text-gray-400';

    }
}

function drawFaceBox(detection) {
    const ctx = elements.overlay.getContext('2d');
    const displaySize = {
        width: elements.overlay.width,
        height: elements.overlay.height
    };
    
    // Clear previous drawings
    ctx.clearRect(0, 0, displaySize.width, displaySize.height);
    
    // Resize detection to match display
    const resizedDetection = faceapi.resizeResults(detection, displaySize);
    
    // Draw face box
    const box = resizedDetection.detection.box;
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 3;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    
    // Draw confidence
    ctx.fillStyle = '#10b981';
    ctx.font = 'bold 16px Arial';
    const confidence = (resizedDetection.detection.score * 100).toFixed(1) + '%';
    ctx.fillText(confidence, box.x, box.y - 10);
}

function clearCanvas() {
    const ctx = elements.overlay.getContext('2d');
    ctx.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
}

// ============================================
// Attendance Processing
// ============================================

async function processAttendance() {
    if (!state.lastDetection) {
        showMessage('Wajah tidak terdeteksi. Posisikan wajah Anda.', 'warning');
        return;
    }
    
    if (state.isProcessing) return;
    
    state.isProcessing = true;
    
    try {
        showMessage('Memproses wajah...', 'info');
        
        // Get face descriptor
        const descriptor = Array.from(state.lastDetection.descriptor);
        
        console.log('Sending descriptor to server:', descriptor.length, 'dimensions');
        
        // Send to server
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/attendance/checkin`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                descriptor: descriptor,
                timestamp: new Date().toISOString()
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Success
            console.log('Check-in successful:', result);
            showMessage(
                `Berhasil! Selamat datang, ${result.employee_name}`,
                'success'
            );
            
            // Update last check-in
            updateLastCheckin(result);
            
            // Stop camera
            // stopCamera();
            
            // Refresh stats
            await loadStats();
            
        } else {
            // Error
            console.error('Check-in failed:', result);
            showMessage(result.detail || 'Wajah tidak dikenali', 'error');
        }
        

    } catch (error) {
        console.error('Attendance error:', error);
        showMessage('Error: ' + error.message, 'error');

    } finally {
        state.isProcessing = false;
    }
}

function updateLastCheckin(data) {
    elements.lastEmployeeName.textContent = data.employee_name;
    elements.lastEmployeeId.textContent = data.employee_id;
    elements.lastTimestamp.textContent = formatDateTime(data.timestamp);
    elements.lastConfidence.textContent = (data.confidence * 100).toFixed(1) + '%';
    elements.lastCheckin.classList.remove('hidden');
}

// ============================================
// History Management
// ============================================

async function loadHistory() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/attendance/history?limit=50`);
        
        if (!response.ok) throw new Error('Failed to load history');
        
        const data = await response.json();
        displayHistory(data);
        
    } catch (error) {
        console.error('History error:', error);
        elements.historyList.innerHTML = `
            <div class="text-center py-12 text-red-600">
                <p>Gagal memuat riwayat: ${error.message}</p>
            </div>
        `;
    }
}

function displayHistory(records) {
    if (!records || records.length === 0) {
        elements.historyList.innerHTML = `
            <div class="text-center py-12 text-gray-500">
                <svg class="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                </svg>
                <p>Tidak ada riwayat absensi</p>
            </div>
        `;
        return;
    }
    
    elements.historyList.innerHTML = records.map(record => `
        <div class="bg-gray-50 border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
            <div class="flex justify-between items-center">
                <div>
                    <p class="font-medium text-gray-900">${record.employee_name}</p>
                    <p class="text-sm text-gray-600">ID: ${record.employee_id}</p>
                    ${record.confidence ? `<p class="text-xs text-gray-500 mt-1">Confidence: ${(record.confidence * 100).toFixed(1)}%</p>` : ''}
                </div>
                <div class="text-right">
                    <p class="text-sm text-gray-600">${formatDateTime(record.timestamp)}</p>
                    <span class="inline-block mt-2 px-3 py-1 rounded-full text-xs font-medium ${
                        record.type === 'check_in' 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-orange-100 text-orange-800'
                    }">
                        ${record.type === 'check_in' ? 'Check In' : 'Check Out'}
                    </span>
                </div>
            </div>
        </div>
    `).join('');
}

// ============================================
// Statistics
// ============================================

async function loadStats() {
    try {
        // Get general stats
        const statsResponse = await fetch(`${CONFIG.API_BASE_URL}/api/admin/stats`);
        const stats = await statsResponse.json();
        
        // Get root info
        const rootResponse = await fetch(`${CONFIG.API_BASE_URL}/`);
        const rootInfo = await rootResponse.json();
        
        // Update UI
        elements.statTotalEmployees.textContent = stats.total_employees || '0';
        elements.statTodayAttendance.textContent = stats.today_attendance || '0';
        elements.statTotalAttendance.textContent = stats.total_attendance_records || '0';
        elements.statModel.textContent = stats.model || 'Unknown';
        elements.statThreshold.textContent = stats.threshold || 'Unknown';
        elements.statVersion.textContent = rootInfo.version || 'Unknown';
        
    } catch (error) {
        console.error('Stats error:', error);
    }
}

// ============================================
// Server Status
// ============================================

async function checkServerStatus() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/`, {
            method: 'GET',
            mode: 'cors',
        });
        
        if (response.ok) {
            updateServerStatus(true);
        } else {
            updateServerStatus(false);
        }
    } catch (error) {
        console.error('Server check error:', error);
        updateServerStatus(false);
    }
}

function updateServerStatus(isOnline) {
    const statusDot = elements.serverStatus.querySelector('div');
    const statusText = elements.serverStatus.querySelector('span');
    
    if (isOnline) {
        statusDot.className = 'w-3 h-3 bg-green-500 rounded-full animate-pulse';
        statusText.textContent = 'Server Online';
        statusText.className = 'text-sm text-green-600 font-medium';
    } else {
        statusDot.className = 'w-3 h-3 bg-red-500 rounded-full';
        statusText.textContent = 'Server Offline';
        statusText.className = 'text-sm text-red-600 font-medium';
        showMessage('Server tidak dapat dihubungi. Pastikan server berjalan di ' + CONFIG.API_BASE_URL, 'error');
    }
}

// ============================================
// UI Helpers
// ============================================

function showMessage(message, type = 'info') {
    const colors = {
        success: 'bg-green-50 text-green-800 border-green-200',
        error: 'bg-red-50 text-red-800 border-red-200',
        warning: 'bg-yellow-50 text-yellow-800 border-yellow-200',
        info: 'bg-blue-50 text-blue-800 border-blue-200',
    };
    
    const icons = {
        success: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>',
        error: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"></path>',
        warning: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>',
        info: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>',
    };
    
    elements.statusMessage.className = `p-4 rounded-lg border flex items-center gap-3 ${colors[type]}`;
    elements.statusMessage.innerHTML = `
        <svg class="w-6 h-6 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            ${icons[type]}
        </svg>
        <span class="font-medium">${message}</span>
    `;
    elements.statusMessage.classList.remove('hidden');
    
    // Auto-hide after 5 seconds for non-error messages
    if (type !== 'error') {
        setTimeout(() => {
            elements.statusMessage.classList.add('hidden');
        }, 9000);
    }
}

function formatDateTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('id-ID', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

// ============================================
// Tab Management
// ============================================

function switchTab(tabName) {
    // Hide all content
    elements.contentCheckin.classList.add('hidden');
    elements.contentHistory.classList.add('hidden');
    elements.contentStats.classList.add('hidden');
    
    // Reset all tabs
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.className = 'tab-button flex-1 py-4 px-6 font-medium text-gray-600 hover:bg-gray-50 transition-colors';
    });
    
    // Show selected content and highlight tab
    switch(tabName) {
        case 'checkin':
            elements.contentCheckin.classList.remove('hidden');
            elements.tabCheckin.className = 'tab-button flex-1 py-4 px-6 font-medium text-blue-600 border-b-2 border-blue-600 bg-blue-50 transition-colors';
            break;
        case 'history':
            elements.contentHistory.classList.remove('hidden');
            elements.tabHistory.className = 'tab-button flex-1 py-4 px-6 font-medium text-blue-600 border-b-2 border-blue-600 bg-blue-50 transition-colors';
            loadHistory();
            break;
        case 'stats':
            elements.contentStats.classList.remove('hidden');
            elements.tabStats.className = 'tab-button flex-1 py-4 px-6 font-medium text-blue-600 border-b-2 border-blue-600 bg-blue-50 transition-colors';
            loadStats();
            break;
    }
}

// ============================================
// Event Listeners
// ============================================

function setupEventListeners() {
    // Camera controls
    elements.btnStartCamera.addEventListener('click', startCamera);
    elements.btnStopCamera.addEventListener('click', stopCamera);
    
    // Tabs
    elements.tabCheckin.addEventListener('click', () => switchTab('checkin'));
    elements.tabHistory.addEventListener('click', () => switchTab('history'));
    elements.tabStats.addEventListener('click', () => switchTab('stats'));
    
    // History refresh
    elements.btnRefreshHistory.addEventListener('click', loadHistory);
    
    // History filters
    elements.filterEmployeeId.addEventListener('input', filterHistory);
    elements.filterType.addEventListener('change', filterHistory);
    
    // Window resize
    window.addEventListener('resize', () => {
        if (state.isCameraActive) {
            setupCanvas();
        }
    });
}

function filterHistory() {
    // Implement filtering logic
    loadHistory();
}

// ============================================
// Start Application
// ============================================

// Wait for DOM and face-api.js to load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    // Check if face-api is loaded
    const checkFaceApi = setInterval(() => {
        if (typeof faceapi !== 'undefined') {
            clearInterval(checkFaceApi);
            init();
        }
    }, 100);
}