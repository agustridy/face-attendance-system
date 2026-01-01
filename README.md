## Absensi Pengenalan Wajah DeepFace Facenet512 FastAPI dan Face-api.js Client Tanpa kirim Foto

[![Tonton di YouTube](https://img.youtube.com/vi/9qk5vxwOqN4/0.jpg)](https://www.youtube.com/watch?v=9qk5vxwOqN4)

## ✨ Fitur

### Backend (FastAPI + DeepFace)
- ✅ Face recognition menggunakan **Facenet512** (512-dimensional embeddings)
- ✅ Pre-processing foto pegawai otomatis
- ✅ Validasi face descriptor real-time
- ✅ RESTful API dengan dokumentasi interaktif
- ✅ Attendance tracking (check-in/check-out)
- ✅ History dan statistik absensi
- ✅ JSON-based storage (mudah di-migrate ke database)
- ✅ CORS support untuk cross-origin requests

### Frontend (Face-api.js)
- ✅ Real-time face detection di browser
- ✅ Face descriptor extraction (512-dimensional)
- ✅ Tidak perlu upload foto (privacy-first)
- ✅ Responsive UI dengan Tailwind CSS
- ✅ Attendance history viewer
- ✅ Multi-browser support

### Keamanan & Privacy
- 🔒 Hanya face descriptor yang dikirim (bukan foto)
- 🔒 Data biometrik terenkripsi
- 🔒 No photo storage di server 
- 🔒 Configurable similarity threshold

## 🚀 Instalasi

### Prerequisites
- Python 3.8+
- Node.js (untuk development frontend)
- Webcam

### Backend Setup
```bash
# Clone repository
git clone [repository-url]
cd face-attendance-system

# Install Python dependencies
pip install -r requirements.txt

# Jalankan server
python main.py
```

### Frontend Setup
```bash
# Buka folder client
cd client

# Install dependencies (jika menggunakan package manager)
npm install  # atau yarn install

# Buka index.html di browser
# Atau gunakan live server
```

## 📁 Struktur Proyek
```
face-attendance-system/
├── main.py                    # Server FastAPI
├── requirements.txt           # Python dependencies
├── employee_photos/           # Folder foto pegawai (upload)
├── face_descriptors.json      # Database descriptor wajah
├── attendance_records.json    # Riwayat absensi
├── models/                    # Model Facenet512 untuk TF.js
│   └── facenet512_model/
│       ├── model.json
│       └── *.bin
├── client/                    # Frontend application
│   ├── index.html
│   ├── app.js
│   └── style.css
└── README.md
```

## 🔧 Konfigurasi

### Server Configuration (main.py)
```python
# Configuration
EMPLOYEE_PHOTOS_DIR = Path("employee_photos")
DESCRIPTORS_FILE = Path("face_descriptors.json")
ATTENDANCE_FILE = Path("attendance_records.json")
SIMILARITY_THRESHOLD = 0.4  # Untuk cosine distance
```

### Client Configuration (app.js)
```javascript
const CONFIG = {
    API_BASE_URL: 'http://localhost:8080',
    MODEL_URL: 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model',
    DETECTION_INTERVAL: 500, // ms
    MIN_CONFIDENCE: 0.5,
};
```

## 📋 Cara Penggunaan

### 1. Setup Awal (Admin)
1. Upload foto pegawai ke folder `employee_photos/`
   - Format nama: `EMP001_John_Doe.jpg`
   - Ekstensi: .jpg, .jpeg, atau .png
2. Jalankan server: `python main.py`
3. Akses endpoint: `POST /api/admin/process-photos`
4. Server akan:
   - Ekstrak descriptor wajah menggunakan Facenet512
   - Simpan descriptor ke `face_descriptors.json`
   - Hapus foto dari server (opsional)

### 2. Daily Usage (Pegawai)
1. Buka `client/index.html` di browser
2. Klik "Start Camera"
3. Posisikan wajah di depan kamera
4. Sistem akan:
   - Deteksi wajah secara real-time
   - Ekstrak descriptor 512-dimensional
   - Kirim descriptor ke server
   - Cocokkan dengan database
   - Catat absensi jika dikenali

### 3. Monitoring (Admin)
- Lihat riwayat absensi: `GET /api/attendance/history`
- Statistik hari ini: `GET /api/attendance/today`
- Info database: `GET /api/admin/database-info`
- Statistik sistem: `GET /api/admin/stats`

## 🔌 API Endpoints

### Authentication & Setup
- `GET /` - Info server
- `POST /api/admin/process-photos` - Proses semua foto pegawai
- `GET /api/admin/database-info` - Info database descriptor
- `DELETE /api/admin/clear-attendance` - Hapus semua riwayat absensi
- `GET /api/admin/stats` - Statistik sistem

### Attendance
- `POST /api/attendance/checkin` - Check-in absensi
- `POST /api/attendance/checkout` - Check-out absensi
- `GET /api/attendance/history` - Riwayat absensi
- `GET /api/attendance/today` - Absensi hari ini

## 🧠 Teknologi yang Digunakan

### Backend
- **FastAPI** - Modern web framework untuk Python
- **DeepFace** - Deep learning face recognition
- **Facenet512** - 512-dimensional face embeddings
- **NumPy** - Scientific computing
- **Pydantic** - Data validation

### Frontend
- **Face-api.js** - Face detection in browser
- **TensorFlow.js** - Machine learning in browser
- **Tailwind CSS** - Utility-first CSS framework
- **JavaScript ES6+** - Modern JavaScript

## 🔒 Keamanan & Privacy

### Privacy-First Design
1. **No Photo Storage**: Foto hanya diproses sekali, lalu dihapus
2. **Descriptor Only**: Hanya mathematical representation yang disimpan
3. **Local Processing**: Face detection di browser client
4. **Encrypted Storage**: Data biometrik terenkripsi

### Security Features
- CORS configuration
- Input validation dengan Pydantic
- Error handling yang aman
- Configurable similarity threshold

## 🚀 Deployment

### Local Development
```bash
# Backend
python main.py

# Frontend
# Buka client/index.html di browser
# Atau gunakan live server extension
```

### Production Considerations
1. **Database**: Migrate dari JSON ke PostgreSQL/MySQL
2. **Authentication**: Tambah JWT authentication
3. **HTTPS**: Enable SSL/TLS
4. **Load Balancing**: Untuk multiple instances
5. **Monitoring**: Logging dan monitoring

## 🐛 Troubleshooting

### Common Issues
1. **Camera not working**: Pastikan izin kamera diberikan
2. **Face not detected**: Pastikan pencahayaan cukup dan wajah terlihat jelas
3. **Server connection error**: Pastikan server berjalan di port 8080
4. **Model loading failed**: Check internet connection untuk download model

### Debug Mode
```python
# Tambah di main.py untuk debug
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance

### Accuracy
- Facenet512: ~99% accuracy pada LFW dataset
- Cosine distance threshold: 0.4 (configurable)
- Real-time processing: < 500ms per detection

### Scalability
- Support untuk ratusan pegawai
- JSON storage mudah di-migrate ke database
- Stateless API design

## 🤝 Kontribusi

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 Lisensi

MIT License - lihat LICENSE file untuk detail

## 🙏 Credits

- **DeepFace** oleh Serengil
- **Face-api.js** oleh Vincent Mühler
- **FastAPI** oleh Sebastián Ramírez
- **Facenet512** model oleh Google Research

## Support Me
[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=agustridy&button_colour=FFDD00&font_colour=000000&font_family=Arial&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/agustridy)

---

**Dibuat dengan ❤️ untuk komunitas programmer Indonesia**

*Sistem absensi modern dengan privacy-first approach*
