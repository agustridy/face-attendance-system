# Menggunakan FastAPI DeepFace FastNet512 untuk Sistem Absensi Wajah
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import numpy as np
import json
from pathlib import Path
from deepface import DeepFace
from fastapi.staticfiles import StaticFiles


# Configuration
EMPLOYEE_PHOTOS_DIR = Path("employee_photos")
DESCRIPTORS_FILE = Path("face_descriptors.json")
ATTENDANCE_FILE = Path("attendance_records.json")
SIMILARITY_THRESHOLD = 0.4  # Untuk cosine distance (lebih rendah = lebih mirip)

# Models
class AttendanceRequest(BaseModel):
    descriptor: List[float]
    timestamp: str

class AttendanceRecord(BaseModel):
    employee_id: str
    employee_name: str
    timestamp: str
    type: str
    confidence: float

class ProcessPhotosResponse(BaseModel):
    processed: int
    failed: List[str]
    message: str

# Storage
face_database = {}
attendance_records = []

def load_face_database():
    """Load face descriptors from file"""
    global face_database
    if DESCRIPTORS_FILE.exists():
        with open(DESCRIPTORS_FILE, 'r') as f:
            data = json.load(f)
            face_database = {
                k: {
                    'descriptor': np.array(v['descriptor']),
                    'name': v['name']
                }
                for k, v in data.items()
            }
        print(f"Loaded {len(face_database)} face descriptors")
    else:
        face_database = {}

def save_face_database():
    """Save face descriptors to file"""
    data = {
        k: {
            'descriptor': v['descriptor'].tolist(),
            'name': v['name']
        }
        for k, v in face_database.items()
    }
    with open(DESCRIPTORS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def load_attendance_records():
    """Load attendance records from file"""
    global attendance_records
    if ATTENDANCE_FILE.exists():
        with open(ATTENDANCE_FILE, 'r') as f:
            attendance_records = json.load(f)
    else:
        attendance_records = []

def save_attendance_records():
    """Save attendance records to file"""
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(attendance_records, f, indent=2)

def process_employee_photo(photo_path: Path):
    """
    Process single employee photo using DeepFace
    Returns: (employee_id, descriptor, name) or None if failed
    """
    try:
        # Get employee ID from filename
        employee_id = photo_path.stem
        
        # Extract face embedding using DeepFace
        # Model options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib
        embedding_objs = DeepFace.represent(
            img_path=str(photo_path),
            model_name="Facenet512",  # 512-dimensional embedding
            enforce_detection=True
        )
        
        if not embedding_objs:
            print(f"No face found in {photo_path.name}")
            return None
        
        # Get first face embedding
        descriptor = np.array(embedding_objs[0]["embedding"])
        
        # Get employee name from filename
        name_parts = employee_id.split('_')[1:]
        name = ' '.join(name_parts) if name_parts else employee_id
        
        return employee_id, descriptor, name
        
    except Exception as e:
        print(f"Error processing {photo_path.name}: {str(e)}")
        return None

def cosine_distance(a, b):
    """Calculate cosine distance between two vectors"""
    # Normalize vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    
    # Calculate cosine similarity
    similarity = np.dot(a_norm, b_norm)
    
    # Convert to distance (0 = identical, 2 = opposite)
    distance = 1 - similarity
    
    return distance

def find_matching_face(input_descriptor: np.ndarray):
    """
    Find matching face in database using cosine distance
    Returns: (employee_id, name, distance) or (None, None, None)
    """
    if not face_database:
        return None, None, None
    
    best_match_id = None
    best_match_name = None
    best_distance = float('inf')
    
    for emp_id, data in face_database.items():
        stored_descriptor = data['descriptor']
        
        # Calculate cosine distance
        distance = cosine_distance(input_descriptor, stored_descriptor)
        
        if distance < best_distance:
            best_distance = distance
            best_match_id = emp_id
            best_match_name = data['name']
    
    # Check if best match is below threshold
    if best_distance < SIMILARITY_THRESHOLD:
        return best_match_id, best_match_name, best_distance
    
    return None, None, None

# Startup event
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup"""
    EMPLOYEE_PHOTOS_DIR.mkdir(exist_ok=True)
    load_face_database()
    load_attendance_records()
    print("Server started successfully")
    print(f"Using DeepFace with Facenet512 model")
    yield
    # Shutdown code here if needed

app = FastAPI(title="Face Attendance API", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Face Attendance API",
        "version": "2.0 (DeepFace)",
        "model": "Facenet512",
        "employees_registered": len(face_database),
        "total_attendance": len(attendance_records)
    }

@app.post("/api/admin/process-photos", response_model=ProcessPhotosResponse)
async def process_all_photos():
    """Process all photos in employee_photos directory"""
    if not EMPLOYEE_PHOTOS_DIR.exists():
        raise HTTPException(status_code=400, detail="Employee photos directory not found")
    
    photo_files = list(EMPLOYEE_PHOTOS_DIR.glob("*.jpg")) + \
                  list(EMPLOYEE_PHOTOS_DIR.glob("*.jpeg")) + \
                  list(EMPLOYEE_PHOTOS_DIR.glob("*.png"))
    
    if not photo_files:
        raise HTTPException(status_code=400, detail="No photos found in directory")
    
    processed_count = 0
    failed_files = []
    
    print(f"Processing {len(photo_files)} photos...")
    
    for photo_path in photo_files:
        print(f"Processing: {photo_path.name}")
        result = process_employee_photo(photo_path)
        if result:
            employee_id, descriptor, name = result
            face_database[employee_id] = {
                'descriptor': descriptor,
                'name': name
            }
            processed_count += 1
            print(f"✓ Processed: {name} ({employee_id})")
        else:
            failed_files.append(photo_path.name)
            print(f"✗ Failed: {photo_path.name}")
    
    # Save to file
    save_face_database()
    
    return ProcessPhotosResponse(
        processed=processed_count,
        failed=failed_files,
        message=f"Successfully processed {processed_count} photos"
    )

@app.get("/api/admin/database-info")
async def get_database_info():
    """Get information about face database"""
    employees = [
        {
            "employee_id": emp_id,
            "name": data['name'],
            "descriptor_size": len(data['descriptor'])
        }
        for emp_id, data in face_database.items()
    ]
    
    return {
        "total_employees": len(face_database),
        "model": "Facenet512",
        "descriptor_dimensions": 512,
        "employees": employees
    }

@app.post("/api/attendance/checkin")
async def check_in(request: AttendanceRequest):
    """Process check-in attendance"""
    if not face_database:
        raise HTTPException(
            status_code=400, 
            detail="No employees registered. Please process photos first."
        )
    
    input_descriptor = np.array(request.descriptor)
    
    # Validate descriptor size (128 for face-api.js, 512 for Facenet512)
    if len(input_descriptor) not in [128, 512]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid descriptor size. Expected 128 or 512, got {len(input_descriptor)}"
        )
    
    # If descriptor is 128 (from face-api.js), we need to handle differently
    # For now, we'll reject and ask client to use compatible model
    if len(input_descriptor) != 512:
        raise HTTPException(
            status_code=400,
            detail=f"Client must use Facenet512 model for compatibility. Descriptor size should be 512. , got {len(input_descriptor)}"
        )
    
    # Find matching face
    employee_id, employee_name, distance = find_matching_face(input_descriptor)
    print(f"Check-in attempt: {employee_id}, distance: {distance}")
    
    if not employee_id:
        raise HTTPException(
            status_code=404,
            detail="Face not recognized. Please contact administrator."
        )
    
    # Calculate confidence
    confidence = 1.0 - (distance / SIMILARITY_THRESHOLD)
    confidence = max(0.0, min(1.0, confidence))
    
    # Create attendance record
    record = {
        "employee_id": employee_id,
        "employee_name": employee_name,
        "timestamp": request.timestamp,
        "type": "check_in",
        "confidence": round(confidence, 2),
        "distance": round(float(distance), 4)
    }
    
    attendance_records.append(record)
    save_attendance_records()
    
    return record

@app.post("/api/attendance/checkout")
async def check_out(request: AttendanceRequest):
    """Process check-out attendance"""
    if not face_database:
        raise HTTPException(
            status_code=400,
            detail="No employees registered. Please process photos first."
        )
    
    input_descriptor = np.array(request.descriptor)
    
    if len(input_descriptor) != 512:
        raise HTTPException(
            status_code=400,
            detail="Client must use Facenet512 model. Descriptor size should be 512."
        )
    
    employee_id, employee_name, distance = find_matching_face(input_descriptor)
    
    if not employee_id:
        raise HTTPException(
            status_code=404,
            detail="Face not recognized. Please contact administrator."
        )
    
    confidence = 1.0 - (distance / SIMILARITY_THRESHOLD)
    confidence = max(0.0, min(1.0, confidence))
    
    record = {
        "employee_id": employee_id,
        "employee_name": employee_name,
        "timestamp": request.timestamp,
        "type": "check_out",
        "confidence": round(confidence, 2),
        "distance": round(float(distance), 4)
    }
    
    attendance_records.append(record)
    save_attendance_records()
    
    return record

@app.get("/api/attendance/history")
async def get_attendance_history(
    employee_id: Optional[str] = None,
    limit: int = 50
):
    """Get attendance history"""
    records = attendance_records
    
    if employee_id:
        records = [r for r in records if r['employee_id'] == employee_id]
    
    sorted_records = sorted(
        records,
        key=lambda x: x['timestamp'],
        reverse=True
    )[:limit]
    
    return sorted_records

@app.get("/api/attendance/today")
async def get_today_attendance():
    """Get today's attendance records"""
    today = datetime.now().date().isoformat()
    
    today_records = [
        r for r in attendance_records
        if r['timestamp'].startswith(today)
    ]
    
    return {
        "date": today,
        "total_records": len(today_records),
        "records": today_records
    }

@app.delete("/api/admin/clear-attendance")
async def clear_attendance():
    """Clear all attendance records"""
    global attendance_records
    attendance_records = []
    save_attendance_records()
    return {"message": "All attendance records cleared"}

@app.get("/api/admin/stats")
async def get_statistics():
    """Get system statistics"""
    today = datetime.now().date().isoformat()
    today_records = [
        r for r in attendance_records
        if r['timestamp'].startswith(today)
    ]
    
    return {
        "total_employees": len(face_database),
        "total_attendance_records": len(attendance_records),
        "today_attendance": len(today_records),
        "threshold": SIMILARITY_THRESHOLD,
        "model": "Facenet512"
    }

app.mount("/models", StaticFiles(directory="models"), name="models")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)