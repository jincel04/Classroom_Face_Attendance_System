# Classroom_Face_Attendance_System
Classroom Face Attendance (30-40s Target)
High-throughput classroom attendance system for a standard laptop/PC webcam.

This project uses:

OpenCV for camera/video/image I/O and visualization
MTCNN for deep face detection + alignment
InceptionResnetV1 (FaceNet-style) embeddings for recognition
SQLite + CSV for attendance persistence/reporting
Designed to mark attendance for 20-100 students in ~30-40 seconds (hardware and scene dependent).

Features
Real-time attendance from:
Live webcam (--mode live)
Group photo (--mode image)
Video file (--mode video)
Dataset enrollment from webcam or folder
Multiple face samples per student
Embedding build and prototype generation
Session-level deduplication (student marked once per session)
CSV export and console summary reports
Benchmark script for latency/FPS estimation
Frontend APIs for:
backend roster CRUD (DB as source of truth)
live attendance start/stop/status/events/frame streaming
enrollment auth + rate limiting protections
Project Structure
attendance_system/
  __init__.py
  attendance.py         # session runner (live/video/image)
  camera.py             # threaded webcam capture
  config.py             # tunable runtime settings
  db.py                 # SQLite schema + queries
  embedding_builder.py  # build embeddings/prototypes from aligned face crops
  enrollment.py         # webcam/folder enrollment
  preprocess.py         # MTCNN detection + alignment helpers
  recognition.py        # embedding inference + prototype matching
  report.py             # text summary + CSV export helpers
scripts/
  init_db.py
  enroll_student.py
  build_embeddings.py
  run_attendance.py
  export_report.py
  benchmark.py
tests/
  test_db.py
data/
  raw/                  # captured raw images per student
  aligned/              # aligned face crops per student
  reports/              # CSV and annotated outputs
  sessions/
Installation
Create and activate a virtual environment.
python -m venv .venv
.venv\Scripts\Activate.ps1
If you are using Command Prompt (cmd.exe) instead of PowerShell:

python -m venv .venv
call .venv\Scripts\activate.bat
Install dependencies.
pip install --upgrade pip
pip install -r requirements.txt
Initialize the database and folders.
python scripts/init_db.py
Frontend + Camera Enrollment
Run the frontend server (includes API endpoints used by frontend/enrollment.html):

python scripts/run_frontend_server.py --host 127.0.0.1 --port 8000
Optional security:

$env:ATTENDANCE_ADMIN_TOKEN = "change-me"
python scripts/run_frontend_server.py
When ATTENDANCE_ADMIN_TOKEN is set, protected write APIs require X-Admin-Token header. If not set, protected write APIs are limited to loopback clients.

Then open:

http://127.0.0.1:8000/enrollment.html for student enrollment + camera capture
http://127.0.0.1:8000/index.html for live attendance page (real backend detections + frame stream)
Enrollment Workflow
Option A: Webcam enrollment (recommended)
Capture 20-40 front-facing samples per student.

python scripts/enroll_student.py --student-id S001 --name "Alice Kim" --source webcam --num-images 25
--student-id can be any enrollment number or ID (for example 1, 23, CSE42).
No S00 prefix is required.

Fast defaults are enabled for webcam enrollment:

capture interval: 0.25s
minimum probability: 0.90
webcam resolution: 640x480
Notes:

Keep only the target student in frame during enrollment.
Vary expression/head angle slightly.
Avoid blur and harsh backlight.
Option B: Folder enrollment
python scripts/enroll_student.py --student-id S002 --name "Bob Lee" --source folder --input-dir .\samples\bob
Rebuild embeddings
All students:

python scripts/build_embeddings.py
Single student:

python scripts/build_embeddings.py --student-id S001
Run Attendance
Live webcam
python scripts/run_attendance.py --mode live --camera-index 0
By default, live attendance keeps running until you stop it (q in the OpenCV window, or Ctrl+C in terminal/headless mode).

Group photo
python scripts/run_attendance.py --mode image --image-path .\class_photo.jpg
Video file
python scripts/run_attendance.py --mode video --video-path .\classroom_clip.mp4
Output:

Session summary in terminal
CSV in data/reports/attendance_<session_id>.csv
Annotated image for --mode image
Export Report for Existing Session
Latest session:

python scripts/export_report.py
Specific session:

python scripts/export_report.py --session-id <SESSION_ID> --output-csv .\my_report.csv
Performance Tuning (Critical for 30-40s)
Use these knobs in scripts/run_attendance.py:

--frame-resize 0.7 to 0.85: smaller frame for faster detection
--sample-every-n-frames 2 or 3: fewer frames processed
--min-votes 2: mark quickly while reducing false positives
--duration 30 to 40: optional cap for short fixed sessions
Recommended starting profile for CPU:

python scripts/run_attendance.py --mode live --frame-resize 0.75 --sample-every-n-frames 2 --min-votes 2
Recommended starting profile for GPU:

python scripts/run_attendance.py --mode live --device cuda:0 --frame-resize 0.85 --sample-every-n-frames 1
Expected Accuracy and Speed
With good enrollment quality and classroom lighting:

Recognition precision: typically high (often >95% for clear frontal faces)
Session runtime: typically ~30-40s for 20-100 students
Real outcomes depend on:

Occlusion (masks, hands, side profile)
Motion blur and focus
Resolution and camera field of view
Distance from camera
Enrollment data quality/diversity
Benchmarking
Measure inference latency:

python scripts/benchmark.py --camera-index 0 --max-frames 120 --frame-resize 0.75 --sample-every-n-frames 2
Or with a video:

python scripts/benchmark.py --video-path .\classroom_clip.mp4 --max-frames 150
Testing
Run tests:

pytest -q
Current tests validate:

Attendance deduplication in a session
Prototype load path from SQLite
Attendance Data Model
SQLite tables:

students: master roster
embeddings: per-image embeddings
student_prototypes: mean normalized vector per student
attendance_sessions: session metadata
attendance_records: per-session present records (unique by student/session)
Error Handling and Practical Guidance
If no face is found during enrollment:
Improve lighting and face angle
Increase capture image count
Lower --min-probability slightly (e.g. 0.90)
If false positives increase:
Raise --recognition-threshold (e.g. 0.65)
Raise --mark-threshold (e.g. 0.72)
Increase --min-votes to 3
If attendance is too slow:
Reduce --frame-resize to 0.70
Increase --sample-every-n-frames to 3
Use GPU (--device cuda:0) if available
Notes
This is an engineering template and should be calibrated on your own classroom/camera before production use.
For strict deployments, add liveness checks and anti-spoofing safeguards.
Follow local privacy and consent policies before capturing biometric data.
