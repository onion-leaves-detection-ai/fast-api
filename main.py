from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import pusher

from typing import List
# Fix for PyTorch 2.6+: allow model class for loading
import torch
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pusher
pusher_client = pusher.Pusher(
    app_id='2021947',
    key='e019d7e7760033b03392',
    secret='cb3d9f86650651582039',
    cluster='ap1',
    ssl=False
)
# Load model
MODEL_PATH = "my_model/my_model.pt"
model = YOLO(MODEL_PATH)
labels = model.names

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(file_path)
    detections = results[0].boxes
    output = []
    boxes = []

    for det in detections:
        class_id = int(det.cls.item())
        label = labels[class_id]
        confidence = float(det.conf.item())

        # Accurate box from det.xyxy (absolute pixel coordinates)
        box_xyxy = det.xyxy[0].tolist()  # [x1, y1, x2, y2]

        output.append({
            "label": label,
            "confidence": round(confidence, 3)
        })

        boxes.append({
            "box": box_xyxy,
            "label": label,
            "confidence": round(confidence, 3)
        })

    # 🔔 Send via Pusher to frontend
    pusher_client.trigger(
        'detection-channel',
        'new-detection',
        {
            "filename": file.filename,
            "results": output[0] if output else {},
            "boxes": boxes
        }
    )

    return {
        "filename": file.filename,
        "results": output
    }
