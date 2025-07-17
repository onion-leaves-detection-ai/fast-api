from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
import shutil
import pusher
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Pusher
pusher_client = pusher.Pusher(
    app_id='2021947',
    key='e019d7e7760033b03392',
    secret='cb3d9f86650651582039',
    cluster='ap1',
    ssl=False
)

# Load model (TorchScript compatible)
MODEL_PATH = "my_model/my_model.torchscript"
model = YOLO(MODEL_PATH)

labels = {
    0: "Anthracnose Twister",
    1: "Botrytis Leaf Blight",
    2: "Downy Mildew",
    3: "Healthy",
    4: "Purple Blotch",
    5: "Stemphylium Leaf Blight",
}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/detect")
async def detect_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # Generate unique filename to avoid collisions
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = model(file_path)
        detections = results[0].boxes

        found = []
        grouped = {}
        box_map = {}

        for det in detections:
            if det.cls is None or det.conf is None:
                continue

            cls_id = int(det.cls.item())
            conf = float(det.conf.item())
            box = det.xyxy[0].tolist()
            label = labels.get(cls_id, f"Class {cls_id}")

            found.append({"label": label, "confidence": round(conf, 3)})
            grouped.setdefault(label, []).append(conf)
            box_map[label] = box

        final_label = "No detection"
        final_conf = 0
        final_box = []

        for label, scores in grouped.items():
            avg = sum(scores) / len(scores)
            if avg > final_conf:
                final_label = label
                final_conf = avg
                final_box = box_map[label]

        payload = {
            "filename": file.filename,
            "results": final_label,
            "box": final_box,
            "found": found
        }

        # Trigger pusher in background
        background_tasks.add_task(
            pusher_client.trigger,
            'detection-channel',
            'new-detection',
            payload
        )

        return payload

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
