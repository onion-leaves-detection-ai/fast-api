from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import pusher

app = FastAPI()

# CORS for any frontend client (localhost, LAN, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pusher configuration
pusher_client = pusher.Pusher(
    app_id='2021947',
    key='e019d7e7760033b03392',
    secret='cb3d9f86650651582039',
    cluster='ap1',
    ssl=False
)

# Load TorchScript model
MODEL_PATH = "my_model/my_model.torchscript"
model = YOLO(MODEL_PATH)
labels = model.names

# Ensure uploads folder exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = model(file_path)
        detections = results[0].boxes

        found = []
        grouped = {}
        box_map = {}

        for det in detections:
            if det.cls is None:
                continue  # Skip if no class detected

            label = labels[int(det.cls.item())]
            conf = float(det.conf.item())
            box = det.xyxy[0].tolist()

            found.append({
                "label": label,
                "confidence": round(conf, 3)
            })

            if label not in grouped:
                grouped[label] = []
                box_map[label] = box
            grouped[label].append(conf)

        final_label = "No detection"
        final_conf = 0
        final_box = []

        if grouped:
            for label, scores in grouped.items():
                avg = sum(scores) / len(scores)
                if avg > final_conf:
                    final_label = label
                    final_conf = avg
                    final_box = box_map[label]

        # Send to Pusher
        pusher_client.trigger(
            'detection-channel',
            'new-detection',
            {
                "filename": file.filename,
                "results": final_label,
                "box": final_box,
                "found": found
            }
        )

        return {
            "filename": file.filename,
            "results": final_label,
            "box": final_box,
            "found": found
        }

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
