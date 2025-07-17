from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
import io
import pusher

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

# Load YOLO TorchScript model once
MODEL_PATH = "my_model/my_model.torchscript"
model = YOLO(MODEL_PATH)

# Define label map
labels = {
    0: "Anthracnose Twister",
    1: "Botrytis Leaf Blight",
    2: "Downy Mildew",
    3: "Healthy",
    4: "Purple Blotch",
    5: "Stemphylium Leaf Blight",
}

@app.post("/detect")
async def detect_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    image_bytes = await file.read()
    results = model(io.BytesIO(image_bytes))
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

    response_payload = {
        "filename": file.filename,
        "results": final_label,
        "box": final_box,
        "found": found
    }

    background_tasks.add_task(
        pusher_client.trigger,
        'detection-channel',
        'new-detection',
        response_payload
    )

    return response_payload
