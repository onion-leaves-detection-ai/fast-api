from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
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

# Load model
MODEL_PATH = "my_model/my_model.torchscript"
model = YOLO(MODEL_PATH)

# ✅ Manually define label map (replace with your actual class labels)
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
            if det.cls is None or det.conf is None:
                continue  # skip if incomplete

            cls_id = int(det.cls.item())
            conf = float(det.conf.item())
            box = det.xyxy[0].tolist()

            label = labels.get(cls_id, f"Class {cls_id}")
            found.append({"label": label, "confidence": round(conf, 3)})

            if label not in grouped:
                grouped[label] = []
                box_map[label] = box
            grouped[label].append(conf)

        final_label = "No detection"
        final_conf = 0
        final_box = []

        for label, scores in grouped.items():
            avg = sum(scores) / len(scores)
            if avg > final_conf:
                final_label = label
                final_conf = avg
                final_box = box_map[label]

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
