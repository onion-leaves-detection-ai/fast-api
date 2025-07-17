from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
import os
import pusher
from PIL import Image
import numpy as np

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pusher setup
pusher_client = pusher.Pusher(
    app_id='2021947',
    key='e019d7e7760033b03392',
    secret='cb3d9f86650651582039',
    cluster='ap1',
    ssl=False
)

# Model and labels
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

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        # Read file in memory
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(image)

        # Perform detection
        results = model(np_image)
        detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data is not None else []

        found = []
        grouped = {}
        box_map = {}

        for det in detections:
            if len(det) < 6:
                continue  # Skip invalid detection

            x1, y1, x2, y2, conf, cls_id = det[:6]
            label = labels.get(int(cls_id), f"Class {int(cls_id)}")
            confidence = float(conf)
            box = [float(x1), float(y1), float(x2), float(y2)]

            found.append({"label": label, "confidence": round(confidence, 3)})

            if label not in grouped:
                grouped[label] = []
                box_map[label] = box
            grouped[label].append(confidence)

        final_label = "No detection"
        final_conf = 0
        final_box = []

        for label, scores in grouped.items():
            avg_conf = sum(scores) / len(scores)
            if avg_conf > final_conf:
                final_label = label
                final_conf = avg_conf
                final_box = box_map[label]

        # Send result to Pusher
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

    except Exception as e:
        return {"error": str(e)}
