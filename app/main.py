
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
from model.inference import predict
from model.cam import CAM_ALG

app = FastAPI(title="SETI Signal Classifier API")
CLASS_NAMES = ['brightpixel','narrowband','narrowbanddrd','noise','squarepulsednarrowband','squiggle','squigglesquarepulsednarrowband']

# Get app directory
app_dir = os.path.dirname(os.path.abspath(__file__))

# Serve static files (CSS, JS)
# app.mount("/static", StaticFiles(directory=app_dir), name="static")
app.mount("/static", StaticFiles(directory=os.path.join(app_dir, "static")), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(app_dir, "index.html"))

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # cls, prob = predict(image)
    img_base64, cls = CAM_ALG(image)

    return {
        "class_id": CLASS_NAMES[cls],
        "cam_image": img_base64
    }
