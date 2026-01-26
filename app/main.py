# from fastapi import FastAPI, File, UploadFile
# from PIL import Image
# import io
# import torch
# import torchvision.transforms as T
# import numpy as np
# import os

# app = FastAPI()

# MODEL_PATH = os.environ.get("weight", "model_weights_vgg16pre.pth")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.jit.load(MODEL_PATH, map_location=device)
# model.eval()

# transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
#     # T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# CLASS_NAMES = ['brightpixel','narrowband','narrowbanddrd','noise','squarepulsednarrowband','squiggle','squigglesquarepulsednarrowband']

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     data = await file.read()
#     img = Image.open(io.BytesIO(data)).convert("RGB")
#     x = transform(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         out = model(x)
#         probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
#         pred_idx = int(probs.argmax())
#     return {"pred": CLASS_NAMES[pred_idx], "scores": probs.tolist()}


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
app.mount("/static", StaticFiles(directory=app_dir), name="static")

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
