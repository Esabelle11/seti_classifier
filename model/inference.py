import torch
# from data.transforms import get_transform
from model.loader import load_model
import torchvision.transforms as T
import os
from huggingface_hub import hf_hub_download


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# weight_path = os.path.join(BASE_DIR, "..", "weight", "model_weights_vgg16pre.pth")
weight_path = hf_hub_download(
    repo_id="Esabelle/seti_classifier_vgg16_model",
    filename="model_weights_vgg16pre.pth"
)


model = load_model(weight_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# transform = get_transform()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    # T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.softmax(output, dim=1)
        cls = prob.argmax(dim=1).item()

    return cls, prob[0].tolist()
