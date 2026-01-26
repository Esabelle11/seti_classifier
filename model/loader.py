import torch
from model.vgg16 import VGG16

DEVICE = torch.device("cpu")  # 强制 CPU

def load_model(weight_path: str):
    model = VGG16()
    model.load_state_dict(
        torch.load(weight_path, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model
