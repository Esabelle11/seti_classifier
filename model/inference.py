import torch
# from data.transforms import get_transform
from model.loader import load_model
import torchvision.transforms as T


model = load_model("weight\model_weights_vgg16pre.pth")
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
