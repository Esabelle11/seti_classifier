import torch
from model.loader import load_model
import torchvision.transforms as T
import numpy as np
import cv2
import base64
import io
from PIL import Image

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(BASE_DIR, "..", "weight", "model_weights_vgg16pre.pth")


model = load_model(weight_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# transform = get_transform()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    # T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


def CAM_ALG(image):
    image = transform(image).unsqueeze(0).to(device)

    
    # output of features and predict label
    model.eval()
    features = model.vgg16.features(image) #torch.Size([1, 256, 6, 6])
    output = model.vgg16.classifier(features.flatten())
    # print(output)
    #print(output.shape)
    #pred_label= torch.argmax(output).item()
    
    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g
 
    # 预测得分最高的那一类对应的输出score
    pred_label = torch.argmax(output).item()
    pred_class = output[pred_label]
    #print(pred_class)
 
    features.register_hook(extract)
    pred_class.backward() # 计算梯度
    grads = features_grad   # 获取梯度
    #print(grads)
    
    
    # Do Global Aberage Pooling
    GAP_features = torch.nn.functional.adaptive_avg_pool2d(grads,(1,1))
    #print(GAP_features.shape) #torch.Size([1, 256, 1, 1])
    
    #Weight_sum_average on the features and features after GAP
    Weighted_sum = features*GAP_features
    Weighted_sum = Weighted_sum.squeeze().cpu().detach().numpy() #(256, 6, 6)
    cam = np.mean(Weighted_sum, axis=0) #(6, 6)
    #print(cam)
    cam=np.maximum(cam,0)
    #cam/= np.max(cam)
    #print( np.max(np.maximum(cam,0.0001)))
    cam/= np.maximum(np.max(cam),0.00000000001)

    # Resize the CAM to the original image size and Min-Max Normalization
    cam = cv2.resize(cam, (224, 224))
    
    # Apply heatmap to the original image
    heatmap = cv2.applyColorMap(np.uint8(254 * (1 - cam)), cv2.COLORMAP_JET)
    image = np.transpose(image.squeeze().cpu().numpy(), (1, 2, 0))
    heatmap =heatmap/ 254 # tonormalize
    superimposed_img = heatmap * 0.4 + image


    # output
    pil_img = Image.fromarray((superimposed_img*255).astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_base64, pred_label

    # return superimposed_img, pred_label