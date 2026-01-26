import torch
from model import VGG16

def export_torchscript(model_path, output_path):
    model = VGG16()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    example = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, example)
    traced.save(output_path)
    print(f'TorchScript model saved to {output_path}')

if __name__ == '__main__':
    # 路径根据实际情况调整
    model_path = '/kaggle/working/model_weights_vgg16pre.pth'
    output_path = '/kaggle/working/model_ts.pt'
    export_torchscript(model_path, output_path)
