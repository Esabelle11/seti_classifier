import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SetiDataset(Dataset):
    def __init__(self, directory, train=False):
        self.images, self.labels = self.get_images(directory)
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        if train:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                # T.Normalize(norm_mean, norm_std),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                # T.Normalize(norm_mean, norm_std)
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = Image.fromarray(image)
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

    def get_images(self, directory):
        images = []
        labels = []
        label_mapping = {
            'brightpixel': 0,
            'narrowband': 1,
            'narrowbanddrd': 2,
            'noise': 3,
            'squarepulsednarrowband': 4,
            'squiggle': 5,
            'squigglesquarepulsednarrowband': 6
        }
        for label in os.listdir(directory):
            if label in label_mapping:
                for image_files in os.listdir(os.path.join(directory, label)):
                    image_path = os.path.join(directory, label, image_files)
                    image = cv2.imread(image_path)
                    images.append(image)
                    labels.append(label_mapping[label])
        return images, labels

def get_classlabels(class_code):
    label_mapping = {
        0: 'brightpixel',
        1: 'narrowband',
        2: 'narrowbanddrd',
        3: 'noise',
        4: 'squarepulsednarrowband',
        5: 'squiggle',
        6: 'squigglesquarepulsednarrowband'
    }
    return label_mapping[class_code]

def get_class_labels():
    return ['brightpixel', 'narrowband', 'narrowbanddrd', 'noise', 'squarepulsednarrowband', 'squiggle', 'squigglesquarepulsednarrowband']
