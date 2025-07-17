from torch.utils.data import Dataset
import torch
import numpy as np
from torch import Tensor
from typing import Tuple
from PIL import Image
import os

class NormalDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith('.png')]
           
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx) -> Tensor:
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img_tensor = torch.FloatTensor(np.array(img)) / 255.0
        return img_tensor.unsqueeze(0)


class AnomalDataset(Dataset):
    def __init__(self, img_dir, seg_dir):
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.img_paths = [os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith('.png')]
        self.seg_paths = [os.path.join(self.seg_dir, os.path.basename(img_path)) for img_path in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img_tensor = torch.FloatTensor(np.array(img)) / 255.0
        seg_path = self.seg_paths[idx]
        seg = Image.open(seg_path)
        seg_tensor = torch.FloatTensor(np.array(seg)) / 255.0
        return img_tensor.unsqueeze(0), seg_tensor.unsqueeze(0)


class AnomalDataset_Test(Dataset):
    def __init__(self, img_dir, seg_dir):
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        img_files = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]
        img_files_sorted = sorted(
            img_files,
            key=lambda x: int(x.split('_')[-1].split('.')[0]) 
        )
        self.img_paths = [os.path.join(self.img_dir, f) for f in img_files_sorted]
        self.seg_paths = [
            os.path.join(self.seg_dir, os.path.basename(img_path))
            for img_path in self.img_paths
        ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img_tensor = torch.FloatTensor(np.array(img)) / 255.0
        seg_path = self.seg_paths[idx]
        seg = Image.open(seg_path)
        seg_tensor = torch.FloatTensor(np.array(seg)) / 255.0
        return img_tensor.unsqueeze(0), seg_tensor.unsqueeze(0)
