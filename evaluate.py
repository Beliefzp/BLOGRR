import torch
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from medpy.metric.binary import dc
from typing import Tuple
from torch import Tensor
import kornia

class evaluate_BLOGRR:
    def __init__(self, config, model, threshold_model):
        self.config = config
        self.model = model
        self.threshold_model = threshold_model
    
    def val_step(self, input) -> Tuple[dict, Tensor]:
        self.model.eval()
        self.threshold_model.eval()
        with torch.no_grad():
            input_recon = self.model(input)
            threshold = self.threshold_model(input)
        anomaly_map = (input - input_recon).abs().mean(1, keepdim=True)
        anomaly_map = kornia.filters.gaussian_blur2d(anomaly_map, kernel_size=(5,5), sigma=(self.config.sigma, self.config.sigma))
        threshold = threshold.unsqueeze(2).unsqueeze(3)    
        threshold = threshold.expand(-1, -1, self.config.image_size, self.config.image_size)
        pred = (anomaly_map > threshold).float()
        mask = torch.stack([inp > inp.min() for inp in input])
        anomaly_map *= mask
        pred *= mask
        return anomaly_map, input_recon, pred, threshold
    
    def visual_result(self, input, seg):
        output = self.val_step(input)
        anomaly_map, input_recon, pred, threshold = output
        dc_value = dc(pred.cpu().numpy(), seg.cpu().numpy())  # 计算 DC 值
        print("dice is:", dc_value)
        subfolder_name = f"{dc_value:.2f}"
        subfolder = os.path.join(self.config.result_save, subfolder_name)
        os.makedirs(subfolder, exist_ok=True)
        input_save = input.detach().cpu().numpy()
        seg_save = seg.detach().cpu().numpy()
        input_recon_save = input_recon.detach().cpu().numpy()
        anomaly_map_save = anomaly_map.detach().cpu().numpy()
        pred_save = pred.detach().cpu().numpy()
        threshold_save = threshold.detach().cpu().numpy()
        np.save(os.path.join(subfolder, 'input.npy'), input_save)
        np.save(os.path.join(subfolder, 'mask.npy'), seg_save)
        np.save(os.path.join(subfolder, 'input_recon.npy'), input_recon_save)
        np.save(os.path.join(subfolder, 'anomaly_map.npy'), anomaly_map_save)
        np.save(os.path.join(subfolder, 'pred.npy'), pred_save)
        np.save(os.path.join(subfolder, 'threshold.npy'), threshold_save)