import os
import torch
from torch.utils.data import Dataset
import binvox_rw
import pandas as pd
import numpy as np

class VoxelDataset(Dataset):
    def __init__(self, csv_file, voxel_dir):
        self.df = pd.read_csv(csv_file)
        self.voxel_dir = voxel_dir
        # voxel_files 생성
        self.voxel_files = [os.path.join(self.voxel_dir, fname) for fname in self.df['filename']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        with open(self.voxel_files[idx], 'rb') as f:
            voxel_data = binvox_rw.read_as_3d_array(f).data
            voxel_data = voxel_data.astype(np.float32)
            voxel_tensor = torch.tensor(voxel_data).unsqueeze(0)  # 채널 차원을 추가
            return voxel_tensor
