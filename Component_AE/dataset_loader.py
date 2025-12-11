import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import binvox_rw


class VoxelDataset(Dataset):
    """
    Dataset for loading voxelized .binvox models.

    Each entry corresponds to a voxel grid stored in a .binvox file.
    The CSV file must contain a column 'filename' with voxel file names.
    """

    def __init__(self, csv_file: str, voxel_dir: str):
        self.df = pd.read_csv(csv_file)
        self.voxel_dir = voxel_dir
        self.voxel_files = [
            os.path.join(voxel_dir, fname) for fname in self.df["filename"]
        ]

    def __len__(self):
        return len(self.voxel_files)

    def __getitem__(self, idx):
        filepath = self.voxel_files[idx]

        with open(filepath, "rb") as f:
            voxel = binvox_rw.read_as_3d_array(f).data

        voxel = voxel.astype(np.float32)
        voxel_tensor = torch.tensor(voxel).unsqueeze(0)  # shape: [1, D, H, W]

        return voxel_tensor
