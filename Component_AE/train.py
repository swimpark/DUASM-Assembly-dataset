"""
Component-level voxel autoencoder training script.

- Continues training from existing encoder/decoder checkpoints if available.
- Oversamples low-occupancy voxel models to improve reconstruction quality.
- Logs training loss to TensorBoard and saves loss history to CSV.
"""

import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import binvox_rw

from autoencoder_model import Encoder, Decoder
from dataset_loader import VoxelDataset
import options


# -------------------------------------------------------------------------
# Configuration (update paths for your environment if needed)
# -------------------------------------------------------------------------

opt = options.args_parser()
writer = SummaryWriter()

# Base directory of this script: JMS_Code/Component_AE
COMPONENT_AE_DIR = Path(__file__).resolve().parent
# Project root assumed one level above JMS_Code
PROJECT_ROOT = COMPONENT_AE_DIR.parents[1]

# Voxel dataset directory:
#   PROJECT_ROOT/Dataset_Autodesk/.../1_binvox/train
VOXEL_DIR = (
    PROJECT_ROOT
    / "Dataset_Autodesk"
    / "data_preparation"
    / "Classified_dataset"
    / "3_6types_Dataset"
    / "stl_vox"
    / "(Actual_Dataset)Merge_6types"
    / "1_binvox"
    / "train"
)

# Output directory for checkpoints, logs, and CSV files.
# Adjust version folder name if you use a different experiment.
# Output directory and assembly edge CSV locations
RESULT_DIR = COMPONENT_AE_DIR / "trained_model"

csv_path = RESULT_DIR / "train_dataset.csv"
low_occ_csv_path = RESULT_DIR / "low_occupancy_files.csv"
loss_csv_path = RESULT_DIR / "loss_records.csv"
occupancy_stats_path = RESULT_DIR / "occupancy_stats.csv"

encoder_save_path_template = RESULT_DIR / "encoder_{}.pth"
decoder_save_path_template = RESULT_DIR / "decoder_{}.pth"

# Pretrained checkpoints to continue from (if available)
encoder_path = RESULT_DIR / "encoder_700.pth"
decoder_path = RESULT_DIR / "decoder_700.pth"

save_interval = 50  # epochs


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def read_binvox(file_path: Path) -> np.ndarray:
    with open(file_path, "rb") as f:
        model = binvox_rw.read_as_3d_array(f)
    return model.data


def calculate_occupancy_stats(voxel_data: np.ndarray) -> Tuple[int, float]:
    total_voxels = voxel_data.size
    occupied_voxels = int(np.sum(voxel_data))
    normalized_ratio = occupied_voxels / total_voxels
    return occupied_voxels, normalized_ratio


def identify_low_occupancy_files(
    file_paths: List[Path], threshold: float = 0.05
) -> List[str]:
    """
    Compute occupancy ratios and return file names below the given threshold.
    Also saves per-file occupancy statistics to CSV.
    """
    occupancy_data = []
    low_occupancy_files: List[str] = []

    for file_path in tqdm(file_paths, desc="Calculating occupancy stats"):
        voxel_data = read_binvox(file_path)
        _, occupancy_ratio = calculate_occupancy_stats(voxel_data)
        file_name = file_path.name
        occupancy_data.append((file_name, occupancy_ratio))

        if occupancy_ratio <= threshold:
            low_occupancy_files.append(file_name)

    occupancy_df = pd.DataFrame(
        occupancy_data, columns=["filename", "occupancy_ratio"]
    )
    occupancy_df.to_csv(occupancy_stats_path, index=False)

    if low_occupancy_files:
        print(f"Found {len(low_occupancy_files)} low-occupancy files (<= {threshold}).")
    else:
        print("No low-occupancy files found.")

    return low_occupancy_files


# -------------------------------------------------------------------------
# Dataset preparation
# -------------------------------------------------------------------------

ensure_dir(RESULT_DIR)

# Create CSV listing all voxel files if it does not exist
if not csv_path.exists():
    file_list = [p.name for p in VOXEL_DIR.glob("*.binvox")]
    df = pd.DataFrame(file_list, columns=["filename"])
    df.to_csv(csv_path, index=False)
    print(f"Saved train file list to {csv_path}.")

# Load train file list
df = pd.read_csv(csv_path)
train_files = [VOXEL_DIR / fname for fname in df["filename"]]

# Identify low-occupancy files and oversample them
low_occupancy_files = identify_low_occupancy_files(train_files, threshold=0.05)

low_occupancy_dataset = None
if low_occupancy_files:
    # Repeat low-occupancy samples 3 times
    repeated_files = low_occupancy_files * 3
    low_occ_df = pd.DataFrame(repeated_files, columns=["filename"])
    low_occ_df.to_csv(low_occ_csv_path, index=False)
    print(f"Saved repeated low-occupancy file list to {low_occ_csv_path}.")

    low_occupancy_dataset = VoxelDataset(
        csv_file=str(low_occ_csv_path),
        voxel_dir=str(VOXEL_DIR),
    )

# Base train dataset
train_dataset = VoxelDataset(
    csv_file=str(csv_path),
    voxel_dir=str(VOXEL_DIR),
)

# Combine base dataset and oversampled subset
if low_occupancy_dataset is not None:
    full_dataset = ConcatDataset([train_dataset, low_occupancy_dataset])
else:
    full_dataset = train_dataset

train_loader = DataLoader(full_dataset, batch_size=1, shuffle=True)


# -------------------------------------------------------------------------
# Model, optimizer, and checkpoint loading
# -------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

encoder = Encoder().to(device)
decoder = Decoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=opt.lr,
    betas=opt.betas,
)

start_epoch = 0

if encoder_path.exists() and decoder_path.exists():
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    print(f"Loaded pretrained encoder from {encoder_path}")
    print(f"Loaded pretrained decoder from {decoder_path}")
else:
    print("No pretrained weights found. Starting from scratch.")


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

num_epochs = opt.epochs
iterations: List[int] = []
train_losses: List[float] = []

try:
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        encoder.train()
        decoder.train()

        with tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}]",
            unit="batch",
        ) as pbar:
            for batch in pbar:
                batch = batch.to(device).float()

                latent = encoder(batch)
                output = decoder(latent)
                loss = criterion(output, batch)

                if torch.isnan(loss):
                    tqdm.write(
                        f"Warning: NaN detected in loss at epoch {epoch + 1}, "
                        f"iteration {len(iterations) + 1}"
                    )
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iterations.append(len(iterations) + 1)
                train_losses.append(loss.item())

                writer.add_scalar("Train/Loss", loss.item(), iterations[-1])
                pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_losses[-1]:.6f}, Time: {epoch_time:.2f}s"
        )

        # Save encoder and decoder weights
        torch.save(
            encoder.state_dict(),
            encoder_save_path_template.format(epoch + 1),
        )
        torch.save(
            decoder.state_dict(),
            decoder_save_path_template.format(epoch + 1),
        )

        # Save loss records periodically
        if (epoch + 1) % save_interval == 0:
            loss_df = pd.DataFrame(
                {"Iteration": iterations, "Train Loss": train_losses}
            )
            loss_df.to_csv(loss_csv_path, index=False)
            print(f"Saved loss records to {loss_csv_path} (up to epoch {epoch + 1}).")

except Exception as e:
    print(f"An error occurred during training: {e}")

finally:
    writer.close()
    print("Training complete.")
