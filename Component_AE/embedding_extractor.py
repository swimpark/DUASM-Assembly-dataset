"""
Extract sub-part embeddings from a pre-trained component-level autoencoder
and save them as a CSV file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import binvox_rw
from autoencoder_model import Encoder


class EmbeddingExtractor:
    def __init__(self, model_path, voxel_dir, csv_save_path, assembly_csv_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.voxel_dir = Path(voxel_dir)
        self.csv_save_path = Path(csv_save_path)
        self.assembly_csv_path = Path(assembly_csv_path)

        self.encoder = Encoder().to(self.device)
        self.encoder.load_state_dict(
            torch.load(self._to_str(model_path), map_location=self.device)
        )
        self.encoder.eval()

        self.assembly_df = pd.read_csv(self.assembly_csv_path)

    @staticmethod
    def _to_str(path_like: Path | str) -> str:
        return str(path_like)

    def parse_sub_parts(self):
        """Return list of (Assembly_ID, sub_part_filename) pairs."""
        sub_parts_data = []
        for _, row in self.assembly_df.iterrows():
            assembly_id = row["Assembly_ID"]
            sub_parts = row["Sub_parts"].split(", ")
            for sub_part in sub_parts:
                sub_part = sub_part.strip()
                if sub_part:
                    sub_parts_data.append((assembly_id, sub_part))
        return sub_parts_data

    def load_voxel_data(self, filepath: Path) -> torch.Tensor:
        if not filepath.exists():
            raise FileNotFoundError(f"Voxel data file not found: {filepath}")

        try:
            with open(filepath, "rb") as f:
                voxel_model = binvox_rw.read_as_3d_array(f)
            voxel_data = torch.from_numpy(voxel_model.data.astype(np.float32))
            return voxel_data
        except Exception as e:
            raise RuntimeError(f"Error loading voxel data from {filepath}: {e}")

    def extract_embeddings_and_save(self):
        """
        Extract embeddings for each sub-part and save to CSV.

        Output columns:
            - Assembly_ID
            - Sub_part
            - Vector (flattened latent vector as Python list)
        """
        sub_parts_data = self.parse_sub_parts()
        embedding_records = []

        with torch.no_grad():
            for assembly_id, sub_part in sub_parts_data:
                voxel_path = self.voxel_dir / sub_part
                voxel_data = self.load_voxel_data(voxel_path).to(self.device).float()

                if len(voxel_data.shape) == 3:
                    voxel_data = voxel_data.unsqueeze(0).unsqueeze(0)

                embedding = self.encoder(voxel_data)
                embedding_vector = embedding.cpu().numpy().flatten().tolist()

                embedding_records.append(
                    {
                        "Assembly_ID": assembly_id,
                        "Sub_part": sub_part,
                        "Vector": embedding_vector,
                    }
                )

        embedding_df = pd.DataFrame(embedding_records)
        self.csv_save_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_df.to_csv(self.csv_save_path, index=False)
        print(f"Embeddings saved to {self.csv_save_path}")


if __name__ == "__main__":
    # Base directory of this script: JMS_Code/Component_AE
    COMPONENT_AE_DIR = Path(__file__).resolve().parent
    # Project root assumed one level above JMS_Code
    PROJECT_ROOT = COMPONENT_AE_DIR.parents[1]

    # Encoder checkpoint (update filename if a different checkpoint is used)
    MODEL_PATH = COMPONENT_AE_DIR / "trained_model" / "encoder_700.pth"

    # Voxel dataset root (update to user's path)
    VOXEL_BASE_DIR = (
        PROJECT_ROOT
        / "Dataset_Autodesk"
        / "data_preparation"
        / "Classified_dataset"
        / "3_6types_Dataset"
        / "stl_vox"
        / "(Actual_Dataset)Merge_6types"
        / "1_binvox"
    )
    TRAIN_VOXEL_DIR = VOXEL_BASE_DIR / "train"
    TEST_VOXEL_DIR = VOXEL_BASE_DIR / "test"

    # Output directory and assembly edge CSV locations
    SAVE_DIR = COMPONENT_AE_DIR / "trained_model"

    train_embed_save_path = SAVE_DIR / "CAE_train_extracted_embeddings.csv"
    train_assembly_csv_path = SAVE_DIR / "train_assembly_edges.csv"

    extractor = EmbeddingExtractor(
        model_path=MODEL_PATH,
        voxel_dir=TRAIN_VOXEL_DIR,
        csv_save_path=train_embed_save_path,
        assembly_csv_path=train_assembly_csv_path,
    )
    extractor.extract_embeddings_and_save()

    test_embed_save_path = SAVE_DIR / "CAE_test_extracted_embeddings.csv"
    test_assembly_csv_path = SAVE_DIR / "test_assembly_edges.csv"

    extractor = EmbeddingExtractor(
        model_path=MODEL_PATH,
        voxel_dir=TEST_VOXEL_DIR,
        csv_save_path=test_embed_save_path,
        assembly_csv_path=test_assembly_csv_path,
    )
    extractor.extract_embeddings_and_save()
