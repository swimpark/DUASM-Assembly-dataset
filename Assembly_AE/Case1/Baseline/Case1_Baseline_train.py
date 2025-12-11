"""
Case 1 – baseline GCN classifier training script.

Location:
    Assembly_AE/Case1/Baseline/Case1_Baseline_train.py

Assumed layout:

    Assembly_AE/
      Baseline_config.py
      Baseline_model.py
      Case1/
        CAE_train_extracted_embeddings.csv
        train_assembly_edges.csv
        train_assembly_metrics.csv
        Baseline/
          Case1_Baseline_train.py
          Case1_Baseline_test.py
          trained_model/
            Classifier_model_epoch_0.pth (optional, for continued training)

This script:
  1) Builds assembly-level graphs with metrics for Case 1.
  2) Trains a baseline GCN classifier with metrics.
  3) Optionally continues training from an existing checkpoint.
  4) Saves intermediate checkpoints and a CSV of training losses.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

# -------------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------------

BASELINE_DIR = Path(__file__).resolve().parent          # .../Case1/Baseline
CASE_DIR = BASELINE_DIR.parent                          # .../Case1
ASSEMBLY_AE_DIR = CASE_DIR.parent                       # .../Assembly_AE
MODEL_DIR = BASELINE_DIR / "trained_model"              # .../Case1/Baseline/trained_model

if str(ASSEMBLY_AE_DIR) not in sys.path:
    sys.path.append(str(ASSEMBLY_AE_DIR))

from Baseline_model import GCNClassifierWithMetrics      # type: ignore
from Baseline_config import (                            # type: ignore
    in_channels,
    latent_dim,
    epochs,
    learning_rate,
)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

MODEL_INIT_NAME = "Classifier_model_epoch_0.pth"  # used only if it exists
SAVE_INTERVAL = 50

# metric scaling (must match evaluation script)
K_METRICS = 10.0
SCALER = 0.5


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

def load_embedding_data(
    embedding_file: Path,
    edge_file: Path,
    metrics_file: Path,
) -> Tuple[List[Data], List[Dict[str, torch.Tensor]], List[int]]:
    """
    Load assembly-level graphs, metrics, and supplier labels.

    Each graph has:
      - x: node features (component embeddings)
      - edge_index: connectivity
      - metrics_*: assembly-level metrics
      - y: supplier label (0-based)
    """
    embedding_df = pd.read_csv(embedding_file)
    edge_df = pd.read_csv(edge_file)
    metrics_df = pd.read_csv(metrics_file)

    graphs: List[Data] = []
    metrics: List[Dict[str, torch.Tensor]] = []
    labels: List[int] = []

    for _, metric_row in metrics_df.iterrows():
        assembly_id = metric_row["Assembly_ID"]

        tolerance = torch.tensor(
            [[metric_row["Restrictive_Tolerance"] * K_METRICS * SCALER]],
            dtype=torch.float,
        )
        cost = torch.tensor(
            [[metric_row["Assembly_Cost"] * K_METRICS * SCALER]],
            dtype=torch.float,
        )
        time = torch.tensor(
            [[metric_row["Assembly_Time"] * K_METRICS * SCALER]],
            dtype=torch.float,
        )
        quantity = torch.tensor(
            [[metric_row["Quantity"] * SCALER]],
            dtype=torch.float,
        )

        supplier = int(metric_row["Supplier"]) - 1  # 0-based

        node_series = embedding_df[embedding_df["Assembly_ID"] == assembly_id]["Vector"]
        if node_series.empty:
            print(f"Warning: missing node features for Assembly_ID {assembly_id}")
            continue
        node_features = node_series.apply(eval).tolist()

        edge_series = edge_df[edge_df["Assembly_ID"] == assembly_id]["Edge_Index"]
        if edge_series.empty:
            print(f"Warning: missing edge data for Assembly_ID {assembly_id}")
            continue
        edge_index_list = edge_series.apply(eval).values[0]

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        graph = Data(
            x=x,
            edge_index=edge_index,
            metrics_tolerance=tolerance,
            metrics_cost=cost,
            metrics_time=time,
            metrics_quantity=quantity,
            y=torch.tensor([supplier], dtype=torch.long),
        )

        graphs.append(graph)
        metrics.append(
            {
                "tolerance": tolerance,
                "cost": cost,
                "time": time,
                "quantity": quantity,
            }
        )
        labels.append(supplier)

    return graphs, metrics, labels


# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------

def train_assembly_model(
    graphs: List[Data],
    metrics: List[Dict[str, torch.Tensor]],
    labels: List[int],
    save_dir: Path,
    save_interval: int = SAVE_INTERVAL,
    pretrained_model_path: Path | None = None,
) -> None:
    """
    Train GCN classifier with assembly-level metrics.

    If pretrained_model_path is provided and exists, training continues
    from that checkpoint. Otherwise the model is initialized from scratch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNClassifierWithMetrics(in_channels, latent_dim, num_classes=3).to(device)

    start_epoch = 0
    if pretrained_model_path is not None and pretrained_model_path.exists():
        print(f"Loading pretrained model from {pretrained_model_path}")
        state = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(state)
        try:
            start_epoch = int(pretrained_model_path.stem.split("_")[-1])
        except ValueError:
            start_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_losses: List[Tuple[int, float]] = []

    model.train()
    num_samples = len(graphs)

    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0.0

        with tqdm(
            zip(graphs, metrics, labels),
            desc=f"Epoch {epoch + 1}/{start_epoch + epochs}",
            unit="batch",
            total=num_samples,
        ) as pbar:
            for graph, metric, label in pbar:
                optimizer.zero_grad()

                x = graph.x.to(device)
                edge_index = graph.edge_index.to(device)

                tolerance = metric["tolerance"].to(device)
                cost = metric["cost"].to(device)
                time = metric["time"].to(device)
                quantity = metric["quantity"].to(device)

                node_dummy = torch.zeros(x.size(0), dtype=torch.long, device=device)

                logits = model(
                    x,
                    edge_index,
                    node_dummy,
                    tolerance,
                    cost,
                    time,
                    quantity,
                )

                target = torch.tensor(label, dtype=torch.long, device=device).unsqueeze(0)
                loss = criterion(logits, target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_samples
        epoch_losses.append((epoch + 1, avg_loss))
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            ckpt_path = save_dir / f"Classifier_model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved at {ckpt_path}")

    final_path = save_dir / "Classifier_model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved at {final_path}")

    loss_csv_path = save_dir / "training_losses.csv"
    with loss_csv_path.open(mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Average Loss"])
        writer.writerows(epoch_losses)
    print(f"Training losses saved to {loss_csv_path}")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_embedding_file = CASE_DIR / "CAE_train_extracted_embeddings.csv"
    train_edge_file = CASE_DIR / "train_assembly_edges.csv"
    train_metrics_file = CASE_DIR / "train_assembly_metrics.csv"

    graphs, metrics, labels = load_embedding_data(
        train_embedding_file,
        train_edge_file,
        train_metrics_file,
    )

    init_model_path = MODEL_DIR / MODEL_INIT_NAME
    pretrained = init_model_path if init_model_path.exists() else None

    train_assembly_model(
        graphs,
        metrics,
        labels,
        save_dir=MODEL_DIR,
        pretrained_model_path=pretrained,
    )
