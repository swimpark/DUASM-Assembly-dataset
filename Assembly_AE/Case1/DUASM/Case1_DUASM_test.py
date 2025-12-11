"""
Assembly-level embedding evaluation script (DUASM, Case 1).

File location:
    JMS_Code/Assembly_AE/Case1/DUASM/Case1_DUASM_test.py

Assumed layout:

    JMS_Code/
      Assembly_AE/
        DUASM_model.py
        DUASM_config.py
        Case1/
          CAE_train_extracted_embeddings.csv
          CAE_test_extracted_embeddings.csv
          train_assembly_edges.csv
          test_assembly_edges.csv
          train_assembly_metrics.csv
          test_assembly_metrics.csv
          DUASM/
            Case1_DUASM_test.py
            trained_model/
              GAE_model_epoch_300.pth

This script:
  1) Loads a trained DUASM GAE model.
  2) Builds graph data with assembly-level metrics.
  3) Generates assembly embeddings (mean-pooled node embeddings).
  4) Performs threshold-based nearest-neighbor classification.
  5) Prints a classification report and plots a confusion matrix.
"""

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent          # .../Case1/DUASM
CASE_DIR = CURRENT_DIR.parent                          # .../Case1
ASSEMBLY_AE_DIR = CASE_DIR.parent                      # .../Assembly_AE
MODEL_DIR = CURRENT_DIR / "trained_model"              # .../Case1/DUASM/trained_model

if str(ASSEMBLY_AE_DIR) not in sys.path:
    sys.path.append(str(ASSEMBLY_AE_DIR))

from DUASM_model import (                              # type: ignore
    GCNEncoder,
    Decoder,
    GAE,
    AssemblyMetricsEmbedding,
)
from DUASM_config import in_channels, latent_dim       # type: ignore

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Must match the training script
k_metrics = 10.0
scaler = 1.0

CLASS_NAMES = ["Supplier 1", "Supplier 2", "Supplier 3"]


# -------------------------------------------------------------------------
# Data loading helpers
# -------------------------------------------------------------------------

def load_all_data_with_metrics(
    embedding_file: Path,
    edge_file: Path,
    metrics_file: Path,
) -> List[Data]:
    """
    Load node features, edge indices, and assembly-level metrics into
    PyG Data objects. Supplier labels are stored as 0-based indices.
    """
    embedding_df = pd.read_csv(embedding_file)
    edge_df = pd.read_csv(edge_file)
    metrics_df = pd.read_csv(metrics_file)

    graphs: List[Data] = []

    for _, row in metrics_df.iterrows():
        assembly_id = row["Assembly_ID"]
        supplier = int(row["Supplier"])

        # Node embeddings for this assembly
        node_series = embedding_df[embedding_df["Assembly_ID"] == assembly_id]["Vector"]
        if node_series.empty:
            continue
        node_features = node_series.apply(eval).tolist()

        # Edge index for this assembly
        edge_series = edge_df[edge_df["Assembly_ID"] == assembly_id]["Edge_Index"]
        if edge_series.empty:
            continue
        edge_index_list = edge_series.apply(eval).values[0]

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        # Assembly-level metrics (same scaling as in training)
        tolerance = torch.tensor(
            [[row["Restrictive_Tolerance"] * k_metrics * scaler]],
            dtype=torch.float,
        )
        cost = torch.tensor(
            [[row["Assembly_Cost"] * k_metrics * scaler]],
            dtype=torch.float,
        )
        time = torch.tensor(
            [[row["Assembly_Time"] * k_metrics * scaler]],
            dtype=torch.float,
        )
        quantity = torch.tensor(
            [[row["Quantity"] * scaler]],
            dtype=torch.float,
        )

        graph = Data(
            x=x,
            edge_index=edge_index,
            metrics_tolerance=tolerance,
            metrics_cost=cost,
            metrics_time=time,
            metrics_quantity=quantity,
            supplier=torch.tensor([supplier - 1], dtype=torch.long),  # 0-based
        )
        graphs.append(graph)

    return graphs


# -------------------------------------------------------------------------
# Embedding and evaluation utilities
# -------------------------------------------------------------------------

def embed_data_with_metrics(model: GAE, data_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Generate a single embedding per assembly graph by averaging node embeddings.
    """
    embeddings = []
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            tolerance = batch.metrics_tolerance.to(device)
            cost = batch.metrics_cost.to(device)
            time = batch.metrics_time.to(device)
            quantity = batch.metrics_quantity.to(device)

            # x_hat is node-level reconstruction; its latent representation
            # is what the model internally uses. Here we simply use the
            # reconstructed x_hat as an embedding proxy and average over nodes.
            x_hat, _, _ = model(x, edge_index, tolerance, cost, time, quantity)
            embeddings.append(x_hat.mean(dim=0).cpu())

    return torch.stack(embeddings)


def compute_k_neighbor_threshold(train_embeddings: torch.Tensor, k: int = 4) -> float:
    """
    Compute the average k-th nearest neighbor distance in the training embeddings.
    """
    dists = torch.cdist(train_embeddings, train_embeddings, p=2)
    kth = torch.kthvalue(dists, k, dim=1).values
    return kth.mean().item()


def evaluate_topk_euclidean_threshold(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    train_labels: List[int],
    train_assembly_ids: List[int],
    threshold: float,
):
    """
    For each query embedding, select all training samples within a given distance
    threshold. If none exist, use the closest neighbor.
    """
    dists = torch.cdist(val_embeddings, train_embeddings, p=2)

    topk_predictions = []
    topk_indices = []
    topk_assembly_ids = []
    topk_suppliers = []
    topk_distances = []

    for i in range(dists.shape[0]):
        idx = torch.where(dists[i] <= threshold)[0].cpu().numpy()
        dist_vals = dists[i][idx].cpu().numpy()

        if len(idx) == 0:
            closest_idx = torch.argmin(dists[i]).item()
            idx = np.array([closest_idx])
            dist_vals = np.array([dists[i][closest_idx].item()])

        order = np.argsort(dist_vals)
        idx = idx[order]
        dist_vals = dist_vals[order]

        topk_indices.append(idx)
        topk_predictions.append([train_labels[j] for j in idx])
        topk_assembly_ids.append([train_assembly_ids[j] for j in idx])
        topk_suppliers.append([train_labels[j] + 1 for j in idx])  # back to 1-based
        topk_distances.append(dist_vals.tolist())

    return topk_predictions, topk_indices, topk_assembly_ids, topk_suppliers, topk_distances


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Print classification report and plot normalized confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    pd.options.display.float_format = "{:.4f}".format
    print(df_report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 16},
    )
    plt.xlabel("Predicted Labels", fontsize=16)
    plt.ylabel("True Labels", fontsize=16)
    plt.title("Confusion Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# Main evaluation function
# -------------------------------------------------------------------------

def validate_embeddings_with_metrics(model_path: Path) -> None:
    """
    Full evaluation pipeline for a trained assembly-level GAE.
    """
    # File locations (all under Case1)
    train_embedding_file = CASE_DIR / "CAE_train_extracted_embeddings.csv"
    test_embedding_file = CASE_DIR / "CAE_test_extracted_embeddings.csv"
    train_edge_file = CASE_DIR / "train_assembly_edges.csv"
    test_edge_file = CASE_DIR / "test_assembly_edges.csv"
    train_metrics_file = CASE_DIR / "train_assembly_metrics.csv"
    test_metrics_file = CASE_DIR / "test_assembly_metrics.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    encoder = GCNEncoder(in_channels, latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    metrics_embed = AssemblyMetricsEmbedding(latent_dim).to(device)
    model = GAE(encoder, decoder, metrics_embed).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # Load graphs
    train_graphs = load_all_data_with_metrics(
        train_embedding_file,
        train_edge_file,
        train_metrics_file,
    )
    valid_graphs = load_all_data_with_metrics(
        test_embedding_file,
        test_edge_file,
        test_metrics_file,
    )

    # Labels
    train_labels = [g.supplier.item() for g in train_graphs]
    valid_labels = [g.supplier.item() for g in valid_graphs]

    # Assembly IDs
    train_assembly_ids = pd.read_csv(train_metrics_file)["Assembly_ID"].tolist()
    valid_assembly_ids = pd.read_csv(test_metrics_file)["Assembly_ID"].tolist()

    # DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=False)
    val_loader = DataLoader(valid_graphs, batch_size=1, shuffle=False)

    # Embeddings
    train_embeddings = embed_data_with_metrics(model, train_loader, device)
    val_embeddings = embed_data_with_metrics(model, val_loader, device)

    # Threshold based on training distribution
    threshold = compute_k_neighbor_threshold(train_embeddings, k=4)
    print(f"Distance threshold (k=4): {threshold:.6f}")

    # Nearest neighbors with threshold
    (
        topk_predictions,
        topk_indices,
        topk_assembly_ids,
        topk_suppliers,
        topk_distances,
    ) = evaluate_topk_euclidean_threshold(
        train_embeddings,
        val_embeddings,
        train_labels,
        train_assembly_ids,
        threshold,
    )

    # Single-label prediction: closest neighbor
    y_pred = [preds[0] if len(preds) > 0 else -1 for preds in topk_predictions]

    print("\nClassification report:")
    print(classification_report(valid_labels, y_pred, target_names=CLASS_NAMES))

    plot_confusion_matrix(valid_labels, y_pred, class_names=CLASS_NAMES)

    # Detailed neighbor list
    print("\nQuery results with neighbor assemblies and distances:")
    for q_idx, indices in enumerate(topk_indices):
        print(f"Query Assembly ID: {valid_assembly_ids[q_idx]}")
        print(f"  True Supplier: Supplier {valid_labels[q_idx] + 1}")
        for rank, _ in enumerate(indices, start=1):
            print(f"  Rank {rank}:")
            print(f"    Predicted Assembly ID: {topk_assembly_ids[q_idx][rank - 1]}")
            print(f"    Predicted Supplier: Supplier {topk_suppliers[q_idx][rank - 1]}")
        print()


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    model_path = MODEL_DIR / "GAE_model_epoch_300.pth"
    validate_embeddings_with_metrics(model_path)
