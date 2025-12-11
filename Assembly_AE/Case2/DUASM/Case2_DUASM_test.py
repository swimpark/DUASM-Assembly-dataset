"""
Case 2 – DUASM embedding evaluation script.

Location:
    Assembly_AE/Case2/DUASM/Case2_DUASM_test.py

Assumed layout:

    Assembly_AE/
      DUASM_config.py
      DUASM_model.py
      Case2/
        DUASM/
          trained_model/
            GAE_model_epoch_600.pth
            train_assembly_edges.csv
            train_assembly_metrics.csv
            test_assembly_metrics_cost.csv
            test_assembly_metrics_cost_qty.csv
            test_assembly_metrics_only_tol.csv
            test_assembly_metrics_origin.csv
            test_assembly_metrics_qty.csv
            test_assembly_metrics_time.csv
            test_assembly_metrics_time_qty.csv
          Case2_DUASM_train.py
          Case2_DUASM_test.py
          CAE_train_extracted_embeddings.csv
          CAE_test_extracted_embeddings.csv
          test_assembly_edges.csv
          test_assembly_metrics.csv

This script:
  1) Loads a trained DUASM GAE model and Case 2 data.
  2) Builds assembly-level graphs with manufacturing metrics.
  3) Generates embeddings and evaluates supplier prediction with
     cosine similarity and thresholding.
  4) Prints CSV-style classification results and plots confusion
     matrices and ROC-style curves.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Avoid name collision with sklearn.classification_report
from sklearn.metrics import classification_report as sk_clf_report
from sklearn.metrics import multilabel_confusion_matrix

# -------------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------------

DUASM_DIR = Path(__file__).resolve().parent                  # .../Case2/DUASM
CASE_DIR = DUASM_DIR.parent                                  # .../Case2
ASSEMBLY_AE_DIR = CASE_DIR.parent                            # .../Assembly_AE
MODEL_DIR = DUASM_DIR / "trained_model"                      # .../Case2/DUASM/trained_model

if str(ASSEMBLY_AE_DIR) not in sys.path:
    sys.path.append(str(ASSEMBLY_AE_DIR))

from DUASM_model import GCNEncoder, Decoder, GAE, AssemblyMetricsEmbedding  # type: ignore
from DUASM_config import in_channels, latent_dim                             # type: ignore


# -------------------------------------------------------------------------
# Global configuration
# -------------------------------------------------------------------------

k_metrics = 10.0            # Metric scale factor (must match training)
num_suppliers = 3
k = 1                       # k value for k-NN based threshold

# Metric weighting factors for tolerance, time, cost, and quantity.
# Set a scale to zero to effectively mask that metric.
scaler_tol = 20.0
scaler_time = 20.0
scaler_cost = 20.0
scaler_qty = 0.0


# -------------------------------------------------------------------------
# Data loading and embedding utilities
# -------------------------------------------------------------------------

def load_all_data_with_metrics(
    embedding_file: Path,
    edge_file: Path,
    metrics_file: Path,
) -> List[Data]:
    """
    Build assembly-level graph objects with metrics and multi-label suppliers.

    Each graph contains:
      - x: node features (component embeddings)
      - edge_index: assembly connectivity
      - metrics_tolerance, metrics_cost, metrics_time, metrics_quantity
      - supplier: multi-hot supplier vector (length = num_suppliers)
      - assembly_id: Assembly_ID identifier
    """
    embedding_df = pd.read_csv(embedding_file)
    edge_df = pd.read_csv(edge_file)
    metrics_df = pd.read_csv(metrics_file)

    graphs: List[Data] = []

    for _, row in metrics_df.iterrows():
        assembly_id = row["Assembly_ID"]

        # Supplier list is stored as a comma-separated string such as "1,2"
        supplier_str = str(row["Supplier"])
        supplier_list = list(map(int, supplier_str.split(",")))
        supplier_vector = torch.zeros(num_suppliers, dtype=torch.float)
        for s in supplier_list:
            supplier_vector[s - 1] = 1

        node_series = embedding_df[embedding_df["Assembly_ID"] == assembly_id]["Vector"]
        node_features = node_series.apply(eval).tolist()

        edge_data = edge_df[edge_df["Assembly_ID"] == assembly_id]
        if edge_data.empty:
            print(f"Warning: no edge data for Assembly_ID {assembly_id}")
            continue

        edge_index_list = edge_data["Edge_Index"].apply(eval).values[0]
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        tolerance = torch.tensor(
            [[row["Restrictive_Tolerance"] * k_metrics * scaler_tol]],
            dtype=torch.float,
        )
        time = torch.tensor(
            [[row["Assembly_Time"] * k_metrics * scaler_time]],
            dtype=torch.float,
        )
        cost = torch.tensor(
            [[row["Assembly_Cost"] * k_metrics * scaler_cost]],
            dtype=torch.float,
        )
        quantity = torch.tensor(
            [[row["Quantity"] * scaler_qty]],
            dtype=torch.float,
        )

        graph = Data(
            x=x,
            edge_index=edge_index,
            metrics_tolerance=tolerance,
            metrics_cost=cost,
            metrics_time=time,
            metrics_quantity=quantity,
            supplier=supplier_vector,
        )
        graph.assembly_id = assembly_id
        graphs.append(graph)

    return graphs


def embed_data_with_metrics(
    model: GAE,
    data_loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract a single embedding per assembly graph by averaging node embeddings.
    """
    embeddings: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            tolerance = batch.metrics_tolerance.to(device)
            cost = batch.metrics_cost.to(device)
            time = batch.metrics_time.to(device)
            quantity = batch.metrics_quantity.to(device)

            x_hat, _, _ = model(x, edge_index, tolerance, cost, time, quantity)
            embeddings.append(x_hat.mean(dim=0).cpu())

    return torch.stack(embeddings)


# -------------------------------------------------------------------------
# Cosine similarity and thresholding
# -------------------------------------------------------------------------

def compute_k_neighbor_threshold_cosine(
    train_embeddings: torch.Tensor,
    k_neighbors: int = k,
) -> float:
    """
    Compute a global cosine-similarity threshold based on k-th nearest neighbors.

    Steps:
      1) L2-normalize all train embeddings.
      2) For each embedding, compute cosine similarity with all others.
      3) Record the k-th highest similarity (excluding self).
      4) Use the average of these values as a global threshold.
    """
    train_norm = torch.nn.functional.normalize(train_embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(train_norm, train_norm.t())
    similarity_matrix[torch.eye(similarity_matrix.size(0), dtype=torch.bool)] = -float("inf")

    kth_similarities = []
    for i in range(similarity_matrix.size(0)):
        kth_value = torch.topk(similarity_matrix[i], k_neighbors, largest=True).values[-1].item()
        kth_similarities.append(kth_value)

    threshold = float(np.mean(kth_similarities))
    return threshold


def evaluate_topk_cosine_threshold(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    train_labels: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict suppliers for validation embeddings by union over neighbors above threshold.

    For each validation sample:
      1) Find training samples whose cosine similarity is >= threshold.
      2) If none exist, fall back to the single most similar training sample.
      3) Aggregate labels with logical OR for a multi-label prediction.
      4) Also compute a per-supplier fraction score for analysis.
    """
    train_norm = torch.nn.functional.normalize(train_embeddings, p=2, dim=1)
    val_norm = torch.nn.functional.normalize(val_embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(val_norm, train_norm.t())

    y_pred_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []

    for i in range(sim_matrix.size(0)):
        indices = torch.where(sim_matrix[i] >= threshold)[0].cpu().numpy()
        if len(indices) == 0:
            indices = torch.topk(sim_matrix[i], 1).indices.cpu().numpy()

        close_vectors = train_labels[indices]
        union_vector = np.any(close_vectors, axis=0).astype(int)
        fraction_vector = np.mean(close_vectors, axis=0)

        y_pred_list.append(union_vector)
        y_score_list.append(fraction_vector)

    y_pred = np.stack(y_pred_list, axis=0)
    y_score = np.stack(y_score_list, axis=0)

    return y_pred, y_score


def evaluate_knn_threshold_per_supplier_fpr_cosine(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    threshold: float,
) -> Tuple[list[float], list[float]]:
    """
    Compute TPR and FPR per supplier using a cosine-similarity threshold.

    For each validation sample, neighbors are training samples whose
    similarity exceeds the threshold. Their labels are OR-aggregated.
    """
    train_norm = torch.nn.functional.normalize(train_embeddings, p=2, dim=1)
    val_norm = torch.nn.functional.normalize(val_embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(val_norm, train_norm.t())

    y_pred_list: list[np.ndarray] = []

    for i in range(sim_matrix.size(0)):
        indices = torch.where(sim_matrix[i] >= threshold)[0].cpu().numpy()
        if len(indices) == 0:
            y_pred_list.append(np.zeros(num_suppliers, dtype=int))
        else:
            close_vectors = train_labels[indices]
            pred_vector = np.any(close_vectors, axis=0).astype(int)
            y_pred_list.append(pred_vector)

    y_pred = np.stack(y_pred_list, axis=0)

    TPR: list[float] = []
    FPR: list[float] = []

    for i in range(val_labels.shape[1]):
        TP = np.sum((y_pred[:, i] == 1) & (val_labels[:, i] == 1))
        FN = np.sum((y_pred[:, i] == 0) & (val_labels[:, i] == 1))
        FP = np.sum((y_pred[:, i] == 1) & (val_labels[:, i] == 0))
        TN = np.sum((y_pred[:, i] == 0) & (val_labels[:, i] == 0))

        tpr = TP / (TP + FN + 1e-9)
        fpr = FP / (FP + TN + 1e-9)

        TPR.append(float(tpr))
        FPR.append(float(fpr))

    return TPR, FPR


def print_threshold_suppliers_per_query_cosine(
    val_embeddings: torch.Tensor,
    train_embeddings: torch.Tensor,
    train_labels: np.ndarray,
    valid_graphs: list[Data],
    train_graphs: list[Data],
    threshold: float,
    model: GAE,
) -> None:
    """
    For each query assembly, print neighbor suppliers and metric reconstruction
    results for both query and neighbors.
    """
    train_norm = torch.nn.functional.normalize(train_embeddings, p=2, dim=1)
    val_norm = torch.nn.functional.normalize(val_embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(val_norm, train_norm.t()).cpu().numpy()

    device = next(model.parameters()).device

    for i in range(sim_matrix.shape[0]):
        query_assembly_id = valid_graphs[i].assembly_id
        query_supplier = valid_graphs[i].supplier.numpy()
        query_suppliers = np.where(query_supplier == 1)[0] + 1
        query_suppliers_str = ",".join(map(str, query_suppliers))

        print(f"\n=== Query Assembly ID: {query_assembly_id} | Query Supplier(s): {query_suppliers_str} ===")

        query_tol = valid_graphs[i].metrics_tolerance.item()
        query_time = valid_graphs[i].metrics_time.item()
        query_cost = valid_graphs[i].metrics_cost.item()
        query_qty = valid_graphs[i].metrics_quantity.item()

        query_x = valid_graphs[i].x.to(device)
        query_edge_index = valid_graphs[i].edge_index.to(device)
        q_tol = valid_graphs[i].metrics_tolerance.to(device)
        q_cost = valid_graphs[i].metrics_cost.to(device)
        q_time = valid_graphs[i].metrics_time.to(device)
        q_qty = valid_graphs[i].metrics_quantity.to(device)

        with torch.no_grad():
            _, _, (query_tol_rec, query_cost_rec, query_time_rec, query_qty_rec) = model(
                query_x,
                query_edge_index,
                q_tol,
                q_cost,
                q_time,
                q_qty,
            )

        query_tol_rec = query_tol_rec.item()
        query_cost_rec = query_cost_rec.item()
        query_time_rec = query_time_rec.item()
        query_qty_rec = query_qty_rec.item()

        query_diff_tol = abs(query_tol - query_tol_rec)
        query_diff_time = abs(query_time - query_time_rec)
        query_diff_cost = abs(query_cost - query_cost_rec)
        query_diff_qty = abs(query_qty - query_qty_rec)

        print(
            f"Tol:{query_tol:.3f}/{query_tol_rec:.3f} (Δ {query_diff_tol:.3f}) | "
            f"Time:{query_time:.3f}/{query_time_rec:.3f} (Δ {query_diff_time:.3f}) | "
            f"Cost:{query_cost:.3f}/{query_cost_rec:.3f} (Δ {query_diff_cost:.3f}) | "
            f"Qty:{query_qty:.3f}/{query_qty_rec:.3f} (Δ {query_diff_qty:.3f})"
        )

        sim_i = sim_matrix[i]
        indices = np.where(sim_i >= threshold)[0]
        if len(indices) == 0:
            indices = np.array([np.argmax(sim_i)])
            print("No neighbors above the threshold; using the most similar neighbor instead.")

        sorted_indices = indices[np.argsort(-sim_i[indices])]

        print(
            "Rank | Neighbor Assembly ID | Neighbor Supplier(s) | Cosine Similarity | "
            "Input (Tol/Time/Cost/Qty) | Reconstructed (Tol/Time/Cost/Qty) | Diff"
        )

        for rank, idx in enumerate(sorted_indices, start=1):
            neighbor_assembly_id = train_graphs[idx].assembly_id
            neighbor_supplier = train_graphs[idx].supplier.numpy()
            neighbor_suppliers = np.where(neighbor_supplier == 1)[0] + 1
            suppliers_str = ",".join(map(str, neighbor_suppliers))
            neighbor_sim = sim_i[idx]

            db_tol = train_graphs[idx].metrics_tolerance.item()
            db_time = train_graphs[idx].metrics_time.item()
            db_cost = train_graphs[idx].metrics_cost.item()
            db_qty = train_graphs[idx].metrics_quantity.item()

            x = train_graphs[idx].x.to(device)
            edge_index = train_graphs[idx].edge_index.to(device)
            tol = train_graphs[idx].metrics_tolerance.to(device)
            cost = train_graphs[idx].metrics_cost.to(device)
            time_metric = train_graphs[idx].metrics_time.to(device)
            qty = train_graphs[idx].metrics_quantity.to(device)

            with torch.no_grad():
                _, _, (tol_rec, cost_rec, time_rec, qty_rec) = model(
                    x,
                    edge_index,
                    tol,
                    cost,
                    time_metric,
                    qty,
                )

            tol_rec = tol_rec.item()
            cost_rec = cost_rec.item()
            time_rec = time_rec.item()
            qty_rec = qty_rec.item()

            diff_tol = abs(db_tol - tol_rec)
            diff_time = abs(db_time - time_rec)
            diff_cost = abs(db_cost - cost_rec)
            diff_qty = abs(db_qty - qty_rec)

            print(
                f"{rank:4d} | {str(neighbor_assembly_id):20s} | {suppliers_str:20s} | "
                f"{neighbor_sim:.4f} | "
                f"{db_tol:.3f}/{tol_rec:.3f} (Δ {diff_tol:.3f}) | "
                f"{db_time:.3f}/{time_rec:.3f} (Δ {diff_time:.3f}) | "
                f"{db_cost:.3f}/{cost_rec:.3f} (Δ {diff_cost:.3f}) | "
                f"{db_qty:.3f}/{qty_rec:.3f} (Δ {diff_qty:.3f})"
            )


# -------------------------------------------------------------------------
# ROC-style evaluation helpers
# -------------------------------------------------------------------------

def compute_best_threshold_per_supplier_roc_by_similarity_discrete(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    class_names: list[str],
    num_points: int = 500,
) -> list[float]:
    """
    Scan cosine-similarity thresholds and find a per-supplier threshold
    that minimizes the distance to the ideal ROC point (FPR=0, TPR=1).
    """
    train_norm = torch.nn.functional.normalize(train_embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(train_norm, train_norm.t())
    sim_matrix[torch.eye(sim_matrix.size(0), dtype=torch.bool)] = -float("inf")

    min_sim = sim_matrix.max(dim=1).values.min().item()
    max_sim = sim_matrix.max().item()
    thresholds = np.linspace(min_sim, max_sim, num_points)

    best_thresholds: list[float] = []

    for supplier_idx, supplier_name in enumerate(class_names):
        best_dist = float("inf")
        best_thr = thresholds[0]

        for thr in thresholds:
            tpr_list, fpr_list = evaluate_knn_threshold_per_supplier_fpr_cosine(
                train_embeddings,
                val_embeddings,
                train_labels,
                val_labels,
                thr,
            )
            tpr = tpr_list[supplier_idx]
            fpr = fpr_list[supplier_idx]
            dist = np.sqrt((1 - tpr) ** 2 + (fpr - 0) ** 2)

            if dist < best_dist:
                best_dist = dist
                best_thr = thr

        best_thresholds.append(best_thr)
        print(f"Supplier {supplier_name} ROC-similarity optimal threshold = {best_thr:.4f}")

    return best_thresholds


def evaluate_supplier_specific_classification_cosine(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    train_labels: np.ndarray,
    supplier_thresholds: list[float],
) -> np.ndarray:
    """
    Apply supplier-specific cosine thresholds using the maximum similarity
    to positive training samples for each supplier.
    """
    train_norm = torch.nn.functional.normalize(train_embeddings, p=2, dim=1)
    val_norm = torch.nn.functional.normalize(val_embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(val_norm, train_norm.t()).cpu().numpy()

    n_val = sim_matrix.shape[0]
    y_pred = np.zeros((n_val, num_suppliers), dtype=int)

    for i in range(n_val):
        for s in range(num_suppliers):
            pos_indices = np.where(train_labels[:, s] == 1)[0]
            if len(pos_indices) > 0:
                sims = sim_matrix[i, pos_indices]
                if np.max(sims) >= supplier_thresholds[s]:
                    y_pred[i, s] = 1

    return y_pred


def compute_supplier_average_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute mean accuracy over suppliers in a multi-label setting.
    """
    accuracies: list[float] = []
    n_samples = y_true.shape[0]

    for i in range(y_true.shape[1]):
        TP = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1))
        TN = np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 0))
        accuracy = (TP + TN) / n_samples
        accuracies.append(float(accuracy))
        print(f"Supplier {i + 1} Accuracy = {accuracy * 100:.1f}")

    return float(np.mean(accuracies))


def plot_multi_label_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
) -> None:
    """
    Plot a 2x2 confusion matrix for each supplier as a separate subplot.
    """
    cm_list = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 4))

    for i, ax in enumerate(axes):
        cm = cm_list[i].astype(float)

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-9
            cm = cm / row_sums
            fmt = ".2f"
            vmin, vmax = 0, 1
        else:
            fmt = "d"
            vmin, vmax = None, None

        cm = cm[::-1, ::-1]
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            annot_kws={"size": 18},
        )
        ax.set_title(class_names[i], fontsize=18)
        ax.set_xlabel("Predicted", fontsize=16)
        ax.set_ylabel("True", fontsize=16)
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(["Positive", "Negative"], fontsize=14, va="center")
        ax.set_yticklabels(["Positive", "Negative"], fontsize=14, va="center")

    plt.tight_layout()
    plt.show()


def plot_tpr_fpr_roc_subplots_cosine(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    class_names: list[str],
) -> None:
    """
    Plot ROC-style TPR-FPR curves by sweeping cosine-similarity thresholds
    from -1 to 1 for each supplier.
    """
    train_norm = torch.nn.functional.normalize(train_embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(train_norm, train_norm.t())
    sim_matrix[torch.eye(sim_matrix.size(0), dtype=torch.bool)] = -float("inf")

    thresholds = np.linspace(-1.0, 1.0, num=500)
    print("ROC curve threshold range: [-1.0, 1.0]")

    fig, axes = plt.subplots(1, len(class_names), figsize=(4 * len(class_names), 4), squeeze=False)

    for supplier_idx, supplier_name in enumerate(class_names):
        tpr_values: list[float] = []
        fpr_values: list[float] = []

        for thr in thresholds:
            y_pred = evaluate_supplier_specific_classification_cosine(
                train_embeddings,
                val_embeddings,
                train_labels,
                [thr] * num_suppliers,
            )

            TP = np.sum((y_pred[:, supplier_idx] == 1) & (val_labels[:, supplier_idx] == 1))
            FN = np.sum((y_pred[:, supplier_idx] == 0) & (val_labels[:, supplier_idx] == 1))
            FP = np.sum((y_pred[:, supplier_idx] == 1) & (val_labels[:, supplier_idx] == 0))
            TN = np.sum((y_pred[:, supplier_idx] == 0) & (val_labels[:, supplier_idx] == 0))

            tpr = TP / (TP + FN + 1e-9)
            fpr = FP / (FP + TN + 1e-9)

            tpr_values.append(float(tpr))
            fpr_values.append(float(fpr))

        auc_value = float(np.trapz(tpr_values, fpr_values))

        ax = axes[0, supplier_idx]
        ax.plot(fpr_values, tpr_values, label=f"AUC={auc_value:.3f}")
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(supplier_name)
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# High-level loaders and evaluators
# -------------------------------------------------------------------------

def load_data_and_embeddings():
    """
    Load the trained model and Case 2 train/test data, then compute embeddings.

    User note:
      The paths below assume that all Case 2 CSV files are located in:
          Assembly_AE/Case2/  (embeddings)
          Assembly_AE/Case2/DUASM/trained_model/  (train edges/metrics, tset edges, masked test metrics)
      and the trained GAE checkpoint is located in:
          Assembly_AE/Case2/DUASM/trained_model/

      If your dataset is stored in a different location, update DUASM_DIR
      and MODEL_DIR at the top of this script.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_save_path = MODEL_DIR / "GAE_model_epoch_600.pth"

    # Embeddings
    train_embedding_file = CASE_DIR / "CAE_train_extracted_embeddings.csv"
    test_embedding_file = CASE_DIR / "CAE_test_extracted_embeddings.csv"

    # Edges
    train_edge_file = MODEL_DIR / "train_assembly_edges.csv"
    test_edge_file = MODEL_DIR / "test_assembly_edges.csv"

    # Metrics
    train_metrics_file = MODEL_DIR / "train_assembly_metrics.csv"
    test_metrics_file = MODEL_DIR / "test_assembly_metrics_cost_time.csv"  # Select the test metrics CSV that matches the masking condition you want to evaluate.

    encoder = GCNEncoder(in_channels, latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    metrics_embed = AssemblyMetricsEmbedding(latent_dim).to(device)
    model = GAE(encoder, decoder, metrics_embed).to(device)

    model.load_state_dict(torch.load(model_save_path, map_location=device), strict=False)
    model.eval()

    train_graphs = load_all_data_with_metrics(train_embedding_file, train_edge_file, train_metrics_file)
    valid_graphs = load_all_data_with_metrics(test_embedding_file, test_edge_file, test_metrics_file)

    train_labels = np.stack([g.supplier.numpy() for g in train_graphs], axis=0)
    valid_labels = np.stack([g.supplier.numpy() for g in valid_graphs], axis=0)

    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=False)
    val_loader = DataLoader(valid_graphs, batch_size=1, shuffle=False)

    train_embeddings = embed_data_with_metrics(model, train_loader, device)
    val_embeddings = embed_data_with_metrics(model, val_loader, device)

    return (
        train_embeddings,
        val_embeddings,
        train_labels,
        valid_labels,
        train_graphs,
        valid_graphs,
        model,
        device,
    )


def validate_embeddings_with_metrics() -> None:
    """
    Main evaluation entry.

    Steps:
      1) Load embeddings and labels for Case 2.
      2) Compute a global cosine-similarity threshold using k-nearest neighbors.
      3) Evaluate multi-label supplier prediction with this threshold.
      4) Print CSV-style metrics for easy copy into tables.
      5) Plot confusion matrices and detailed neighbor-level results.
    """
    (
        train_embeddings,
        val_embeddings,
        train_labels,
        valid_labels,
        train_graphs,
        valid_graphs,
        model,
        _,
    ) = load_data_and_embeddings()

    cosine_threshold = compute_k_neighbor_threshold_cosine(train_embeddings, k_neighbors=k)
    print(f"\n=== (1) k-NN-based cosine similarity threshold = {cosine_threshold:.4f} ===")

    y_pred_cosine, _ = evaluate_topk_cosine_threshold(
        train_embeddings,
        val_embeddings,
        train_labels,
        cosine_threshold,
    )

    report_dict = sk_clf_report(
        valid_labels,
        y_pred_cosine,
        target_names=["Supplier 1", "Supplier 2", "Supplier 3"],
        zero_division=0,
        output_dict=True,
    )

    supplier_accuracy = {}
    n_samples = valid_labels.shape[0]

    for i, sup in enumerate(["Supplier 1", "Supplier 2", "Supplier 3"]):
        TP = np.sum((y_pred_cosine[:, i] == 1) & (valid_labels[:, i] == 1))
        TN = np.sum((y_pred_cosine[:, i] == 0) & (valid_labels[:, i] == 0))
        acc = (TP + TN) / n_samples * 100
        supplier_accuracy[sup] = acc

    supplier_avg_acc = float(np.mean(list(supplier_accuracy.values())))

    print("\n=== CSV Format for PPT Table Conversion ===")
    print("Supplier,Precision,Recall,F1-Score,Support,Accuracy")

    for sup in ["Supplier 1", "Supplier 2", "Supplier 3"]:
        precision = report_dict[sup]["precision"] * 100
        recall = report_dict[sup]["recall"] * 100
        f1score = report_dict[sup]["f1-score"] * 100
        support = int(report_dict[sup]["support"])
        acc = supplier_accuracy[sup]
        print(f"{sup},{precision:.1f},{recall:.1f},{f1score:.1f},{support},{acc:.1f}")

    for avg_key in ["micro avg", "macro avg", "weighted avg", "samples avg"]:
        if avg_key in report_dict:
            precision = report_dict[avg_key]["precision"] * 100
            recall = report_dict[avg_key]["recall"] * 100
            f1score = report_dict[avg_key]["f1-score"] * 100
            support = int(report_dict[avg_key]["support"])
            print(f"{avg_key},{precision:.1f},{recall:.1f},{f1score:.1f},{support}")

    print(f"Supplier-average Prediction Accuracy: {supplier_avg_acc:.1f}")
    print("=== End of CSV ===\n")

    plot_multi_label_confusion_matrix(
        valid_labels,
        y_pred_cosine,
        class_names=["Supplier 1", "Supplier 2", "Supplier 3"],
    )

    print("\n=== Detailed Query -> Supplier Selection Results (Cosine Similarity) ===")
    print_threshold_suppliers_per_query_cosine(
        val_embeddings=val_embeddings,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        valid_graphs=valid_graphs,
        train_graphs=train_graphs,
        threshold=cosine_threshold,
        model=model,
    )


def validate_embeddings_with_roc_thresholds() -> None:
    """
    Optional ROC-based evaluation.

    This function:
      1) Finds supplier-specific cosine thresholds by discrete ROC search.
      2) Prints a classification report and supplier-average accuracy.
      3) Plots ROC-style curves for each supplier.
    """
    (
        train_embeddings,
        val_embeddings,
        train_labels,
        val_labels,
        _,
        _,
        _,
        _,
    ) = load_data_and_embeddings()

    print("\n=== (2) ROC-similarity based per-supplier threshold search ===")
    roc_thresholds_similarity = compute_best_threshold_per_supplier_roc_by_similarity_discrete(
        train_embeddings,
        val_embeddings,
        train_labels,
        val_labels,
        class_names=["Supplier 1", "Supplier 2", "Supplier 3"],
        num_points=500,
    )

    y_pred_roc_sim = evaluate_supplier_specific_classification_cosine(
        train_embeddings,
        val_embeddings,
        train_labels,
        roc_thresholds_similarity,
    )

    print("\n[Classification Report: ROC-similarity thresholds]")
    print(
        sk_clf_report(
            val_labels,
            y_pred_roc_sim,
            target_names=["Supplier 1", "Supplier 2", "Supplier 3"],
            zero_division=0,
        )
    )

    supplier_avg_acc_roc = compute_supplier_average_accuracy(val_labels, y_pred_roc_sim)
    print(f"Supplier-average Prediction Accuracy (ROC-similarity thresholds): {supplier_avg_acc_roc * 100:.1f}")

    print("\n=== (3) ROC subplot (cosine similarity, threshold in [-1, 1]) ===")
    plot_tpr_fpr_roc_subplots_cosine(
        train_embeddings,
        val_embeddings,
        train_labels,
        val_labels,
        class_names=["Supplier 1", "Supplier 2", "Supplier 3"],
    )


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # (1) Global k-NN based cosine threshold evaluation
    validate_embeddings_with_metrics()

    # (2) ROC-based threshold search and ROC plots
    # Uncomment the line below if ROC-style analysis is needed.
    # validate_embeddings_with_roc_thresholds()
