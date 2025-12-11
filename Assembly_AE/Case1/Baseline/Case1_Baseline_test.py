"""
Case 1 – baseline GCN classifier evaluation script.

Location:
    Assembly_AE/Case1/Baseline/Case1_Baseline_test.py

Assumed layout:

    Assembly_AE/
      Baseline_config.py
      Baseline_model.py
      Case1/
        CAE_train_extracted_embeddings.csv
        CAE_test_extracted_embeddings.csv
        train_assembly_edges.csv
        test_assembly_edges.csv
        train_assembly_metrics.csv
        test_assembly_metrics.csv
        Baseline/
          Case1_Baseline_train.py
          Case1_Baseline_test.py
          trained_model/
            Classifier_model_epoch_300.pth

This script:
  1) Loads a trained baseline GCN classifier.
  2) Builds assembly-level graphs with metrics.
  3) Evaluates classification performance on the test set.
  4) Prints a percentage-based classification report.
  5) Plots a row-normalized confusion matrix.
  6) Prints per-assembly predictions with probabilities.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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
from Baseline_config import in_channels, latent_dim      # type: ignore

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# metric scaling (must match training script)
K_METRICS = 10.0
SCALER = 1.0

CLASS_NAMES = ["Supplier 1", "Supplier 2", "Supplier 3"]


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

def load_all_data_with_metrics(
    embedding_file: Path,
    edge_file: Path,
    metrics_file: Path,
) -> Tuple[List[Data], List[Dict[str, torch.Tensor]], np.ndarray]:
    """
    Load assembly-level graphs, metrics, and labels.

    Each graph has:
      - x, edge_index
      - metrics_* tensors
      - y: supplier label (0-based)
      - assembly_id: stored for reporting
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
        graph.assembly_id = assembly_id

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

    return graphs, metrics, np.array(labels, dtype=int)


# -------------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------------

def evaluate_model(
    model: GCNClassifierWithMetrics,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate classifier and return predicted and true labels.
    """
    all_preds: List[int] = []
    all_labels: List[int] = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            tolerance = batch.metrics_tolerance.to(device)
            cost = batch.metrics_cost.to(device)
            time = batch.metrics_time.to(device)
            quantity = batch.metrics_quantity.to(device)

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

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def evaluate_model_with_probabilities(
    model: GCNClassifierWithMetrics,
    data_loader: DataLoader,
    device: torch.device,
):
    """
    For each test graph, report predicted label and probability.
    """
    results = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            tolerance = batch.metrics_tolerance.to(device)
            cost = batch.metrics_cost.to(device)
            time = batch.metrics_time.to(device)
            quantity = batch.metrics_quantity.to(device)

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
            probs = torch.softmax(logits, dim=1)

            pred_label = torch.argmax(probs, dim=1).cpu().numpy()
            pred_prob = torch.max(probs, dim=1)[0].cpu().numpy()
            true_label = batch.y.cpu().numpy()

            assembly_id = getattr(batch, "assembly_id", ["N/A"])[0]

            results.append(
                {
                    "assembly_id": assembly_id,
                    "true_label": int(true_label[0]),
                    "predicted_label": int(pred_label[0]),
                    "predicted_prob": float(pred_prob[0]),
                }
            )

    return results


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot row-normalized confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 20},
    )
    plt.xlabel("Predicted Labels", fontsize=16)
    plt.ylabel("True Labels", fontsize=16)
    plt.title("Confusion Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    # model path
    model_path = MODEL_DIR / "Classifier_model_epoch_300.pth"

    # CSV paths under Case1
    train_embedding_file = CASE_DIR / "CAE_train_extracted_embeddings.csv"
    test_embedding_file = CASE_DIR / "CAE_test_extracted_embeddings.csv"
    train_edge_file = CASE_DIR / "train_assembly_edges.csv"
    test_edge_file = CASE_DIR / "test_assembly_edges.csv"
    train_metrics_file = CASE_DIR / "train_assembly_metrics.csv"
    test_metrics_file = CASE_DIR / "test_assembly_metrics.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GCNClassifierWithMetrics(in_channels, latent_dim, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    train_graphs, _, _ = load_all_data_with_metrics(
        train_embedding_file,
        train_edge_file,
        train_metrics_file,
    )
    valid_graphs, _, valid_labels = load_all_data_with_metrics(
        test_embedding_file,
        test_edge_file,
        test_metrics_file,
    )

    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=False)
    val_loader = DataLoader(valid_graphs, batch_size=1, shuffle=False)

    y_pred, y_true = evaluate_model(model, val_loader, device)

    # classification report in percentage
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    overall_accuracy = report.pop("accuracy", None)
    df_report = pd.DataFrame(report).transpose()

    for col in ["precision", "recall", "f1-score"]:
        if col in df_report.columns:
            df_report[col] = df_report[col] * 100.0

    pd.options.display.float_format = "{:.2f}".format

    print("\nClassification report (percentage):")
    print(df_report)
    if overall_accuracy is not None:
        print(f"\nOverall accuracy: {overall_accuracy * 100:.2f}%")

    # per-supplier accuracy
    supplier_accuracies = {}
    for supplier in np.unique(y_true):
        idx = y_true == supplier
        acc = np.mean(y_pred[idx] == supplier) * 100.0
        supplier_accuracies[f"Supplier {supplier + 1}"] = acc
    avg_supplier_acc = np.mean(list(supplier_accuracies.values()))

    print("\nPer-supplier accuracy (percentage):")
    for name, acc in supplier_accuracies.items():
        print(f"{name}: {acc:.2f}%")
    print(f"Supplier-average prediction accuracy: {avg_supplier_acc:.2f}%")

    plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES)

    # detailed per-assembly results
    results = evaluate_model_with_probabilities(model, val_loader, device)
    print("\nDetailed test results with probabilities:")
    for res in results:
        print(
            f"Assembly ID: {res['assembly_id']}, "
            f"True: Supplier {res['true_label'] + 1}, "
            f"Predicted: Supplier {res['predicted_label'] + 1} "
            f"(Probability: {res['predicted_prob'] * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()
