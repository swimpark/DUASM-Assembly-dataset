"""
Assembly-level graph autoencoder training script (DUASM, Case 1).

File location:
    JMS_Code/Assembly_AE/Case1/DUASM/Case1_DUASM_train.py

Assumed layout:

    JMS_Code/
      Assembly_AE/
        DUASM_model.py
        DUASM_config.py
        binvox_rw.py
        dataset_loader.py
        Case1/
          CAE_train_extracted_embeddings.csv
          train_assembly_edges.csv
          train_assembly_metrics.csv
          DUASM/
            Case1_DUASM_train.py
            trained_model/
              GAE_model_epoch_0.pth  (optional, for warm start)
"""

import sys
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

# -------------------------------------------------------------------------
# Path resolution
# -------------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent          # .../Case1/DUASM
CASE_DIR = CURRENT_DIR.parent                          # .../Case1
ASSEMBLY_AE_DIR = CASE_DIR.parent                      # .../Assembly_AE

if str(ASSEMBLY_AE_DIR) not in sys.path:
    sys.path.append(str(ASSEMBLY_AE_DIR))

from DUASM_model import (                              # type: ignore
    GCNEncoder,
    Decoder,
    GAE,
    AssemblyMetricsEmbedding,
)
from DUASM_config import (                             # type: ignore
    in_channels,
    latent_dim,
    out_channels,
    epochs,
    learning_rate,
    batch_size,
)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Model checkpoints and logs are stored here
SAVE_DIR = CURRENT_DIR / "trained_model"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Input CSVs are shared at the Case1 level
DATA_DIR = CASE_DIR

pretrained_model_name = "GAE_model_epoch_0.pth"        # optional
save_interval = 50

k_metrics = 10.0
scaler = 0.5

# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------


def load_embedding_data(
    assembly_metrics_file: Path,
    embedding_file: Path,
    edge_file: Path,
) -> Tuple[List[Data], List[Dict[str, torch.Tensor]]]:
    """
    Load assembly node embeddings, edge indices, and assembly-level metrics.

    Returns
    -------
    graphs : list[Data]
        PyG graph objects with node features and edges.
    metrics : list[dict]
        Per-assembly tensors for tolerance, time, cost, quantity.
    """
    assembly_df = pd.read_csv(assembly_metrics_file)
    embedding_df = pd.read_csv(embedding_file)
    edge_df = pd.read_csv(edge_file)

    graphs: List[Data] = []
    metrics: List[Dict[str, torch.Tensor]] = []

    for _, row in assembly_df.iterrows():
        assembly_id = row["Assembly_ID"]

        node_series = embedding_df[embedding_df["Assembly_ID"] == assembly_id]["Vector"]
        if node_series.empty:
            continue
        node_features = node_series.apply(eval).tolist()

        edge_series = edge_df[edge_df["Assembly_ID"] == assembly_id]["Edge_Index"]
        if edge_series.empty:
            continue
        edge_index_list = edge_series.apply(eval).values[0]

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        tol = torch.tensor([[row["Restrictive_Tolerance"] * k_metrics * scaler]])
        t_ = torch.tensor([[row["Assembly_Time"] * k_metrics * scaler]])
        c_ = torch.tensor([[row["Assembly_Cost"] * k_metrics * scaler]])
        q_ = torch.tensor([[row["Quantity"] * scaler]])

        metrics.append(
            {
                "tolerance": tol,
                "time": t_,
                "cost": c_,
                "quantity": q_,
            }
        )
        graphs.append(Data(x=x, edge_index=edge_index))

    return graphs, metrics


# -------------------------------------------------------------------------
# Graph utilities
# -------------------------------------------------------------------------


def get_adjacency_matrix(edge_index: torch.Tensor, num_nodes: int, device: torch.device):
    """Construct adjacency matrix from edge index."""
    adj = torch.zeros((num_nodes, num_nodes), device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj.fill_diagonal_(0)
    return adj


def validate_and_fix_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Make edges undirected and remove duplicates.
    """
    edges = set()
    for i, j in edge_index.t().tolist():
        edges.add((i, j))
        edges.add((j, i))

    cleaned = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    if cleaned.max().item() >= num_nodes:
        raise ValueError("Edge index contains node id out of range.")
    return cleaned


# -------------------------------------------------------------------------
# Loss definition
# -------------------------------------------------------------------------


def compute_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    adj_orig: torch.Tensor,
    adj_hat: torch.Tensor,
    metrics_orig: torch.Tensor,
    metrics_hat,
) -> torch.Tensor:
    """
    Total loss = node reconstruction + edge BCE + metric MSE terms.
    """
    tol_hat, cost_hat, time_hat, qty_hat = metrics_hat

    loss_node = F.mse_loss(x_hat, x)
    loss_edge = F.binary_cross_entropy(adj_hat.view(-1), adj_orig.view(-1))

    loss_tol = F.mse_loss(tol_hat, metrics_orig[:, 0:1])
    loss_cost = F.mse_loss(cost_hat, metrics_orig[:, 1:2])
    loss_time = F.mse_loss(time_hat, metrics_orig[:, 2:3])
    loss_qty = F.mse_loss(qty_hat, metrics_orig[:, 3:4])

    return (
        0.2 * loss_node
        + 1.0 * loss_edge
        + 0.1 * loss_tol
        + 0.1 * loss_cost
        + 0.1 * loss_time
        + 0.1 * loss_qty
    )


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------


def load_pretrained(model: torch.nn.Module, path: Path) -> bool:
    """Load pretrained checkpoint if it exists."""
    if path.exists():
        print(f"Loading pretrained model from {path}")
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return True
    return False


def train_assembly_model(
    graphs: List[Data],
    metrics: List[Dict[str, torch.Tensor]],
    pretrained_model_path: Path | None = None,
) -> None:
    """Train DUASM GAE with assembly-level metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = GCNEncoder(in_channels, latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    metrics_embed = AssemblyMetricsEmbedding(latent_dim=latent_dim).to(device)
    model = GAE(encoder, decoder, metrics_embed).to(device)

    if pretrained_model_path is not None and pretrained_model_path.exists():
        load_pretrained(model, pretrained_model_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses: list[list[float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(range(len(graphs)), desc=f"Epoch {epoch + 1}/{epochs}", unit="graph")

        for idx in pbar:
            graph = graphs[idx]
            metric = metrics[idx]

            optimizer.zero_grad()

            x = graph.x.to(device)
            edge_index = validate_and_fix_edge_index(graph.edge_index, x.size(0)).to(device)

            tol = metric["tolerance"].to(device)
            t_ = metric["time"].to(device)
            c_ = metric["cost"].to(device)
            q_ = metric["quantity"].to(device)

            metrics_orig = torch.cat([tol, c_, t_, q_], dim=-1)

            x_hat, adj_hat, metrics_hat = model(x, edge_index, tol, c_, t_, q_)
            adj_orig = get_adjacency_matrix(edge_index, x.size(0), device)

            loss = compute_loss(x, x_hat, adj_orig, adj_hat, metrics_orig, metrics_hat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(graphs)
        epoch_losses.append([epoch + 1, avg_loss])
        print(f"Epoch {epoch + 1}: average loss = {avg_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            ckpt_path = SAVE_DIR / f"GAE_model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # final model
    final_path = SAVE_DIR / "GAE_model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model: {final_path}")

    # loss log
    with open(SAVE_DIR / "training_losses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "AverageLoss"])
        writer.writerows(epoch_losses)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    assembly_metrics_file = DATA_DIR / "train_assembly_metrics.csv"
    embedding_file = DATA_DIR / "CAE_train_extracted_embeddings.csv"
    edge_file = DATA_DIR / "train_assembly_edges.csv"

    graphs, metrics = load_embedding_data(
        assembly_metrics_file=assembly_metrics_file,
        embedding_file=embedding_file,
        edge_file=edge_file,
    )

    pretrained_path = SAVE_DIR / pretrained_model_name
    train_assembly_model(graphs, metrics, pretrained_model_path=pretrained_path)
