import os
import sys
import torch
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# ----------------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # .../Case3/DUASM
CASE_DIR = THIS_DIR.parent                          # .../Case3
ASSEMBLY_AE_DIR = CASE_DIR.parent                   # .../Assembly_AE
SAVE_DIR = THIS_DIR / "trained_model"               # .../Case3/DUASM/trained_model

if str(ASSEMBLY_AE_DIR) not in sys.path:
    sys.path.append(str(ASSEMBLY_AE_DIR))

from DUASM_model import (  # type: ignore
    GCNEncoder,
    Decoder,
    GAE,
    AssemblyMetricsEmbedding,
)
from DUASM_config import in_channels, latent_dim                     # type: ignore

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
k_metrics     = 10   # Metrics scale factor
num_suppliers = 3
scaler_tol    = 10
scaler_time   = 10
scaler_cost   = 0
scaler_qty    = 0


# ----------------------------------------------------------------------
# Data-loading and embedding
# ----------------------------------------------------------------------
def load_all_data_with_metrics(embedding_file, edge_file, metrics_file):
    embedding_df = pd.read_csv(embedding_file)
    edge_df      = pd.read_csv(edge_file)
    metrics_df   = pd.read_csv(metrics_file)

    graphs = []

    for _, row in metrics_df.iterrows():
        aid = row["Assembly_ID"]

        # supplier one-hot
        sup_vec = torch.zeros(num_suppliers, dtype=torch.float)
        for s in map(int, str(row["Supplier"]).split(",")):
            sup_vec[s - 1] = 1

        # node features
        vecs = (
            embedding_df.query("Assembly_ID == @aid")["Vector"]
            .apply(ast.literal_eval)
            .tolist()
        )
        if len(vecs) == 0:
            print(f"Warning: no embedding vectors for Assembly_ID {aid}")
            continue
        x = torch.tensor(vecs, dtype=torch.float)

        # edges
        ed = edge_df.query("Assembly_ID == @aid")
        if ed.empty:
            print(f"Warning: no edge data for Assembly_ID {aid}")
            continue
        eidx = ast.literal_eval(ed["Edge_Index"].iloc[0])
        edge_index = torch.tensor(eidx, dtype=torch.long).t().contiguous()

        # metrics (scaled)
        tol = torch.tensor(
            [[row["Restrictive_Tolerance"] * k_metrics * scaler_tol]],
            dtype=torch.float,
        )
        tim = torch.tensor(
            [[row["Assembly_Time"] * k_metrics * scaler_time]],
            dtype=torch.float,
        )
        cst = torch.tensor(
            [[row["Assembly_Cost"] * k_metrics * scaler_cost]],
            dtype=torch.float,
        )
        qty = torch.tensor(
            [[row["Quantity"] * k_metrics * scaler_qty]],
            dtype=torch.float,
        )

        g = Data(
            x=x,
            edge_index=edge_index,
            metrics_tolerance=tol,
            metrics_time=tim,
            metrics_cost=cst,
            metrics_quantity=qty,
            supplier=sup_vec,
        )
        g.assembly_id = aid
        graphs.append(g)

    return graphs


def embed_data_with_metrics(model, loader, device):
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            ei = batch.edge_index.to(device)

            tol = batch.metrics_tolerance.to(device).float()
            tim = batch.metrics_time.to(device).float()
            cst = batch.metrics_cost.to(device).float()
            qty = batch.metrics_quantity.to(device).float()

            x_hat, _, _ = model(x, ei, tol, cst, tim, qty)
            embs.append(x_hat.mean(dim=0).cpu())
    return torch.stack(embs)


# ----------------------------------------------------------------------
# nDCG
# ----------------------------------------------------------------------
def compute_ndcg_from_supplier(candidate_suppliers, k=None):
    n = len(candidate_suppliers)
    if k is None or k > n:
        k = n

    actual = candidate_suppliers[:k]
    ideal = sorted(actual, reverse=True)

    dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(actual))
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


# ----------------------------------------------------------------------
# Load model, data, embeddings
# ----------------------------------------------------------------------
def load_data_and_embeddings():
    """
    Assumed files in SAVE_DIR (= Case3/DUASM/trained_model):

        GAE_model_epoch_600.pth
        train_assembly_edges.csv
        train_assembly_metrics.csv
        test_assembly_edges.csv
        test_assembly_metrics.csv

    If your files are located elsewhere, change SAVE_DIR above.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    enc = GCNEncoder(in_channels, latent_dim).to(device)
    dec = Decoder(latent_dim).to(device)
    met = AssemblyMetricsEmbedding(latent_dim).to(device)
    model = GAE(enc, dec, met).to(device)

    ckpt_path = SAVE_DIR / "GAE_model_epoch_600.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # data
    train_graphs = load_all_data_with_metrics(
        CASE_DIR / "CAE_train_extracted_embeddings.csv",
        SAVE_DIR / "train_assembly_edges.csv",
        SAVE_DIR / "train_assembly_metrics.csv",
    )
    valid_graphs = load_all_data_with_metrics(
        CASE_DIR / "CAE_test_extracted_embeddings.csv",
        SAVE_DIR / "test_assembly_edges.csv",
        SAVE_DIR / "test_assembly_metrics.csv",
    )

    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=False)
    val_loader = DataLoader(valid_graphs, batch_size=1, shuffle=False)

    train_emb = embed_data_with_metrics(model, train_loader, device)
    val_emb = embed_data_with_metrics(model, val_loader, device)

    return train_emb, val_emb, train_graphs, valid_graphs, model, device


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    train_emb, val_emb, train_graphs, valid_graphs, model, device = load_data_and_embeddings()

    k_list = [10, 50, 100, 150]
    for k in k_list:
        ndcg_scores = []
        all_orders = []

        for query in valid_graphs:
            # 1) query embedding
            with torch.no_grad():
                xq = query.x.to(device)
                ei = query.edge_index.to(device)
                tol = query.metrics_tolerance.to(device).float()
                tim = query.metrics_time.to(device).float()
                cst = query.metrics_cost.to(device).float()
                qty = query.metrics_quantity.to(device).float()

                x_hat, _, _ = model(xq, ei, tol, cst, tim, qty)
                q_emb = x_hat.mean(dim=0).cpu().unsqueeze(0)

            # 2) top-k candidate indices
            sims = torch.nn.functional.cosine_similarity(q_emb, train_emb, dim=1)
            topk_idx = torch.topk(sims, k, largest=True).indices.cpu().numpy()

            # 3) sort by reconstructed time
            cand = []
            for idx in topk_idx:
                g = train_graphs[idx]
                with torch.no_grad():
                    _, _, (_, _, tim_r, _) = model(
                        g.x.to(device),
                        g.edge_index.to(device),
                        g.metrics_tolerance.to(device).float(),
                        g.metrics_cost.to(device).float(),
                        g.metrics_time.to(device).float(),
                        g.metrics_quantity.to(device).float(),
                    )
                sup = int(g.supplier.argmax()) + 1
                cand.append((sup, tim_r.item()))

            sup_order = [s for s, _ in sorted(cand, key=lambda x: x[1])]
            all_orders.append(sup_order)

            # 4) nDCG@k
            ndcg_scores.append(compute_ndcg_from_supplier(sup_order, k=k))

        mean_ndcg = float(np.mean(ndcg_scores))
        std_ndcg = float(np.std(ndcg_scores))
        rand_order = all_orders[np.random.randint(len(all_orders))]

        print(f"\n=== k={k} ===")
        print(f"Mean nDCG = {mean_ndcg:.4f},  Std = {std_ndcg:.4f}")
        print(f"Sample sup_order: {rand_order}")
