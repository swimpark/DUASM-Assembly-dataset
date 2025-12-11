# -*- coding: utf-8 -*-
"""
PCA-based visualization of DUASM assembly-level embeddings (Case 2).

File location:
    JMS_Code/Validation/PCA_visualization.py

Assumed layout:

    JMS_Code/
      Assembly_AE/
        DUASM_model.py
        DUASM_config.py
        Case2/
          CAE_test_extracted_embeddings.csv
          DUASM/
            trained_model/
              GAE_model_epoch_600.pth
              test_assembly_edges.csv
              test_assembly_metrics_origin.csv
              test_assembly_metrics_cost_time.csv
              test_assembly_metrics_cost_qty.csv
              test_assembly_metrics_time_qty.csv

If you store Case 2 data in a different place, update CASE2_DIR
and DUASM_MODEL_DIR in the path setup section below.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pathlib import Path

# -----------------------------------------------------------------------------#
# Path setup
# -----------------------------------------------------------------------------#

# .../JMS_Code/Validation
VALIDATION_DIR = Path(__file__).resolve().parent
# .../JMS_Code
JMS_CODE_DIR = VALIDATION_DIR.parent
# .../JMS_Code/Assembly_AE
ASSEMBLY_AE_DIR = JMS_CODE_DIR / "Assembly_AE"
# .../JMS_Code/Assembly_AE/Case2
CASE2_DIR = ASSEMBLY_AE_DIR / "Case2"
# .../JMS_Code/Assembly_AE/Case2/DUASM
CASE2_DUASM_DIR = CASE2_DIR / "DUASM"
# .../JMS_Code/Assembly_AE/Case2/DUASM/trained_model
DUASM_MODEL_DIR = CASE2_DUASM_DIR / "trained_model"

# Add Assembly_AE to sys.path for DUASM model imports
if str(ASSEMBLY_AE_DIR) not in sys.path:
    sys.path.append(str(ASSEMBLY_AE_DIR))

# DUASM model definition and configuration
from DUASM_model import GCNEncoder, Decoder, GAE, AssemblyMetricsEmbedding  # type: ignore
from DUASM_config import in_channels, latent_dim                            # type: ignore

# -----------------------------------------------------------------------------#
# Optional: hover tooltips
# -----------------------------------------------------------------------------#
try:
    import mplcursors
    _HAS_MPLCURSORS = True
except Exception:
    _HAS_MPLCURSORS = False

# ============================== Global configuration (test) ============================== #
k_metrics = 10
runtime_scale_20 = 20.0
num_suppliers = 3
RANDOM_SEED = 0
VERIFY_SCALE = True

# DUASM Case 2 model and CSV paths
MODEL_PATH = DUASM_MODEL_DIR / "GAE_model_epoch_600.pth"

TEST_EMB_FILE = CASE2_DIR / "CAE_test_extracted_embeddings.csv"
TEST_EDGE_FILE = DUASM_MODEL_DIR / "test_assembly_edges.csv"

# Metric files inside the trained_model folder
TEST_METRICS_ORIGIN = DUASM_MODEL_DIR / "test_assembly_metrics_origin.csv"
TEST_METRICS_COST_MASK = DUASM_MODEL_DIR / "test_assembly_metrics_time_qty.csv"
TEST_METRICS_TIME_MASK = DUASM_MODEL_DIR / "test_assembly_metrics_cost_qty.csv"
TEST_METRICS_QTY_MASK = DUASM_MODEL_DIR / "test_assembly_metrics_cost_time.csv"

# Highlight conditions for the target region
TARGET_SUPPLIER_LABEL = 3
ALT_SUPPLIER_LABEL = 2
TARGET_KEYWORD = "latch"

# ================================ Loading and embeddings ================================ #
def load_graphs_with_metrics(embedding_file, edge_file, metrics_file):
    """
    Build PyG graph objects with component embeddings, assembly edges,
    and scaled assembly-level metrics.
    """
    embedding_df = pd.read_csv(embedding_file)
    edge_df = pd.read_csv(edge_file)
    metrics_df = pd.read_csv(metrics_file)

    graphs = []
    for _, row in metrics_df.iterrows():
        assembly_id = str(row["Assembly_ID"])

        # Supplier multi-label (comma separated string such as "1,2")
        supplier_str = str(row["Supplier"])
        supplier_list = list(map(int, supplier_str.split(",")))
        supplier_vector = torch.zeros(num_suppliers, dtype=torch.float)
        for s in supplier_list:
            supplier_vector[s - 1] = 1

        # Node features
        node_features = (
            embedding_df[embedding_df["Assembly_ID"] == assembly_id]["Vector"]
            .apply(eval)
            .tolist()
        )
        if len(node_features) == 0:
            continue
        x = torch.tensor(node_features, dtype=torch.float)

        # Edges
        edge_data = edge_df[edge_df["Assembly_ID"] == assembly_id]
        if edge_data.empty:
            continue
        edge_index = (
            torch.tensor(
                edge_data["Edge_Index"].apply(eval).values[0],
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )

        # Assembly-level metrics (scaled for runtime)
        tol = float(row["Restrictive_Tolerance"]) * k_metrics * runtime_scale_20
        tim = float(row["Assembly_Time"]) * k_metrics * runtime_scale_20
        cst = float(row["Assembly_Cost"]) * k_metrics * runtime_scale_20
        qty = float(row["Quantity"]) * runtime_scale_20

        g = Data(
            x=x,
            edge_index=edge_index,
            metrics_tolerance=torch.tensor([[tol]], dtype=torch.float),
            metrics_cost=torch.tensor([[cst]], dtype=torch.float),
            metrics_time=torch.tensor([[tim]], dtype=torch.float),
            metrics_quantity=torch.tensor([[qty]], dtype=torch.float),
            supplier=supplier_vector,
        )
        g.assembly_id = assembly_id
        graphs.append(g)
    return graphs


def embed_graphs(model, graphs, device):
    """
    Run DUASM on each graph and average component-level embeddings
    to obtain a single assembly-level embedding.
    """
    if len(graphs) == 0:
        return torch.empty(0, latent_dim)
    loader = DataLoader(graphs, batch_size=1, shuffle=False)
    embs = []
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            ei = batch.edge_index.to(device)
            t = batch.metrics_tolerance.to(device)
            c = batch.metrics_cost.to(device)
            ti = batch.metrics_time.to(device)
            q = batch.metrics_quantity.to(device)
            x_hat, _, _ = model(x, ei, t, c, ti, q)
            embs.append(x_hat.mean(dim=0).cpu())
    return torch.stack(embs)


def load_model():
    """
    Load the trained DUASM model for Case 2.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAE(
        GCNEncoder(in_channels, latent_dim).to(device),
        Decoder(latent_dim).to(device),
        AssemblyMetricsEmbedding(latent_dim).to(device),
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    return model, device

# =================================== Utilities =================================== #
def graphs_to_dataframe(graphs, embeddings_tensor):
    """
    Convert graphs and embeddings into a flat DataFrame for analysis and plotting.
    """
    rows = []
    Z = embeddings_tensor.cpu().numpy()
    for i, g in enumerate(graphs):
        sup_vec = g.supplier.numpy()
        sup_label = int(np.argmax(sup_vec)) + 1
        rows.append(
            {
                "assembly_id": g.assembly_id,
                "supplier_vec": sup_vec,
                "supplier_label": sup_label,
                "tolerance": float(g.metrics_tolerance.item()),
                "cost": float(g.metrics_cost.item()),
                "time": float(g.metrics_time.item()),
                "quantity": float(g.metrics_quantity.item()),
                "embedding": Z[i],
            }
        )
    return pd.DataFrame(rows)


def pca_fit_transform(X, n_components=2):
    """
    Fit PCA on X and return the 2D projection and the fitted PCA object.
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X2 = pca.fit_transform(X)
    return X2, pca


def pca_transform(pca, X):
    """
    Apply a fitted PCA object to new data.
    """
    return pca.transform(X)


def select_aligned_subset(target_ids, graphs_list, emb_list, pca=None):
    """
    Select assemblies that appear in all metric conditions and return
    the aligned embeddings for each condition.
    """
    maps = [{g.assembly_id: i for i, g in enumerate(gs)} for gs in graphs_list]
    ids_used = [aid for aid in target_ids if all(aid in m for m in maps)]
    if len(ids_used) == 0:
        return [], [np.empty((0, 2))] * len(graphs_list)

    arrays = []
    for gi, ei in zip(graphs_list, emb_list):
        idx = [{g.assembly_id: i for i, g in enumerate(gi)}[aid] for aid in ids_used]
        vecs = np.stack([ei[i].numpy() for i in idx], axis=0)
        arrays.append(vecs if pca is None else pca_transform(pca, vecs))
    return ids_used, arrays


def _add_cursor(scatter, labels_fmt):
    """
    Attach hover tooltip to a scatter plot if mplcursors is available.
    """
    if not _HAS_MPLCURSORS:
        return
    cur = mplcursors.cursor(scatter, hover=True)

    @cur.connect("add")
    def _on_add(sel):
        i = sel.index
        sel.annotation.set(text=labels_fmt(i))
        sel.annotation.get_bbox_patch().set_alpha(0.92)


def sanity_check_metrics_scale(origin_csv, cost_csv, time_csv, qty_csv):
    """
    Print basic statistics for each metric CSV to confirm that masking
    behaves as expected before runtime scaling.
    """
    try:
        do = pd.read_csv(origin_csv)
        dc = pd.read_csv(cost_csv)
        dt = pd.read_csv(time_csv)
        dq = pd.read_csv(qty_csv)
        msg = []
        for name, df in [
            ("origin", do),
            ("cost0", dc),
            ("time0", dt),
            ("qty0", dq),
        ]:
            msg.append(
                f"{name:6s} | mean tol={df['Restrictive_Tolerance'].mean():.3f}, "
                f"time={df['Assembly_Time'].mean():.3f}, "
                f"cost={df['Assembly_Cost'].mean():.3f}, "
                f"qty={df['Quantity'].mean():.3f}"
            )
        print("[SCALE CHECK] Raw CSV means (before runtime x20):")
        print("\n".join(msg))
    except Exception as e:
        print(f"[SCALE CHECK] Skipped due to error: {e}")

# -------------------------- Cluster visualization utilities -------------------------- #
def _centroid(X):
    """
    Return the centroid of a 2D point cloud.
    """
    return np.mean(X, axis=0) if X.shape[0] > 0 else np.array([np.nan, np.nan])


def _cov_ellipse_params(X, conf=0.95):
    """
    Approximate parameters of a 95 percent confidence ellipse
    for a 2D Gaussian distribution using chi-square value 5.991.
    """
    if X.shape[0] < 3:
        return None
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    r2 = 5.991
    width, height = 2 * np.sqrt(vals * r2)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return width, height, angle


def draw_cov_ellipse(ax, X, edgecolor="k", alpha=0.15, lw=1.0):
    """
    Draw a covariance ellipse that covers most of the points in X.
    """
    p = _cov_ellipse_params(X)
    if p is None:
        return
    w, h, ang = p
    ctr = _centroid(X)
    e = Ellipse(
        xy=ctr,
        width=w,
        height=h,
        angle=ang,
        facecolor=edgecolor,
        alpha=alpha,
        edgecolor=edgecolor,
        lw=lw,
    )
    ax.add_patch(e)


def link_centroids_double_arrow(ax, X_a, X_b):
    """
    Draw a double-headed arrow between the centroids of two clusters
    and return the Euclidean distance between them.
    """
    if X_a.shape[0] == 0 or X_b.shape[0] == 0:
        return np.nan
    c0 = _centroid(X_a)
    c1 = _centroid(X_b)
    ax.annotate(
        "",
        xy=c1,
        xytext=c0,
        arrowprops=dict(arrowstyle="<->", lw=1.6, alpha=0.9),
    )
    d = float(np.linalg.norm(c1 - c0))
    return d

# -------------------------- Silhouette score utility -------------------------- #
def silhouette_by_supplier(embeddings: np.ndarray, supplier_labels: np.ndarray):
    """
    Compute a silhouette score for supplier labels.
    Suppliers that appear fewer than two times are ignored.
    """
    uniq, counts = np.unique(supplier_labels, return_counts=True)
    valid_labels = uniq[counts >= 2]
    mask = np.isin(supplier_labels, valid_labels)
    X = embeddings[mask]
    y = supplier_labels[mask]
    if len(np.unique(y)) < 2:
        return np.nan, {}
    score = silhouette_score(X, y, metric="euclidean")
    counts_dict = {int(k): int(v) for k, v in zip(uniq, counts)}
    return float(score), counts_dict

# ============================== Figure generation (test) ============================== #
def make_figures_test():
    """
    Generate a 2x2 PCA plot that compares the baseline metric condition
    with cost, time, and quantity masked conditions.
    """
    if VERIFY_SCALE:
        sanity_check_metrics_scale(
            TEST_METRICS_ORIGIN,
            TEST_METRICS_COST_MASK,
            TEST_METRICS_TIME_MASK,
            TEST_METRICS_QTY_MASK,
        )

    model, device = load_model()

    # Baseline metrics
    graphs_origin = load_graphs_with_metrics(
        TEST_EMB_FILE, TEST_EDGE_FILE, TEST_METRICS_ORIGIN
    )
    emb_origin = embed_graphs(model, graphs_origin, device)
    if emb_origin.numel() == 0:
        print("[ERROR] origin embeddings are empty.")
        return
    df_origin = graphs_to_dataframe(graphs_origin, emb_origin)

    # Global supplier separability on the full test set
    Y_sup = df_origin["supplier_label"].values.astype(int)
    X_lat = np.stack(df_origin["embedding"].values, axis=0)
    s_global, cnts = silhouette_by_supplier(X_lat, Y_sup)
    print(
        f"[SILHOUETTE] Global supplier separability (origin, TEST): "
        f"{s_global:.4f}  | counts={cnts}"
    )

    # PCA fit on baseline embeddings
    X2_all, pca = pca_fit_transform(X_lat, n_components=2)

    # Flip the coordinate system to align visual orientation
    X2_all = -X2_all

    # Assemblies that match the target supplier and keyword
    mask_target_origin = (
        (df_origin["supplier_label"] == TARGET_SUPPLIER_LABEL)
        & (df_origin["assembly_id"].str.lower().str.contains(TARGET_KEYWORD.lower()))
    )
    if not mask_target_origin.any():
        print("[INFO] No target assemblies found in origin condition.")
        return
    target_ids = df_origin.loc[mask_target_origin, "assembly_id"].tolist()
    target_set = set(target_ids)

    # Metric-masked conditions
    graphs_cost = load_graphs_with_metrics(
        TEST_EMB_FILE, TEST_EDGE_FILE, TEST_METRICS_COST_MASK
    )
    graphs_time = load_graphs_with_metrics(
        TEST_EMB_FILE, TEST_EDGE_FILE, TEST_METRICS_TIME_MASK
    )
    graphs_qty = load_graphs_with_metrics(
        TEST_EMB_FILE, TEST_EDGE_FILE, TEST_METRICS_QTY_MASK
    )

    emb_cost = embed_graphs(model, graphs_cost, device)
    emb_time = embed_graphs(model, graphs_time, device)
    emb_qty = embed_graphs(model, graphs_qty, device)

    # DataFrame and PCA coordinates for each condition
    def _df_and_pca_all(graphs, emb_tensor, pca_obj):
        if emb_tensor.numel() == 0 or len(graphs) == 0:
            return pd.DataFrame(
                columns=["assembly_id", "supplier_label"]
            ), np.empty((0, 2))
        df = graphs_to_dataframe(graphs, emb_tensor)
        X = (
            np.stack(df["embedding"].values, axis=0)
            if len(df) > 0
            else np.empty((0, latent_dim))
        )

        if X.shape[0] > 0:
            X2 = pca_transform(pca_obj, X)
            # Apply the same sign flip as the baseline
            X2 = -X2
        else:
            X2 = np.empty((0, 2))

        return df, X2

    df_cost, X2_cost_all = _df_and_pca_all(graphs_cost, emb_cost, pca)
    df_time, X2_time_all = _df_and_pca_all(graphs_time, emb_time, pca)
    df_qty, X2_qty_all = _df_and_pca_all(graphs_qty, emb_qty, pca)

    # Assemblies with the same product ID as the target set
    same_prod_o = df_origin["assembly_id"].isin(target_set).values
    labels_o = df_origin["supplier_label"].values
    mask_alt_o = same_prod_o & (labels_o == ALT_SUPPLIER_LABEL)
    mask_target_o = same_prod_o & (labels_o == TARGET_SUPPLIER_LABEL)

    # Axes limits for all panels
    xlim = [-8000, 3000]
    ylim = [np.min(X2_all[:, 1]) - 1, np.max(X2_all[:, 1]) + 1]

    # 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    titles = ["Full", "Cost masked", "Time masked", "Quantity masked"]
    panels = [
        (axs[0, 0], X2_all, df_origin),
        (axs[0, 1], X2_cost_all, df_cost),
        (axs[1, 0], X2_time_all, df_time),
        (axs[1, 1], X2_qty_all, df_qty),
    ]
    colors = ["C0", "C1", "C2", "C3"]

    # Store centroid distances for each condition
    centroid_dists = {}

    for (ax, X2, df_cond), title, color in zip(panels, titles, colors):
        if X2.shape[0] == 0 or len(df_cond) == 0:
            ax.set_title(f"{title} (no data)")
            continue

        # Filter assemblies that share the same product ID
        same_prod = df_cond["assembly_id"].isin(target_set).values
        labels = df_cond["supplier_label"].values
        mask_alt = same_prod & (labels == ALT_SUPPLIER_LABEL)
        mask_target = same_prod & (labels == TARGET_SUPPLIER_LABEL)

        # Scatter points
        if np.any(mask_alt):
            ax.scatter(
                X2[mask_alt, 0],
                X2[mask_alt, 1],
                s=22,
                alpha=0.55,
                c="#6c6c6c",
                marker="o",
                label=f"Supplier{ALT_SUPPLIER_LABEL}",
            )
        if np.any(mask_target):
            sc_t = ax.scatter(
                X2[mask_target, 0],
                X2[mask_target, 1],
                s=26,
                alpha=0.95,
                c=color,
                marker="o",
                label=f"Supplier{TARGET_SUPPLIER_LABEL} target",
            )
            if _HAS_MPLCURSORS:
                ids_arr = df_cond.loc[mask_target, "assembly_id"].values
                _add_cursor(
                    sc_t,
                    lambda i, ids_arr=ids_arr, title=title: f"{title}\nassembly_id={ids_arr[i]}",
                )

        # Covariance ellipses
        if np.any(mask_alt):
            draw_cov_ellipse(
                ax, X2[mask_alt], edgecolor="#6c6c6c", alpha=0.18, lw=1.0
            )
        if np.any(mask_target):
            draw_cov_ellipse(
                ax, X2[mask_target], edgecolor=color, alpha=0.15, lw=1.2
            )

        # Double arrow between centroids
        d = link_centroids_double_arrow(
            ax,
            X2[mask_alt] if np.any(mask_alt) else np.empty((0, 2)),
            X2[mask_target] if np.any(mask_target) else np.empty((0, 2)),
        )
        centroid_dists[title] = d

        # Axis formatting
        ax.set_title(title, fontsize=11)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Axis labels on outer plots only
    axs[1, 0].set_xlabel("PC1")
    axs[1, 1].set_xlabel("PC1")
    axs[0, 0].set_ylabel("PC2")
    axs[1, 0].set_ylabel("PC2")

    # Shared legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    # Print centroid distances for each panel
    dist_series = pd.Series(
        centroid_dists, name="centroid_distance_S2_to_S3"
    )
    print("\n[Centroid distances within each panel (same product only)]")
    print(dist_series.to_string(float_format=lambda v: f"{v:0.4f}"))

    plt.show()

# ================================== Main entry ================================== #
if __name__ == "__main__":
    make_figures_test()
