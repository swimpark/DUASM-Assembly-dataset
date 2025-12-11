import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

# GCN Encoder with Residual Connection
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(GCNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = GCNConv(in_channels, 64)  # n x in_channels -> n x 64
        self.conv2 = GCNConv(64, 16)  # n x 64 -> n x 32

        # Residual Layers
        self.residual_projection1 = torch.nn.Linear(in_channels, 64)  # First residual projection
        self.residual_projection2 = torch.nn.Linear(64, 16)  # Second residual projection

        # Residual weights
        self.alpha1 = torch.nn.Parameter(torch.tensor(0.5))  # First residual weight
        self.alpha2 = torch.nn.Parameter(torch.tensor(0.5))  # Second residual weight

    def forward(self, x, edge_index):
        # First Residual Connection
        x_res1 = torch.nn.functional.silu(self.residual_projection1(x))  # Project and apply SiLU

        # First GCN Layer
        x = torch.nn.functional.silu(self.conv1(x, edge_index))

        x = x + x_res1 * self.alpha1  # Add first residual connection

        # Second Residual Connection
        x_res2 = torch.nn.functional.silu(self.residual_projection2(x))  # Project and apply SiLU

        # Second GCN Layer
        z = self.conv2(x, edge_index)

        z = z + x_res2 * self.alpha2  # Add second residual connection
        return z  # (n x 16)

# Metrics Embedding
class AssemblyMetricsEmbedding(nn.Module):
    def __init__(self, latent_dim):
        super(AssemblyMetricsEmbedding, self).__init__()
        self.latent_dim = latent_dim

        # MLPs for each metric
        self.tolerance_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 8),  # 1 -> 16
            torch.nn.SiLU(),
            torch.nn.Linear(8, 16),  # 1 -> 16
        )
        self.cost_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 8),  # 1 -> 16
            torch.nn.SiLU(),
            torch.nn.Linear(8, 16),  # 1 -> 16
        )
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 8),  # 1 -> 16
            torch.nn.SiLU(),
            torch.nn.Linear(8, 16),  # 1 -> 16
        )
        self.quantity_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 8),  # 1 -> 16
            torch.nn.SiLU(),
            torch.nn.Linear(8, 16),  # 1 -> 16
        )

    def forward(self, tolerance, cost, time, quantity, num_nodes):
        # Process each metric
        tol_emb = self.tolerance_mlp(tolerance)  # (1x1 -> n x 32)
        cost_emb = self.cost_mlp(cost)  # (1x2 -> n x 32)
        time_emb = self.time_mlp(time)  # (1x2 -> n x 32)
        quantity_emb = self.quantity_mlp(quantity)  # (1x1 -> n x 32)

        # Repeat embeddings for all nodes
        tol_emb = tol_emb.repeat(num_nodes, 1)  # (n x 32)
        cost_emb = cost_emb.repeat(num_nodes, 1)  # (n x 32)
        time_emb = time_emb.repeat(num_nodes, 1)  # (n x 32)
        quantity_emb = quantity_emb.repeat(num_nodes, 1)  # (n x 32)

        # Concatenate metrics embeddings
        combined_emb = torch.cat([tol_emb, cost_emb, time_emb, quantity_emb], dim=1)  # (n x 128)
        return combined_emb

# GCN Classifier with Metrics and Residual Connection
class GCNClassifierWithMetrics(nn.Module):
    def __init__(self, in_channels, latent_dim, num_classes):
        super(GCNClassifierWithMetrics, self).__init__()
        self.encoder = GCNEncoder(in_channels, latent_dim)
        self.metrics_embed = AssemblyMetricsEmbedding(latent_dim)

        # Fully Connected Layers for Classification
        self.fc1 = nn.Linear(latent_dim + 64, 80)  # Combine graph and metrics embeddings
        self.fc2 = nn.Linear(80, num_classes)

        # Global Pooling
        self.global_pool = global_mean_pool

    def forward(self, x, edge_index, batch, tolerance, cost, time, quantity, apply_softmax=False):
        # Graph Embedding
        node_embeddings = self.encoder(x, edge_index)

        # Graph-level Embedding
        graph_embedding = self.global_pool(node_embeddings, batch)

        # Metrics Embedding
        metrics_embedding = self.metrics_embed(tolerance, cost, time, quantity, graph_embedding.size(0))

        # Combine Graph and Metrics Embeddings
        combined_embedding = torch.cat([graph_embedding, metrics_embedding], dim=1)

        # Classification
        out = torch.nn.functional.silu(self.fc1(combined_embedding))
        out = self.fc2(out)

        # 🔥 추론 시 Softmax 적용 (출력값 합을 1로 만들기 위해)
        if apply_softmax:
            out = torch.nn.functional.softmax(out, dim=1)
        return out
