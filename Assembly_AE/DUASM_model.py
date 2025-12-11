import torch
from torch_geometric.nn import GCNConv

# GCN Encoder
class GCNEncoder(torch.nn.Module):
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


class AssemblyMetricsEmbedding(torch.nn.Module):
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


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        # Node reconstruction
        self.node_decoder = torch.nn.Sequential(
            torch.nn.Linear(80, 80),
            torch.nn.SiLU(),
            torch.nn.Linear(80, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 128),
        )

        # Edge reconstruction
        self.edge_decoder = torch.nn.Sequential(
            torch.nn.Linear(80 * 2, 80),  # z_node_split 두 개 연결
            torch.nn.SiLU(),
            torch.nn.Linear(80, 1)
        )

        # Metrics reconstruction for each part
        self.tolerance_decoder = torch.nn.Sequential(
            torch.nn.Linear(80, 80),  # nx32 입력
            torch.nn.SiLU(),
            torch.nn.Linear(80, 80),  # nx32 입력
            torch.nn.SiLU(),
            torch.nn.Linear(80, 1),  # nx32 입력
        )
        self.cost_decoder = torch.nn.Sequential(
            torch.nn.Linear(80, 80),  # nx32 입력
            torch.nn.SiLU(),
            torch.nn.Linear(80, 80),  # nx32 입력
            torch.nn.SiLU(),
            torch.nn.Linear(80, 1),  # nx32 입력
        )
        self.time_decoder = torch.nn.Sequential(
            torch.nn.Linear(80, 80),  # nx32 입력
            torch.nn.SiLU(),
            torch.nn.Linear(80, 80),  # nx32 입력
            torch.nn.SiLU(),
            torch.nn.Linear(80, 1),  # nx32 입력
        )
        self.quantity_decoder = torch.nn.Sequential(
            torch.nn.Linear(80, 80),  # nx32 입력
            torch.nn.SiLU(),
            torch.nn.Linear(80, 80),  # nx32 입력
            torch.nn.SiLU(),
            torch.nn.Linear(80, 1),  # nx32 입력
        )

    def forward(self, z_combined, num_nodes):

        # Further split metrics embedding
        z_tolerance, z_cost, z_time, z_quantity = [z_combined] * 4

        # Node feature reconstruction
        x_hat = self.node_decoder(z_combined)

        # Edge reconstruction
        idx = torch.arange(num_nodes, device=z_combined.device)
        idx_pairs = torch.cartesian_prod(idx, idx)
        z_i = z_combined[idx_pairs[:, 0]]
        z_j = z_combined[idx_pairs[:, 1]]
        edge_features = torch.cat([z_i, z_j], dim=1)
        adj_logits = self.edge_decoder(edge_features).view(num_nodes, num_nodes)
        adj_hat = torch.sigmoid(adj_logits)

        # Metrics reconstruction
        # Reduce nx32 to 1x32 by mean
        tolerance_emb = z_tolerance.mean(dim=0, keepdim=True)  # 1x160
        cost_emb = z_cost.mean(dim=0, keepdim=True)  # 1x160
        time_emb = z_time.mean(dim=0, keepdim=True)  # 1x160
        quantity_emb = z_quantity.mean(dim=0, keepdim=True)  # 1x160

        # Decode reduced embeddings to original sizes
        tolerance_hat = self.tolerance_decoder(tolerance_emb)  # 1x1
        cost_hat = self.cost_decoder(cost_emb)  # 1x1
        time_hat = self.time_decoder(time_emb)  # 1x1
        quantity_hat = self.quantity_decoder(quantity_emb)  # 1x1

        return x_hat, adj_hat, (tolerance_hat, cost_hat, time_hat, quantity_hat)


class GAE(torch.nn.Module):
    def __init__(self, encoder, decoder, metrics_embed):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.metrics_embed = metrics_embed

        # 추가된 MLPs (Concat 이후 MLP 로 축소 -> shape, tolerance, time, cost, quantity 정보 융합)
        self.mlp_128_to_64 = torch.nn.Sequential(
            torch.nn.Linear(80, 64),  # 축소
            torch.nn.SiLU(),

        )

        self.mlp_64_to_128 = torch.nn.Sequential(
            torch.nn.Linear(64, 80),  # 복원
            torch.nn.SiLU(),
        )

    def forward(self, x, edge_index, tolerance, cost, time, quantity):
        num_nodes = x.size(0)

        # Node embedding from GCN Encoder
        z_node = self.encoder(x, edge_index)  # (n x 32)

        # Assembly metrics embedding
        z_metrics = self.metrics_embed(tolerance, cost, time, quantity, num_nodes)  # (n x 96)

        # Concatenate node and metrics embeddings
        z_combined = torch.cat([z_node, z_metrics], dim=1)  # (n x 128)

        # 추가된 축소 및 복원 단계
        z_reduced = self.mlp_128_to_64(z_combined)  # (n x 64)

        z_restored = self.mlp_64_to_128(z_reduced)  # (n x 128)

        # Decode for reconstruction
        x_hat, adj_hat, metrics_hat = self.decoder(z_restored, num_nodes)
        return x_hat, adj_hat, metrics_hat
