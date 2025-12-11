import torch
import torch.nn as nn

torch.manual_seed(2)


class Encoder(nn.Module):
    """
    3D convolutional encoder for voxel inputs.

    Input:
        x: [B, 1, D, H, W] voxel grid (e.g., 128 x 128 x 128)

    Output:
        z: [B, hidden_dim] latent embedding (z-score normalized per sample)
    """
    def __init__(
        self,
        in_channels: int = 1,
        dim: int = 128,
        out_conv_channels: int = 512,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        torch.manual_seed(3)

        conv1_channels = out_conv_channels // 8
        conv2_channels = out_conv_channels // 4
        conv3_channels = out_conv_channels // 2

        self.out_conv_channels = out_conv_channels
        self.out_dim = dim // 16  # after four stride-2 conv layers

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, conv1_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(conv1_channels, conv2_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(conv2_channels, conv3_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(conv3_channels, out_conv_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.vector = nn.Sequential(
            nn.Linear(
                out_conv_channels * self.out_dim * self.out_dim * self.out_dim,
                hidden_dim,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns a per-sample z-score normalized latent vector.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(-1, self.out_conv_channels * self.out_dim * self.out_dim * self.out_dim)
        x = self.vector(x)

        # Per-sample z-score normalization of the latent vector
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-8)

        return x


class Decoder(nn.Module):
    """
    3D convolutional decoder for voxel reconstruction.

    Input:
        z: [B, z_dim] latent embedding

    Output:
        x_recon: [B, out_channels, D, H, W] voxel grid
    """
    def __init__(
        self,
        in_channels: int = 512,
        dim: int = 128,
        z_dim: int = 128,
        out_channels: int = 1,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        torch.manual_seed(3)

        self.in_channels = in_channels
        self.out_dim = dim
        self.in_dim = dim // 16  # input spatial size after linear layer

        conv1_out_channels = in_channels // 2
        conv2_out_channels = conv1_out_channels // 2
        conv3_out_channels = conv2_out_channels // 2

        self.linear1 = nn.Sequential(
            nn.Linear(z_dim, in_channels * self.in_dim * self.in_dim * self.in_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                conv1_out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(conv1_out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(
                conv1_out_channels,
                conv2_out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(conv2_out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(
                conv2_out_channels,
                conv3_out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(conv3_out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(
                conv3_out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )

        if activation == "sigmoid":
            self.out = nn.Sigmoid()
        else:
            self.out = nn.Sigmoid()  # can be changed to nn.Tanh if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = x.view(-1, self.in_channels, self.in_dim, self.in_dim, self.in_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.out(x)


if __name__ == "__main__":
    encoder = Encoder()
    decoder = Decoder()
    print(encoder)
    print(decoder)
