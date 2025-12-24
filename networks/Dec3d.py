import torch.nn as nn

class Decoder3D(nn.Module):
    """
    FeatureDecoder reconstructs a sequence from encoded features using a projection layer,
    a ConvTranspose1D decoder, and a channel attention mechanism.

    Args:
        out_seq_len (int): Desired output sequence length.
        feature_dim (int): Input and output feature dimension.
    """

    def __init__(self, out_seq_len=50176, feature_dim=1152):
        super().__init__()
        self.out_seq_len = out_seq_len
        self.feature_dim = feature_dim
        reduced_len = out_seq_len // 32  # Assumes 5 upsampling layers with stride=2

        # Projection Layer: Compress temporal length
        self.projection = nn.Sequential(
            nn.Linear(out_seq_len, reduced_len),
            nn.GELU(),
            nn.LayerNorm(reduced_len),
            nn.Dropout(0.1),
            nn.Linear(reduced_len, reduced_len),
            nn.GELU(),
            nn.LayerNorm(reduced_len)
        )

        # Transpose Convolution Decoder: Expand temporal length
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(feature_dim, 768, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(768),
            nn.GELU(),

            nn.ConvTranspose1d(768, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.ConvTranspose1d(128, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # Channel Attention (Squeeze-and-Excitation style)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim // 8, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

       

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, seq_len, feature_dim)
        Returns:
            Tensor of shape (B, out_seq_len, feature_dim)
        """
        x = x.permute(0, 2, 1)                # (B, feature_dim, seq_len)
        x = self.projection(x)                # (B, feature_dim, reduced_len)
        x = self.decoder(x)                   # (B, feature_dim, out_seq_len)

        attn = self.attention(x)              # (B, feature_dim, 1)
        x = x * attn                          # Apply channel-wise attention

        x = x.permute(0, 2, 1)                # (B, out_seq_len, feature_dim)
        return x

