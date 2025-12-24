import torch.nn as nn


class FeatureMappigMLP(nn.Module):
    """
    Cross-modal feature mapping network implemented as a lightweight MLP.

    This module projects features from one modality into the feature space
    of another modality (e.g., 2D → 3D or 3D → 2D) while preserving semantic
    consistency between modalities.
    """

    def __init__(self, in_features=None, out_features=None,
                 act_layer=nn.GELU, dropout=0.1):
        super().__init__()

        # Intermediate dimensionality used for smoother cross-modal projection
        # Acts as a bottleneck between input and output feature spaces
        self.reduced_dim = (in_features + out_features) // 2

        # Non-linear activation function
        self.act_fcn = act_layer()

        # Linear layer that projects input features to the reduced dimension
        self.input = nn.Linear(in_features, self.reduced_dim)

        # Projection block:
        # A deeper MLP with normalization and dropout to stabilize training
        # and improve cross-modal alignment
        self.projection = nn.Sequential(
            nn.Linear(self.reduced_dim, self.reduced_dim),
            nn.GELU(),
            nn.LayerNorm(self.reduced_dim),
            nn.Dropout(dropout),

            nn.Linear(self.reduced_dim, self.reduced_dim),
            nn.GELU(),
            nn.LayerNorm(self.reduced_dim)
        )

        # Final linear layer projecting features to the target modality dimension
        self.output = nn.Linear(self.reduced_dim, out_features)

    def forward(self, x):
        """
        Forward pass for cross-modal feature mapping.

        Args:
            x (Tensor): Input features of shape [B, N, in_features]

        Returns:
            Tensor: Mapped features of shape [B, N, out_features]
        """

        # Initial linear projection + non-linearity
        x = self.input(x)
        x = self.act_fcn(x)

        # Apply the projection block to enforce semantic alignment
        x = self.projection(x)

        # Final projection to the target feature space
        x = self.output(x)

        return x
