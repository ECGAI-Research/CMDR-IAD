import torch
import torch.nn as nn
import torch.nn.functional as F
class TunedSparseAttention1D(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=128, mlp_ratio=2.0,
                 attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparseAttention1D(dim, num_heads, window_size, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(proj_drop),

            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# =====================================================
# SparseAttention1D (B, N, C)
# =====================================================
class SparseAttention1D(nn.Module):
    """
    Windowed multi-head self-attention for 1D sequences.
    Input:  (B, N, C)
    Output: (B, N, C)
    """
    def __init__(self, dim, num_heads=8, window_size=128, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        # Output projection
        self.proj = nn.Linear(dim, dim)
        # Dropouts
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x: (B, N, C)
        return: (B, N, C)
        """
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # each (B, H, N, D)

        # --- Split into local windows ---
        win_tokens = self.window_size
        pad_len = (win_tokens - N % win_tokens) % win_tokens
        if pad_len > 0:
            Q = F.pad(Q, (0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))

        Np = Q.shape[2]
        num_windows = Np // win_tokens

        Q = Q.view(B, H, num_windows, win_tokens, D)
        K = K.view(B, H, num_windows, win_tokens, D)
        V = V.view(B, H, num_windows, win_tokens, D)

        # --- Attention within windows ---
        attn = torch.einsum("bhwqd,bhwkd->bhwqk", Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum("bhwqk,bhwkd->bhwqd", attn, V)

        # --- Merge back ---
        out = out.reshape(B, H, Np, D)[:, :, :N, :]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


# =====================================================
# Decoder2D using SparseAttention1D
# =====================================================
class Decoder2D(nn.Module):
    """
    2D decoder with windowed attention + ConvTranspose2D refinement.
    Input:  (B, N, C)
    Output: (B, N, C)
    """
    def __init__(self, dim, img_size=224, num_heads=8, window_size=128):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.reduced_dim = dim // 2  # compressed channel size (e.g., 384)

        # ====== Projection (reduce feature dimension) ======
        self.projection = nn.Sequential(
            nn.Linear(dim, self.reduced_dim),
            nn.GELU(),
            nn.LayerNorm(self.reduced_dim),
            nn.Dropout(0.1),
            nn.Linear(self.reduced_dim, self.reduced_dim),
            nn.GELU(),
            nn.LayerNorm(self.reduced_dim)
        )

        # ====== Sparse Attention Refinement ======
        self.attn = TunedSparseAttention1D(
         dim=self.reduced_dim,
         num_heads=num_heads,
         window_size=window_size,
         mlp_ratio=2.0,
         attn_drop=0.1,
         proj_drop=0.1
)


        # ====== ConvTranspose2D Refinement ======
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.reduced_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.ConvTranspose2d(128, self.reduced_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.reduced_dim),
            nn.GELU(),
        )

        # ====== Final Reconstruction ======
        self.reconstruct = nn.Sequential(
            nn.Linear(self.reduced_dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        """
        x: (B, N, C)
        return: (B, N, C)
        """
        B, N, C = x.shape
        H = W = int(self.img_size)

        # --- projection ---
        x = self.projection(x)     # (B, N, reduced_dim)
        x = self.attn(x)           # windowed attention refinement

        # --- reshape to 2D ---
        x_2d = x.transpose(1, 2).reshape(B, self.reduced_dim, H, W)  # (B, reduced_dim, H, W)
        x_2d = self.conv_decoder(x_2d)                                # local conv refinement
        x = x_2d.flatten(2).transpose(1, 2)                           # (B, N, reduced_dim)

        # --- reconstruction ---
        x = self.reconstruct(x)                                       # (B, N, dim)
        return x
