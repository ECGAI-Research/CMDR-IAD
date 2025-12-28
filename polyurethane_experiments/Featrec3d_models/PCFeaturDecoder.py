import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDecoder(nn.Module):
    def __init__(self, out_seq_len=4096, feature_dim=1152):  # <-- 32×32 = 1024
        super().__init__()
        reduced_len = out_seq_len // 32 if out_seq_len >= 32 else out_seq_len

        self.projection = nn.Sequential(
            nn.Linear(out_seq_len, reduced_len),
            nn.GELU(),
            nn.LayerNorm(reduced_len),
            nn.Dropout(0.1),
            nn.Linear(reduced_len, reduced_len),
            nn.GELU(),
            nn.LayerNorm(reduced_len)
        )

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

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim // 8, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, feature_dim, seq_len)
        x = self.projection(x)  # (B, feature_dim, reduced_len)
        x = self.decoder(x)     # (B, feature_dim, out_seq_len)

        attn = self.attention(x)
        x = x * attn

        x = x.permute(0, 2, 1)  # (B, out_seq_len, feature_dim)
        return x
class FeatureDecoder_9216(nn.Module):
    def __init__(self, out_seq_len=9216, feature_dim=1152):  # <-- 96×96 = 9216
        super().__init__()
        reduced_len = out_seq_len // 32 if out_seq_len >= 32 else out_seq_len

        self.projection = nn.Sequential(
            nn.Linear(out_seq_len, reduced_len),
            nn.GELU(),
            nn.LayerNorm(reduced_len),
            nn.Dropout(0.1),
            nn.Linear(reduced_len, reduced_len),
            nn.GELU(),
            nn.LayerNorm(reduced_len)
        )

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

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim // 8, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, feature_dim, seq_len)
        x = self.projection(x)  # (B, feature_dim, reduced_len)
        x = self.decoder(x)     # (B, feature_dim, out_seq_len)

        attn = self.attention(x)
        x = x * attn

        x = x.permute(0, 2, 1)  # (B, out_seq_len, feature_dim)
        return x
