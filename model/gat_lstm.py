"""
GAT-LSTM: Graph Attention Network for spatial sensor relationships
combined with LSTM for temporal patterns over the last 6 minutes.

Architecture:
  Input: (N, T, F) — N nodes, T=6 time steps, F=25 features per node
  → For each time step t: GAT over all N nodes  → (N, gat_hidden)
  → Stack T outputs                              → (N, T, gat_hidden)
  → LSTM per node over time dimension            → (N, lstm_hidden)
  → Linear head                                  → (N, 1) fire risk score
"""

import torch
import torch.nn as nn
from .gat_layer import MultiHeadGAT


class GATLSTM(nn.Module):
    """
    Spatial-temporal wildfire risk prediction model.

    n_nodes:     number of sensor nodes (100)
    n_features:  features per node per timestep (25: 8 sensors × 3 stats + fire_risk)
    n_timesteps: LSTM sequence length (6)
    gat_hidden:  GAT output dimension (64)
    lstm_hidden: LSTM hidden dimension (128)
    n_heads:     number of GAT attention heads (4)
    dropout:     dropout rate
    """

    def __init__(
        self,
        n_features: int = 25,
        n_timesteps: int = 6,
        gat_hidden: int = 64,
        lstm_hidden: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_timesteps = n_timesteps

        # Spatial: process each time step with GAT
        self.gat = MultiHeadGAT(n_features, gat_hidden, n_heads=n_heads, dropout=dropout)

        # Temporal: LSTM over GAT-encoded time steps
        self.lstm = nn.LSTM(
            input_size=gat_hidden,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,   # (batch, seq, features)
            dropout=dropout,
        )

        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,    # (batch, N, T, F)
        adj: torch.Tensor,  # (N, N)
    ) -> torch.Tensor:      # (batch, N, 1)
        batch_size, N, T, F = x.shape

        gat_out_list = []
        for t in range(T):
            x_t = x[:, :, t, :]                          # (batch, N, F)
            g = self.gat(x_t, adj)                       # (batch, N, gat_hidden)
            gat_out_list.append(g)

        # Stack time steps: (batch, N, T, gat_hidden)
        gat_seq = torch.stack(gat_out_list, dim=2)

        # LSTM per node: reshape to (batch*N, T, gat_hidden)
        gat_flat = gat_seq.view(batch_size * N, T, -1)
        lstm_out, _ = self.lstm(gat_flat)       # (batch*N, T, lstm_hidden)
        last_hidden = lstm_out[:, -1, :]        # (batch*N, lstm_hidden)

        # Predict
        risk = self.classifier(last_hidden)     # (batch*N, 1)
        return risk.view(batch_size, N, 1)      # (batch, N, 1)
