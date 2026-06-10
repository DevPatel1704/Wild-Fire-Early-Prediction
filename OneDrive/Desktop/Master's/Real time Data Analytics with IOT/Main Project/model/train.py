"""
Training script for the GAT-LSTM wildfire risk model.

Usage:
    python -m model.train --epochs 50 --batch-size 16 --csv data/raw/simulated_readings.csv
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from loguru import logger

from .gat_lstm import GATLSTM
from .dataset import WildfireDataset


def train(
    csv_path: str = "data/raw/simulated_readings.csv",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    checkpoint_dir: str = "model/checkpoints",
    device_name: str = "auto",
    max_samples: int = 0,
    nrows: int = 0,
):
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Training on device: {device}")

    # Dataset
    dataset = WildfireDataset(csv_path=csv_path, n_timesteps=6, normalize=True, nrows=nrows)
    if max_samples > 0 and max_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:max_samples]
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
        logger.info(f"Subsampled dataset to {max_samples} samples.")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    logger.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # Model
    sample_x, _, _ = dataset[0]
    n_features = sample_x.shape[-1]  # F
    model = GATLSTM(
        n_features=n_features,
        n_timesteps=6,
        gat_hidden=64,
        lstm_hidden=128,
        n_heads=4,
        dropout=0.2,
    ).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for x, adj, y in train_loader:
            # x: (B, N, T, F), adj: (B, N, N), y: (B, N)
            x = x.to(device)
            adj = adj[0].to(device)   # adj is same for all — take first
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x, adj).squeeze(-1)   # (B, N)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, adj, y in val_loader:
                x = x.to(device)
                adj = adj[0].to(device)
                y = y.to(device)
                pred = model(x, adj).squeeze(-1)
                val_loss += criterion(pred, y).item()
                all_preds.extend(pred.cpu().flatten().tolist())
                all_labels.extend(y.cpu().flatten().tolist())

        try:
            auc = roc_auc_score(
                [1 if v >= 0.5 else 0 for v in all_labels], all_preds
            )
        except ValueError:
            auc = 0.0

        scheduler.step(val_loss / max(len(val_loader), 1))
        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)

        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | AUC: {auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            ckpt_path = os.path.join(checkpoint_dir, "gat_lstm_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "auc": auc,
                "n_features": n_features,
            }, ckpt_path)
            logger.info(f"  Saved best checkpoint (AUC={auc:.4f}) → {ckpt_path}")

    logger.info(f"Training complete. Best AUC: {best_auc:.4f}")
    return best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/simulated_readings.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit training samples (0=all)")
    parser.add_argument("--nrows", type=int, default=0, help="Limit CSV rows loaded (0=all)")
    args = parser.parse_args()

    logger.add("logs/training.log", rotation="50 MB")
    train(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        nrows=args.nrows,
    )
