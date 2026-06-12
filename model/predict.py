"""
Inference wrapper: loads a trained checkpoint and exposes a simple
predict_batch() method used by the stream processor.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from .gat_lstm import GATLSTM


class FireRiskPredictor:
    """Thread-safe inference wrapper around GATLSTM."""

    def __init__(self, checkpoint_path: str = None, device_name: str = "auto"):
        ckpt = checkpoint_path or os.getenv("MODEL_CHECKPOINT_PATH", "model/checkpoints/gat_lstm_best.pt")
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        self.model: Optional[GATLSTM] = None
        self._node_order: Optional[List[str]] = None
        self._adj: Optional[torch.Tensor] = None
        self._load(ckpt)

    def _load(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found at {path}. Running without model.")
            return
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            n_features = ckpt.get("n_features", 24)
            self.model = GATLSTM(n_features=n_features).to(self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            logger.info(f"Model loaded from {path} (epoch={ckpt.get('epoch')}, AUC={ckpt.get('auc', '?')})")
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            self.model = None

    def set_graph(self, node_ids: List[str], adj: np.ndarray):
        """Set the sensor graph used for inference."""
        self._node_order = node_ids
        self._adj = torch.tensor(adj, dtype=torch.float32).to(self.device)

    def predict_batch(
        self,
        feature_tensors: List[Tuple[str, np.ndarray]],
    ) -> Dict[str, float]:
        """
        feature_tensors: list of (node_id, array(T, F))
        Returns: dict of node_id → risk_score (0–1)
        """
        if self.model is None or not feature_tensors:
            return {}

        if self._adj is None:
            logger.warning("Adjacency matrix not set. Call set_graph() first.")
            return {}

        node_ids = [nid for nid, _ in feature_tensors]
        # (1, N, T, F)
        x_np = np.stack([arr for _, arr in feature_tensors], axis=0)
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Subset adjacency to current nodes
        n = len(node_ids)
        adj = self._adj[:n, :n] if self._adj.shape[0] >= n else self._adj

        with torch.no_grad():
            preds = self.model(x, adj).squeeze(-1).squeeze(0)  # (N,)
            scores = preds.cpu().numpy()

        return {node_id: float(scores[i]) for i, node_id in enumerate(node_ids)}

    @property
    def is_ready(self) -> bool:
        return self.model is not None
