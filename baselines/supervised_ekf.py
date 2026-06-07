"""
Supervised NN baseline for EKF noise estimation.
This will train a neural network to predict Q_true/R_true from innovation windows.
Tests whether RL is necessary vs supervised learning.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SupervisedNoisePredictor(nn.Module):
    """MLP that predicts Q/R diagonal from innovation window."""

    def __init__(
        self,
        innovation_dim: int = 2,
        window_size: int = 50,
        n_q: int = 6,
        n_r: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        input_dim = innovation_dim * window_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_q + n_r),
            nn.Softplus(),  # ensure positive outputs
        )

    def forward(self, innovation_window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            innovation_window: (B, W, D) innovation buffer.
        Returns:
            params: (B, n_q + n_r) predicted Q/R diagonal elements.
        """
        x = innovation_window.flatten(start_dim=1)
        return self.net(x)


class SupervisedNNAdapter:
    """Supervised NN adapter for EKF noise estimation.

    This will wrap the trained SupervisedNoisePredictor for use as a baseline.
    """

    def __init__(
        self,
        Q_init: np.ndarray,
        R_init: np.ndarray,
        model_path: str = None,
        window_size: int = 50,
        innovation_dim: int = 2,
    ):
        self.Q = Q_init.copy()
        self.R = R_init.copy()
        self.n_q = Q_init.shape[0]
        self.n_r = R_init.shape[0]
        self.window_size = window_size
        self._innovations = []

        if TORCH_AVAILABLE:
            self.model = SupervisedNoisePredictor(
                innovation_dim=innovation_dim,
                window_size=window_size,
                n_q=self.n_q,
                n_r=self.n_r,
            )
            if model_path is not None:
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
            self.model.eval()
        else:
            self.model = None

    def adapt(
        self,
        innovation: np.ndarray,
        P: np.ndarray,
        S: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._innovations.append(innovation.copy())

        if self.model is None or len(self._innovations) < self.window_size:
            return self.Q, self.R

        # Get last W innovations
        window = np.array(self._innovations[-self.window_size :])

        with torch.no_grad():
            inp = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            params = self.model(inp).numpy().squeeze(0)

        self.Q = np.diag(params[: self.n_q])
        self.R = np.diag(params[self.n_q :])
        return self.Q, self.R

    def reset(self):
        self._innovations.clear()

    @staticmethod
    def train_model(
        train_data: list[dict],
        innovation_dim: int = 2,
        window_size: int = 50,
        n_q: int = 6,
        n_r: int = 2,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> "SupervisedNoisePredictor":
        """Train the supervised model on labeled data.

        Args:
            train_data: List of dicts with 'innovation_window' and 'true_params'.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training")

        model = SupervisedNoisePredictor(
            innovation_dim, window_size, n_q, n_r
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for sample in train_data:
                inp = torch.tensor(
                    sample["innovation_window"], dtype=torch.float32
                ).unsqueeze(0)
                target = torch.tensor(
                    sample["true_params"], dtype=torch.float32
                ).unsqueeze(0)

                pred = model(inp)
                loss = loss_fn(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return model

        