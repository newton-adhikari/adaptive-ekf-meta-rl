"""
The main contribution: 
    PID-Lagrangian Consistency-Constrained Policy Optimization (PID-CCPO).

This enforces NEES consistency as a hard constraint via
PID-controlled Lagrangian relaxation rather than fragile reward shaping.

This combines:
  - SAC (Soft Actor-Critic) for the base RL objective.
  - PID-Lagrangian dual variable for the consistency constraint.
  - Cost critic for predicting constraint violations.
"""


class RunningNormalizer:
    """
    Online running mean/variance using Welford's algorithm.

    this will normalize inputs to approximately zero mean and unit variance.
    """

    def __init__(self, shape: tuple, eps: float = 1e-8):
        self.shape = shape
        self.eps = eps
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray):
        """Update running statistics with a batch of data.

        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.shape):
            x = x[None]  # add batch dim

        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / max(total, 1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / max(total, 1)
        new_var = m2 / max(total, 1)

        self.mean = new_mean
        self.var = np.maximum(new_var, self.eps)
        self.count = total

    def normalize(self, x):
        """
        the function which normalizes input using running statistics.

        Works with both numpy arrays and torch tensors.
        """
        if isinstance(x, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.tensor(np.sqrt(self.var) + self.eps, dtype=x.dtype, device=x.device)
            return (x - mean) / std
        else:
            return (x - self.mean) / (np.sqrt(self.var) + self.eps)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: dict):
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]


class PIDLagrangian:
    """This is PID controller for the Lagrangian dual variable λ.

    Updates λ based on constraint violation error using PID control,
    which is more stable than vanilla gradient-based dual updates.
    """

    def __init__(
        self,
        delta: float = 0.1,
        k_p: float = 0.1,
        k_i: float = 0.01,
        k_d: float = 0.01,
        integral_max: float = 10.0,
    ):
        self.delta = delta
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.integral_max = integral_max

        self._lambda = 0.0
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, avg_violation: float) -> float:
        """
        this updates λ based on current average constraint violation.
        """
        error = avg_violation - self.delta
        self._integral += error
        # Clamp integral to prevent windup
        self._integral = max(-self.integral_max, min(self.integral_max, self._integral))

        derivative = error - self._prev_error
        self._prev_error = error

        self._lambda = max(
            0.0,
            self.k_p * error + self.k_i * self._integral + self.k_d * derivative,
        )
        return self._lambda

    @property
    def value(self) -> float:
        return self._lambda

    def reset(self):
        self._lambda = 0.0
        self._integral = 0.0
        self._prev_error = 0.0



class PIDCCPOAgent:
    """
    This is the CC-MetaEKF agent: 
        SAC + PID-Lagrangian consistency constraint.
    
    """

    def __init__(
        self
    ):
        