"""
Multi-task replay buffer for meta-RL training.
Stores transitions grouped by task for context-based meta-learning.
"""


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    innovation_window: np.ndarray  # (W, D)
    filter_state: np.ndarray       # [NEES, NIS, tr(P), diag(S)]
    constraint_violation: float     # 1.0 if NEES outside bounds, else 0.0
    task_id: int


@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    innovation_windows: torch.Tensor
    filter_states: torch.Tensor
    constraint_violations: torch.Tensor


class MultiTaskReplayBuffer:
    """Replay buffer that stores transitions per task.

    - Random batch across all tasks (for critic updates).
    - Per-task context windows (for encoder).
    - Balanced sampling across tasks.

    """

    def __init__(self):
        pass