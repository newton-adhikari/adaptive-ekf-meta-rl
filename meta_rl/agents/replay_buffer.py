"""
Multi-task replay buffer for meta-RL training.
Stores transitions grouped by task for context-based meta-learning.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

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

    def __init__(self, capacity: int = 100_000, max_tasks: int = 500):
        self.capacity = capacity
        self.max_tasks = max_tasks
        self._buffers: dict[int, list[Transition]] = defaultdict(list)
        self._positions: dict[int, int] = defaultdict(int)
        self._total_size = 0