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

    def add(self, transition: Transition):
        """add a transition to the buffer."""
        task_id = transition.task_id
        buf = self._buffers[task_id]

        if len(buf) < self.capacity:
            buf.append(transition)
            self._total_size += 1
        else:
            pos = self._positions[task_id] % self.capacity
            buf[pos] = transition
            self._positions[task_id] = pos + 1

    def add_episode(self, transitions: list[Transition]):
        """add a full episode of transitions."""
        for t in transitions:
            self.add(t)

    def sample(
        self, batch_size: int, device: str = "cpu"
    ) -> Batch:
        """to sample a random batch across all tasks."""
        all_transitions = []
        for buf in self._buffers.values():
            all_transitions.extend(buf)

        indices = np.random.randint(0, len(all_transitions), size=batch_size)
        batch_transitions = [all_transitions[i] for i in indices]
        return self._to_batch(batch_transitions, device)

    def sample_task(
        self, task_id: int, batch_size: int, device: str = "cpu"
    ) -> Optional[Batch]:
        """sample transitions from a specific task."""
        if task_id not in self._buffers or len(self._buffers[task_id]) == 0:
            return None

        buf = self._buffers[task_id]
        indices = np.random.randint(0, len(buf), size=batch_size)
        batch_transitions = [buf[i] for i in indices]
        return self._to_batch(batch_transitions, device)

    def sample_context(
        self, task_id: int, context_size: int, device: str = "cpu"
    ) -> Optional[Batch]:
        """sample context transitions for encoder from a specific task.

        Returns the most recent transitions (for temporal coherence).
        """
        if task_id not in self._buffers or len(self._buffers[task_id]) == 0:
            return None

        buf = self._buffers[task_id]
        n = min(context_size, len(buf))
        batch_transitions = buf[-n:]
        return self._to_batch(batch_transitions, device)

    def sample_balanced(
        self, batch_size: int, device: str = "cpu"
    ) -> Batch:
        """sample balanced batch (equal transitions per task)."""
        task_ids = list(self._buffers.keys())
        if not task_ids:
            raise ValueError("Buffer is empty")

        per_task = max(1, batch_size // len(task_ids))
        batch_transitions = []

        for tid in task_ids:
            buf = self._buffers[tid]
            n = min(per_task, len(buf))
            indices = np.random.randint(0, len(buf), size=n)
            batch_transitions.extend([buf[i] for i in indices])

        # Trim to exact batch_size
        if len(batch_transitions) > batch_size:
            batch_transitions = batch_transitions[:batch_size]

        return self._to_batch(batch_transitions, device)

    def _to_batch(
        self, transitions: list[Transition], device: str
    ) -> Batch:
        return Batch(
            states=torch.tensor(
                np.array([t.state for t in transitions]), dtype=torch.float32
            ).to(device),
            actions=torch.tensor(
                np.array([t.action for t in transitions]), dtype=torch.float32
            ).to(device),
            rewards=torch.tensor(
                np.array([t.reward for t in transitions]), dtype=torch.float32
            ).unsqueeze(-1).to(device),
            next_states=torch.tensor(
                np.array([t.next_state for t in transitions]), dtype=torch.float32
            ).to(device),
            dones=torch.tensor(
                np.array([t.done for t in transitions]), dtype=torch.float32
            ).unsqueeze(-1).to(device),
            innovation_windows=torch.tensor(
                np.array([t.innovation_window for t in transitions]),
                dtype=torch.float32,
            ).to(device),
            filter_states=torch.tensor(
                np.array([t.filter_state for t in transitions]),
                dtype=torch.float32,
            ).to(device),
            constraint_violations=torch.tensor(
                np.array([t.constraint_violation for t in transitions]),
                dtype=torch.float32,
            ).unsqueeze(-1).to(device),
        )

    @property
    def size(self) -> int:
        return self._total_size

    @property
    def num_tasks(self) -> int:
        return len(self._buffers)