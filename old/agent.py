"""
Tabular Q-learning agent.

Q-table shape: (ns_bins, ew_bins, phase_bit, timer_bins, actions)
             = (5, 5, 2, 3, 2) = 300 values — trivially small.
"""
import numpy as np
from config import (
    ALPHA, GAMMA, EPS_START, EPS_MIN, EPS_DECAY,
    NUM_QUEUE_BINS, NUM_TIMER_BINS, QTABLE_PATH,
)

_TABLE_SHAPE = (NUM_QUEUE_BINS, NUM_QUEUE_BINS, 2, NUM_TIMER_BINS, 2)


class QLearningAgent:
    N_ACTIONS = 2  # 0 = keep phase,  1 = switch phase

    def __init__(self):
        self.q   = np.zeros(_TABLE_SHAPE)
        self.eps = EPS_START

    # ── Decision ──────────────────────────────────────────────────────────────

    def act(self, state: tuple, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.eps:
            return np.random.randint(self.N_ACTIONS)
        return int(np.argmax(self.q[state]))

    # ── Learning ──────────────────────────────────────────────────────────────

    def update(self, s: tuple, a: int, r: float, s2: tuple):
        td_target = r + GAMMA * np.max(self.q[s2])
        self.q[s][a] += ALPHA * (td_target - self.q[s][a])

    def decay_epsilon(self):
        self.eps = max(EPS_MIN, self.eps * EPS_DECAY)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = QTABLE_PATH):
        np.save(path, self.q)
        print(f"  Q-table saved  →  {path}")

    def load(self, path: str = QTABLE_PATH):
        self.q   = np.load(path)
        self.eps = 0.0   # fully greedy after loading
        print(f"  Q-table loaded ←  {path}")
