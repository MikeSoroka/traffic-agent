"""
4-way signalised intersection environment.

         N
         ↓
W →  [inter]  ← E
         ↑
         S

Phases (internal):
  0  NS_GREEN   N and S lights green, E/W red
  1  NS_YELLOW  transitioning out of NS green
  2  EW_GREEN   E and W lights green, N/S red
  3  EW_YELLOW  transitioning out of EW green

The RL agent receives a decision call only during green phases.
Yellow transitions run automatically.
"""
import numpy as np
from config import (
    SPAWN_PROB, MAX_QUEUE, DEPART_RATE,
    MIN_GREEN, YELLOW_STEPS,
    QUEUE_BIN_SIZE, NUM_QUEUE_BINS, NUM_TIMER_BINS,
)

NS_GREEN  = 0
NS_YELLOW = 1
EW_GREEN  = 2
EW_YELLOW = 3

_PHASE_LABEL = {
    NS_GREEN:  "NS  GREEN",
    NS_YELLOW: "NS  YELLOW",
    EW_GREEN:  "EW  GREEN",
    EW_YELLOW: "EW  YELLOW",
}


class TrafficEnv:
    """
    Gym-style environment (no gym dependency).
    State  : (ns_bin, ew_bin, phase_bit, timer_bin)  — 5×5×2×3 = 150 states
    Actions: 0 = keep current phase,  1 = request switch
    Reward : −(total cars queued this step)
    """

    def __init__(self, spawn_probs: dict | None = None, seed: int | None = None):
        self._rng   = np.random.default_rng(seed)
        self._spawn = spawn_probs or {d: SPAWN_PROB for d in "NSEW"}
        self.reset()

    # ── Public interface ──────────────────────────────────────────────────────

    def reset(self) -> tuple:
        self.q        = {d: 0 for d in "NSEW"}
        self._phase   = NS_GREEN
        self._timer   = 0    # steps spent in current green phase
        self._ytimer  = 0    # steps spent in current yellow phase
        self.step_num = 0
        self.total_q  = 0    # cumulative queue-length (diagnostic)
        return self._encode()

    def step(self, action: int) -> tuple[tuple, float, bool, dict]:
        """
        During yellow phases, `action` is ignored; the light advances
        automatically.  Call is_yellow to check before passing an action.
        """
        if self._is_yellow():
            self._advance_yellow()
        else:
            self._green_step(action)

        self._spawn_cars()

        waiting = sum(self.q.values())
        self.total_q  += waiting
        self.step_num += 1

        return self._encode(), float(-waiting), False, self._info()

    # ── Properties for visualiser ─────────────────────────────────────────────

    @property
    def is_yellow(self) -> bool:
        return self._is_yellow()

    @property
    def phase_label(self) -> str:
        return _PHASE_LABEL[self._phase]

    @property
    def ns_light(self) -> str:
        if self._phase == NS_GREEN:   return "green"
        if self._phase == NS_YELLOW:  return "yellow"
        return "red"

    @property
    def ew_light(self) -> str:
        if self._phase == EW_GREEN:   return "green"
        if self._phase == EW_YELLOW:  return "yellow"
        return "red"

    # ── Internals ─────────────────────────────────────────────────────────────

    def _is_yellow(self) -> bool:
        return self._phase in (NS_YELLOW, EW_YELLOW)

    def _advance_yellow(self):
        self._ytimer += 1
        if self._ytimer >= YELLOW_STEPS:
            self._phase  = EW_GREEN if self._phase == NS_YELLOW else NS_GREEN
            self._timer  = 0
            self._ytimer = 0

    def _green_step(self, action: int):
        if action == 1 and self._timer >= MIN_GREEN:
            # Begin yellow transition
            self._phase  = NS_YELLOW if self._phase == NS_GREEN else EW_YELLOW
            self._ytimer = 0
            return
        # Cars depart
        if self._phase == NS_GREEN:
            self.q["N"] = max(0, self.q["N"] - DEPART_RATE)
            self.q["S"] = max(0, self.q["S"] - DEPART_RATE)
        else:
            self.q["E"] = max(0, self.q["E"] - DEPART_RATE)
            self.q["W"] = max(0, self.q["W"] - DEPART_RATE)
        self._timer += 1

    def _spawn_cars(self):
        for d, p in self._spawn.items():
            if self._rng.random() < p:
                self.q[d] = min(self.q[d] + 1, MAX_QUEUE)

    def _encode(self) -> tuple:
        """Discretise continuous queues and timer into a Q-table index."""
        ns_raw = self.q["N"] + self.q["S"]
        ew_raw = self.q["E"] + self.q["W"]
        ns_bin = min(ns_raw // QUEUE_BIN_SIZE, NUM_QUEUE_BINS - 1)
        ew_bin = min(ew_raw // QUEUE_BIN_SIZE, NUM_QUEUE_BINS - 1)
        phase_bit = 0 if self._phase in (NS_GREEN, NS_YELLOW) else 1
        timer_bin = (
            0 if self._timer < MIN_GREEN else
            1 if self._timer < 3 * MIN_GREEN else
            2
        )
        return (ns_bin, ew_bin, phase_bit, timer_bin)

    def _info(self) -> dict:
        return {
            "queues": dict(self.q),
            "phase":  self._phase,
            "timer":  self._timer,
        }
