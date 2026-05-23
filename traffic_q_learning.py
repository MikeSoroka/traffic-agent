"""
traffic_q_learning.py
=====================
Tabular Q-Learning agent for smart traffic-light control using the
sumo-rl library (https://github.com/LucasAlegre/sumo-rl) as the
Gymnasium-compatible environment.

Quick-start
-----------
1. Install SUMO and set SUMO_HOME:
       export SUMO_HOME="/usr/share/sumo"
2. Install Python deps:
       pip install sumo-rl gymnasium numpy matplotlib
3. Clone example network files (if you don't have your own):
       git clone https://github.com/LucasAlegre/sumo-rl.git
4. Run (headless):
       python traffic_q_learning.py
5. Run with SUMO GUI (set USE_GUI = True below, or pass --gui):
       python traffic_q_learning.py --gui

Outputs
-------
  outputs/training_rewards.png   – reward curve with smoothed trend line
  outputs/q_learning_run_*.csv   – per-step metrics from sumo-rl
  q_table.npy                    – serialised Q-table (reload with np.load)
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; must come before pyplot import
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CONFIGURABLE PARAMETERS
#    Adjust these before running; no other edits required for a basic run.
# ---------------------------------------------------------------------------

# --- Paths ------------------------------------------------------------------
# Default to the single-intersection sample bundled inside the sumo-rl repo.
# Change these to point at your own .net.xml / .rou.xml if you have them.
import importlib.util as _ilu
_SUMO_RL_NETS = os.path.join(
    os.path.dirname(_ilu.find_spec("sumo_rl").origin),
    "nets", "single-intersection",
)
NET_FILE   = os.path.join(_SUMO_RL_NETS, "single-intersection.net.xml")
ROUTE_FILE = os.path.join(_SUMO_RL_NETS, "single-intersection.rou.xml")

# All CSV logs and plots land here.
OUTPUT_DIR     = "outputs"
CSV_PREFIX     = os.path.join(OUTPUT_DIR, "q_learning_run")
QTABLE_PATH    = "q_table.npy"
PLOT_PATH      = os.path.join(OUTPUT_DIR, "training_rewards.png")

# --- Environment ------------------------------------------------------------
USE_GUI        = False    # True  → open the SUMO GUI window (great for demos)
NUM_SECONDS    = 3600     # simulated seconds per episode (≈ 1 hour of traffic)
DELTA_TIME     = 5        # seconds between agent decisions
YELLOW_TIME    = 2        # automatic yellow-phase buffer (seconds)
MIN_GREEN      = 5        # minimum green-phase duration (seconds)

# --- State discretisation ---------------------------------------------------
NUM_BINS = 4              # buckets per observation dimension (more = finer, slower)

# --- Q-Learning hyperparameters ---------------------------------------------
ALPHA         = 0.10      # learning rate  (how fast Q values shift on each update)
GAMMA         = 0.95      # discount factor (how much future rewards are valued)
EPS_START     = 1.0       # initial exploration probability
EPS_DECAY     = 0.98      # multiplicative decay applied after every episode
EPS_MIN       = 0.05      # exploration floor (agent never becomes fully greedy)

# --- Training ---------------------------------------------------------------
TRAIN_EPISODES = 15       # number of full simulation episodes to train on

# ---------------------------------------------------------------------------
# 2. REWARD FUNCTION
#    sumo-rl uses its own default (differential waiting-time) reward.
#    Uncomment the block below and pass reward_fn=custom_reward_fn to
#    SumoEnvironment to override it with the quadratic queue-length penalty
#    described in the project spec.  Both are shown here so you can easily
#    swap them for a comparative experiment in your report.
# ---------------------------------------------------------------------------

def custom_reward_fn(traffic_signal) -> float:
    """
    Penalises the *square* of the total queue length across all controlled
    lanes, which amplifies the cost of large backlogs more aggressively than
    the linear default.  This encourages the agent to clear heavy queues
    quickly even at the expense of lighter ones.

    Parameters
    ----------
    traffic_signal : sumo_rl.TrafficSignal
        The TrafficSignal object injected by sumo-rl for the current agent.

    Returns
    -------
    float
        A negative scalar; closer to 0 is better.
    """
    total_queue: float = sum(
        traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)
        for lane in traffic_signal.lanes
    )
    return -(total_queue ** 2)

# To use the custom reward, change the line below from None to custom_reward_fn:
#   REWARD_FN = custom_reward_fn
REWARD_FN = None           # None → use sumo-rl's built-in differential reward

# ---------------------------------------------------------------------------
# 3. STATE DISCRETISATION
# ---------------------------------------------------------------------------

def discretize_state(obs: np.ndarray, num_bins: int = NUM_BINS) -> tuple:
    """
    Convert sumo-rl's continuous observation vector into a discrete tuple
    that can serve as a dictionary key in the Q-table.

    sumo-rl's default observation for each controlled lane is a float in
    [0, 1] representing normalised density and queue length.  We partition
    [0, 1] into `num_bins` equal-width buckets using numpy's digitize, then
    return the resulting integer indices as an immutable tuple.

    Parameters
    ----------
    obs      : 1-D numpy array of floats, all values in [0.0, 1.0]
    num_bins : number of buckets  (4 → indices 0-3)

    Returns
    -------
    tuple of ints, e.g. (0, 2, 1, 3, …)

    Examples
    --------
    >>> discretize_state(np.array([0.0, 0.49, 0.51, 1.0]), num_bins=4)
    (0, 1, 2, 3)
    """
    # Produce (num_bins - 1) interior boundary points, e.g. [0.25, 0.5, 0.75]
    # for num_bins=4.  np.digitize then assigns each value to bucket 0..3.
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]

    # Clip guarantees that values exactly at 1.0 land in the last bin,
    # not an out-of-range index.
    discrete = tuple(
        int(np.clip(np.digitize(x, bin_edges), 0, num_bins - 1))
        for x in obs
    )
    return discrete

# ---------------------------------------------------------------------------
# 4. Q-TABLE HELPERS
# ---------------------------------------------------------------------------

def get_q_values(q_table: dict, state: tuple, num_actions: int) -> np.ndarray:
    """Return the Q-value row for `state`, initialising to zeros on first visit."""
    if state not in q_table:
        q_table[state] = np.zeros(num_actions, dtype=np.float64)
    return q_table[state]


def select_action(
    q_table: dict,
    state: tuple,
    num_actions: int,
    epsilon: float,
) -> int:
    """
    Epsilon-greedy policy:
      - With probability epsilon → sample a random action (exploration).
      - Otherwise               → choose the action with the highest Q-value
                                  (exploitation), breaking ties at random.
    """
    if np.random.rand() < epsilon:
        return int(np.random.randint(num_actions))

    q_values = get_q_values(q_table, state, num_actions)
    # np.argmax returns the first max; adding small noise breaks ties uniformly
    return int(np.argmax(q_values + np.random.randn(num_actions) * 1e-9))


def bellman_update(
    q_table: dict,
    state: tuple,
    action: int,
    reward: float,
    next_state: tuple,
    num_actions: int,
    alpha: float,
    gamma: float,
) -> None:
    """
    Apply the Bellman equation in-place:

        Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]

    Parameters
    ----------
    q_table    : dict mapping state tuples to numpy Q-value arrays
    state      : current state s
    action     : action taken a
    reward     : observed reward r
    next_state : resulting state s'
    num_actions: size of the action space (needed to initialise new states)
    alpha      : learning rate  α ∈ (0, 1]
    gamma      : discount factor γ ∈ [0, 1]
    """
    current_q  = get_q_values(q_table, state,      num_actions)[action]
    next_max_q = np.max(get_q_values(q_table, next_state, num_actions))

    td_error = reward + gamma * next_max_q - current_q
    get_q_values(q_table, state, num_actions)[action] += alpha * td_error

# ---------------------------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------------------------

def build_env(use_gui: bool) -> "sumo_rl.SumoEnvironment":   # type: ignore[name-defined]
    """Construct and return the sumo-rl environment."""
    from sumo_rl import SumoEnvironment  # imported here so the module is testable without SUMO

    kwargs: dict = dict(
        net_file     = NET_FILE,
        route_file   = ROUTE_FILE,
        out_csv_name = CSV_PREFIX,    # sumo-rl appends episode number and .csv
        single_agent = True,
        use_gui      = use_gui,
        num_seconds  = NUM_SECONDS,
        delta_time   = DELTA_TIME,
        yellow_time  = YELLOW_TIME,
        min_green    = MIN_GREEN,
    )

    # Attach custom reward only when the user has opted in (REWARD_FN is not None).
    if REWARD_FN is not None:
        kwargs["reward_fn"] = REWARD_FN

    return SumoEnvironment(**kwargs)


def train(use_gui: bool = USE_GUI) -> dict:
    """
    Run the full Q-learning training loop.

    Returns
    -------
    dict  – the final Q-table (state tuple → numpy array of Q-values)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = build_env(use_gui)
    num_actions = env.action_space.n

    q_table: dict = {}
    epsilon: float = EPS_START
    rewards_per_episode: list[float] = []

    print(
        f"\n{'='*60}\n"
        f"  Traffic-Light Q-Learning\n"
        f"  Episodes   : {TRAIN_EPISODES}\n"
        f"  Alpha/Gamma: {ALPHA} / {GAMMA}\n"
        f"  Bins       : {NUM_BINS}  |  Actions: {num_actions}\n"
        f"  Reward fn  : {'custom (quadratic queue)' if REWARD_FN else 'sumo-rl default'}\n"
        f"{'='*60}\n"
    )

    for episode in range(1, TRAIN_EPISODES + 1):
        # ── Reset ────────────────────────────────────────────────────────────
        obs, info = env.reset()
        state = discretize_state(obs)

        episode_reward = 0.0
        terminated = False
        truncated  = False
        step_count = 0

        # ── Episode loop ─────────────────────────────────────────────────────
        while not (terminated or truncated):
            action = select_action(q_table, state, num_actions, epsilon)

            # Standard Gymnasium 5-tuple unpacking (gymnasium >= 0.26)
            next_obs, reward, terminated, truncated, info = env.step(action)

            next_state = discretize_state(next_obs)

            bellman_update(
                q_table    = q_table,
                state      = state,
                action     = action,
                reward     = reward,
                next_state = next_state,
                num_actions= num_actions,
                alpha      = ALPHA,
                gamma      = GAMMA,
            )

            state          = next_state
            episode_reward += reward
            step_count     += 1

        # ── End-of-episode bookkeeping ────────────────────────────────────
        rewards_per_episode.append(episode_reward)

        # Decay exploration — never below the floor
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

        unique_states = len(q_table)
        print(
            f"  Ep {episode:3d}/{TRAIN_EPISODES}"
            f"  reward={episode_reward:10.2f}"
            f"  ε={epsilon:.3f}"
            f"  steps={step_count:5d}"
            f"  Q-states={unique_states:6d}"
        )

    env.close()

    # ── Persist Q-table ───────────────────────────────────────────────────────
    # We save as a numpy object array so it round-trips via np.load(allow_pickle=True)
    np.save(QTABLE_PATH, np.array(q_table, dtype=object))
    print(f"\n  Q-table saved → {QTABLE_PATH}  ({len(q_table)} states)")

    # ── Plot & return ─────────────────────────────────────────────────────────
    plot_training_curve(rewards_per_episode)
    return q_table

# ---------------------------------------------------------------------------
# 6. PLOTTING
# ---------------------------------------------------------------------------

def plot_training_curve(rewards: list[float]) -> None:
    """
    Save a PNG showing raw episode rewards and a smoothed trend line.
    The smoothing window is auto-sized to roughly 20 % of the episode count
    so the chart looks clean for both short and long runs.
    """
    n      = len(rewards)
    window = max(1, n // 5)          # at least 1 to avoid a zero-width convolution
    kernel = np.ones(window) / window
    smooth = np.convolve(rewards, kernel, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        range(n), rewards,
        color="#5a9fd4", alpha=0.3, lw=0.8, label="Raw reward",
    )
    ax.plot(
        range(window - 1, n), smooth,
        color="#5a9fd4", lw=2.5, label=f"Smoothed ({window}-ep rolling mean)",
    )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward  (higher = less waiting)", fontsize=12)
    ax.set_title(
        "Q-Learning Training Curve — Single Intersection (sumo-rl)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"  Training curve saved → {PLOT_PATH}")

# ---------------------------------------------------------------------------
# 7. ENTRY POINT
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tabular Q-Learning for SUMO traffic-light control.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=USE_GUI,
        help="Open the SUMO GUI window (useful for demos, slow for training).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=TRAIN_EPISODES,
        metavar="N",
        help=f"Number of training episodes (default: {TRAIN_EPISODES}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Allow the episode count to be overridden from the CLI without editing
    # the source file — handy for quick smoke-tests (--episodes 2) or long
    # overnight runs (--episodes 200).
    TRAIN_EPISODES = args.episodes   # shadows the module-level constant

    train(use_gui=args.gui)
