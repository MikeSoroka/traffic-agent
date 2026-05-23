import os
import importlib.util as _ilu
from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# 1. CONFIGURATION SYSTEM
# ---------------------------------------------------------------------------


@dataclass
class GlobalConfig:
    # --- Master Seed for Reproducibility ---
    MASTER_SEED: int = 0

    # --- Q-Learning Hyperparameters ---
    ALPHA: float = 0.10  # learning rate (alpha)
    GAMMA: float = 0.95  # discount factor (gamma)
    EPS_START: float = 1.0  # initial exploration rate
    EPS_DECAY: float = 0.98  # per-episode multiplicative decay
    EPS_MIN: float = 0.05  # exploration floor
    NUM_BINS: int = 4  # state discretisation bins

    # --- Simulation parameters ---
    NUM_SECONDS: int = 3600  # simulated seconds per episode
    DELTA_TIME: int = 5  # seconds between agent decisions
    YELLOW_TIME: int = 2  # automatic yellow-phase buffer
    MIN_GREEN: int = 5  # minimum green-phase duration

    # --- Intersection & Traffic Intensity ---
    NUM_LANES: int = (
        4  # Number of lanes at the single-intersection (defined in SUMO net)
    )
    TRAFFIC_INTENSITY: float = 1.0  # Scale factor for stochastically generated flows

    # --- Output Directory Layout ---
    OUTPUT_DIR: str = "outputs"
    DATA_DIR: str = "outputs/data"
    MODELS_DIR: str = "outputs/models"
    PLOTS_DIR: str = "outputs/plots"

    TRAIN_EPISODES: int = 15


# Instantiate the global configuration
CONFIG = GlobalConfig()

# Create required directories
for d in [CONFIG.OUTPUT_DIR, CONFIG.DATA_DIR, CONFIG.MODELS_DIR, CONFIG.PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Network files (resolved from the installed sumo_rl package) -----------
_PKG_SPEC = _ilu.find_spec("sumo_rl")
if _PKG_SPEC is None or _PKG_SPEC.origin is None:
    raise ImportError("sumo_rl package not found. Please install dependencies.")
_PKG_DIR = os.path.dirname(_PKG_SPEC.origin)
_NETS = os.path.join(_PKG_DIR, "nets", "single-intersection")
NET_FILE = os.path.join(_NETS, "single-intersection.net.xml")
ROUTE_FILE = os.path.join(_NETS, "single-intersection.rou.xml")

# ---------------------------------------------------------------------------
# 2. STATE DISCRETISATION
# ---------------------------------------------------------------------------


def discretize_state(obs: np.ndarray, num_bins: int = CONFIG.NUM_BINS) -> tuple:
    """
    Map a continuous observation vector ∈ [0,1]^n to a discrete tuple of
    bin indices ∈ {0, …, num_bins-1}^n usable as a Q-table dictionary key.
    """
    edges = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    discrete = tuple(int(np.clip(np.digitize(x, edges), 0, num_bins - 1)) for x in obs)
    return discrete


# ---------------------------------------------------------------------------
# 3. ACTION SELECTION
# ---------------------------------------------------------------------------


def _q(q_table: dict, state: tuple, n_actions: int) -> np.ndarray:
    """Return the Q-value row for `state`, lazily initialised to zeros."""
    if state not in q_table:
        q_table[state] = np.zeros(n_actions, dtype=np.float64)
    return q_table[state]


def select_action(q_table: dict, state: tuple, n_actions: int, epsilon: float) -> int:
    """Epsilon-greedy policy; ties in argmax broken by tiny random noise."""
    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))
    q_vals = _q(q_table, state, n_actions)
    return int(np.argmax(q_vals + np.random.randn(n_actions) * 1e-9))


# ---------------------------------------------------------------------------
# 4. BELLMAN UPDATE
# ---------------------------------------------------------------------------


def bellman_update(
    q_table: dict,
    state: tuple,
    action: int,
    reward: float,
    next_state: tuple,
    n_actions: int,
) -> None:
    """
    In-place Bellman update:
        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
    """
    current = _q(q_table, state, n_actions)[action]
    next_max = np.max(_q(q_table, next_state, n_actions))
    _q(q_table, state, n_actions)[action] += CONFIG.ALPHA * (
        reward + CONFIG.GAMMA * next_max - current
    )


# ---------------------------------------------------------------------------
# 5. ENVIRONMENT BUILDER
# ---------------------------------------------------------------------------


def build_env(config_name: str, reward_fn, use_gui: bool = False):
    """Construct and return a configured SumoEnvironment."""
    from sumo_rl import SumoEnvironment  # deferred so the file imports without SUMO

    csv_prefix = os.path.join(CONFIG.DATA_DIR, f"{config_name}_sumo")
    kwargs: dict = dict(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        out_csv_name=csv_prefix,
        single_agent=True,
        use_gui=use_gui,
        num_seconds=CONFIG.NUM_SECONDS,
        delta_time=CONFIG.DELTA_TIME,
        yellow_time=CONFIG.YELLOW_TIME,
        min_green=CONFIG.MIN_GREEN,
    )
    if reward_fn is not None:
        kwargs["reward_fn"] = reward_fn
    return SumoEnvironment(**kwargs)
