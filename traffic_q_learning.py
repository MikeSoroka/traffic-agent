"""
traffic_q_learning.py
=====================
Tabular Q-Learning for smart traffic-light control via sumo-rl / Gymnasium.

Supports two named reward configurations run back-to-back for comparative
analysis, meeting university project requirements:
  - Training reward curves
  - Per-episode queue length & waiting time metrics
  - Signal-schedule Gantt + traffic-state heatmap (episode 1 vs final)
  - Q-table value & preferred-action heatmap
  - Final-episode bar-chart comparison

Quick-start
-----------
  conda activate traffic-rl
  export SUMO_HOME=/usr/share/sumo

  # Single config (default reward):
  python traffic_q_learning.py

  # Compare both reward functions (required for report):
  python traffic_q_learning.py --compare

  # Fewer episodes for a quick smoke-test:
  python traffic_q_learning.py --compare --episodes 5

Outputs (all written to outputs/)
----------------------------------
  reward_comparison.png
  performance_metrics.png
  signal_schedule_<config>_ep1.png
  signal_schedule_<config>_final.png
  q_table_heatmap_<config>.png
  final_comparison.png          (only with --compare)
  <config>_conn0_ep<N>.csv      (per-step SUMO metrics, one file per episode)
  q_table_<config>.npy
"""

from __future__ import annotations

import argparse
import glob
import importlib.util as _ilu
import os
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. CONFIGURABLE PARAMETERS
# ---------------------------------------------------------------------------

# --- Network files (resolved from the installed sumo_rl package) -----------
_PKG_DIR = os.path.dirname(_ilu.find_spec("sumo_rl").origin)
_NETS    = os.path.join(_PKG_DIR, "nets", "single-intersection")
NET_FILE   = os.path.join(_NETS, "single-intersection.net.xml")
ROUTE_FILE = os.path.join(_NETS, "single-intersection.rou.xml")

# --- Output directory -------------------------------------------------------
OUTPUT_DIR = "outputs"

# --- Simulation parameters --------------------------------------------------
USE_GUI     = False   # override with --gui
NUM_SECONDS = 3600    # simulated seconds per episode
DELTA_TIME  = 5       # seconds between agent decisions
YELLOW_TIME = 2       # automatic yellow-phase buffer
MIN_GREEN   = 5       # minimum green-phase duration

# --- State discretisation ---------------------------------------------------
NUM_BINS = 4          # bins per observation dimension  (0 … NUM_BINS-1)

# --- Q-Learning hyperparameters ---------------------------------------------
ALPHA     = 0.10      # learning rate
GAMMA     = 0.95      # discount factor
EPS_START = 1.0       # initial exploration rate
EPS_DECAY = 0.98      # per-episode multiplicative decay
EPS_MIN   = 0.05      # exploration floor

# --- Training ---------------------------------------------------------------
TRAIN_EPISODES = 15   # override with --episodes N

# ---------------------------------------------------------------------------
# 2. REWARD FUNCTIONS
# ---------------------------------------------------------------------------

def _default_reward(ts) -> float:
    """
    Placeholder — None is passed to SumoEnvironment so it uses its built-in
    differential waiting-time reward.  This function is never actually called;
    it exists only to document what the default computes:

        r = Σ(accumulated_waiting_time_prev) − Σ(accumulated_waiting_time_now)

    A positive value means total waiting decreased this step.
    """
    raise NotImplementedError("Pass reward_fn=None to use sumo-rl's built-in reward.")


def quadratic_queue_reward(ts) -> float:
    """
    Penalise the *square* of total queue length, amplifying the cost of large
    backlogs.  The agent learns to clear heavy queues quickly even at the cost
    of lighter ones — a different trade-off than the differential reward.

    r = −(Σ halting_vehicles_per_lane)²
    """
    total_queue: float = sum(
        ts.sumo.lane.getLastStepHaltingNumber(lane)
        for lane in ts.lanes
    )
    return -(total_queue ** 2)


# Named configurations used for comparative experiments.
# Add more entries here to run additional comparisons without touching
# any other part of the script.
CONFIGS: dict[str, dict] = {
    "default": {
        "label":     "Default Reward (Δ waiting time)",
        "color":     "#5a9fd4",
        "reward_fn": None,                  # sumo-rl built-in
    },
    "quadratic": {
        "label":     "Quadratic Queue Penalty",
        "color":     "#f0a050",
        "reward_fn": quadratic_queue_reward,
    },
}

# ---------------------------------------------------------------------------
# 3. STATE DISCRETISATION
# ---------------------------------------------------------------------------

def discretize_state(obs: np.ndarray, num_bins: int = NUM_BINS) -> tuple:
    """
    Map a continuous observation vector ∈ [0,1]^n to a discrete tuple of
    bin indices ∈ {0, …, num_bins-1}^n usable as a Q-table dictionary key.

    np.digitize assigns values to buckets defined by (num_bins-1) interior
    boundary points, e.g. [0.25, 0.5, 0.75] for num_bins=4.
    np.clip handles the edge case value == 1.0 (which digitize sends to
    bucket num_bins, one past the last valid index).
    """
    edges    = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    discrete = tuple(
        int(np.clip(np.digitize(x, edges), 0, num_bins - 1))
        for x in obs
    )
    return discrete

# ---------------------------------------------------------------------------
# 4. Q-TABLE HELPERS
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
    current  = _q(q_table, state,      n_actions)[action]
    next_max = np.max(_q(q_table, next_state, n_actions))
    _q(q_table, state, n_actions)[action] += ALPHA * (reward + GAMMA * next_max - current)

# ---------------------------------------------------------------------------
# 5. EPISODE RECORDER
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    """
    Lightweight container for one episode's step-by-step data.
    Stored only for the first and final episodes to limit memory use.
    """
    actions:       list = field(default_factory=list)
    observations:  list = field(default_factory=list)
    rewards:       list = field(default_factory=list)
    total_stopped: list = field(default_factory=list)
    mean_wait:     list = field(default_factory=list)

    def record(self, action: int, obs: np.ndarray, reward: float, info: dict) -> None:
        self.actions.append(action)
        self.observations.append(obs.copy())
        self.rewards.append(reward)
        self.total_stopped.append(float(info.get("system_total_stopped", 0)))
        self.mean_wait.append(float(info.get("system_mean_waiting_time", 0.0)))

    @property
    def obs_array(self) -> np.ndarray:
        return np.array(self.observations)          # shape (T, obs_dim)

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

# ---------------------------------------------------------------------------
# 6. ENVIRONMENT BUILDER
# ---------------------------------------------------------------------------

def build_env(config_name: str, reward_fn, use_gui: bool = False):
    """Construct and return a configured SumoEnvironment."""
    from sumo_rl import SumoEnvironment  # deferred so the file imports without SUMO

    csv_prefix = os.path.join(OUTPUT_DIR, config_name)
    kwargs: dict = dict(
        net_file     = NET_FILE,
        route_file   = ROUTE_FILE,
        out_csv_name = csv_prefix,
        single_agent = True,
        use_gui      = use_gui,
        num_seconds  = NUM_SECONDS,
        delta_time   = DELTA_TIME,
        yellow_time  = YELLOW_TIME,
        min_green    = MIN_GREEN,
    )
    if reward_fn is not None:
        kwargs["reward_fn"] = reward_fn
    return SumoEnvironment(**kwargs)

# ---------------------------------------------------------------------------
# 7. TRAINING LOOP
# ---------------------------------------------------------------------------

def train(
    config_name: str,
    episodes: int = TRAIN_EPISODES,
    use_gui: bool = USE_GUI,
) -> dict:
    """
    Train a tabular Q-learning agent for one named configuration.

    Returns a results dict containing:
        rewards          – list of per-episode total rewards
        first_record     – EpisodeRecord for episode 1
        last_record      – EpisodeRecord for episode `episodes`
        q_table          – final Q-table dict
        episode_metrics  – list of per-episode summary dicts (from CSV)
        config_name      – the config key used
        config           – the config dict
    """
    cfg = CONFIGS[config_name]
    env = build_env(config_name, cfg["reward_fn"], use_gui)
    n_actions = env.action_space.n

    q_table: dict = {}
    epsilon       = EPS_START
    rewards_hist: list[float] = []
    first_record: EpisodeRecord | None = None
    last_record:  EpisodeRecord | None = None

    print(
        f"\n{'='*62}\n"
        f"  Config : {config_name}  —  {cfg['label']}\n"
        f"  Episodes: {episodes}   α={ALPHA}  γ={GAMMA}  "
        f"bins={NUM_BINS}  actions={n_actions}\n"
        f"{'='*62}"
    )

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        state     = discretize_state(obs)
        record    = EpisodeRecord() if ep in (1, episodes) else None

        ep_reward  = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action                               = select_action(q_table, state, n_actions, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state                           = discretize_state(next_obs)

            if record is not None:
                record.record(action, obs, reward, info)

            bellman_update(q_table, state, action, reward, next_state, n_actions)

            state      = next_state
            obs        = next_obs
            ep_reward += reward

        if ep == 1:
            first_record = record
        if ep == episodes:
            last_record = record

        rewards_hist.append(ep_reward)
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

        print(
            f"  Ep {ep:3d}/{episodes}"
            f"  reward={ep_reward:10.2f}"
            f"  ε={epsilon:.3f}"
            f"  steps={len(record.actions) if record else '?':>5}"
            f"  Q-states={len(q_table):6d}"
        )

    env.close()

    # Persist Q-table
    qtable_path = f"q_table_{config_name}.npy"
    np.save(qtable_path, np.array(q_table, dtype=object))
    print(f"  Q-table saved → {qtable_path}  ({len(q_table)} states)")

    # Parse sumo-rl CSV files for per-episode summary metrics
    episode_metrics = _parse_csv_metrics(config_name, episodes)

    return dict(
        config_name     = config_name,
        config          = cfg,
        rewards         = rewards_hist,
        first_record    = first_record,
        last_record     = last_record,
        q_table         = q_table,
        n_actions       = n_actions,
        episode_metrics = episode_metrics,
    )

# ---------------------------------------------------------------------------
# 8. CSV PARSING
# ---------------------------------------------------------------------------

def _parse_csv_metrics(config_name: str, episodes: int) -> list[dict]:
    """
    Read the per-step CSV files written by sumo-rl and compute per-episode
    aggregate metrics: avg/max queue length and average waiting time.

    sumo-rl names files as: {OUTPUT_DIR}/{config_name}_conn0_ep{N}.csv
    (N is 1-indexed to match how we call env.reset() per episode).
    """
    prefix = os.path.join(OUTPUT_DIR, config_name)
    results = []
    for ep_idx in range(1, episodes + 1):
        pattern = f"{prefix}_conn*_ep{ep_idx}.csv"
        files   = sorted(glob.glob(pattern))
        if not files:
            continue
        df = pd.read_csv(files[0])
        results.append({
            "episode":          ep_idx,
            "avg_queue":        df["system_total_stopped"].mean(),
            "max_queue":        df["system_total_stopped"].max(),
            "avg_waiting_time": df["system_mean_waiting_time"].mean(),
        })
    return results

# ---------------------------------------------------------------------------
# 9. PLOTTING
# ---------------------------------------------------------------------------

def _smooth(values: list[float], frac: float = 0.20) -> tuple[np.ndarray, int]:
    """Return (smoothed_array, window_size) using a rolling mean."""
    window = max(1, int(len(values) * frac))
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    return smoothed, window


def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── 9a. Training reward curves ──────────────────────────────────────────────

def plot_training_rewards(results: list[dict]) -> None:
    """Overlay smoothed reward curves for every configuration."""
    fig, ax = plt.subplots(figsize=(11, 5))

    for r in results:
        rewards    = r["rewards"]
        color      = r["config"]["color"]
        label      = r["config"]["label"]
        smooth, w  = _smooth(rewards)
        n          = len(rewards)

        ax.plot(range(n), rewards, color=color, alpha=0.18, lw=0.8)
        ax.plot(range(w - 1, n), smooth, color=color, lw=2.5,
                label=f"{label}  (rolling mean, w={w})")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward  (higher → less waiting)", fontsize=12)
    ax.set_title("Q-Learning Training Curves — Configuration Comparison", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "reward_comparison.png")


# ── 9b. Per-episode performance metrics ─────────────────────────────────────

def plot_performance_metrics(results: list[dict]) -> None:
    """
    Two-panel figure:
      Left  — avg queue per episode (shaded band = avg to max)
      Right — avg waiting time per episode
    """
    has_metrics = [r for r in results if r["episode_metrics"]]
    if not has_metrics:
        print("  No CSV metrics found — skipping performance_metrics.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in has_metrics:
        m     = r["episode_metrics"]
        eps   = [x["episode"]          for x in m]
        avg_q = [x["avg_queue"]        for x in m]
        max_q = [x["max_queue"]        for x in m]
        avg_w = [x["avg_waiting_time"] for x in m]
        c     = r["config"]["color"]
        lbl   = r["config"]["label"]

        axes[0].plot(eps, avg_q, color=c, lw=2.2, label=lbl)
        axes[0].fill_between(eps, avg_q, max_q, color=c, alpha=0.13,
                             label=f"{lbl} (max band)")

        axes[1].plot(eps, avg_w, color=c, lw=2.2, label=lbl)

    for ax, title, ylabel in [
        (axes[0], "Queue Length per Episode",      "Vehicles stopped (avg / max band)"),
        (axes[1], "Mean Waiting Time per Episode", "Waiting time (s)"),
    ]:
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "performance_metrics.png")


# ── 9c. Signal-schedule Gantt + traffic-state heatmap ───────────────────────

_PHASE_COLORS = {0: "#4a90d9", 1: "#e8872a"}
_PHASE_LABELS = {0: "NS Green", 1: "EW Green"}


def plot_signal_schedule(
    record: EpisodeRecord | None,
    title: str,
    filename: str,
) -> None:
    """
    Three-panel figure for one episode:
      Top    — Gantt chart: which phase was active at each decision step
      Middle — total stopped vehicles over time
      Bottom — heatmap of the full observation vector over time
                 (rows = observation features, columns = decision steps)
                 colours show normalised density / queue per lane

    This visualises what the agent 'sees' and how it responds across an
    entire episode, making it easy to compare episode 1 (random) vs final
    (trained) in a presentation.
    """
    if record is None:
        return

    actions   = record.actions
    stopped   = record.total_stopped
    obs_arr   = record.obs_array    # (T, obs_dim)
    T         = len(actions)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [1, 1, 2.5]},
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    # ── Top: phase Gantt ─────────────────────────────────────────────────────
    ax = axes[0]
    t  = 0
    while t < T:
        a  = actions[t]
        t2 = t + 1
        while t2 < T and actions[t2] == a:
            t2 += 1
        ax.barh(
            0, t2 - t, left=t, height=0.55,
            color=_PHASE_COLORS[a % 2], alpha=0.85,
            edgecolor="white", linewidth=0.4,
        )
        t = t2

    ax.set_xlim(0, T)
    ax.set_yticks([])
    ax.set_ylabel("Phase", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    patches = [
        mpatches.Patch(color=_PHASE_COLORS[i], label=_PHASE_LABELS[i])
        for i in (0, 1)
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.8)

    # ── Middle: stopped vehicles ─────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(range(T), stopped, color="#c06060", alpha=0.55, label="Stopped vehicles")
    ax.plot(range(T), stopped, color="#c06060", lw=0.9)
    ax.set_xlim(0, T)
    ax.set_ylabel("Stopped vehicles", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ── Bottom: observation heatmap ──────────────────────────────────────────
    ax = axes[2]
    if obs_arr.ndim == 2 and obs_arr.shape[0] > 0:
        n_feat  = obs_arr.shape[1]
        # sumo-rl appends phase/time bits after the per-lane values;
        # split density vs queue vs extra conservatively.
        n_lanes = n_feat // 2
        feat_labels = (
            [f"Density L{i+1}" for i in range(n_lanes)] +
            [f"Queue   L{i+1}" for i in range(n_lanes)] +
            [f"Extra   {i+1}"  for i in range(n_feat - 2 * n_lanes)]
        )

        im = ax.imshow(
            obs_arr.T,            # rows=features, cols=time
            aspect="auto",
            cmap="YlOrRd",
            vmin=0.0, vmax=1.0,
            origin="lower",
            extent=[-0.5, T - 0.5, -0.5, n_feat - 0.5],
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="Normalised value [0, 1]", fraction=0.025)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(feat_labels, fontsize=7)

    ax.set_xlabel("Decision step", fontsize=9)
    ax.set_ylabel("Observation feature", fontsize=9)
    ax.set_title(
        "Traffic-state heatmap  (lane density & queue over time)",
        fontsize=9,
    )

    fig.tight_layout()
    _save(fig, filename)


# ── 9d. Q-table heatmap ─────────────────────────────────────────────────────

def plot_q_table_heatmap(result: dict) -> None:
    """
    Two scatter-plot panels projecting visited Q-table states onto the first
    two observation dimensions:
      Left  — colour = max Q-value (state value estimate)
      Right — colour = preferred action (policy)

    Other dimensions are compressed by taking the mean Q-value across them,
    giving an intuitive summary of what the agent learned.
    """
    q_table   = result["q_table"]
    n_actions = result["n_actions"]
    cname     = result["config_name"]

    if not q_table:
        return

    states   = list(q_table.keys())
    q_vals   = np.array([q_table[s] for s in states])   # (N, n_actions)
    max_q    = q_vals.max(axis=1)
    best_act = q_vals.argmax(axis=1)
    dim0     = np.array([s[0] for s in states])
    dim1     = np.array([s[1] for s in states])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Q-Table Visualisation — {result['config']['label']}\n"
        f"({len(states)} visited states, projected to dims 0 & 1)",
        fontsize=11,
    )

    jitter = np.random.rand(len(states)) * 0.35 - 0.175   # avoid full overlap

    sc1 = axes[0].scatter(
        dim0 + jitter, dim1 + jitter,
        c=max_q, cmap="viridis", s=55, alpha=0.75,
    )
    plt.colorbar(sc1, ax=axes[0], label="max Q(s, ·)  (state value)")
    axes[0].set_title("State Value Estimate")
    axes[0].set_xlabel("Dim 0 — lane 1 density bin")
    axes[0].set_ylabel("Dim 1 — lane 2 density bin")
    axes[0].set_xticks(range(NUM_BINS))
    axes[0].set_yticks(range(NUM_BINS))
    axes[0].grid(alpha=0.3)

    sc2 = axes[1].scatter(
        dim0 + jitter, dim1 + jitter,
        c=best_act, cmap="coolwarm", s=55, alpha=0.75,
        vmin=0, vmax=max(n_actions - 1, 1),
    )
    cbar2 = plt.colorbar(sc2, ax=axes[1], label="Preferred action")
    cbar2.set_ticks([0, 1])
    cbar2.set_ticklabels(["NS Green (0)", "EW Green (1)"])
    axes[1].set_title("Preferred Action (Greedy Policy)")
    axes[1].set_xlabel("Dim 0 — lane 1 density bin")
    axes[1].set_ylabel("Dim 1 — lane 2 density bin")
    axes[1].set_xticks(range(NUM_BINS))
    axes[1].set_yticks(range(NUM_BINS))
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, f"q_table_heatmap_{cname}.png")


# ── 9e. Final-episode comparison bar chart ───────────────────────────────────

def plot_final_comparison(results: list[dict]) -> None:
    """
    Bar chart comparing the *last* episode's aggregate metrics across all
    configurations — the primary quantitative result for the report.
    """
    valid = [r for r in results if r["episode_metrics"]]
    if len(valid) < 2:
        return

    metric_keys   = ["avg_queue",   "max_queue",   "avg_waiting_time"]
    metric_labels = ["Avg Queue\n(vehicles)", "Max Queue\n(vehicles)", "Avg Wait\n(seconds)"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Final-Episode Performance Comparison", fontsize=13, fontweight="bold")

    for ax, mkey, mlabel in zip(axes, metric_keys, metric_labels):
        labels = [r["config"]["label"] for r in valid]
        values = [r["episode_metrics"][-1][mkey]  for r in valid]
        colors = [r["config"]["color"]             for r in valid]

        bars = ax.bar(range(len(labels)), values, color=colors, width=0.5, zorder=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=12, ha="right")
        ax.set_title(mlabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3, zorder=0)

        # Value labels above bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(values) * 0.02,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    fig.tight_layout()
    _save(fig, "final_comparison.png")

# ---------------------------------------------------------------------------
# 10. ORCHESTRATION
# ---------------------------------------------------------------------------

def run_all(config_names: list[str], episodes: int, use_gui: bool) -> list[dict]:
    """Train all requested configurations sequentially and generate plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results: list[dict] = []
    for cname in config_names:
        result = train(cname, episodes, use_gui)
        results.append(result)

    print(f"\n  {'─'*50}")
    print("  Generating plots …")

    # Reward curves
    plot_training_rewards(results)

    # Per-episode metrics
    plot_performance_metrics(results)

    # Per-config: signal schedules + Q-table heatmap
    for r in results:
        cname = r["config_name"]
        lbl   = r["config"]["label"]

        plot_signal_schedule(
            r["first_record"],
            f"Signal Schedule — Episode 1  [{lbl}]",
            f"signal_schedule_{cname}_ep1.png",
        )
        plot_signal_schedule(
            r["last_record"],
            f"Signal Schedule — Final Episode  [{lbl}]",
            f"signal_schedule_{cname}_final.png",
        )
        plot_q_table_heatmap(r)

    # Cross-config comparison (only meaningful with ≥ 2 configs)
    if len(results) > 1:
        plot_final_comparison(results)

    print(f"\n  All outputs written to  {OUTPUT_DIR}/")
    return results

# ---------------------------------------------------------------------------
# 11. ENTRY POINT
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tabular Q-Learning for SUMO traffic-light control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        default="default",
        help="Single reward configuration to train.",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run ALL configurations and generate comparison plots.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=TRAIN_EPISODES,
        metavar="N",
        help="Number of training episodes per configuration.",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        default=USE_GUI,
        help="Open the SUMO GUI window.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args         = _parse_args()
    config_names = list(CONFIGS.keys()) if args.compare else [args.config]
    run_all(config_names, args.episodes, args.gui)
