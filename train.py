"""
Training script — no pygame required.

Usage:
    python train.py

Produces:
    q_table.npy          — saved Q-table (load with agent.load())
    plots/training_curve.png
    plots/comparison.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    TRAIN_EPISODES, STEPS_PER_EP, EVAL_EPISODES,
    FIXED_CYCLE, QTABLE_PATH, PLOT_DIR,
)
from environment import TrafficEnv
from agent import QLearningAgent


# ── Episode runners ───────────────────────────────────────────────────────────

def run_ql_episode(env: TrafficEnv, agent: QLearningAgent, train: bool = True) -> float:
    """Run one episode.  If train=True, update Q-table; else greedy policy."""
    state = env.reset()
    total_r = 0.0
    for _ in range(STEPS_PER_EP):
        if env.is_yellow:
            state, r, _, _ = env.step(0)
        else:
            action = agent.act(state, greedy=not train)
            next_state, r, _, _ = env.step(action)
            if train:
                agent.update(state, action, r, next_state)
            state = next_state
        total_r += r
    return total_r


def run_fixed_episode(env: TrafficEnv, cycle: int = FIXED_CYCLE) -> float:
    """Fixed-time controller: switch phase every `cycle` green steps."""
    state = env.reset()
    total_r = 0.0
    green_steps = 0
    for _ in range(STEPS_PER_EP):
        if env.is_yellow:
            state, r, _, _ = env.step(0)
            green_steps = 0
        else:
            action = 1 if green_steps >= cycle else 0
            state, r, _, _ = env.step(action)
            if not env.is_yellow:
                green_steps += 1
        total_r += r
    return total_r


def run_random_episode(env: TrafficEnv) -> float:
    """Random policy baseline."""
    state = env.reset()
    total_r = 0.0
    for _ in range(STEPS_PER_EP):
        action = np.random.randint(2)
        state, r, _, _ = env.step(action)
        total_r += r
    return total_r


# ── Main training loop ────────────────────────────────────────────────────────

def train() -> QLearningAgent:
    os.makedirs(PLOT_DIR, exist_ok=True)

    env   = TrafficEnv(seed=0)
    agent = QLearningAgent()

    rewards: list[float] = []
    print(f"Training Q-learning agent  ({TRAIN_EPISODES} episodes × {STEPS_PER_EP} steps)")

    for ep in range(1, TRAIN_EPISODES + 1):
        r = run_ql_episode(env, agent, train=True)
        agent.decay_epsilon()
        rewards.append(r)
        if ep % 50 == 0 or ep == 1:
            print(f"  ep {ep:4d}/{TRAIN_EPISODES}   reward={r:8.1f}   ε={agent.eps:.3f}")

    agent.save(QTABLE_PATH)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\nEvaluating ({EVAL_EPISODES} greedy runs each)…")
    ql_r  = [run_ql_episode(TrafficEnv(seed=i+100), agent, train=False) for i in range(EVAL_EPISODES)]
    ft_r  = [run_fixed_episode(TrafficEnv(seed=i+100))                  for i in range(EVAL_EPISODES)]
    rnd_r = [run_random_episode(TrafficEnv(seed=i+100))                 for i in range(EVAL_EPISODES)]

    print(f"  Q-Learning  :  {np.mean(ql_r):8.1f}  ±  {np.std(ql_r):.1f}")
    print(f"  Fixed-time  :  {np.mean(ft_r):8.1f}  ±  {np.std(ft_r):.1f}")
    print(f"  Random      :  {np.mean(rnd_r):8.1f}  ±  {np.std(rnd_r):.1f}")

    _plot_training_curve(rewards)
    _plot_comparison(ql_r, ft_r, rnd_r)
    print(f"\nPlots written to  {PLOT_DIR}/")
    return agent


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_training_curve(rewards: list[float]):
    window = min(20, len(rewards) // 5)
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards, alpha=0.25, color="#5a9fd4", lw=0.8, label="Raw")
    ax.plot(range(window - 1, len(rewards)), smooth,
            color="#5a9fd4", lw=2.2, label=f"Smoothed ({window}-ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward  (higher = less waiting)")
    ax.set_title("Q-Learning Training Curve — 4-Way Intersection")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "training_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_comparison(ql: list, ft: list, rnd: list):
    labels = ["Q-Learning", f"Fixed-Time\n({FIXED_CYCLE}s cycle)", "Random"]
    means  = [np.mean(ql),  np.mean(ft),  np.mean(rnd)]
    stds   = [np.std(ql),   np.std(ft),   np.std(rnd)]
    colors = ["#5a9fd4", "#f0a050", "#c06060"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=7, color=colors, width=0.5, zorder=3)
    ax.set_ylabel("Average Total Reward  (higher = fewer stops)")
    ax.set_title("Policy Comparison — Greedy Evaluation")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + max(stds) * 0.05,
                f"{m:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


if __name__ == "__main__":
    train()
