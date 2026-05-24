import os
import argparse
import numpy as np
import pandas as pd
from config import CONFIG, build_env, discretize_state, select_action, bellman_update


def custom_queue_reward(ts) -> float:
    """Custom reward function for Q-learning"""
    # reward = previous queue - current queue
    current_queue = sum(
        float(ts.sumo.lane.getLastStepHaltingNumber(lane)) for lane in ts.lanes
    )

    previous_queue = getattr(ts, "_last_queue_size", None)

    # update the attribute for the next step
    ts._last_queue_size = current_queue

    # if first step
    if previous_queue is None:
        return 0.0

    return float(previous_queue - current_queue)


CONFIGS = {
    "baseline": {
        "label": "Baseline Reward (Δ waiting time)",
        "reward_fn": None,
    },
    "custom": {
        "label": "Custom Reward (Differential Queue)",
        "reward_fn": custom_queue_reward,
    },
}


def run_training(config_name: str, episodes: int, seed: int):
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown configuration name: {config_name}")

    cfg = CONFIGS[config_name]
    env = build_env(config_name, cfg["reward_fn"], use_gui=False)
    n_actions = env.action_space.n

    np.random.seed(seed)

    q_table = {}
    epsilon = CONFIG.EPS_START
    episode_summary_data = []

    print(
        f"Training config: {config_name}\n"
        f"Lanes: {CONFIG.NUM_LANES} | Traffic intensity: {CONFIG.TRAFFIC_INTENSITY}\n"
        f"Episodes: {episodes} | Alpha: {CONFIG.ALPHA} | Gamma: {CONFIG.GAMMA}\n"
        f"Epsilon start: {CONFIG.EPS_START} | Epsilon decay: {CONFIG.EPS_DECAY} | Epsilon min: {CONFIG.EPS_MIN}\n"
    )

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        state = discretize_state(obs)

        ep_reward = 0.0
        terminated = truncated = False
        step_count = 0

        step_telemetry = []

        while not (terminated or truncated):
            action = select_action(q_table, state, n_actions, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_obs)

            # update Q-table
            bellman_update(q_table, state, action, reward, next_state, n_actions)

            step_telemetry.append(
                {
                    "step": step_count,
                    "action": action,
                    "reward": reward,
                    "total_stopped": float(info.get("system_total_stopped", 0)),
                    "mean_waiting_time": float(
                        info.get("system_mean_waiting_time", 0.0)
                    ),
                    **{f"obs_{i}": float(val) for i, val in enumerate(obs)},
                }
            )

            state = next_state
            obs = next_obs
            ep_reward += reward
            step_count += 1

        # decay epsilon
        epsilon = max(CONFIG.EPS_MIN, epsilon * CONFIG.EPS_DECAY)

        # save step telemetry for this episode
        ep_df = pd.DataFrame(step_telemetry)
        ep_file_path = os.path.join(CONFIG.DATA_DIR, f"{config_name}_ep_{ep}.csv")
        ep_df.to_csv(ep_file_path, index=False)

        # save Q-table for this specific episode to enable playbacks at any episode in demo
        qtable_ep_path = os.path.join(
            CONFIG.MODELS_DIR, f"q_table_{config_name}_ep_{ep}.npy"
        )
        np.save(qtable_ep_path, np.array(q_table, dtype=object))

        # metrics for the episode
        ep_stopped = [row["total_stopped"] for row in step_telemetry]
        ep_wait = [row["mean_waiting_time"] for row in step_telemetry]
        avg_q = sum(ep_stopped) / len(ep_stopped) if ep_stopped else 0
        max_q = max(ep_stopped) if ep_stopped else 0
        avg_w = sum(ep_wait) / len(ep_wait) if ep_wait else 0

        # save to summary data
        episode_summary_data.append(
            {
                "episode": ep,
                "total_reward": ep_reward,
                "avg_queue": avg_q,
                "max_queue": max_q,
                "avg_waiting_time": avg_w,
                "epsilon": epsilon,
                "Q-states": len(q_table),
            }
        )

        print(
            f"Episode {ep:3d}/{episodes} | "
            f"Reward: {ep_reward:9.2f} | "
            f"Avg queue: {avg_q:5.2f} | "
            f"Avg wait: {avg_w:5.2f}s | "
            f"Epsilon: {epsilon:.2f} | "
            f"States: {len(q_table)}"
        )

    env.close()

    # save overall summary
    summary_df = pd.DataFrame(episode_summary_data)
    summary_file_path = os.path.join(CONFIG.DATA_DIR, f"{config_name}_summary.csv")
    summary_df.to_csv(summary_file_path, index=False)
    print(f"Saved episode summary telemetry to {summary_file_path}")

    # export the trained Q-table
    qtable_path = os.path.join(CONFIG.MODELS_DIR, f"q_table_{config_name}.npy")
    np.save(qtable_path, np.array(q_table, dtype=object))
    print(f"Saved trained model matrix to {qtable_path} ({len(q_table)} states)")


def main():
    p = argparse.ArgumentParser(
        description="Training runner for Q-Learning traffic controller."
    )
    p.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        default="baseline",
        help="The reward configuration to train.",
    )
    p.add_argument(
        "--compare", action="store_true", help="Train ALL configurations sequentially."
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=CONFIG.TRAIN_EPISODES,
        help="Number of training episodes.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=CONFIG.MASTER_SEED,
        help="Master seed for reproducibility.",
    )
    args = p.parse_args()

    configs_to_run = list(CONFIGS.keys()) if args.compare else [args.config]

    for cfg_name in configs_to_run:
        run_training(cfg_name, args.episodes, args.seed)


if __name__ == "__main__":
    main()
