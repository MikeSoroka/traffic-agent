import os
import argparse
import numpy as np
from config import CONFIG, build_env, discretize_state, select_action

# Define the configurations map to align with trained rewards
CONFIGS = {
    "default": {
        "label": "Default Reward (Δ waiting time)",
        "reward_fn": None,
    },
    "quadratic": {
        "label": "Quadratic Queue Penalty",
        "reward_fn": None,
    },
}


def evaluate(config_name: str, use_gui: bool, seed: int, episode: int | None = None):
    # 1. Select and load the appropriate Q-table model
    if episode is not None:
        model_name = f"q_table_{config_name}_ep_{episode}.npy"
        # We align the environment reset seed to matches the exact training episode vehicle arrivals
        eval_seed = seed + episode
        print(
            f"\n  [★] Presentation targeting Episode {episode} (Deterministic Seed: {eval_seed})"
        )
    else:
        model_name = f"q_table_{config_name}.npy"
        eval_seed = seed
        print(
            f"\n  [★] Presentation targeting Final Model (Deterministic Seed: {eval_seed})"
        )

    model_path = os.path.join(CONFIG.MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n[!] Trained Q-table model file not found at: {model_path}\n"
            f"    Please run training first to generate this model, e.g.:\n"
            f"    python src/train.py --config {config_name} --episodes {episode if episode else 15}"
        )

    print(f"  [★] Loading trained Q-table from: {model_path}")
    q_table = np.load(model_path, allow_pickle=True).item()
    print(f"  [★] Loaded model containing {len(q_table)} learned states.")

    # 2. Build the environment with GUI rendering enabled/disabled
    print(f"  [★] Starting SUMO Environment (use_gui={use_gui})...")
    env = build_env(config_name, reward_fn=None, use_gui=use_gui)
    n_actions = env.action_space.n

    # Reset with aligned seed to make live visual playback perfectly match the trained history
    obs, info = env.reset(seed=eval_seed)
    state = discretize_state(obs)

    terminated = truncated = False
    ep_reward = 0.0
    ep_stopped = []
    ep_wait = []
    step_count = 0

    print(
        f"\n============================================================\n"
        f"  [LIVE PRESENTATION DEMO] Config: {config_name.upper()}\n"
        f"  Running exploitation cycle (ε=0.0)... Close GUI window or wait to finish.\n"
        f"============================================================"
    )

    while not (terminated or truncated):
        try:
            # Force epsilon = 0.0 for pure exploitation (greedy policy) during the defense/demo
            action = select_action(q_table, state, n_actions, epsilon=0.0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_obs)

            state = next_state
            obs = next_obs
            ep_reward += reward
            step_count += 1

            # Track metrics
            ep_stopped.append(float(info.get("system_total_stopped", 0)))
            ep_wait.append(float(info.get("system_mean_waiting_time", 0.0)))

            # Console feedback every 50 steps
            if step_count % 50 == 0:
                print(
                    f"    Step {step_count:4d} | Stopped Queue: {ep_stopped[-1]:4.1f} | Average Wait: {ep_wait[-1]:4.1f}s"
                )
        except Exception as e:
            # Gracefully catch case where user closes the SUMO GUI window manually
            err_msg = str(e).lower()
            err_class = e.__class__.__name__.lower()
            if (
                "closed" in err_msg
                or "fatal" in err_msg
                or "traci" in err_msg
                or "constructed" in err_msg
                or "fatal" in err_class
                or "traci" in err_class
            ):
                print(
                    "\n  [★] SUMO GUI window closed by the user. Capturing metrics up to this step..."
                )
                break
            else:
                raise e

    try:
        env.close()
    except Exception:
        pass

    # 3. Print out presentation-ready summary metrics
    avg_queue = sum(ep_stopped) / len(ep_stopped) if ep_stopped else 0
    max_queue = max(ep_stopped) if ep_stopped else 0
    avg_wait = sum(ep_wait) / len(ep_wait) if ep_wait else 0

    print(
        f"\n============================================================\n"
        f"  DEMO COMPLETE: {config_name.upper()} RESULTS\n"
        f"============================================================\n"
        f"  Total Steps Completed: {step_count}\n"
        f"  Cumulative Reward:     {ep_reward:.2f}\n"
        f"  Average Queue Length:  {avg_queue:.2f} vehicles\n"
        f"  Maximum Queue Length:  {max_queue:.2f} vehicles\n"
        f"  Average Waiting Time:  {avg_wait:.2f} seconds\n"
        f"============================================================\n"
    )


def main():
    p = argparse.ArgumentParser(
        description="Live evaluation demo utility for smart traffic control."
    )
    p.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        default="quadratic",
        help="Which trained configuration to load.",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Launch the SUMO graphical interface for visualization.",
    )
    p.add_argument(
        "--no-gui",
        action="store_false",
        dest="gui",
        help="Disable the graphical interface.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=CONFIG.MASTER_SEED,
        help="Seed for deterministic evaluation alignment.",
    )
    p.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Specific episode index to evaluate (loads intermediate Q-table).",
    )
    args = p.parse_args()

    evaluate(args.config, args.gui, args.seed, args.episode)


if __name__ == "__main__":
    main()
