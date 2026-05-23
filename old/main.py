"""
Entry point for the Traffic Signal Q-Learning project.

Usage
─────
  python main.py --train            Train the Q-table (headless, fast)
  python main.py --demo             Demo the saved Q-table with Pygame
  python main.py --demo --fixed     Demo fixed-time controller with Pygame
  python main.py --demo --random    Demo random policy with Pygame
  python main.py --train --demo     Train then immediately open demo
"""
import argparse
import os
import sys


def _parse():
    p = argparse.ArgumentParser(description="Q-Learning Traffic Signal Control")
    p.add_argument("--train",  action="store_true", help="Run training")
    p.add_argument("--demo",   action="store_true", help="Run Pygame demo")
    p.add_argument("--fixed",  action="store_true", help="Use fixed-time controller in demo")
    p.add_argument("--random", action="store_true", help="Use random policy in demo")
    p.add_argument("--episodes", type=int, default=None, help="Override TRAIN_EPISODES")
    p.add_argument("--steps",    type=int, default=None, help="Override STEPS_PER_EP")
    return p.parse_args()


def do_train(args):
    if args.episodes is not None:
        import config
        config.TRAIN_EPISODES = args.episodes
    if args.steps is not None:
        import config
        config.STEPS_PER_EP = args.steps
    from train import train
    return train()


def do_demo(args):
    try:
        import pygame
    except ImportError:
        print("pygame is not installed.  Run:  pip install pygame", file=sys.stderr)
        sys.exit(1)

    from config import FPS, DEMO_STEPS, QTABLE_PATH
    from environment import TrafficEnv
    from agent import QLearningAgent
    from train import run_fixed_episode, run_random_episode
    from visualizer import Visualizer, make_screen

    env = TrafficEnv(seed=99)

    if args.random:
        label  = "Random Policy"
        policy = "random"
    elif args.fixed:
        label  = "Fixed-Time Controller"
        policy = "fixed"
    else:
        label  = "Q-Learning (trained)"
        policy = "ql"

    agent = QLearningAgent()
    if policy == "ql":
        if not os.path.exists(QTABLE_PATH):
            print(f"No Q-table found at '{QTABLE_PATH}'.  Run --train first.")
            sys.exit(1)
        agent.load(QTABLE_PATH)

    screen, clock = make_screen()
    viz   = Visualizer(screen)

    state      = env.reset()
    total_r    = 0.0
    step       = 0
    fixed_gs   = 0   # green-step counter for fixed-time

    from config import FIXED_CYCLE
    running = True

    print(f"Running demo: {label}  ({DEMO_STEPS} steps).  Close the window to exit.")

    while running and step < DEMO_STEPS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # ── Choose action ──────────────────────────────────────────────────
        if env.is_yellow:
            action = 0
        elif policy == "ql":
            action = agent.act(state, greedy=True)
        elif policy == "fixed":
            action = 1 if fixed_gs >= FIXED_CYCLE else 0
        else:
            import numpy as np
            action = np.random.randint(2)

        state, r, _, _ = env.step(action)

        if policy == "fixed" and not env.is_yellow:
            fixed_gs = (fixed_gs + 1) % (FIXED_CYCLE + 1)
        elif env.is_yellow:
            fixed_gs = 0

        total_r += r
        step    += 1

        viz.draw(env, episode=1, step=step, total_r=total_r, label=label)
        clock.tick(FPS)

    pygame.quit()
    print(f"Demo finished.  Total reward: {total_r:.1f}  ({step} steps)")


def main():
    args = _parse()

    if not args.train and not args.demo:
        print(__doc__)
        return

    if args.train:
        do_train(args)

    if args.demo:
        do_demo(args)


if __name__ == "__main__":
    main()
