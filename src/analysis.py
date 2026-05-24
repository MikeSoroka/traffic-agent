import os
import sys
import numpy as np
import pandas as pd

# enforce headless backend conditionally before importing pyplot to avoid UI blocking on servers without a display
if __name__ == "__main__" and "IPython" not in sys.modules:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from config import CONFIG

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.size": 10,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    }
)

IS_INTERACTIVE = "IPython" in sys.modules


def generate_training_rewards_plot(save: bool = False):
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"baseline": "#5a9fd4", "custom": "#f0a050"}
    labels = {
        "baseline": "Baseline Reward (Δ waiting time)",
        "custom": "Custom Reward (Differential Queue)",
    }

    found_any = False
    for config_name in colors.keys():
        summary_path = os.path.join(CONFIG.DATA_DIR, f"{config_name}_summary.csv")
        if not os.path.exists(summary_path):
            continue

        found_any = True
        df = pd.read_csv(summary_path)
        rewards = df["total_reward"].tolist()

        c = colors[config_name]
        lbl = labels[config_name]
        n = len(rewards)

        style_opts = {
            "baseline": {"marker": "o"},
            "custom": {"marker": "^"},
        }
        opts = style_opts.get(config_name, {"marker": "s"})

        ax.plot(
            range(n),
            rewards,
            color=c,
            ls="-",
            marker=opts["marker"],
            markevery=max(1, n // 10),
            markersize=6,
            lw=1.8,
            zorder=2,
            label=lbl,
        )

    if not found_any:
        print(
            "No training summary CSV data found in outputs/data/. Run training first."
        )
        plt.close(fig)
        return

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Reward", fontsize=11)
    ax.set_title(
        "Q-Learning Training Curves: Comparative Reward Analysis",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    if save:
        plot_path = os.path.join(CONFIG.PLOTS_DIR, "reward_comparison.pdf")
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparative rewards plot to {plot_path}")
    else:
        plt.show()


def generate_performance_metrics_plot(save: bool = False):
    colors = {"baseline": "#5a9fd4", "custom": "#f0a050"}
    labels = {"baseline": "Baseline Reward", "custom": "Custom Reward"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    found_any = False

    for config_name in colors.keys():
        summary_path = os.path.join(CONFIG.DATA_DIR, f"{config_name}_summary.csv")
        if not os.path.exists(summary_path):
            continue

        found_any = True
        df = pd.read_csv(summary_path)
        eps = df["episode"].tolist()
        avg_q = df["avg_queue"].tolist()
        max_q = df["max_queue"].tolist()
        avg_w = df["avg_waiting_time"].tolist()

        c = colors[config_name]
        lbl = labels[config_name]

        # apply distinct styles to support color-blind differentiation
        style_opts = {
            "baseline": {"ls": "-", "marker": "o"},
            "custom": {"ls": "-", "marker": "^"},
        }
        opts = style_opts.get(config_name, {"ls": "-", "marker": "s"})
        n_eps = len(eps)
        mark_step = max(1, n_eps // 10)

        # left panel - queue lengths
        axes[0].plot(
            eps,
            avg_q,
            color=c,
            ls=opts["ls"],
            marker=opts["marker"],
            markevery=mark_step,
            lw=2.2,
            label=lbl,
        )
        axes[0].fill_between(
            eps, avg_q, max_q, color=c, alpha=0.10, label=f"{lbl} (max queue band)"
        )
        axes[0].plot(eps, max_q, color=c, ls=":", lw=0.8, alpha=0.6)

        # right panel - waiting times
        axes[1].plot(
            eps,
            avg_w,
            color=c,
            ls=opts["ls"],
            marker=opts["marker"],
            markevery=mark_step,
            lw=2.2,
            label=lbl,
        )

    if not found_any:
        print("No training summary CSV data found. Skipping performance metrics plot.")
        plt.close(fig)
        return

    for ax, title, ylabel in [
        (
            axes[0],
            "Stopped Queue Length per Episode",
            "Vehicles stopped (avg / max band)",
        ),
        (axes[1], "Mean Waiting Time per Episode", "Waiting time (seconds)"),
    ]:
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)

    fig.tight_layout()

    if save:
        plot_path = os.path.join(CONFIG.PLOTS_DIR, "performance_metrics.pdf")
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved performance metrics panel to {plot_path}")
    else:
        plt.show()


def plot_episode_details(config_name: str, episode: int, save: bool = False):
    """
    Reconstruction of signal schedule and traffic heatmaps from saved CSV logs.
    Visualizes any episode at any time rule
    """
    ep_file = os.path.join(CONFIG.DATA_DIR, f"{config_name}_ep_{episode}.csv")
    if not os.path.exists(ep_file):
        print(f"Telemetry log file not found: {ep_file}")
        return

    df = pd.read_csv(ep_file)
    actions = df["action"].tolist()
    stopped = df["total_stopped"].tolist()

    # extract observation features (find keys starting with 'obs_')
    obs_cols = sorted(
        [col for col in df.columns if col.startswith("obs_")],
        key=lambda x: int(x.split("_")[1]),
    )
    obs_arr = df[obs_cols].to_numpy()
    T = len(actions)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [1, 1, 2.5]},
    )
    fig.suptitle(
        f"Signal Schedule and Traffic State: {config_name.upper()} (Episode {episode})",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    # top panel: phase Gantt
    ax = axes[0]
    phase_colors = {0: "#4a90d9", 1: "#e8872a"}
    phase_labels = {0: "NS Green", 1: "EW Green"}
    t = 0
    while t < T:
        a = actions[t]
        t2 = t + 1
        while t2 < T and actions[t2] == a:
            t2 += 1
        ax.barh(
            0,
            t2 - t,
            left=t,
            height=0.6,
            color=phase_colors[a % 2],
            alpha=0.90,
            edgecolor="none",
        )
        t = t2

    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_ylabel("Green Phase", fontsize=9)
    ax.set_xticks(range(0, T, max(1, T // 20)))
    ax.grid(axis="x", alpha=0.4, ls=":")
    patches = [
        mpatches.Patch(color=phase_colors[i], label=phase_labels[i]) for i in (0, 1)
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.8)

    # middle panel: halted vehicles over time
    ax = axes[1]
    ax.fill_between(
        range(T), stopped, color="#e74c3c", alpha=0.15, label="Stopped vehicles"
    )
    ax.plot(range(T), stopped, color="#c0392b", lw=1.8)
    ax.set_xlim(0, T)
    ax.set_ylabel("Stopped Vehicles", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # bottom panel: observation heatmap
    ax = axes[2]
    if obs_arr.ndim == 2 and obs_arr.shape[0] > 0:
        n_feat = obs_arr.shape[1]
        feat_labels = [
            "Density East",
            "Density North",
            "Density South",
            "Density West",
            "Queue East",
            "Queue North",
            "Queue South",
            "Queue West",
            "Phase Bit 1",
            "Phase Bit 2",
            "Phase Bit 3",
        ]

        im = ax.imshow(
            obs_arr.T,  # rows=features, cols=time
            aspect="auto",
            cmap="rocket",  # seaborn theme continuous range
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            extent=[-0.5, T - 0.5, -0.5, n_feat - 0.5],
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="Normalised value [0, 1]", fraction=0.025)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(feat_labels, fontsize=7)

        # draw white horizontal dividers between features for visual separation
        for idx in range(1, n_feat):
            ax.axhline(idx - 0.5, color="white", lw=1.0, ls="-")

    ax.set_xlabel("Decision step", fontsize=9)
    ax.set_ylabel("Observation feature", fontsize=9)
    ax.set_title("Traffic-state heatmap (lane density and queue over time)", fontsize=9)
    ax.grid(False)  # turn off grid for pixel-aligned heatmaps
    ax.set_xticks(range(0, T, max(1, T // 20)))

    fig.tight_layout()

    if save:
        plot_path = os.path.join(
            CONFIG.PLOTS_DIR, f"signal_schedule_{config_name}_ep_{episode}.pdf"
        )
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved episode schedule and heatmap plot to {plot_path}")
    else:
        plt.show()


def compare_episodes(config_name: str, ep_a: int, ep_b: int, save: bool = False):
    """
    Side-by-side retrospective analysis comparing two arbitrary episodes.
    Renders signal phases and queue lengths
    """
    file_a = os.path.join(CONFIG.DATA_DIR, f"{config_name}_ep_{ep_a}.csv")
    file_b = os.path.join(CONFIG.DATA_DIR, f"{config_name}_ep_{ep_b}.csv")

    if not os.path.exists(file_a) or not os.path.exists(file_b):
        print(f"Missing telemetry files for comparison: {file_a} or {file_b}")
        return

    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex="col")
    fig.suptitle(
        f"Retrospective Evolution Panel: {config_name.upper()}\n"
        f"Episode {ep_a} (Early/Random) vs Episode {ep_b} (Trained Policy)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    phase_colors = {0: "#4a90d9", 1: "#e8872a"}
    phase_labels = {0: "NS Green", 1: "EW Green"}

    for idx, (df, ep, col) in enumerate([(df_a, ep_a, 0), (df_b, ep_b, 1)]):
        actions = df["action"].tolist()
        stopped = df["total_stopped"].tolist()
        T = len(actions)

        # plot Gantt chart for green phases
        ax_gantt = axes[0, col]
        t = 0
        while t < T:
            a = actions[t]
            t2 = t + 1
            while t2 < T and actions[t2] == a:
                t2 += 1
            ax_gantt.barh(
                0,
                t2 - t,
                left=t,
                height=0.6,
                color=phase_colors[a % 2],
                alpha=0.90,
                edgecolor="none",
            )
            t = t2

        ax_gantt.set_xlim(0, T)
        ax_gantt.set_ylim(-0.5, 0.5)
        ax_gantt.set_yticks([])
        ax_gantt.set_title(
            f"Episode {ep}: Signal Phase Gantt", fontsize=10, fontweight="bold"
        )
        if col == 0:
            ax_gantt.set_ylabel("Active Green", fontsize=9)
        patches = [
            mpatches.Patch(color=phase_colors[i], label=phase_labels[i]) for i in (0, 1)
        ]
        ax_gantt.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.8)
        ax_gantt.grid(axis="x", alpha=0.3)

        # plot stopped queues over time
        ax_queue = axes[1, col]
        ax_queue.fill_between(
            range(T), stopped, color="#e74c3c", alpha=0.15, label="Stopped vehicles"
        )
        ax_queue.plot(range(T), stopped, color="#c0392b", lw=1.8)
        ax_queue.set_xlim(0, T)
        ax_queue.set_xlabel("Decision Step", fontsize=9)
        if col == 0:
            ax_queue.set_ylabel("Stopped Vehicles", fontsize=9)
        ax_queue.grid(alpha=0.3)
        ax_queue.legend(fontsize=8)

    fig.tight_layout()

    if save:
        plot_path = os.path.join(
            CONFIG.PLOTS_DIR, f"comparison_ep_{ep_a}_vs_{ep_b}.pdf"
        )
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparative retrospective plot to {plot_path}")
    else:
        plt.show()


def generate_q_table_scatter_projection(
    config_name: str, label: str, save: bool = False
):
    model_path = os.path.join(CONFIG.MODELS_DIR, f"q_table_{config_name}.npy")
    if not os.path.exists(model_path):
        print(f"Saved Q-table not found at: {model_path}. Skipping plot.")
        return

    # load dictionary from numpy array object
    q_table = np.load(model_path, allow_pickle=True).item()
    if not q_table:
        print("Loaded Q-table is empty.")
        return

    states = list(q_table.keys())
    q_vals = np.array([q_table[s] for s in states])  # (N, n_actions)
    max_q = q_vals.max(axis=1)
    best_act = q_vals.argmax(axis=1)
    dim0 = np.array([s[0] for s in states])
    dim1 = np.array([s[1] for s in states])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Q-Table Value and Policy Scatter Projection: {label}\n"
        f"({len(states)} states visited, projected to incoming East-North densities)",
        fontsize=11,
        fontweight="bold",
    )

    jitter = np.random.rand(len(states)) * 0.1 - 0.1  # avoid direct overlaps

    # left panel - value estimates (rasterized to prevent huge vector PDFs)
    sc1 = axes[0].scatter(
        dim0 + jitter,
        dim1 + jitter,
        c=max_q,
        cmap="mako",  # seaborn theme continuous range
        s=55,
        alpha=0.85,
        edgecolors="#222222",
        linewidths=0.5,
        zorder=2,
        rasterized=True,
    )
    plt.colorbar(sc1, ax=axes[0], label="max Q(s, ·)  (state value)")
    axes[0].set_title("State Value Estimate (Expected Return)")
    axes[0].set_xlabel("Dim 0: East Lane Density Bin")
    axes[0].set_ylabel("Dim 1: North Lane Density Bin")
    axes[0].set_xticks(range(CONFIG.NUM_BINS))
    axes[0].set_yticks(range(CONFIG.NUM_BINS))

    # right panel - policy (rasterized to prevent huge vector PDFs)
    sc2 = axes[1].scatter(
        dim0 + jitter,
        dim1 + jitter,
        c=best_act,
        cmap="vlag",  # seaborn theme diverging range
        s=55,
        alpha=0.85,
        edgecolors="#222222",
        linewidths=0.5,
        vmin=0,
        vmax=1,
        zorder=2,
        rasterized=True,
    )
    cbar2 = plt.colorbar(sc2, ax=axes[1], label="Greedy action selection")
    cbar2.set_ticks([0, 1])
    cbar2.set_ticklabels(["NS Green (0)", "EW Green (1)"])
    axes[1].set_title("Preferred Action (Greedy Policy)")
    axes[1].set_xlabel("Dim 0: East Lane Density Bin")
    axes[1].set_ylabel("Dim 1: North Lane Density Bin")
    axes[1].set_xticks(range(CONFIG.NUM_BINS))
    axes[1].set_yticks(range(CONFIG.NUM_BINS))

    # add light-gray dotted background coordinates grid
    for ax in axes:
        ax.grid(True, which="both", color="#dddddd", ls=":", lw=0.7, zorder=0)

    fig.tight_layout()

    if save:
        plot_path = os.path.join(
            CONFIG.PLOTS_DIR, f"q_table_scatter_projection_{config_name}.pdf"
        )
        fig.savefig(plot_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Q-table scatter projection to {plot_path}")
    else:
        plt.show()


def generate_true_policy_matrix_heatmaps(
    config_name: str, label: str, save: bool = False
):
    """
    Constructs true 4x4 matrix heatmaps of visited Q-state values and greedy policies
    projected onto the East Lane Density Bin (columns) and North Lane Density Bin (rows)
    """
    model_path = os.path.join(CONFIG.MODELS_DIR, f"q_table_{config_name}.npy")
    if not os.path.exists(model_path):
        print(f"Saved Q-table not found at: {model_path}. Skipping plot.")
        return

    q_table = np.load(model_path, allow_pickle=True).item()
    if not q_table:
        print("Loaded Q-table is empty.")
        return

    value_matrix = np.zeros((4, 4))
    policy_matrix = np.zeros((4, 4))

    for row in range(4):  # North Bin (y-axis)
        for col in range(4):  # East Bin (x-axis)
            matching_max_qs = []
            matching_actions = []
            for state, q_values in q_table.items():
                if state[0] == col and state[1] == row:
                    matching_max_qs.append(np.max(q_values))
                    matching_actions.append(np.argmax(q_values))
            if len(matching_max_qs) > 0:
                value_matrix[row, col] = np.mean(matching_max_qs)
                policy_matrix[row, col] = np.mean(matching_actions)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Q-Table True Value and Policy Grid Matrix Heatmaps: {label}\n"
        f"({len(q_table)} states visited, aggregated to 4x4 East-North Density Grid)",
        fontsize=11,
        fontweight="bold",
    )

    # disable background grids explicitly
    axes[0].grid(False)
    axes[1].grid(False)

    # left panel - value matrix heatmap
    sns.heatmap(
        value_matrix,
        ax=axes[0],
        annot=True,
        fmt=".3f",
        cmap="rocket",  # seaborn theme continuous range
        cbar_kws={"label": "Mean max Q(s, ·) (State Value)"},
    )
    axes[0].invert_yaxis()  # set origin to lower so Bin 0 is at bottom
    axes[0].set_title("Mean State Value Estimate", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("East Lane Density Bin", fontsize=9)
    axes[0].set_ylabel("North Lane Density Bin", fontsize=9)

    # right panel - policy matrix heatmap
    sns.heatmap(
        policy_matrix,
        ax=axes[1],
        annot=True,
        fmt=".3f",
        cmap="vlag",  # seaborn theme diverging range
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Mean Action (Greedy Policy)"},
    )
    axes[1].invert_yaxis()  # set origin to lower so Bin 0 is at bottom
    axes[1].set_title(
        "Preferred Action (Greedy Policy Preference)", fontsize=10, fontweight="bold"
    )
    axes[1].set_xlabel("East Lane Density Bin", fontsize=9)
    axes[1].set_ylabel("North Lane Density Bin", fontsize=9)

    fig.tight_layout()

    if save:
        plot_path = os.path.join(
            CONFIG.PLOTS_DIR, f"true_q_table_matrix_heatmaps_{config_name}.pdf"
        )
        fig.savefig(plot_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved true Q-table matrix heatmaps to {plot_path}")
    else:
        plt.show()


def generate_final_comparison_bar_chart(save: bool = False):
    configs = ["baseline", "custom"]
    labels = ["Baseline (Waiting Time)", "Custom (Differential Queue)"]
    colors = ["#5a9fd4", "#f0a050"]

    valid_data = []
    for cfg in configs:
        summary_path = os.path.join(CONFIG.DATA_DIR, f"{cfg}_summary.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            if not df.empty:
                valid_data.append((cfg, df.iloc[-1]))  # get the final episode row

    if len(valid_data) < 2:
        print(
            "Comparative bar chart skipped. Needs both 'baseline' and 'custom' summaries."
        )
        return

    metric_keys = ["avg_queue", "max_queue", "avg_waiting_time"]
    metric_labels = [
        "Avg Queue Length\n(vehicles)",
        "Max Queue Length\n(vehicles)",
        "Avg Waiting Time\n(seconds)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        "Final-Episode Performance Metrics Comparison", fontsize=12, fontweight="bold"
    )

    for ax, mkey, mlabel in zip(axes, metric_keys, metric_labels):
        cfg_labels = [
            labels[0] if item[0] == "baseline" else labels[1] for item in valid_data
        ]
        values = [item[1][mkey] for item in valid_data]
        bar_colors = [
            colors[0] if item[0] == "baseline" else colors[1] for item in valid_data
        ]

        bars = ax.bar(
            range(len(cfg_labels)), values, color=bar_colors, width=0.45, zorder=3
        )
        ax.set_xticks(range(len(cfg_labels)))
        ax.set_xticklabels(cfg_labels, fontsize=8)
        ax.set_title(mlabel, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, zorder=0)

        # draw numeric values on top of the bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(values) * 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout()

    if save:
        plot_path = os.path.join(CONFIG.PLOTS_DIR, "final_comparison.pdf")
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparative performance bar chart to {plot_path}")
    else:
        plt.show()


def generate_q_table_growth_plot(save: bool = False):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"baseline": "#5a9fd4", "custom": "#f0a050"}
    labels = {"baseline": "Baseline Reward", "custom": "Custom Reward"}
    style_opts = {
        "baseline": {"ls": "-", "marker": "o"},
        "custom": {"ls": "-", "marker": "^"},
    }

    for config_name in colors.keys():
        summary_path = os.path.join(CONFIG.DATA_DIR, f"{config_name}_summary.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            # pull the logged state column over the 100 episodes
            if "Q-states" in df.columns:
                n_eps = len(df["episode"])
                mark_step = max(1, n_eps // 10)
                opts = style_opts.get(config_name, {"ls": "-", "marker": "s"})
                ax.plot(
                    df["episode"],
                    df["Q-states"],
                    color=colors[config_name],
                    ls=opts["ls"],
                    marker=opts["marker"],
                    markevery=mark_step,
                    markersize=6,
                    lw=2.5,
                    zorder=2,
                    label=labels[config_name],
                )

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Explored States in Dictionary (|S|)", fontsize=11)
    ax.set_title(
        "Q-Table Expansion Curve: State-Space Discovery Over Time",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, which="both", color="#dddddd", ls=":", lw=0.7, zorder=0)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save:
        plot_path = os.path.join(CONFIG.PLOTS_DIR, "q_table_growth.pdf")
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Q-table discovery curve to {plot_path}")
    else:
        plt.show()


def batch_generate_all_plots(save: bool = False):
    print(f"Generating figures (save={save})...")

    # 1. rewards
    generate_training_rewards_plot(save=save)

    # 2. key aggregates over training
    generate_performance_metrics_plot(save=save)

    # 3. before/after detailed schedules and heatmaps
    # episode 1 representing random exploration behavior
    plot_episode_details("baseline", 1, save=save)
    plot_episode_details("custom", 1, save=save)

    # final episode showing trained behavior
    # auto-detect final episode number from summary files
    for config_name in ["baseline", "custom"]:
        summary_path = os.path.join(CONFIG.DATA_DIR, f"{config_name}_summary.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            final_ep = int(df["episode"].max())
            plot_episode_details(config_name, final_ep, save=save)

    # 4. state value projections
    generate_q_table_scatter_projection(
        "baseline", "Baseline Reward (Δ wait time)", save=save
    )
    generate_q_table_scatter_projection(
        "custom", "Custom Reward (Differential Queue)", save=save
    )
    generate_true_policy_matrix_heatmaps(
        "baseline", "Baseline Reward (Δ wait time)", save=save
    )
    generate_true_policy_matrix_heatmaps(
        "custom", "Custom Reward (Differential Queue)", save=save
    )

    # 5. final episode comparison bars
    generate_final_comparison_bar_chart(save=save)

    # 6. Q-table state discovery growth curve over episodes
    generate_q_table_growth_plot(save=save)

    # 7. retrospective comparative evolution (episode 1 vs final episode)
    for config_name in ["baseline", "custom"]:
        summary_path = os.path.join(CONFIG.DATA_DIR, f"{config_name}_summary.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            final_ep = int(df["episode"].max())
            compare_episodes(config_name, 1, final_ep, save=save)

    if save:
        print(f"All figures generated and saved to: {CONFIG.PLOTS_DIR}/")


if __name__ == "__main__":
    batch_generate_all_plots(save=True)
