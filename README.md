# 🚦 Smart Traffic Light Control with Tabular Q-Learning

An intelligent, adaptive traffic signal management system leveraging model-free **Tabular Q-Learning** to dynamically optimize phase timings at four-way urban road intersections. Built microscopically within the **Eclipse SUMO (Simulation of Urban MObility)** environment via the `sumo-rl` Gymnasium framework.

This project implements a fully parameterized simulation pipeline that allows researchers to benchmark alternative reward strategies against identical stochastic vehicle arrival distributions. The complete system includes a headless training engine, a visual playback demonstration runtime, an interactive analysis compiler, and an academic LaTeX report formatted to **Kaunas University of Technology (KTU)** specifications.

---

## 🏛️ System Architecture & Code Structure

The project is decoupled into independent, highly modular modules to ensure strict separation of concerns, eliminating graphical rendering overhead during primary reinforcement learning cycles.

```text
.
├── src/
│   ├── config.py       # Centralised Configuration Hub & Master Seed reproducibility
│   ├── train.py        # Headless Training Engine (TraCI off-policy learning loop)
│   ├── analysis.py     # Standalone Plot & Figure Compiler (saves vector PDFs)
│   └── demo.py         # Deterministic GUI Demonstration Module (exploration rate ε=0)
├── outputs/            # Dynamic workspace output directory
│   ├── data/           # Step-by-step CSV telemetry logs saved after episodes
│   ├── models/         # Serialised Q-table lookup dictionaries (*.npy)
│   ├── plots/          # Generated vector report figures
│   └── nets/           # Dynamically compiled SUMO road network XML files
├── pyproject.toml      # System dependency definitions (managed via uv)
```

---

## 📦 Installation & Environmental Setup

This project uses `uv` for fast, reproducible Python packaging and virtual environment management.

### 1. System Prerequisites
Ensure that the **SUMO microscopic simulator** (including command-line utilities like `netconvert`) is installed on your operating system.

**For Ubuntu/Linux:**
```bash
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

### 2. Python Environment & Dependencies
Clone the repository and install all required python libraries (`matplotlib`, `numpy`, `pandas`, `seaborn`, and `sumo-rl`) inside a local virtual environment using `uv`:

```bash
# Install dependencies and sync virtual environment
uv sync
```

### 3. Environment Variables
To allow Python to communicate with the underlying SUMO TraCI binaries, you must configure your local paths. Copy the template `.env.example` into a local `.env` file:

```bash
cp .env.example .env
```

Ensure your `.env` contains the correct absolute path to your SUMO installation (the default path is shown below):
```ini
SUMO_HOME="/usr/share/sumo"
LIBSUMO_AS_TRACI=1
```

---

## 🚀 Execution Guide

Always prep your terminal environment variables before running script commands.

### 1. Training Headless RL Agents (`src/train.py`)
Run the training engine to optimize greedy lookup tables over a $100$-episode horizon. Telemetry logs and binary model snapshots will be permanently written to `outputs/data/` and `outputs/models/`.

```bash
# Source environmental variables
export SUMO_HOME=/usr/share/sumo
export PATH=$SUMO_HOME/bin:$PATH
export LIBSUMO_AS_TRACI=1

# Benchmark both agents sequentially under identical stochastically generated queues
uv run src/train.py --compare --episodes 100

# Train a single Baseline Agent (Differential Waiting Time reward)
# uv run src/train.py --config baseline

# Train a single Custom Agent (Differential Queue Length reward)
# uv run src/train.py --config custom
```

**Custom CLI Parameters:**
* `--episodes N`: Override total training cycles (default: `100`).
* `--num-lanes L`: Dynamically compiles the cross-intersection with custom lane counts (default: `1`).
* `--traffic-intensity I`: Scales Poisson vehicle generation flows (default: `1.0`).

---

### 📈 2. Compiling Academic Vector Plots (`src/analysis.py`)
Parse historical telemetry logs and model parameters, compiling publication-grade vector PDFs instantly. This analysis engine runs completely offline without launching active traffic simulations or SUMO GUI layers.

```bash
export SUMO_HOME=/usr/share/sumo
export PATH=$SUMO_HOME/bin:$PATH
export LIBSUMO_AS_TRACI=1

uv run src/analysis.py
```

**Generated Visualizations:**
* `reward_comparison.pdf`: Training reward convergence timeline.
* `performance_metrics.pdf`: Average waiting times and stopped-vehicle queues with dynamic max-queue shadow boundaries.
* `signal_schedule_[baseline|custom]_ep_[1|100].pdf`: Micro-level green-phase Gantt chart overlaid above detailed lane-density pixel heatmaps.
* `q_table_growth.pdf`: Sub-linear state-space exploration curves.
* `q_table_scatter_projection_[baseline|custom].pdf`: 2D scatter plots of visited states, showing value expectations (Seaborn *Mako* theme) and action choices (Seaborn *Vlag* theme).
* `true_q_table_matrix_heatmaps_[baseline|custom].pdf`: Aggregated $4 \times 4$ Q-value and greedy policy grid matrix heatmaps.
* `final_comparison.pdf`: Side-by-side Pareto efficiency comparison metrics during final evaluation cycles.

---

### 📺 3. Deteministic Visual GUI Playback (`src/demo.py`)
Launch the SUMO graphic user interface to visual audit a trained model's real-time traffic coordination capabilities under a strict, non-exploratory greedy policy ($\epsilon = 0$).

```bash
export SUMO_HOME=/usr/share/sumo
export PATH=$SUMO_HOME/bin:$PATH
export LIBSUMO_AS_TRACI=1

# Run live playback for the custom agent
uv run src/demo.py --config custom

# Run live playback for the baseline agent
uv run src/demo.py --config baseline
```

---

## 📝 Mathematical Reward Formulations

To evaluate the systemic effects of alternative reinforcement learning objectives, two distinct formulations are compared back-to-back under deterministic arrival runs:

### 1. Baseline: Differential Waiting Time
Optimizes for changes in aggregate delay accumulated on all incoming approach corridors:
$$r_t = \sum_{i} W_i^{t-1} - \sum_{i} W_i^t$$
where $W_i^t$ represents the total accumulated waiting time of vehicle $i$ at control step $t$.

### 2. Custom: Differential Queue Length
Optimizes for the direct change in total stationary queuing footprint:
$$r_t = Q_{t-1} - Q_t \quad \text{where} \quad Q_t = \sum_{l} h_l^t$$
where $h_l^t$ measures the explicit number of completely halted passenger cars detected on approach lane $l$ at step $t$. By utilizing a differential metric, the agent filters out permanent background vehicle noise and unlocks a clean, high-precision learning gradient.
