import os
import importlib.util as _ilu
from dataclasses import dataclass
import numpy as np


@dataclass
class GlobalConfig:
    MASTER_SEED: int = 0

    # Q-learning hyperparameters
    ALPHA: float = 0.20  # learning rate (alpha)
    GAMMA: float = 0.95  # discount factor (gamma)
    EPS_START: float = 1.0  # initial exploration rate
    EPS_DECAY: float = 0.96  # per-episode multiplicative decay
    EPS_MIN: float = 0.05  # exploration floor
    NUM_BINS: int = 4  # state discretisation bins

    # simulation parameters
    NUM_SECONDS: int = 3600  # simulated seconds per episode
    DELTA_TIME: int = 5  # seconds between agent decisions
    YELLOW_TIME: int = 2  # automatic yellow-phase buffer
    MIN_GREEN: int = 5  # minimum green-phase duration

    # intersection and traffic intensity
    # number of lanes at the single-intersection (defined in SUMO net)
    NUM_LANES: int = 1
    TRAFFIC_INTENSITY: float = 1.0  # scale factor for stochastically generated flows

    # output directory layout
    OUTPUT_DIR: str = "outputs"
    DATA_DIR: str = os.path.join(OUTPUT_DIR, "data")
    MODELS_DIR: str = os.path.join(OUTPUT_DIR, "models")
    PLOTS_DIR: str = os.path.join(OUTPUT_DIR, "plots")
    NETS_DIR: str = os.path.join(OUTPUT_DIR, "nets")

    TRAIN_EPISODES: int = 100

    # structural traffic rules
    SPEED_LIMIT_M_S: float = 13.89  # 50 km/h urban velocity matching local regulations
    # seconds between vehicle generations at intensity 1.0
    BASE_SPAWN_PERIOD: float = 4.0

    # route selection splits
    SPLIT_STRAIGHT: float = 0.70  # 70% of vehicles travel across the intersection
    SPLIT_LEFT: float = 0.15  # 15% turn left
    SPLIT_RIGHT: float = 0.15  # 15% turn right

    # intersection coordinates and geometry
    # length of road approaches from intersection center
    APPROACH_LENGTH_M: float = 150.0
    EDGE_PRIORITY: int = 7  # priority value assigned to edges

    # vehicle dynamics (passenger cars)
    VEH_ACCEL: float = 2.6  # acceleration (m/s^2)
    VEH_DECEL: float = 4.5  # deceleration (m/s^2)
    VEH_SIGMA: float = 0.5  # driver imperfection (0 to 1)
    VEH_LENGTH: float = 5.0  # vehicle physical length in meters
    VEH_MIN_GAP: float = 2.5  # minimum front-to-back safety gap in meters

    # safety floors for traffic generation
    # absolute floor for Poisson period to prevent SUMO overload
    MIN_SPAWN_PERIOD: float = 0.5
    MIN_INTENSITY_FLOOR: float = 0.1  # safety threshold to avoid division by zero


# instantiate the global configuration
CONFIG = GlobalConfig()

# create required directories
for d in (
    CONFIG.OUTPUT_DIR,
    CONFIG.DATA_DIR,
    CONFIG.MODELS_DIR,
    CONFIG.PLOTS_DIR,
    CONFIG.NETS_DIR,
):
    os.makedirs(d, exist_ok=True)

# network files (resolved from the installed sumo_rl package)
_PKG_SPEC = _ilu.find_spec("sumo_rl")
if _PKG_SPEC is None or _PKG_SPEC.origin is None:
    raise ImportError("sumo_rl package not found. Please install dependencies.")
_PKG_DIR = os.path.dirname(_PKG_SPEC.origin)
_NETS = os.path.join(_PKG_DIR, "nets", "single-intersection")
NET_FILE = os.path.join(_NETS, "single-intersection.net.xml")
ROUTE_FILE = os.path.join(_NETS, "single-intersection.rou.xml")


def discretize_state(obs: np.ndarray, num_bins: int = CONFIG.NUM_BINS) -> tuple:
    """
    Map a continuous observation vector ∈ [0,1]^n to a discrete tuple of
    bin indices ∈ {0, …, num_bins-1}^n usable as a Q-table dictionary key
    """
    edges = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    discrete = tuple(int(np.clip(np.digitize(x, edges), 0, num_bins - 1)) for x in obs)
    return discrete


def _q(q_table: dict, state: tuple, n_actions: int) -> np.ndarray:
    # return the Q-value row for state, lazily initialised to zeros
    if state not in q_table:
        q_table[state] = np.zeros(n_actions, dtype=np.float64)
    return q_table[state]


def select_action(q_table: dict, state: tuple, n_actions: int, epsilon: float) -> int:
    # epsilon-greedy policy, ties in argmax broken by tiny random noise
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
    In-place Bellman update.
    Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
    """
    current = _q(q_table, state, n_actions)[action]
    next_max = np.max(_q(q_table, next_state, n_actions))
    _q(q_table, state, n_actions)[action] += CONFIG.ALPHA * (
        reward + CONFIG.GAMMA * next_max - current
    )


def generate_custom_scenario(
    num_lanes: int, traffic_intensity: float
) -> tuple[str, str]:
    """
    Dynamically compile custom SUMO network and stochastically generated routes
    based on user-configured intersection parameters
    """
    import subprocess

    # generate custom nodes
    nodes_path = os.path.join(CONFIG.NETS_DIR, "custom_nodes.nod.xml")
    nodes_content = f"""
    <nodes>
        <node id="center" x="0" y="0" type="traffic_light"/>
        <node id="north" x="0" y="{CONFIG.APPROACH_LENGTH_M}" type="priority"/>
        <node id="south" x="0" y="-{CONFIG.APPROACH_LENGTH_M}" type="priority"/>
        <node id="east" x="{CONFIG.APPROACH_LENGTH_M}" y="0" type="priority"/>
        <node id="west" x="-{CONFIG.APPROACH_LENGTH_M}" y="0" type="priority"/>
    </nodes>    
    """
    with open(nodes_path, "w") as f:
        f.write(nodes_content)

    # generate custom edges with user-defined number of lanes
    edges_path = os.path.join(CONFIG.NETS_DIR, "custom_edges.edg.xml")
    edges_content = f"""
    <edges>
        <edge id="N2C" from="north" to="center" priority="{CONFIG.EDGE_PRIORITY}" numLanes="{num_lanes}" speed="{CONFIG.SPEED_LIMIT_M_S}"/>
        <edge id="S2C" from="south" to="center" priority="{CONFIG.EDGE_PRIORITY}" numLanes="{num_lanes}" speed="{CONFIG.SPEED_LIMIT_M_S}"/>
        <edge id="E2C" from="east" to="center" priority="{CONFIG.EDGE_PRIORITY}" numLanes="{num_lanes}" speed="{CONFIG.SPEED_LIMIT_M_S}"/>
        <edge id="W2C" from="west" to="center" priority="{CONFIG.EDGE_PRIORITY}" numLanes="{num_lanes}" speed="{CONFIG.SPEED_LIMIT_M_S}"/>
        
        <edge id="C2N" from="center" to="north" priority="{CONFIG.EDGE_PRIORITY}" numLanes="{num_lanes}" speed="{CONFIG.SPEED_LIMIT_M_S}"/>
        <edge id="C2S" from="center" to="south" priority="{CONFIG.EDGE_PRIORITY}" numLanes="{num_lanes}" speed="{CONFIG.SPEED_LIMIT_M_S}"/>
        <edge id="C2E" from="center" to="east" priority="{CONFIG.EDGE_PRIORITY}" numLanes="{num_lanes}" speed="{CONFIG.SPEED_LIMIT_M_S}"/>
        <edge id="C2W" from="center" to="west" priority="{CONFIG.EDGE_PRIORITY}" numLanes="{num_lanes}" speed="{CONFIG.SPEED_LIMIT_M_S}"/>
    </edges>
    """
    with open(edges_path, "w") as f:
        f.write(edges_content)

    # compile network using netconvert
    net_path = os.path.join(CONFIG.NETS_DIR, "custom_net.net.xml")
    cmd = [
        "netconvert",
        f"--node-files={nodes_path}",
        f"--edge-files={edges_path}",
        f"--output-file={net_path}",
        "--no-warnings",
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # generate stochastic route flows based on traffic intensity
    period = max(
        CONFIG.MIN_SPAWN_PERIOD,
        CONFIG.BASE_SPAWN_PERIOD / max(CONFIG.MIN_INTENSITY_FLOOR, traffic_intensity),
    )

    # allocate traffic stochastically based on route splits in configuration
    period_straight = period / CONFIG.SPLIT_STRAIGHT
    period_left = period / CONFIG.SPLIT_LEFT
    period_right = period / CONFIG.SPLIT_RIGHT

    rou_path = os.path.join(CONFIG.NETS_DIR, "custom_routes.rou.xml")
    rou_content = f"""
    <routes>
        <vType id="car" accel="{CONFIG.VEH_ACCEL}" decel="{CONFIG.VEH_DECEL}" sigma="{CONFIG.VEH_SIGMA}" length="{CONFIG.VEH_LENGTH}" minGap="{CONFIG.VEH_MIN_GAP}" maxSpeed="{CONFIG.SPEED_LIMIT_M_S}" guiShape="passenger"/>
        
        <!-- Straight Routes -->
        <route id="N2S" edges="N2C C2S"/>
        <route id="S2N" edges="S2C C2N"/>
        <route id="E2W" edges="E2C C2W"/>
        <route id="W2E" edges="W2C C2E"/>
        
        <!-- Left Turn Routes -->
        <route id="N2E" edges="N2C C2E"/>
        <route id="S2W" edges="S2C C2W"/>
        <route id="E2S" edges="E2C C2S"/>
        <route id="W2N" edges="W2C C2N"/>
        
        <!-- Right Turn Routes -->
        <route id="N2W" edges="N2C C2W"/>
        <route id="S2E" edges="S2C C2E"/>
        <route id="E2N" edges="E2C C2N"/>
        <route id="W2S" edges="W2C C2S"/>
        
        <!-- Straight Flows -->
        <flow id="flow_N2S" type="car" route="N2S" begin="0" end="{CONFIG.NUM_SECONDS}" period="{period_straight:.2f}"/>
        <flow id="flow_S2N" type="car" route="S2N" begin="0" end="{CONFIG.NUM_SECONDS}" period="{period_straight:.2f}"/>
        <flow id="flow_E2W" type="car" route="E2W" begin="0" end="{CONFIG.NUM_SECONDS}" period="{period_straight:.2f}"/>
        <flow id="flow_W2E" type="car" route="W2E" begin="0" end="{CONFIG.NUM_SECONDS}" period="{period_straight:.2f}"/>
        
        <!-- Left Turn Flows -->
        <flow id="flow_N2E" type="car" route="N2E" begin="0" end="{CONFIG.NUM_SECONDS}" period="{period_left:.2f}"/>
        <flow id="flow_S2W" type="car" route="S2W" begin="0" end="{CONFIG.NUM_SECONDS}" period="{period_left:.2f}"/>
        <flow id="flow_E2S" type="car" route="E2S" begin="0" end="{CONFIG.NUM_SECONDS}" period="{period_left:.2f}"/>
        <flow id="flow_W2N" type="car" route="W2N" begin="0" end="{CONFIG.NUM_SECONDS}" period="{period_left:.2f}"/>
        
        <!-- Right Turn Flows -->
        <flow id="flow_N2W" type="car" route="N2W" begin="0" end="{CONFIG.NUM_SECONDS}" period="{period_right:.2f}"/>
        <flow id="flow_S2E" type="car" route="S2E" begin="0" end="{CONFIG.NUM_SECONDS}" period="{period_right:.2f}"/>
        <flow id="flow_E2N" type="car" route="E2N" begin="0" end="{CONFIG.NUM_SECONDS}" period="{period_right:.2f}"/>
        <flow id="flow_W2S" type="car" route="W2S" begin="0" end="{CONFIG.NUM_SECONDS}" period="{period_right:.2f}"/>
    </routes>
    """
    with open(rou_path, "w") as f:
        f.write(rou_content)

    return net_path, rou_path


def build_env(config_name: str, reward_fn, use_gui: bool = False):
    """Construct and return a configured SumoEnvironment"""
    from sumo_rl import SumoEnvironment

    # dynamically compile the user-defined road network and route files
    custom_net_file, custom_route_file = generate_custom_scenario(
        CONFIG.NUM_LANES, CONFIG.TRAFFIC_INTENSITY
    )

    csv_prefix = os.path.join(CONFIG.DATA_DIR, f"{config_name}_sumo")
    kwargs: dict = dict(
        net_file=custom_net_file,
        route_file=custom_route_file,
        out_csv_name=csv_prefix,
        single_agent=True,
        use_gui=use_gui,
        num_seconds=CONFIG.NUM_SECONDS,
        delta_time=CONFIG.DELTA_TIME,
        yellow_time=CONFIG.YELLOW_TIME,
        min_green=CONFIG.MIN_GREEN,
        sumo_warnings=False,
        additional_sumo_cmd="--delay 90",
    )
    if reward_fn is not None:
        kwargs["reward_fn"] = reward_fn
    return SumoEnvironment(**kwargs)
