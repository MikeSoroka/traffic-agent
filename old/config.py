# ── Simulation ────────────────────────────────────────────────────────────────
SPAWN_PROB    = 0.30   # P(car arrives per lane per step)
MAX_QUEUE     = 12     # hard cap per lane
DEPART_RATE   = 1      # cars departing per step when light is green
MIN_GREEN     = 5      # minimum green-phase steps before switch is allowed
YELLOW_STEPS  = 3      # duration of yellow light (steps)

# ── State Discretisation ──────────────────────────────────────────────────────
# NS queue and EW queue are each binned into NUM_QUEUE_BINS buckets.
# Combined queue (e.g. N+S) is divided by QUEUE_BIN_SIZE then clamped.
QUEUE_BIN_SIZE = 3     # cars per bin  → 0-2→0, 3-5→1, 6-8→2, 9-11→3, 12+→4
NUM_QUEUE_BINS = 5
# How long the current green phase has been running
NUM_TIMER_BINS = 3     # 0 = just started (<MIN_GREEN), 1 = medium, 2 = long

# ── Q-Learning ────────────────────────────────────────────────────────────────
ALPHA         = 0.15   # learning rate
GAMMA         = 0.97   # discount factor
EPS_START     = 1.0
EPS_MIN       = 0.05
EPS_DECAY     = 0.992  # multiplied per episode

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_EPISODES = 200   # intentionally short; increase on the training machine
STEPS_PER_EP   = 500
EVAL_EPISODES  = 20    # greedy evaluation runs after training
FIXED_CYCLE    = 20    # steps per phase for the fixed-time baseline

QTABLE_PATH    = "q_table.npy"
PLOT_DIR       = "plots"

# ── Visualiser (pygame) ───────────────────────────────────────────────────────
SCREEN_W  = 1000
SCREEN_H  = 700
FPS       = 10
DEMO_STEPS = 800      # how many env steps to run in demo mode

# Intersection geometry (pixels)
CX, CY      = 400, 350   # intersection centre
ROAD_HALF   = 70          # half of road width  → road spans CX±70, CY±70
PANEL_X     = 810         # left edge of stats panel

COLORS = dict(
    bg        = (28,  28,  38),
    road      = (62,  62,  72),
    inter     = (75,  75,  85),
    marking   = (200, 200, 200),
    car_ns    = (90,  160, 240),
    car_ew    = (240, 140,  60),
    green     = (55,  215,  85),
    yellow    = (250, 210,  50),
    red       = (225,  50,  50),
    panel_bg  = (18,  18,  28),
    text      = (225, 225, 225),
    text_dim  = (130, 130, 150),
    bar_bg    = (45,  45,  58),
)
