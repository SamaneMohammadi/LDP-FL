"""
All settings for LDP-FL with CSS, following the paper
(Section "Usecase Description and Simulation Setting").
"""

# --- Federated setup ---------------------------------------------------------
NUM_CLIENTS = 91            # one client per CREMA-D speaker
CLIENTS_PER_ROUND = 50     # K selected each round (the paper uses 50; 7 for the attack study)
NUM_ROUNDS = 200           # global training epochs
BATCH_SIZE = 20            # local minibatch size B
LEARNING_RATE = 0.1        # eta (FedSGD)
LOCAL_EPOCHS = 1
VAL_SPLIT = 0.2            # 80/20 per client

# --- Model (MLP) -------------------------------------------------------------
INPUT_DIM = 988            # OpenSMILE emobase feature length
HIDDEN_DIMS = [256, 128]
NUM_CLASSES = 4            # neutral, sad, happy, angry
DROPOUT = 0.2
EMOTIONS = ["ANG", "HAP", "NEU", "SAD"]

# --- Local differential privacy (DP-SGD) -------------------------------------
# noise N(0, sigma^2 C^2 I); clip to C. Paper sweeps these.
CLIP_NORMS = [1.0, 2.0, 4.0]
NOISE_SCALES = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
DELTAS = [1e-3, 1e-4, 1e-5]
DEFAULT_CLIP = 2.0
DEFAULT_SIGMA = 3.0
DEFAULT_DELTA = 1e-5

# --- Client selection strategy (CSS) -----------------------------------------
# half the selected clients are the largest by sample size, half are random.
SELECTION = "css"          # "css" or "random"

# --- Model inversion attack (Algorithm 3) ------------------------------------
ATTACK_ITERS = 200         # T
ATTACK_LR = 0.1            # eta
ATTACK_PATIENCE = 100      # beta (stop if no improvement in this many steps)
ATTACK_GAMMA = 0.99        # cost threshold

# --- Reproducibility ---------------------------------------------------------
SEED = 8
