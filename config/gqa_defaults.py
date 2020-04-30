from yacs.config import CfgNode as CN

_C = CN()


# -----------------------------------------------------------------------------
# MAC
# -----------------------------------------------------------------------------
_C.MAC = CN()
_C.MAC.DIM = 512
_C.MAC.EMBD_DIM = 300
_C.MAC.TRAINED_EMBD_PATH = '/datasets/GoogleNews-vectors-negative300.bin'
_C.MAC.KNOW_DIM = 2048
_C.MAC.ENC_INPUT_DROPOUT = 0.15
_C.MAC.STEM_DROPOUT = 0.18
_C.MAC.STEM_FILTER_SIZE = 1
_C.MAC.STEM_PAD = 0
_C.MAC.QUESTION_DROPOUT = 0.08
_C.MAC.MEM_DROPOUT = 0.15
_C.MAC.READ_DROPOUT = 0.15
_C.MAC.OUTPUT_DROPOUT = 0.15
_C.MAC.GATE_DROPOUT = 0.0
_C.MAC.SELF_ATT = False
_C.MAC.MEMORY_GATE = False
_C.MAC.MEMORY_GATE_DIMS = []
_C.MAC.MEMORY_GATE_INPUT_CONT = True
_C.MAC.MEMORY_GATE_INPUT_MEM = False
_C.MAC.MEMORY_GATE_INPUT_Q = False
_C.MAC.MEMORY_GATE_BIAS = 1.0
_C.MAC.USE_ACT = False
_C.MAC.INIT_CNTRL_AS_Q = True
_C.MAC.MEMORY_VAR_DROPOUT = True


# -----------------------------------------------------------------------------
# ACT
# -----------------------------------------------------------------------------
_C.ACT = CN()
_C.ACT.MAX_ITER = 4
_C.ACT.SMOOTH = True
_C.ACT.HALT_TRAIN = False
_C.ACT.HALT_TEST = True
_C.ACT.PENALTY_TYPE = "unit"
_C.ACT.EXP_PENALTY_PHASE = 7
_C.ACT.LINEAR_PENALTY_PHASE = 5
_C.ACT.PENALTY_COEF = 1e-3
_C.ACT.MIN_PENALTY = 1.0
_C.ACT.BASELINE = CN()
_C.ACT.BASELINE.EPSILON = 1e-2


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.N_VOCAB = 2938


# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
_C.OUTPUT = CN()
_C.OUTPUT.DIM = 1846


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.BATCH_SIZE = 64
_C.DATALOADER.DATA_PERCENT = 1.0
_C.DATALOADER.FEATURES_PATH = "/datasets/GQA"
_C.DATALOADER.TRAIN_SPLIT = "all"
_C.DATALOADER.VAL_SPLIT = "all"


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 5
_C.SOLVER.LR = 1e-4
_C.SOLVER.GATE_LR = 0.0
_C.SOLVER.GRAD_CLIP = 8
_C.SOLVER.USE_SCHEDULER = True


# ---------------------------------------------------------------------------- #
# weight saving/loading options
# ---------------------------------------------------------------------------- #
_C.SAVE_PATH = 'checkpoint_GQA/mac_4'
_C.SAVE_ATTENTIONS = False
_C.LOAD = False
_C.LOAD_PATH = ""
_C.DEVICE = "cuda"


# ---------------------------------------------------------------------------- #
# Comet logging
# ---------------------------------------------------------------------------- #
_C.COMET = CN()
_C.COMET.EXPERIMENT_NAME = 'MAC 4'
_C.COMET.PROJECT_NAME = 'mac-gqa'
