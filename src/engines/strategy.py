from enum import Enum


class MoveSelectionStrategy(str, Enum):
    VALUE = "value"
    AVS = "avs"
    AVS2 = "avs2"
    POLICY = "policy"
    OPT_POLICY_SPLIT = "opt_policy_split"
    NEGAMAX = "negamax"
    ALPHA_BETA = "alpha_beta"
    ALPHA_BETA_NODE = "alpha_beta_node"
    MCTS = "mcts"
    MTDF = "mtdf"
    PVS = "pvs" 