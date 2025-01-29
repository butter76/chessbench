"""PyTorch implementation of the training algorithm for action-value prediction."""

import torch
from torch.utils.data import DataLoader, Dataset
torch.set_default_dtype(torch.bfloat16)
torch.set_printoptions(profile="full")
import sys
from tqdm import tqdm

from searchless_chess.src import config as config_lib
from searchless_chess.src import data_loader

class ChessDataset(Dataset):
    """PyTorch Dataset wrapper for chess data."""
    
    def __init__(self, config: config_lib.DataConfig):
        self.config = config
        self.data_iter = data_loader.build_data_loader(config).__iter__()
        
    def __getitem__(self, _):
        try:
            state, win_prob = next(self.data_iter)
        except StopIteration:
            self.data_iter = data_loader.build_data_loader(self.config).__iter__()
            state, win_prob = next(self.data_iter)
        return torch.from_numpy(state), torch.from_numpy(win_prob).to(torch.bfloat16)
    
    def __len__(self):
        return sys.maxsize