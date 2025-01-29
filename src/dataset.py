"""PyTorch implementation of the training algorithm for action-value prediction."""

import torch
from torch.utils.data import Dataset
import os
import bagz
from itertools import cycle

from searchless_chess.src import config as config_lib
from searchless_chess.src import constants
from searchless_chess.src.data_loader import _process_fen

class ChessDataset(Dataset):
    """PyTorch Dataset wrapper for streaming chess data."""
    
    def __init__(self, config: config_lib.DataConfig):
        self.config = config
        data_path = os.path.join(
            os.getcwd(),
            f'../data/{config.split}/{config.policy}_data.bag'
        )
        self.data_source = bagz.BagDataSource(data_path)
        # Create an infinite iterator using cycle
        self.iterator = cycle(self.data_source)
        self.length = len(self.data_source)
        
    def __getitem__(self, _):
        # Get next item from infinite iterator
        element = next(self.iterator)
        fen, win_prob = constants.CODERS[self.config.policy].decode(element)
        state = _process_fen(fen)
        return torch.from_numpy(state), torch.tensor(win_prob, dtype=torch.bfloat16)
    
    def __len__(self):
        return self.length