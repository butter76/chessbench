"""PyTorch implementation of the training algorithm for action-value prediction."""

import copy
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
torch.set_default_dtype(torch.bfloat16)
torch.set_printoptions(profile="full")
import sys
from tqdm import tqdm

from searchless_chess.src import config as config_lib
from searchless_chess.src import data_loader
from searchless_chess.src import tokenizer
from searchless_chess.src import utils

class ChessDataset(Dataset):
    """PyTorch Dataset wrapper for chess data."""
    
    def __init__(self, config: config_lib.DataConfig):
        self.config = config
        self.data_iter = data_loader.build_data_loader(config).__iter__()
        
    def __getitem__(self, _):
        state, win_prob = next(self.data_iter)
        return torch.from_numpy(state), torch.from_numpy(win_prob).to(torch.bfloat16)
    
    def __len__(self):
        return sys.maxsize


class TransformerConfig:
    """PyTorch equivalent of transformer config."""
    
    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        embedding_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        widening_factor: int = 4,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.widening_factor = widening_factor
        self.dropout = dropout


class ChessTransformer(nn.Module):
    """PyTorch implementation of the transformer model."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.seq_len = tokenizer.SEQUENCE_LENGTH
        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seq_len, config.embedding_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * config.widening_factor,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(config.embedding_dim * self.seq_len, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x + self.pos_embedding
        x = self.transformer(x)

        # Flatten the output
        batch_size = x.shape[0]
        flat_x = x.reshape(batch_size, -1)  # [batch, seq_len * embedding_dim]

        return self.value_head(flat_x)


def train(
    train_config: config_lib.TrainConfig,
    model_config: TransformerConfig,
    device: Optional[str] = None,
) -> nn.Module:
    """Trains the model and returns it."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    model = ChessTransformer(model_config).to(device)
    model_ema = copy.deepcopy(model)
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        eps=1e-8,
    )
    
    # Setup data
    dataset = ChessDataset(train_config.data)
    dataloader = DataLoader(
        dataset,
        num_workers=train_config.data.worker_count,
    )

    # In the train function, modify the training loop:
    num_epochs = train_config.num_steps // train_config.ckpt_frequency
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(total=train_config.ckpt_frequency, desc=f'Epoch {epoch+1}/{num_epochs}')
        for step_in_epoch in range(train_config.ckpt_frequency):
            step = epoch * train_config.ckpt_frequency + step_in_epoch

            x, win_prob = next(iter(dataloader))
                
            x = x.to(device)
            win_prob = win_prob.to(device)

            x = x.squeeze(0)
            win_prob = win_prob.squeeze(0)
            
            # Forward pass
            value = model(x)
            value = value.squeeze(-1)


            
            # Compute loss
            criterion = torch.nn.MSELoss()
            loss = criterion(value, win_prob)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if train_config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    train_config.max_grad_norm
                )
            optimizer.step()
            
            # Update EMA
            with torch.no_grad():
                for p_ema, p in zip(model_ema.parameters(), model.parameters()):
                    p_ema.data.mul_(0.99).add_(p.data, alpha=0.01)

            pbar.update(1)
                
        pbar.close()

        # Checkpointing
        checkpoint = {
            'model': model.state_dict(),
            'model_ema': model_ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
        }
        checkpoint_dir = os.path.join(
            os.getcwd(),
            f'../checkpoints/local/action_value'
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, f'checkpoint_{step}.pt')
        )
    
    return model


def main():
    """Main training function."""
    # Set constants
    num_return_buckets = 128
    policy = 'action_value'
    
    # Create model config
    model_config = TransformerConfig(
        vocab_size=len(tokenizer._CHARACTERS),
        output_size=num_return_buckets,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        widening_factor=4,
        dropout=0.1,
    )
    
    # Create training config
    train_config = config_lib.TrainConfig(
        learning_rate=1e-4,
        data=config_lib.DataConfig(
            batch_size=128,
            shuffle=True,
            worker_count=0,  # 0 disables multiprocessing
            num_return_buckets=num_return_buckets,
            policy=policy,
            split='train',
        ),
        log_frequency=1,
        num_steps=20,
        ckpt_frequency=5,
        save_frequency=10,
    )
    
    # Train model
    model = train(
        train_config=train_config,
        model_config=model_config,
    )
    
    print("Training completed!")
    return model


if __name__ == "__main__":
    main()